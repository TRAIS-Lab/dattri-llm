"""Gradient collection for verl PPO training using HookManager.

verl's actor worker runs inside a Ray remote process and requires FSDP on
CUDA.  This file shows both:

PART 1 — Standalone demo (runs immediately on CPU)
    Demonstrates that HookManager captures per-sample gradients from the exact
    forward+backward pattern that verl's update_actor() performs internally:
    model(input_ids, attention_mask) → compute_response_nll_loss → backward.

PART 2 — Full cluster integration (requires verl + ray + CUDA)
    GradientCapturingActorWorker subclasses verl's AsyncActorRolloutRefWorker
    and wires up HookManager in init_model().  GradientCapturingTaskRunner
    registers this worker class, then passes it to run_ppo() via the
    task_runner_class argument — the only entry point that avoids monkey-patching.

Run (standalone demo, no GPU needed):
    conda activate dattri
    cd /scratch/shixuanl/dattri_llm
    python examples/example_verl.py

Run (full cluster):
    # Requires: pip install verl ray omegaconf + CUDA
    # Pass --full flag or set VERL_FULL=1
"""

from __future__ import annotations

import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.utils import hash_sample
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, OffloadCallback
from dattri_llm.trainers.verl import VERL_GPT2_MLP_PATTERNS, compute_response_nll_loss

# ============================================================================
# PART 1 — Standalone demo
#
#    Uses sshleifer/tiny-gpt2 (2-layer, CPU-safe) to reproduce what
#    GradientCapturingActorWorker does in each update_actor() call:
#      1. Forward pass with input_ids + attention_mask
#      2. compute_response_nll_loss restricts loss to response tokens
#      3. backward() → HookManager captures activations + output gradients
# ============================================================================

MODEL_ID = "sshleifer/tiny-gpt2"
print(f"Loading {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).train()

torch.manual_seed(0)
B, T_prompt, T_response = 2, 8, 8
T = T_prompt + T_response

prompts        = torch.randint(0, tokenizer.vocab_size, (B, T_prompt))
responses      = torch.randint(0, tokenizer.vocab_size, (B, T_response))
input_ids      = torch.cat([prompts, responses], dim=1)   # (B, T)
attention_mask = torch.ones(B, T, dtype=torch.long)
response_mask  = torch.zeros(B, T, dtype=torch.long)
response_mask[:, T_prompt:] = 1                            # mask out prompt tokens

with tempfile.TemporaryDirectory() as grad_dir:
    hook_cfg = HookManagerConfig(
        recording_type="per_sample",
        hook_types=["mlp_io"],
        mlp_name_patterns=VERL_GPT2_MLP_PATTERNS,
    )
    manager   = GradientFileManager(grad_dir)
    offload   = OffloadCallback(every_n_steps=1, file_manager=manager)
    collector = HookManager(model, config=hook_cfg, callbacks=[offload])

    print(f"\nHooked {len(collector.layer_names)} GPT-2 MLP layer(s):")
    for name in collector.layer_names:
        print(f"  {name}")

    N_STEPS = 2
    print(f"\nSimulating {N_STEPS} update_actor() steps …")
    with collector.collect():
        for step in range(N_STEPS):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = compute_response_nll_loss(
                outputs.logits, input_ids, response_mask.float()
            )
            loss.backward()
            model.zero_grad()
            print(f"  step {step}  response-masked NLL = {loss.item():.4f}")

    print(f"\nSteps collected : {collector.steps_collected}")
    print(f"Index entries   : {len(manager.index)} unique sample hashes")
    collector.remove()

    query   = {"input_ids": input_ids, "attention_mask": attention_mask}
    records = manager.load_all(query)
    print(f"Records loaded  : {len(records)} (B={B} × {N_STEPS} steps)")

    h0      = hash_sample(query, sample_idx=0)
    recs_s0 = manager.load_all_by_hash(h0)
    print(f"Sample 0 appears in {len(recs_s0)} record(s)")

    if len(recs_s0) >= 2:
        g0  = recs_s0[0].gradient.materialize().aggregate(dim="token", mode="sum")
        g1  = recs_s0[1].gradient.materialize().aggregate(dim="token", mode="sum")
        sim = g0.similarity(g1, metric="cosine", reduce="layer")
        print("Cosine similarity sample 0 — step 0 vs step 1:")
        for layer, val in sim.items():
            print(f"  {layer}  cos_sim={val.item():.6f}")

# ============================================================================
# PART 2 — Full cluster integration  (requires verl + ray + CUDA)
# ============================================================================

try:
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import TaskRunner, run_ppo
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False

if _VERL_AVAILABLE:
    from dattri_llm.trainers.verl import VERL_LLAMA_MLP_PATTERNS

    # ---------------------------------------------------------------------- #
    # Custom actor worker                                                      #
    # ---------------------------------------------------------------------- #

    class GradientCapturingActorWorker(AsyncActorRolloutRefWorker):
        """verl actor worker that wires up HookManager after init_model().

        Hooks are registered on self.actor_module (the unwrapped policy
        nn.Module extracted from FSDP by the base class).  The collect()
        context is opened once and kept alive for the worker's full lifetime.
        """

        def __init__(self, *args,
                     gradient_save_dir: str = "/tmp/gradients",
                     offload_every_n_steps: int = 32,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._gradient_save_dir = gradient_save_dir
            self._offload_every_n_steps = offload_every_n_steps
            self._collector: HookManager | None = None

        def init_model(self) -> None:
            super().init_model()       # builds + shards the actor model

            hook_cfg = HookManagerConfig(
                recording_type="per_sample",
                hook_types=["mlp_io"],
                mlp_name_patterns=VERL_LLAMA_MLP_PATTERNS,
            )
            manager = GradientFileManager(self._gradient_save_dir)
            offload = OffloadCallback(
                every_n_steps=self._offload_every_n_steps,
                file_manager=manager,
            )
            self._collector = HookManager(
                self.actor_module,     # unwrapped nn.Module, not the FSDP wrapper
                config=hook_cfg,
                callbacks=[offload],
            )
            self._collect_ctx = self._collector.collect()
            self._collect_ctx.__enter__()

        def __del__(self) -> None:
            if self._collector is not None:
                try:
                    self._collect_ctx.__exit__(None, None, None)
                    self._collector.remove()
                except Exception:
                    pass

    # ---------------------------------------------------------------------- #
    # Custom TaskRunner — registers GradientCapturingActorWorker              #
    # ---------------------------------------------------------------------- #

    GRAD_DIR = tempfile.mkdtemp(prefix="verl_grads_")

    class GradientCapturingTaskRunner(TaskRunner):
        """TaskRunner that swaps in GradientCapturingActorWorker."""

        def add_actor_rollout_worker(self, config):
            actor_cls, rg_cls = super().add_actor_rollout_worker(config)

            from verl.trainer.ppo.ray_trainer import Role
            for role in list(self.role_worker_mapping):
                if role in (Role.ActorRollout, Role.ActorRolloutRef):
                    self.role_worker_mapping[role] = ray.remote(
                        GradientCapturingActorWorker
                    )
            return GradientCapturingActorWorker, rg_cls

    # ---------------------------------------------------------------------- #
    # Minimal PPO config + run_ppo()                                          #
    #                                                                         #
    # Load the ppo_trainer.yaml shipped with verl as a base (it contains all #
    # required skeleton fields), then override only the fields we need.       #
    # ---------------------------------------------------------------------- #

    import verl
    _verl_pkg = os.path.dirname(verl.__file__)
    # Use the pre-resolved config (all defaults already merged).
    _base_yaml = os.path.join(_verl_pkg, "trainer", "config", "_generated_ppo_trainer.yaml")

    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(_base_yaml)

    with open_dict(cfg):
        cfg.algorithm.adv_estimator                              = "grpo"
        cfg.data.train_files                                     = "/path/to/train.parquet"
        cfg.data.val_files                                       = "/path/to/val.parquet"
        cfg.data.train_batch_size                                = 256
        cfg.actor_rollout_ref.model.path                        = "Qwen/Qwen2.5-1.5B-Instruct"
        cfg.actor_rollout_ref.rollout.name                      = "vllm"
        cfg.actor_rollout_ref.rollout.n                         = 8
        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu    = 4
        cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 4
        cfg.trainer.total_epochs                                 = 1
        cfg.trainer.n_gpus_per_node                             = 8
        cfg.trainer.default_local_dir                            = os.path.join(GRAD_DIR, "checkpoints")

    # run_ppo with our custom TaskRunner — no monkey-patching required.
    run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(GradientCapturingTaskRunner))

    # After training, load gradient records from all ranks.
    manager = GradientFileManager(GRAD_DIR)
    print(f"\nPost-training: {len(manager.index)} unique sample hashes indexed.")

else:
    print("\nverl not installed — skipping full cluster integration.")
    print("Install with:  pip install verl ray omegaconf")
    print("Requires CUDA + a running Ray cluster for the full PPO run.")
