"""Gradient collection for OLMo training using HookManager.

OLMo's feed-forward projections (ff_proj / ff_out inside
transformer.blocks.<i>) do not match the default MLP-keyword heuristic.
OLMO_MLP_PATTERNS supplies the correct regex strings for HookManagerConfig
so that the collector targets exactly those layers.

This example builds a tiny OLMo model entirely in Python (no config YAML),
constructs a real OLMo Trainer, and wraps Trainer.fit() with a
HookManager collect() context so every backward pass is captured.

Run (with ai2-olmo installed):
    conda activate dattri
    cd /scratch/shixuanl/dattri_llm
    python examples/example_olmo.py
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# ============================================================================
# OLMo import guard
# ============================================================================

try:
    from olmo.config import DistributedStrategy, ModelConfig, TrainConfig
    from olmo.data.iterable_dataset import IterableDataset as OLMoIterableDataset
    from olmo.model import OLMo
    from olmo.optim import build_optimizer, build_scheduler
    from olmo.train import Trainer
except ImportError:
    raise SystemExit(
        "ai2-olmo is required for this example.\n"
        "Install with:  pip install ai2-olmo\n"
        "or see:        https://github.com/allenai/OLMo"
    )

from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.utils import hash_sample
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, OffloadCallback
from dattri_llm.trainers.olmo import OLMO_MLP_PATTERNS

# ============================================================================
# 1. Single-process distributed init
#
#    OLMo's Trainer calls dist.reduce() inside train_step(), so a process
#    group must exist even for single-GPU / CPU runs.
# ============================================================================

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
dist.init_process_group("gloo", rank=0, world_size=1)

# ============================================================================
# 2. Hard-coded tiny TrainConfig (no YAML file needed)
#
#    Minimal model: 2 layers, d_model=64, 2 heads — runs on CPU in seconds.
#    Checkpointing is disabled (save_num_checkpoints_to_keep=0) so the example
#    needs no persistent save_folder.
# ============================================================================

VOCAB, SEQ, BATCH = 256, 32, 2

cfg = TrainConfig(
    model=ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
    ),
    save_folder=tempfile.mkdtemp(prefix="olmo_ckpt_"),
    max_duration=4,                       # 4 steps = 2 epochs over 4 samples
    device_train_batch_size=BATCH,
    global_train_batch_size=BATCH,
    precision="fp32",
    distributed_strategy=DistributedStrategy.ddp,
    save_interval=None,                   # no periodic checkpoint
    save_interval_unsharded=None,
    save_interval_ephemeral=None,
    save_num_checkpoints_to_keep=0,       # no final checkpoint
)

# ============================================================================
# 3. Build model, dist_model, optimizer, scheduler, and dataloader
#
#    OLMo's Trainer is a dataclass; pass pre-built components.
#    OLMoIterableDataset wraps any sequence of sample dicts and provides the
#    total_size attribute that Trainer.fit() requires.
# ============================================================================

device = torch.device("cpu")

model     = OLMo(cfg.model).to(device)
dist_model = DDP(model)                   # single-rank DDP for dist.reduce
optim     = build_optimizer(cfg, model)
scheduler = build_scheduler(cfg)

# Synthetic token sequences — replace with real data for actual training.
torch.manual_seed(0)
sequences = [{"input_ids": torch.randint(0, VOCAB, (SEQ,))} for _ in range(4)]
ds         = OLMoIterableDataset(sequences, global_batch_size=BATCH,
                                  shuffle=False, drop_last=True)
loader     = DataLoader(ds, batch_size=BATCH)

trainer = Trainer(
    cfg=cfg,
    model=model,
    dist_model=dist_model,
    optim=optim,
    scheduler=scheduler,
    train_loader=loader,
    device=device,
    evaluators=[],
)

# ============================================================================
# 4. Wire up HookManager + OffloadCallback, then call trainer.fit()
#
#    HookManager is attached to the unwrapped model (not dist_model) so hooks
#    see the real Linear modules regardless of DDP wrapping.
#    collector.collect() wraps trainer.fit() — every backward pass inside the
#    OLMo training loop is captured automatically.
# ============================================================================

with tempfile.TemporaryDirectory() as grad_dir:
    hook_cfg = HookManagerConfig(
        recording_type="per_sample",
        hook_types=["mlp_io"],
        mlp_name_patterns=OLMO_MLP_PATTERNS,
    )
    manager   = GradientFileManager(grad_dir)
    offload   = OffloadCallback(every_n_steps=1, file_manager=manager)
    collector = HookManager(model, config=hook_cfg, callbacks=[offload])

    print(f"OLMO_MLP_PATTERNS hooks {len(collector.layer_names)} layer(s):")
    for name in collector.layer_names:
        print(f"  {name}")

    print("\nRunning trainer.fit() with gradient collection …")
    with collector.collect():
        trainer.fit()

    print(f"\nSteps collected : {collector.steps_collected}")
    print(f"Index entries   : {len(manager.index)} unique sample hashes")
    collector.remove()

    # ========================================================================
    # 5. Load and inspect saved records
    #
    #    OLMo's model_forward receives {input_ids} (attention_mask is optional
    #    and not passed in train_batch).  Build the query with the same key.
    # ========================================================================

    all_ids = torch.stack([s["input_ids"] for s in sequences])
    query   = {"input_ids": all_ids}
    records = manager.load_all(query)
    print(f"Records loaded  : {len(records)}")

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

dist.destroy_process_group()
