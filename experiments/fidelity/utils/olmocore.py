"""The fidelity setting under OLMo-core's trainer.

An OLMo model is continually pretrained on the setting's WikiText-2 blocks
(512 blocks of 128 tokens, one epoch at batch 32, AdamW with the setting's
learning-rate schedule) by OLMo-core's ``Trainer``: its transformer train
module, its numpy data loader and its checkpoints.  One process trains
without data-parallel wrapping; several shard the model with FSDP (one
process per GPU, launched with ``torchrun``).  Everything is fp32 with TF32
off and deterministic kernels.  A training block is left out through the
data loader's label mask: the loss of a process is the mean over its
unmasked tokens.  The trainer's dry-run batch is disabled, so the steps
under the hooks are the training steps.

    torchrun --nproc-per-node 4 utils/truth/olmo.py prepare --scale olmo2-1b
    torchrun --nproc-per-node 4 utils/truth/olmo.py train --scale olmo2-1b --left-out none --tag ref
    torchrun --nproc-per-node 4 utils/truth/olmo.py train --scale olmo2-1b --left-out 7 --tag 7
    python utils/truth/olmo.py selected --scale olmo2-1b      # the 50 blocks to leave out
    python utils/truth/olmo.py truth --scale olmo2-1b         # runs -> results/<scale>/lr1e-05_seed0/matrices.pt
    python utils/attribution/olmo.py export --scale olmo3-7b --microbatch 32
    python utils/attribution/olmo.py adamw --scale olmo3-7b --microbatch 32 --hook-layers all
    python utils/attribution/olmo.py adamw-score --scale olmo3-7b

``prepare`` writes the token files and the initial checkpoint (the released
weights in OLMo-core's format) under ``$OLMOCORE_WORK_DIR/<scale>`` (default
``results/work/<scale>``).  ``train`` trains once from that checkpoint
(``--left-out none``, or the index of the block left out; one process group
per run, as the trainer holds one train module per process) and saves the
validation losses under ``runs/<tag>``.  ``compare`` prints the difference
of the validation losses between the first tag and the others.  ``truth``
collects the reference run (tag ``ref``) and the run of every selected block
(tag = its index) into the ground truth the other drivers read: ``tsloo``
(the change of each validation loss), the ``(block, step)`` pairs, and the
reference losses.  ``--microbatch`` is the number of blocks per
forward/backward of a process (the step is the same sum of gradients).

``export`` trains once and saves the trained model as a Hugging Face
checkpoint (``runs/final/final_hf``) for the drivers that take
``--final-model`` (dattri-llm's EK-FAC, Bergson's EK-FAC and TrackStar).

``adamw`` and ``adamw-score`` are dattri-llm's masked AdamW-influence.  The
trainer is not modified: ``trainer.fit()`` runs inside ``HookManager.collect()``
with the optimizer-state recorder, which stores the masked per-sample
gradients of every training step and the optimizer's moments (``--n-masks``
masks of ``--mask-dim`` coordinates per layer; ``--hook-layers all`` hooks
every layer a default HookManager hooks, ``blocks`` the linear layers of the
transformer blocks).  The query gradients are captured the same way on the
trained model, each process taking a share of the queries.  ``adamw-score``
is one process: it scores the stores with ``AdamWInfluenceAttributor`` through
the protocol of ``utils/protocol.py`` and writes ``results/<scale>/
lr1e-05_seed0_masked<tag>``.  One micro-batch per step is required here: the
recorder pairs the optimizer's moments with the hooks' steps.

``capture`` and ``ekfac`` store the factorized gradients of the training
blocks and the queries on the trained model (every linear layer of the
transformer blocks, through a rank-64 LoGra projection with ``--projection
64``), each process taking a share, and score the stores with
``EKFACAttributor.attribute_from_cache`` in one process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf import load_hf_model, save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, Scheduler
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
N_TRAIN, BLOCK, N_VAL, BATCH, N_SELECTED = 512, 128, 256, 32, 50
MODELS = {
    "olmo2-1b": ("allenai/OLMo-2-0425-1B", "olmo2_1B_v2"),
    "olmo2-7b": ("allenai/OLMo-2-1124-7B", "olmo2_7B"),
    "olmo3-7b": ("allenai/Olmo-3-1025-7B", "olmo3_7B"),
}


def work_dir(scale: str) -> pathlib.Path:
    return pathlib.Path(os.environ.get("OLMOCORE_WORK_DIR", RESULTS / "work")) / scale


def blocks(model_id: str, split: str, n: int, seed: int) -> torch.Tensor:
    """``n`` random 128-token WikiText-2 blocks (as ``settings._blocks``)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:  # fall back to the namespaced dataset id
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = torch.tensor(tok("\n\n".join(ds["text"]))["input_ids"])
    n_blocks = ids.numel() // BLOCK
    ids = ids[: n_blocks * BLOCK].view(n_blocks, BLOCK)
    gen = torch.Generator().manual_seed(seed)
    return ids[torch.randperm(n_blocks, generator=gen)[:n]]


def deterministic() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def model_config(scale: str) -> TransformerConfig:
    factory = getattr(TransformerConfig, MODELS[scale][1])
    return factory(vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
                   dtype=DType.float32, attn_backend=AttentionBackendName.torch)


@dataclasses.dataclass
class SettingSchedule(Scheduler):
    """The schedule of the fidelity setting (``settings.lr_factor``): linear
    warmup over the first 10% of the steps, then linear decay that reaches
    zero after the last step.  The trainer counts steps from 1."""

    def get_lr(self, initial_lr, current: int, t_max: int):
        step, warm = current - 1, round(0.1 * t_max)
        if step < warm:
            return initial_lr * (step + 1) / max(1, warm)
        return initial_lr * max(0.0, (t_max - step) / max(1, t_max - warm))


def train_module_config(lr: float, n_steps: int, microbatch: int = BATCH) -> TransformerTrainModuleConfig:
    """*microbatch*: blocks per forward/backward on a rank (the step is the same sum of gradients).
    One process trains without data-parallel wrapping; several shard with FSDP."""
    world = dist.get_world_size()
    return TransformerTrainModuleConfig(
        rank_microbatch_size=min(microbatch, BATCH // world) * BLOCK,
        max_sequence_length=BLOCK,
        optim=AdamWConfig(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True),
        scheduler=SettingSchedule(),
        dp_config=None if world == 1 else TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=None, reduce_dtype=DType.float32),
        compile_model=False, autocast_precision=None, max_grad_norm=None,
    )


class Record(Callback):
    """The block indices and the loss of every step."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []
        self.losses: list[float] = []

    def pre_step(self, batch) -> None:
        index = batch["index"].to(self.trainer.device)
        gathered = [torch.empty_like(index) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, index)
        self.batches.append(torch.cat(gathered).tolist())

    def post_train_batch(self) -> None:
        self.losses.append(float(self.trainer.get_metric("train/CE loss")))


def prepare(a: argparse.Namespace) -> None:
    model_id = MODELS[a.scale][0]
    work = work_dir(a.scale)
    if get_rank() == 0:
        work.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", N_TRAIN), ("validation", N_VAL)):
            ids = blocks(model_id, split, n, a.seed)
            ids.numpy().astype(np.uint32).tofile(work / f"{split}.npy")
    dist.barrier()
    model = model_config(a.scale).build(init_device="meta")
    train_module = train_module_config(a.lr, 16, a.microbatch).build(model)
    state = get_model_state_dict(train_module.model)
    load_hf_model(model_id, state, num_embeddings=model.vocab_size, work_dir=str(work / "hf"))
    set_model_state_dict(train_module.model, state)
    save_model_and_optim_state(str(work / "init" / "model_and_optim"), train_module.model,
                               save_overwrite=True)
    if get_rank() == 0:
        print(f"prepared {work}", flush=True)


@torch.no_grad()
def val_losses(model, x_va: torch.Tensor) -> torch.Tensor:
    """Per-block mean-token loss of the validation blocks (every rank runs all of them)."""
    model.eval()
    out = []
    for i in range(0, x_va.shape[0], BATCH):
        ids = x_va[i : i + BATCH]
        labels = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=-100)
        ce = model(ids, labels=labels, ignore_index=-100, loss_reduction="none",
                   return_logits=False).ce_loss
        out.append(ce.view(ids.shape[0], -1)[:, :-1].double().mean(1))
    model.train()
    return torch.cat(out).cpu()


LINEAR_LAYERS = r"^blocks\.\d+\.(attention\.w_(q|k|v|out)|feed_forward\.w[123])$"
DAMPING = 0.1  # of the mean corrected eigenvalue, per layer


def build_trainer(a: argparse.Namespace, run: pathlib.Path, callbacks: dict):
    """The trainer of one run over the label mask in *run*, from the initial checkpoint."""
    work = work_dir(a.scale)
    seed_all(a.seed)
    model = model_config(a.scale).build(init_device="meta")
    train_module = train_module_config(a.lr, N_TRAIN // BATCH, a.microbatch).build(model)
    dataset = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig.dolma2(), paths=[str(work / "train.npy")],
        label_mask_paths=[str(run / "label_mask.npy")], sequence_length=BLOCK,
        dtype=NumpyDatasetDType.uint32, work_dir=str(run / "dataset"),
    ).build()
    data_loader = NumpyDataLoaderConfig(global_batch_size=BATCH * BLOCK, seed=a.seed, num_workers=0,
                                        work_dir=str(run / "loader")).build(
        dataset, dp_process_group=train_module.dp_process_group)
    config = TrainerConfig(save_folder=str(run / "save"), work_dir=str(run / "trainer"),
                           max_duration=Duration.epochs(1), metrics_collect_interval=1,
                           no_checkpoints=True, no_evals=True)
    for name, callback in callbacks.items():
        config = config.with_callback(name, callback)
    # Without a mock batch the trainer skips its dry-run batch, which would
    # otherwise be captured as a step of its own.
    data_loader.get_mock_batch = _no_mock_batch
    trainer = config.build(train_module, data_loader)
    load_model_and_optim_state(str(work / "init" / "model_and_optim"), train_module.model)
    return trainer, train_module


def _no_mock_batch():
    raise NotImplementedError


def write_mask(run: pathlib.Path, left_out: int | None) -> None:
    if get_rank() == 0:
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        mask = np.ones((N_TRAIN, BLOCK), dtype=np.bool_)
        if left_out is not None:
            mask[left_out] = False
        mask.tofile(run / "label_mask.npy")
    dist.barrier()


def load_blocks(scale: str, split: str, n: int) -> torch.Tensor:
    ids = np.fromfile(work_dir(scale) / f"{split}.npy", dtype=np.uint32).astype(np.int64)
    return torch.from_numpy(ids).view(n, BLOCK)


def sample_hash(ids: torch.Tensor) -> str:
    """The identity of a block in the stores: the content hash of its labels
    (``sample_hash_fields=["labels"]``; the trainer also passes the block's
    index and its label mask to the model, which are not part of it)."""
    from dattri_llm.utils.hashing import hash_sample

    return hash_sample({"labels": torch.nn.functional.pad(ids[1:], (0, 1), value=-100)})


def method_name(a: argparse.Namespace) -> str:
    return "ekfac_k64" if a.projection == "64" else "ekfac"


def export_hf(a: argparse.Namespace, run: pathlib.Path, model) -> None:
    """The trained model as a Hugging Face checkpoint at ``<run>/final_hf``."""
    save_hf_model(str(run / "final_hf"), get_model_state_dict(model), model, save_overwrite=True)
    if get_rank() == 0:  # the export leaves the context length unset (-1); take the released model's
        from transformers import AutoConfig

        path = run / "final_hf" / "config.json"
        config_json = json.loads(path.read_text())
        config_json["max_position_embeddings"] = AutoConfig.from_pretrained(
            MODELS[a.scale][0]).max_position_embeddings
        path.write_text(json.dumps(config_json, indent=2))


def export(a: argparse.Namespace) -> None:
    run = work_dir(a.scale) / "runs" / "final"
    write_mask(run, None)
    trainer, train_module = build_trainer(a, run, {})
    trainer.fit()
    export_hf(a, run, train_module.model)
    if get_rank() == 0:
        print(f"exported {run / 'final_hf'}", flush=True)


def capture(a: argparse.Namespace) -> None:
    from dattri_llm.gradient.callbacks import OffloadCallback
    from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
    from dattri_llm.gradient.storage_manager import GradientStorageManager

    run = work_dir(a.scale) / "runs" / method_name(a)
    write_mask(run, None)
    proj = None if a.projection == "none" else {"__default__": {
        "style": "logra", "proj_dim": 64, "proj_max_batch_size": 32, "proj_type": "rademacher", "proj_seed": 0}}
    config = HookManagerConfig(linear_io=[LINEAR_LAYERS], projection_kwargs=proj, capture_style="factorized")
    record = Record()
    trainer, train_module = build_trainer(a, run, {"record": record})
    model = train_module.model
    stores = {side: GradientStorageManager(str(run / side)) for side in ("train", "test")}

    t0 = time.time()
    trainer.fit()
    torch.cuda.synchronize()
    train_s = time.time() - t0
    export_hf(a, run, model)

    # Both sides on the trained model: each rank captures its share of the blocks.
    rank, world = get_rank(), dist.get_world_size()
    x_q = load_blocks(a.scale, "validation", N_VAL)[: a.n_queries]
    sides = {"train": (load_blocks(a.scale, "train", N_TRAIN), 8), "test": (x_q, a.eval_batch)}
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    for side, (x, batch) in sides.items():
        mine = x[rank::world].to(trainer.device)
        hooks = HookManager(model, config=config, callbacks=[OffloadCallback(1, stores[side])],
                            sample_hash_fields=["labels"])
        with hooks.collect():
            for i in range(0, mine.shape[0], batch):
                ids = mine[i : i + batch]
                labels = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=-100)
                loss = model(ids, labels=labels, ignore_index=-100, loss_reduction="none",
                             return_logits=False).loss  # differentiable (``ce_loss`` is detached)
                (loss.view(ids.shape[0], -1)[:, :-1].mean(1).sum()).backward()
                model.zero_grad(set_to_none=True)
        hooks.remove()
    torch.cuda.synchronize()
    capture_s = time.time() - t1
    dist.barrier()
    if rank == 0:
        (run / "capture.json").write_text(json.dumps({
            "world_size": world, "train_s": round(train_s, 1), "capture_s": round(capture_s, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "train_losses": record.losses}))
        print(f"trained {train_s:.1f}s, captured both sides {capture_s:.1f}s", flush=True)


def ekfac(a: argparse.Namespace) -> None:
    from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor
    from dattri_llm.attribution.arguments import AttributionArguments
    method = method_name(a)
    run = work_dir(a.scale) / "runs" / method
    captured = json.loads((run / "capture.json").read_text())
    x_q = load_blocks(a.scale, "validation", N_VAL)[: a.n_queries]
    x_tr = load_blocks(a.scale, "train", N_TRAIN)
    args = AttributionArguments(output_dir=str(run / "attribution"), per_device_train_batch_size=8,
                                per_device_eval_batch_size=a.eval_batch, dataloader_pin_memory=False)
    t1 = time.time()
    score = EKFACAttributor(args).attribute_from_cache(
        str(run / "train"), str(run / "test"), damping=DAMPING, relative_damping=True,
        factor_cache_residency=a.factor_cache_residency, loop_over_test=a.loop_over_test,
        preconditioned_test_cache_residency="disk" if a.loop_over_test else None)
    ids, matrix = score.agnostic_matrix()
    torch.cuda.synchronize()
    score_s = time.time() - t1
    pos = {h: i for i, h in enumerate(ids)}
    order = [pos[sample_hash(x_tr[i])] for i in range(N_TRAIN)]
    cols = [score.test_ids.index(sample_hash(x_q[q])) for q in range(a.n_queries)]
    scores = matrix.cpu().float()[order][:, cols]
    out = RESULTS / a.scale / f"lr{a.lr:g}_seed{a.seed}_{method}"
    out.mkdir(parents=True, exist_ok=True)
    truth = torch.load(RESULTS / a.scale / f"lr{a.lr:g}_seed{a.seed}" / "matrices.pt", weights_only=False)
    pred = torch.stack([scores[i] for i, _ in truth["pairs"]])
    rho = spearman(pred, truth["tsloo"][:, : a.n_queries])
    result = {"name": a.scale, "seed": a.seed, "lr": a.lr, "model": MODELS[a.scale][0], "trainer": "olmo-core",
              "world_size": captured["world_size"], "method": method, "n_queries": a.n_queries, "damping": DAMPING,
              "projection": a.projection, method: float(rho.mean()), f"{method}_std_over_val": float(rho.std()),
              "train_s": captured["train_s"], "capture_s": captured["capture_s"],
              "score_s": round(score_s, 1), "attribute_s": round(captured["capture_s"] + score_s, 1),
              "capture_peak_gb": captured["peak_gb"],
              "score_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    torch.save({"pred": pred, "pairs": truth["pairs"]}, out / "matrices.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


class RecordMoments(Callback):
    """Records the optimizer's moments after every step (the hooks' step
    count runs from 0; the trainer's from 1)."""

    def __init__(self, recorder) -> None:
        self.recorder = recorder

    def post_train_batch(self) -> None:
        self.recorder.record_post(self.trainer.global_step - 1)


def adamw(a: argparse.Namespace) -> None:
    """Masked AdamW-influence, capture: the trainer runs inside the hooks with
    the optimizer recorder; the queries are captured on the trained model."""
    from dattri_llm.gradient.callbacks import OffloadCallback, OptimizerStateCallback
    from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
    from dattri_llm.gradient.optimizer_state import OptimizerSnapshot
    from dattri_llm.gradient.storage_manager import GradientStorageManager

    run = work_dir(a.scale) / "runs" / f"adamw{a.mask_tag}"
    write_mask(run, None)
    trainer, train_module = build_trainer(a, run, {"record": Record()})
    model, optimizer = train_module.model, train_module.optim
    selector = REGISTER_ALL if a.hook_layers == "all" else [LINEAR_LAYERS]
    hm = HookManager(model, config=HookManagerConfig(linear_io=selector))
    names = list(hm.layer_names)
    hm.remove()
    widths = {n: OptimizerSnapshot(model, optimizer).width(n) for n in names}
    projection = {n: {"style": "mask", "proj_dim": min(widths[n], a.n_masks * a.mask_dim), "proj_seed": a.seed}
                  for n in names}
    config = HookManagerConfig(linear_io=selector, projection_kwargs=projection)
    rank, world = get_rank(), dist.get_world_size()
    if rank == 0:
        for d in ("train_grads", "test_grads", "masks", "scores"):
            shutil.rmtree(run / d, ignore_errors=True)
        (run / "projection.json").write_text(json.dumps({"projection": projection, "hook_layers": a.hook_layers,
                                                         "n_masks": a.n_masks, "mask_dim": a.mask_dim}))
    dist.barrier()
    recorder = OptimizerStateCallback(model, optimizer, projection_kwargs=projection)
    trainer.add_callback("moments", RecordMoments(recorder))
    store = GradientStorageManager(str(run / "train_grads"))
    torch.cuda.synchronize()
    t0 = time.time()
    hooks = HookManager(model, config=config, callbacks=[OffloadCallback(1, store), recorder],
                        sample_hash_fields=["labels"])
    with hooks.collect():
        trainer.fit()
    hooks.remove()
    dynamics = recorder.dynamics()
    torch.save(dynamics, run / "train_grads" / f"adamw_dynamics_rank{rank}.pt")
    train_s = time.time() - t0

    x_q = load_blocks(a.scale, "validation", N_VAL)[: a.n_queries]
    mine = x_q[rank::world].to(trainer.device)
    hooks = HookManager(model, config=config,
                        callbacks=[OffloadCallback(1, GradientStorageManager(str(run / "test_grads")))],
                        sample_hash_fields=["labels"])
    with hooks.collect():
        for i in range(0, mine.shape[0], a.eval_batch):
            ids = mine[i : i + a.eval_batch]
            labels = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=-100)
            loss = model(ids, labels=labels, ignore_index=-100, loss_reduction="none", return_logits=False).loss
            (loss.view(ids.shape[0], -1)[:, :-1].mean(1).sum()).backward()
            model.zero_grad(set_to_none=True)
    hooks.remove()
    torch.cuda.synchronize()
    capture_s = time.time() - t0
    dist.barrier()
    if rank == 0:
        (run / "capture.json").write_text(json.dumps({
            "world_size": world, "train_s": round(train_s, 1), "capture_s": round(capture_s, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "n_steps": len(dynamics)}))
        print(f"adamw captured: trajectory under hooks {train_s:.1f}s, with the queries {capture_s:.1f}s, "
              f"{len(dynamics)} steps of dynamics, {len(names)} layers", flush=True)


def adamw_score(a: argparse.Namespace) -> None:
    """Masked AdamW-influence, scoring (one process) from the capture of ``adamw``."""
    import types

    import protocol

    run = work_dir(a.scale) / "runs" / f"adamw{a.mask_tag}"
    meta = json.loads((run / "projection.json").read_text())
    captured = json.loads((run / "capture.json").read_text())
    # The recorded dynamics are the same on every rank; take rank 0's.
    dynamics = torch.load(run / "train_grads" / "adamw_dynamics_rank0.pt", weights_only=False)
    torch.save(dynamics, run / "train_grads" / protocol.DYNAMICS_FILE)
    truth = torch.load(RESULTS / a.scale / f"lr{a.lr:g}_seed{a.seed}" / "matrices.pt", weights_only=False)
    x_tr, x_q = load_blocks(a.scale, "train", N_TRAIN), load_blocks(a.scale, "validation", N_VAL)[: a.n_queries]
    train_map = {sample_hash(x_tr[i]): i for i in range(N_TRAIN)}
    test_map = {sample_hash(x_q[q]): q for q in range(a.n_queries)}
    s = types.SimpleNamespace(out_dir=run, recompute=False, n_masks=meta["n_masks"], mask_dim=meta["mask_dim"],
                              batch_size=BATCH, val_batch_size=a.eval_batch, device="cuda",
                              hook_layers=None if meta["hook_layers"] == "all" else [LINEAR_LAYERS],
                              selected=truth["selected"], seed=a.seed)
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    torch.cuda.synchronize()
    t0 = time.time()
    rows = protocol.attribute(s, train_map, test_map, meta["projection"], log)
    pairs = truth["pairs"]
    scores = protocol._matrix(rows, train_map, test_map, pairs, a.n_queries)
    torch.cuda.synchronize()
    score_s = time.time() - t0
    rho = protocol.spearman_per_column(scores, protocol.SIGN * truth["tsloo"][:, : a.n_queries])
    method = protocol.METHOD
    out = RESULTS / a.scale / f"lr{a.lr:g}_seed{a.seed}_masked{a.mask_tag}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"name": a.scale, "seed": a.seed, "lr": a.lr, "model": MODELS[a.scale][0], "trainer": "olmo-core",
              "world_size": captured["world_size"], "mask": f"{meta['n_masks']}x{meta['mask_dim']}/layer",
              "hook_layers": meta["hook_layers"], "n_queries": a.n_queries, "n_selected": len(pairs),
              method: float(rho.mean()), f"{method}_std_over_val": float(rho.std()),
              "train_s": captured["train_s"], "capture_s": captured["capture_s"], "score_s": round(score_s, 1),
              "attribute_s": round(captured["capture_s"] - captured["train_s"] + score_s, 1),
              "capture_peak_gb": captured["peak_gb"],
              "score_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    torch.save({"pred": scores, "pairs": pairs}, out / "matrices.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman across rows, one value per column."""
    def rank(x):
        return x.argsort(0).argsort(0).double()
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(0, keepdim=True), rb - rb.mean(0, keepdim=True)
    return (ra * rb).sum(0) / (ra.norm(dim=0) * rb.norm(dim=0) + 1e-12)


def train_once(a: argparse.Namespace, left_out: int | None, tag: str) -> dict:
    work = work_dir(a.scale)
    run = work / "runs" / tag
    if get_rank() == 0:
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        mask = np.ones((N_TRAIN, BLOCK), dtype=np.bool_)
        if left_out is not None:
            mask[left_out] = False
        mask.tofile(run / "label_mask.npy")
    dist.barrier()
    seed_all(a.seed)
    model = model_config(a.scale).build(init_device="meta")
    train_module = train_module_config(a.lr, N_TRAIN // BATCH, a.microbatch).build(model)
    dataset = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig.dolma2(), paths=[str(work / "train.npy")],
        label_mask_paths=[str(run / "label_mask.npy")], sequence_length=BLOCK,
        dtype=NumpyDatasetDType.uint32, work_dir=str(run / "dataset"),
    ).build()
    data_loader = NumpyDataLoaderConfig(global_batch_size=BATCH * BLOCK, seed=a.seed, num_workers=0,
                                        work_dir=str(run / "loader")).build(
        dataset, dp_process_group=train_module.dp_process_group)
    record = Record()
    trainer = TrainerConfig(save_folder=str(run / "save"), work_dir=str(run / "trainer"),
                            max_duration=Duration.epochs(1), metrics_collect_interval=1,
                            no_checkpoints=True, no_evals=True).with_callback("record", record).build(
        train_module, data_loader)
    load_model_and_optim_state(str(work / "init" / "model_and_optim"), train_module.model)
    t0 = time.time()
    trainer.fit()
    seconds = time.time() - t0
    x_va = torch.from_numpy(np.fromfile(work / "validation.npy", dtype=np.uint32).astype(np.int64))
    losses = val_losses(train_module.model, x_va.view(N_VAL, BLOCK).to(trainer.device))
    out = {"tag": tag, "left_out": left_out, "seconds": seconds, "train_losses": record.losses,
           "batches": record.batches, "world_size": dist.get_world_size(), "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    if get_rank() == 0:
        torch.save({"val_losses": losses, **out}, run / "run.pt")
        print(f"[{tag}] {seconds:.1f}s  loss {record.losses[0]:.6f} -> {record.losses[-1]:.6f}  "
              f"val {losses.mean():.8f}  peak {out['peak_gb']:.1f} GB", flush=True)
    del trainer, train_module, model, data_loader, dataset
    torch.cuda.empty_cache()
    return {"val_losses": losses, **out}


def compare(a: argparse.Namespace) -> None:
    runs = [torch.load(work_dir(a.scale) / "runs" / tag / "run.pt") for tag in a.tags.split(",")]
    ref = runs[0]
    for r in runs[1:]:
        diff = (r["val_losses"] - ref["val_losses"]).abs()
        print(f"{r['tag']} vs {ref['tag']}: max |d val loss| {diff.max():.3e}  mean {diff.mean():.3e}  "
              f"identical {bool(torch.equal(r['val_losses'], ref['val_losses']))}  "
              f"same batches {r['batches'] == ref['batches']}  "
              f"same train losses {r['train_losses'] == ref['train_losses']}", flush=True)


def selected_blocks(seed: int) -> list[int]:
    """The blocks with a ground truth (as ``settings.build``)."""
    gen = torch.Generator().manual_seed(seed + 7)
    return torch.randperm(N_TRAIN, generator=gen)[:N_SELECTED].tolist()


def truth(a: argparse.Namespace) -> None:
    runs = work_dir(a.scale) / "runs"
    ref = torch.load(runs / "ref" / "run.pt")
    step_of = {i: t for t, batch in enumerate(ref["batches"]) for i in batch}
    selected = selected_blocks(a.seed)
    rows, seconds = [], ref["seconds"]
    for i in selected:
        r = torch.load(runs / str(i) / "run.pt")
        assert r["batches"] == ref["batches"], f"run {i} saw another data order"
        rows.append((r["val_losses"] - ref["val_losses"]).float())
        seconds += r["seconds"]
    tsloo = torch.stack(rows)
    out = RESULTS / a.scale / f"lr{a.lr:g}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"tsloo": tsloo, "selected": torch.tensor(selected),
                "pairs": [(i, step_of[i]) for i in selected],
                "ref_losses": ref["val_losses"].float(), "batches": ref["batches"]}, out / "matrices.pt")
    result = {"name": a.scale, "seed": a.seed, "lr": a.lr, "model": MODELS[a.scale][0],
              "trainer": "olmo-core", "world_size": ref["world_size"], "n_selected": len(selected),
              "n_val": N_VAL, "tsloo_s": round(seconds, 1), "tsloo_abs_mean": float(tsloo.abs().mean()),
              "tsloo_abs_max": float(tsloo.abs().max()), "tsloo_only": True}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(result)


TRUTH_MODES = ("prepare", "train", "compare", "selected", "truth")
ATTRIBUTION_MODES = ("export", "capture", "ekfac", "adamw", "adamw-score")


def main(modes: tuple[str, ...] = TRUTH_MODES + ATTRIBUTION_MODES, doc: str | None = None) -> None:
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("mode", choices=list(modes))
    ap.add_argument("--hook-layers", default="all", choices=["all", "blocks"],
                    help="adamw: every layer a default HookManager hooks, or the linear layers of the blocks")
    ap.add_argument("--n-masks", type=int, default=10)
    ap.add_argument("--mask-dim", type=int, default=512, help="coordinates per layer per mask")
    ap.add_argument("--mask-tag", default="", help="adamw: suffix of the run and result directories")
    ap.add_argument("--projection", default="64", choices=["none", "64"])
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=8, help="queries per capture step on a rank")
    ap.add_argument("--microbatch", type=int, default=BATCH, help="blocks per forward/backward on a rank")
    ap.add_argument("--loop-over-test", action="store_true",
                    help="ekfac: precondition the queries once into a disk cache and re-stream them per "
                         "training block (``--eval-batch`` of them on the device at a time)")
    ap.add_argument("--factor-cache-residency", default=None, choices=["memory", "tiered", "disk"],
                    help="ekfac: where the fitted factors are held (default: on the device)")
    ap.add_argument("--scale", required=True, choices=sorted(MODELS))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--left-out", default="none", help="none, or the index of the block left out")
    ap.add_argument("--tag", default=None, help="name of the run directory (default: --left-out)")
    ap.add_argument("--tags", default="", help="compare: comma-separated run tags, the reference first")
    a = ap.parse_args()
    if a.mode in ("compare", "selected", "truth", "ekfac", "adamw-score"):
        if a.mode == "selected":
            print(" ".join(map(str, selected_blocks(a.seed))))
        else:
            {"compare": compare, "truth": truth, "ekfac": ekfac, "adamw-score": adamw_score}[a.mode](a)
        return
    prepare_training_environment(shared_filesystem=True)
    deterministic()
    try:
        if a.mode == "prepare":
            prepare(a)
        elif a.mode in ("capture", "export", "adamw"):
            {"capture": capture, "export": export, "adamw": adamw}[a.mode](a)
        else:
            train_once(a, None if a.left_out == "none" else int(a.left_out), a.tag or a.left_out)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
