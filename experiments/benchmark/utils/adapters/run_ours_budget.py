"""dattri_llm under a fixed multi-GPU budget (``throughput.py``): one process
group per scale, every requested method on one fixed workload.

The model is loaded, frozen (all but the input embedding) and FSDP-wrapped
once.  Each method runs ``warmup_steps`` untimed steps through its full
pipeline and then ``steps`` measured steps at the given per-GPU
``batch``; the time is the whole attribution call, including the gather of
the score rows on rank 0.  One result row per method goes to
``results.jsonl``.

Every rank scores its own training shard against every query, on the fly:
GradDot through ``attribute()`` (the train streamer shards the set, the test
probe runs unsharded on every rank); K-FAC/EK-FAC through ``attribute()`` with
a gradient store (the materialized rank-64 block goes to a rank-local store,
the covariances are collected at capture and reduced across ranks, and the
fit and the scores are read from the store).
The covariances, the EK-FAC spectrum and the final score rows are the only
data exchanged between ranks.

Task fields: ``model``, ``params_b``, ``dataset``, ``batch`` (per GPU),
``steps``, and optionally ``methods`` (default graddot, kfac, ekfac),
``warmup_steps`` (default 2), ``block_size``, ``n_test``, ``seed`` and
``dtype``.  Row fields: ``n_gpus``, ``batch_per_gpu``, ``samples``, ``steps``,
``time_s``, ``throughput`` (samples per second), ``score_shape`` and ``device``.

Environment variables: ``DATTRI_FSDP_CPU_INIT=1`` keeps the model on the CPU
until FSDP shards it; ``STORE_RESIDENCY`` sets the residency of the
K-FAC/EK-FAC gradient stores (default ``memory``).

    torchrun --nproc_per_node=4 run_ours_budget.py --task-file <plan> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import models  # noqa: E402
import run_ours_fsdp as F  # noqa: E402
from data import load_task_data  # noqa: E402
from log import device_details  # noqa: E402

from dattri_llm.gradient.hooks import HookManagerConfig  # noqa: E402
from dattri_llm.task import AttributionTask  # noqa: E402

def _warm_cusolver(dev: str) -> None:
    """Create cuSOLVER's handle at start-up, untimed.

    The handle is per-process set-up that allocates outside PyTorch's caching
    allocator; a small ``eigh`` creates it while the device memory is free.
    """
    torch.linalg.eigh(torch.eye(64, device=dev, dtype=torch.float32))
    torch.cuda.synchronize()


def run(task: dict, out_root: Path) -> None:
    local_rank, rank = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    dev = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(dev))
    _warm_cusolver(dev)  # while the device memory is free

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[dattri_llm budget] {msg}", flush=True)

    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM

    model_id, params_b = task["model"], task["params_b"]
    methods = list(task.get("methods", ["graddot", "kfac", "ekfac"]))
    block, n_test, seed = task.get("block_size", 512), task.get("n_test", 1), task.get("seed", 0)
    warm_steps, steps = int(task.get("warmup_steps", 2)), int(task["steps"])
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[models.dtype_for(params_b, task.get("dtype"))]

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if os.environ.get("DATTRI_FSDP_CPU_INIT", "0") != "1":
        model = model.to(dev)
    model = model.eval()
    model.requires_grad_(False)  # all but the input embedding (see run_ours_fsdp)
    model.get_input_embeddings().requires_grad_(True)
    fsdp_model = FSDP(
        model, device_id=torch.cuda.current_device(), sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                                           transformer_layer_cls=F.transformer_block_classes(model)),
        use_orig_params=True,
    )
    layers = F.linear_layer_names(model)
    log(f"{model_id}: built and sharded over {world} GPUs in {time.perf_counter() - t0:.0f}s, "
        f"{len(layers)} hooked layers")
    proj = {"__default__": {"style": "logra", "proj_dim": F.PROJ_DIM, "proj_max_batch_size": 32,
                            "proj_type": "rademacher", "proj_seed": 0}}

    batch = int(task["batch"])
    log(f"batch {batch}/GPU ({batch * world} per step)")

    # the warm-up and the measured workload: disjoint slices of the training set
    step = batch * world
    sizes = {"warm": warm_steps * step, "measured": steps * step}
    train_ds, test_ds = load_task_data(model_id, task["dataset"], block, sum(sizes.values()), n_test, seed)
    offsets, at = {}, 0
    for name, n in sizes.items():
        offsets[name] = range(at, at + n)
        at += n
    out_root.mkdir(parents=True, exist_ok=True)
    run_live(task, out_root, model, fsdp_model, layers, proj, batch, sizes, offsets,
             train_ds, test_ds, methods, n_test, world, dev, log)
    dist.barrier()
    dist.destroy_process_group()


def run_live(task, out_root, model, fsdp_model, layers, proj, batch, sizes, offsets,
             train_ds, test_ds, methods, n_test, world, dev, log) -> None:
    """Every rank scores its shard against every query; the rows are gathered."""
    rank = dist.get_rank()
    args = F.AttributionArguments(output_dir=tempfile.mkdtemp(prefix="budget_live_"),
                                  per_device_train_batch_size=batch, per_device_eval_batch_size=n_test,
                                  dataloader_pin_memory=False)
    assert args.world_size == world, (args.world_size, world)
    # The wrapped model as the task: hooks on the module, forward on the
    # wrapper, the frozen parameters as the checkpoint, the direct loss.
    live_task = AttributionTask(F.direct_loss, fsdp_model)
    base = {"graddot": F.TracInAttributor, "kfac": F.KFACAttributor, "ekfac": F.EKFACAttributor}
    # GradDot: the capture-time cost model picks the representation per layer.
    auto_config = HookManagerConfig(linear_io=layers, projection_kwargs=proj, capture_style="auto",
                                    include_frozen=True)
    # K-FAC/EK-FAC: the capture stores the materialized projected block and
    # collects the projected covariances in the same pass.
    mat_config = HookManagerConfig(linear_io=layers, projection_kwargs=proj, capture_style="materialized",
                                   include_frozen=True)
    # The compact blocks (64x64 per layer per sample) are held in host memory
    # by default; ``STORE_RESIDENCY`` selects another residency.
    residency = os.environ.get("STORE_RESIDENCY", "memory")

    def graddot(ds):
        return base["graddot"](args, task=live_task).attribute(ds, test_ds, hook_config=auto_config)

    def kron(method, ds):
        return base[method](args, task=live_task).attribute(
            ds, test_ds, hook_config=mat_config, gradient_cache_residency=residency, damping=F.DAMPING)

    for method in methods:
        total, shape = {}, None
        for name in sizes:  # the warm-up first, then the measured workload
            ds = Subset(train_ds, offsets[name])
            dist.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            score = graddot(ds) if method == "graddot" else kron(method, ds)
            gathered = F.gather_rows(score, world)
            torch.cuda.synchronize()
            total[name] = time.perf_counter() - t
            if rank == 0:
                ids, rows = gathered
                shape = list(rows.shape)
                torch.save({"ids": ids, "score": rows}, out_root / f"score-{method}-{name}.pt")
            log(f"  {method} {name}: {sizes[name]} samples in {total[name]:.1f}s (rows {shape})")
        name = "measured"
        if rank == 0:
            row = {"lib": "dattri_llm", "task": {**task, "method": method, "batch": batch}, "status": "ok",
                   "n_gpus": world, "batch_per_gpu": batch,
                   "samples": sizes[name], "steps": sizes[name] // (batch * world), "time_s": round(total[name], 2),
                   "score_shape": shape, "throughput": round(sizes[name] / total[name], 2),
                   "device": device_details()}
            with (out_root / "results.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        log(f"{method}: {sizes[name] / total[name]:.2f} samples/s, batch {batch}/GPU")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    payload = json.loads(Path(a.task_file).read_text())
    run(payload.get("task", payload), Path(a.out))


if __name__ == "__main__":
    main()
