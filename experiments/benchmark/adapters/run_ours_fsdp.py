"""dattri_llm FSDP adapter for the universal benchmark.

Runs attribution under FSDP across N GPUs (``torchrun --nproc_per_node=N``).

The library's streamer/`attribute()` path builds the forward with
``torch.func.functional_call``, which is incompatible with FSDP (it bypasses
FSDP's parameter all-gather and cannot resolve the ``_fsdp_wrapped_module``
names).  The FSDP-correct route -- the one the library's FSDP *tests* use -- is
HookManager-direct capture: register the ghost (`linear_io`) hooks on the
unwrapped model, wrap it in FSDP, and run a plain ``fsdp_model(**batch)`` forward
+ backward; the hooks fire on the (surviving) submodules and OffloadCallback
persists each per-sample gradient to a per-rank store.  Scoring then runs on
rank 0 from the merged store (DiskGradientSource concatenates every rank's
shards), yielding the FSDP-correctness invariant: FSDP scores == single-device.

    torchrun --nproc_per_node=2 adapters/run_ours_fsdp.py --task-file plan.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

import models
from data import load_task_data
from log import BenchRun

from dattri.task import AttributionTask
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.utils.hashing import hash_sample

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64
DAMPING = 1e-3
LIB = "dattri_llm"


def linear_layer_names(model) -> list[str]:
    return [f"{n}$" for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "embed" not in n]


def transformer_block_classes(model) -> set:
    """The repeated decoder-block classes, for FSDP's transformer wrap policy.

    Without a policy FSDP treats the whole model as ONE unit, so the forward
    all-gather reconstructs every parameter on every rank -- sharding is undone
    exactly at the memory peak.  HF exposes the block class via
    ``_no_split_modules``; the ModuleList fallback covers models that do not.
    """
    names = set(getattr(model, "_no_split_modules", None) or ())
    cls = {type(m) for _, m in model.named_modules() if type(m).__name__ in names}
    if not cls:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
                cls = {type(m[0])}
                break
    return cls


def capture_split(model, fsdp_model, ds, batch_size, layers, proj, out_dir, dev):
    """Ghost-capture one split under FSDP into a per-rank on-disk store."""
    fm = GradientStorageManager(out_dir)  # auto rank_N/ subdirs under dist
    hm = HookManager(
        model, config=HookManagerConfig(linear_io=layers, projection=proj),
        callbacks=[OffloadCallback(offload_interval=1, file_manager=fm,
                                   recording_type="per_sample")])
    sampler = DistributedSampler(ds, shuffle=False)  # disjoint shard per rank
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    with hm.collect():
        for b in loader:
            ids = b["input_ids"].to(dev)
            am = b["attention_mask"].to(dev)
            fsdp_model.zero_grad(set_to_none=True)
            out = fsdp_model(input_ids=ids, attention_mask=am)
            logits = out.logits if hasattr(out, "logits") else out
            labels = ids.masked_fill(am == 0, -100)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(),
                reduction="sum", ignore_index=-100)
            loss.backward()
    hm.remove()
    fsdp_model.zero_grad(set_to_none=True)
    return out_dir


def run(task: dict, out_root: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    dev = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(dev))

    model_id, params_b, method = task["model"], task["params_b"], task["method"]
    n_train = task.get("n_train", 1024)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)

    tag = f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}-{method}-fsdp{world}"
    run_dir = out_root / "runs" / tag
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch,
                      "proj_dim": PROJ_DIM, "fsdp_world": world},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB) if rank == 0 else None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[models.dtype_for(params_b)]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # CPU init: a model larger than one device cannot be materialized on GPU
    # before FSDP shards it (72B in bf16 is ~145GB vs a 141GB H200).  FSDP's
    # ``device_id`` moves each unit to the device as it shards, so GPU peak
    # during init is one block plus the local shard, not the whole model.
    cpu_init = os.environ.get("DATTRI_FSDP_CPU_INIT", "0") == "1"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if not cpu_init:
        model = model.to(dev)
    model = model.eval()
    proj = {"__default__": {"style": "logra_factorized", "proj_dim": PROJ_DIM,
                            "proj_max_batch_size": 32, "proj_type": "rademacher",
                            "proj_seed": 0}}

    train_ds, test_ds = load_task_data(model_id, task["dataset"], block_size,
                                       n_train, n_test, seed)

    # Wrap FSDP once; use_orig_params keeps the hooked submodules addressable.
    # The auto-wrap policy is what makes sharding pay off: one unit per decoder
    # block gathers one block at a time instead of the whole model.
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_block_classes(model))
    fsdp_model = FSDP(model, device_id=torch.cuda.current_device(),
                      sharding_strategy=ShardingStrategy.FULL_SHARD,
                      auto_wrap_policy=wrap_policy, use_orig_params=True)

    # Hooked-layer names MUST be read after wrapping: nested wrapping rewrites
    # every path (``layers.0.mlp`` -> ``layers.0._fsdp_wrapped_module.mlp``) and
    # these patterns are anchored, so pre-wrap names match nothing and capture
    # silently yields an empty store.
    layers = linear_layer_names(model)

    cache_dir = out_root / "store" / tag
    train_dir = str(cache_dir / "train")
    test_dir = str(cache_dir / "test")
    if rank == 0:
        shutil.rmtree(cache_dir, ignore_errors=True)
    dist.barrier()

    def _capture():
        capture_split(model, fsdp_model, train_ds, batch, layers, proj, train_dir, dev)
        capture_split(model, fsdp_model, test_ds, batch, layers, proj, test_dir, dev)

    if rank == 0:
        with bench.phase("cache", n_train + n_test):
            _capture()
    else:
        _capture()
    dist.barrier()  # all ranks' shards flushed before rank 0 reads the merge
    dist.destroy_process_group()  # scoring is single-process; tear the group down
    if rank != 0:
        return

    # Rank 0 scores alone from the MERGED store.  Clear the distributed env so
    # the attributor's args see world_size==1 and issue no collectives (rank 1
    # has exited).  The task model is unused by attribute_from_cache (it reads
    # gradients off disk), so a dummy stands in.
    for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
              "GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(v, None)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="bench_fsdp_"),
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        dataloader_pin_memory=False, use_cpu=False)
    dummy = torch.nn.Linear(1, 1)
    task_obj = AttributionTask(
        loss_func=lambda p, b: torch.tensor(0.0), model=dummy,
        checkpoints=[{}], target_func=lambda p, b: torch.tensor(0.0))
    cls = {"graddot": TracInAttributor, "tracin": TracInAttributor,
           "gradcos": TracInAttributor, "kfac": KFACAttributor,
           "ekfac": EKFACAttributor}[method]
    attributor = cls(args, task=task_obj)
    bench.record_disk("store", cache_dir)
    with bench.phase("score", n_train + n_test):
        if method in ("graddot", "tracin", "gradcos"):
            score = attributor.attribute_from_cache(
                train_dir, test_dir, normalized_grad=(method == "gradcos"))
        else:
            score = attributor.attribute_from_cache(
                train_dir, test_dir, damping=DAMPING)
    # Align rows/cols to INPUT order by content hash (the sharded sampler stores
    # them out of order); makes the matrix element-wise identical to single-GPU.
    th = [hash_sample({"input_ids": train_ds[i]["input_ids"],
                       "attention_mask": train_ds[i]["attention_mask"]})
          for i in range(len(train_ds))]
    vh = [hash_sample({"input_ids": test_ds[i]["input_ids"],
                       "attention_mask": test_ds[i]["attention_mask"]})
          for i in range(len(test_ds))]
    matrix = score.query(th, vh).cpu().float()
    torch.save({"score": matrix}, run_dir / "score.pt")
    bench.set(score_shape=list(matrix.shape), n_linear_layers=len(layers))
    bench.finish(status="ok")
    print(f"[done] {tag}: score {tuple(matrix.shape)} on {world} GPUs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task")
    g.add_argument("--task-file", dest="task_file")
    ap.add_argument("--out", default=str(BENCH / "out"))
    a = ap.parse_args()
    if a.task_file:
        payload = json.loads(Path(a.task_file).read_text())
        task = payload.get("task", payload)
    else:
        task = json.loads(a.task)
    run(task, Path(a.out))


if __name__ == "__main__":
    main()
