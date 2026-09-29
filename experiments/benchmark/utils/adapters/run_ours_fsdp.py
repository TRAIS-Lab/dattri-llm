"""dattri_llm FSDP adapter for the benchmark.

Runs attribution under FSDP across N GPUs (``torchrun --nproc_per_node=N``)
and logs through ``log.BenchRun``.

The model is wrapped in FSDP (one unit per decoder block) and handed to the
attributor as a :class:`dattri_llm.task.AttributionTask`.  Every rank scores
its shard of the training set against every query; rank 0 gathers the score
rows and aligns the matrix to input order by content hash.  GradDot runs
``attribute()``; K-FAC/EK-FAC run ``attribute()`` with a disk gradient store,
which holds the materialized rank-``PROJ_DIM`` block and collects the
covariances during capture.

    method -> attributor:  graddot -> TracInAttributor,  kfac -> KFACAttributor,
                           ekfac -> EKFACAttributor

Task fields: ``model``, ``params_b``, ``dataset``, ``method``, and optionally
``dtype``, ``n_train``, ``n_test``, ``block_size``, ``batch``, ``seed``,
``warmup_train``, ``measure_train`` and ``freeze``.  The timed phase is
``attribute``.  Gradients are projected to ``PROJ_DIM`` per side;
K-FAC/EK-FAC use damping ``DAMPING``.  ``DATTRI_FSDP_CPU_INIT=1`` keeps the
model on the CPU until FSDP shards it.

    torchrun --nproc_per_node=2 run_ours_fsdp.py --task-file plan.json --out <dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import sys
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
from torch.utils.data import Subset

import models
from data import load_task_data
from log import BenchRun

from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.task import AttributionTask
from dattri_llm.utils.hashing import hash_sample

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64
DAMPING = 1e-3
LIB = "dattri_llm"


def linear_layer_names(model) -> list[str]:
    return [
        f"{n}$"
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "embed" not in n
    ]


def transformer_block_classes(model) -> set:
    """The repeated decoder-block classes, for FSDP's transformer wrap policy.

    Read from the HF model's ``_no_split_modules``; for a model without it,
    the element class of the first ``ModuleList`` with more than one element.
    """
    names = set(getattr(model, "_no_split_modules", None) or ())
    cls = {type(m) for _, m in model.named_modules() if type(m).__name__ in names}
    if not cls:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
                cls = {type(m[0])}
                break
    return cls


def direct_loss(model, batch) -> torch.Tensor:
    """Token-summed next-token loss of *model* (the FSDP wrapper) on *batch*."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1),
        labels[:, 1:].flatten(),
        reduction="sum",
        ignore_index=-100,
    )


def gather_rows(score, world: int) -> tuple[list[str], torch.Tensor] | None:
    """Every rank's ``(train_ids, rows)`` concatenated on rank 0 (``None`` elsewhere)."""
    ids, matrix = score.agnostic_matrix()
    parts: list = [None] * world
    dist.all_gather_object(parts, (list(ids), matrix.cpu().float()))
    if dist.get_rank() != 0:
        return None
    return [i for ids_r, _ in parts for i in ids_r], torch.cat([m for _, m in parts], dim=0)


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

    tag = f"{task.get('family', '?')}-{task.get('scale', '?')}-{task['dataset']}-{method}-fsdp{world}"
    run_dir = out_root / "runs" / tag
    bench = (
        BenchRun(
            {
                **task,
                "n_train": n_train,
                "n_test": n_test,
                "block_size": block_size,
                "batch": batch,
                "proj_dim": PROJ_DIM,
                "fsdp_world": world,
            },
            results_path=out_root / "results.jsonl",
            run_dir=run_dir,
            lib=LIB,
        )
        if rank == 0
        else None
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = models.dtype_for(params_b, task.get("dtype"))
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    if bench is not None:
        bench.set(dtype=dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # ``DATTRI_FSDP_CPU_INIT=1`` (for a model larger than one device) keeps
    # the model on the CPU; FSDP's ``device_id`` then moves each unit to the
    # device as it is sharded.
    cpu_init = os.environ.get("DATTRI_FSDP_CPU_INIT", "0") == "1"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if not cpu_init:
        model = model.to(dev)
    model = model.eval()
    # ``task["freeze"]`` (default True) freezes every parameter but the input
    # embedding, which keeps gradients flowing through the activations, and
    # captures the frozen layers (``include_frozen``).  Autograd then computes
    # no weight gradient and FSDP allocates no sharded gradients.  With
    # ``False`` every parameter stays trainable.
    freeze = bool(task.get("freeze", True))
    if freeze:
        model.requires_grad_(False)
        model.get_input_embeddings().requires_grad_(True)
    # Capture style per method: GradDot captures with ``capture_style="auto"``
    # (the library's cost rule picks factorized or materialized per layer);
    # K-FAC/EK-FAC store the materialized projected block and collect the
    # covariances during capture.
    capture_style = "materialized" if method in ("kfac", "ekfac") else "auto"
    proj = {
        "__default__": {
            "style": "logra",
            "proj_dim": PROJ_DIM,
            "proj_max_batch_size": 32,
            "proj_type": "rademacher",
            "proj_seed": 0,
        }
    }
    if bench is not None:
        bench.set(capture_style=capture_style, freeze=freeze)

    train_ds, test_ds = load_task_data(
        model_id, task["dataset"], block_size, n_train, n_test, seed
    )

    # One FSDP unit per decoder block; ``use_orig_params`` keeps the hooked
    # submodules addressable.
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_block_classes(model),
    )
    fsdp_model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        use_orig_params=True,
    )

    # Hooked-layer names are read after wrapping: nested wrapping rewrites
    # every path (``layers.0.mlp`` -> ``layers.0._fsdp_wrapped_module.mlp``)
    # and the patterns are anchored.
    layers = linear_layer_names(model)

    cache_dir = out_root / "store" / tag
    if rank == 0:
        shutil.rmtree(cache_dir, ignore_errors=True)
    dist.barrier()

    # Warm-up / measured split, as in run_ours.py: the first ``warmup_train``
    # samples run through the whole pipeline untimed, then the next
    # ``measure_train`` samples are timed.  ``measure_train`` defaults to
    # ``n_train - warmup_train``.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or (n_train - n_warm))
    if n_warm + n_meas > n_train:
        msg = f"n_train={n_train} < warmup_train + measure_train = {n_warm + n_meas}"
        raise ValueError(msg)
    if bench is not None:
        bench.set(warmup_train=n_warm, measure_train=n_meas)
    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))

    # The wrapped model as the task: hooks on the module, forward on the
    # wrapper, the current parameters as the checkpoint.
    task_obj = AttributionTask(direct_loss, fsdp_model)
    hook_config = HookManagerConfig(
        linear_io=layers, projection_kwargs=proj, capture_style=capture_style,
        include_frozen=freeze,
    )
    cls = {
        "graddot": TracInAttributor,
        "kfac": KFACAttributor,
        "ekfac": EKFACAttributor,
    }[method]
    calls = 0

    def attribute(ds):
        """One attribution of *ds* against the test set; each call has its own
        output directory (and, for K-FAC/EK-FAC, its own rank-local store)."""
        nonlocal calls
        calls += 1
        args = AttributionArguments(
            output_dir=str(cache_dir / f"run{calls}" / f"rank{rank}"),
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,  # queries in batches of the training batch size
            dataloader_pin_memory=False,
        )
        attributor = cls(args, task=task_obj)
        if method == "graddot":
            score = attributor.attribute(ds, test_ds, hook_config=hook_config)
        else:
            score = attributor.attribute(
                ds, test_ds, hook_config=hook_config,
                gradient_cache_residency="disk", damping=DAMPING,
            )
        return score, gather_rows(score, world)

    if n_warm:  # untimed
        attribute(Subset(train_ds, range(n_warm)))
        if rank == 0:
            shutil.rmtree(cache_dir / "run1", ignore_errors=True)
        dist.barrier()
        torch.cuda.empty_cache()
    if rank == 0:
        with bench.phase("attribute", n_meas):
            score, gathered = attribute(measured)
    else:
        score, gathered = attribute(measured)
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return
    bench.record_disk("store", cache_dir)
    train_ids, rows = gathered

    # Align rows/columns to input order by content hash; the sharded sampler
    # stores the samples in a different order.
    th = [
        hash_sample(
            {
                "input_ids": measured[i]["input_ids"],
                "attention_mask": measured[i]["attention_mask"],
            }
        )
        for i in range(len(measured))
    ]
    vh = [
        hash_sample(
            {
                "input_ids": test_ds[i]["input_ids"],
                "attention_mask": test_ds[i]["attention_mask"],
            }
        )
        for i in range(len(test_ds))
    ]
    row_of = {h: i for i, h in enumerate(train_ids)}
    matrix = rows[[row_of[h] for h in th]][:, [score.test_index[h] for h in vh]]
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
