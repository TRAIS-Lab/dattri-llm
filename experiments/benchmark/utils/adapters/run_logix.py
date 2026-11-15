"""LogIX adapter for the benchmark.

Runs LogIX's LoGra pipeline -- rank-64 LoRA compression (``add_lora``),
extraction (gradients and, for K-FAC/EK-FAC, Hessian statistics), then
``compute_influence_all`` -- on a (HF model, dataset) task and logs through
``log.BenchRun``.  LoGra's compression is a per-side rank-64 projection of
each linear layer.  With ``proj_mode="full"`` the LoRA step is skipped.

    method -> logix hessian:  graddot -> "none",  kfac -> "kfac",  ekfac -> "ekfac"

GradDot is ``hessian="none"`` with ``precondition=False``: dot products of the
logged gradients.  K-FAC and EK-FAC score with ``precondition=True``.

Task fields: ``model``, ``params_b``, ``dataset``, ``method``, and optionally
``dtype``, ``proj_mode``, ``n_train``, ``n_test``, ``block_size``, ``batch``,
``seed``, ``warmup_train`` and ``measure_train``.  The timed phases are
``extract`` (``add_lora`` + extraction) and ``score``.

Warm-up / measured split, as in run_ours.py: the first ``warmup_train``
samples go through the full pipeline untimed (own ``LogIX`` instance and
project), then the next ``measure_train`` samples are timed.  LogIX wraps
linear layers in place when LoRA is added, so between the two passes the
warm-up instance's hooks are cleared and its LoRA wrappers removed
(``LoraLinear._linear`` is the original layer).

The model dtype follows the task.  LogIX creates its LoRA modules in float32,
so they are cast to the model's dtype after ``add_lora``.  LogIX writes its
log through numpy, which has no bfloat16, so for a bfloat16 model the log is
stored in float32 (LogIX's ``logging.log_dtype`` option).  The layer name
filter ``["att", "mlp"]`` selects the attention and MLP linears of GPT-2,
GPT-NeoX (Pythia), Qwen and Llama models.

The task's ``logix_io`` selects how the training gradients reach the scorer.
``log`` (LogIX's documented pipeline) writes every per-sample gradient to
LogIX's on-disk log during extraction and scores from its log loader.
``memory`` never stores a training gradient: GradDot scores each batch's
gradients against the query gradients as they are logged, in one pass over
the training set; K-FAC and EK-FAC first accumulate the Kronecker statistics
(``save="none"``, so nothing is logged), precondition the query gradients once,
and then recompute the training gradients in a second pass and score them
batch by batch, as Kronfluence and Bergson do.  Only the statistics themselves
reach the disk, through LogIX's ``finalize``.

Multi-GPU (under torchrun), by the task's ``parallelism``: ``ddp`` runs one
model replica per GPU and ``fsdp`` shards the model across the GPUs (FSDP, one
unit per decoder block, applied after LogIX's hooks and LoRA modules are in
place: ``add_lora`` replaces the hooked layers and saves the model's state
dict, which a sharded model's parameter names do not survive).  Either way
each rank extracts its shard of the training set, LogIX merges the per-rank
covariance state, and rank 0 scores from the merged logs and records the row.
Under ``fsdp`` every rank runs the query forward/backward (a sharded forward
needs all ranks), the LoRA insertion and the sharding precede the timed
``extract`` phase, and the timed run starts from a freshly built model, as the
warm-up's sharded model cannot have its LoRA modules removed.

    python run_logix.py --task-file plan.json --out <dir>
    torchrun --nproc_per_node=4 run_logix.py --task-file plan.json --out <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import gc
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset

import models
from data import load_task_data
from log import BenchRun
from versions import require

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

LIB = "logix"
LORA_RANK = 64
_HESSIAN = {"kfac": "kfac", "graddot": "none", "ekfac": "ekfac"}
NAME_FILTER = ["att", "mlp"]

# LogIX's saved state holds populated defaultdicts, which
# ``torch.load(weights_only=True)`` does not rebuild.  Every file loaded here
# is this run's own output, so ``weights_only`` defaults to False.
_orig_load = torch.load


def _trusting_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _trusting_load


def shift_ce_sum(logits, targets):
    sl = logits[..., :-1, :].contiguous()
    st = targets[..., 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                           reduction="sum", ignore_index=-100)


def build_model(model_id: str, params_b: float, dtype_override: str | None,
                *, cpu: bool = False):
    """The model (on the device, or on the CPU with ``cpu`` for one that FSDP
    moves shard by shard), its tokenizer and the dtype name."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = models.dtype_for(params_b, dtype_override)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=getattr(torch, dtype_name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return (model if cpu else model.cuda()).eval(), tok, dtype_name


def _world() -> tuple[int, int]:
    """``(rank, world_size)`` under torchrun, ``(0, 1)`` otherwise."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def _transformer_block_classes(model) -> set:
    """The repeated decoder-block classes, for FSDP's transformer wrap policy
    (as ``run_ours_fsdp.transformer_block_classes``)."""
    names = set(getattr(model, "_no_split_modules", None) or ())
    cls = {type(m) for _, m in model.named_modules() if type(m).__name__ in names}
    if not cls:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
                cls = {type(m[0])}
                break
    return cls


def _shard(model):
    """Shard *model* with FSDP, one unit per decoder block; ``use_orig_params``
    keeps LogIX's hooked modules and LoRA parameters addressable."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return FSDP(model, device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=_transformer_block_classes(model)),
                use_orig_params=True)


def _lora_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if type(m).__name__ in ("LoraLinear", "LoraConv2d", "LoraEmbedding")]


def _cast_lora(model, dtype: torch.dtype) -> None:
    """Cast LogIX's LoRA modules (created in float32) to the model's dtype."""
    for _, m in _lora_modules(model):
        m.to(dtype)


def _remove_lora(model) -> int:
    """Undo ``add_lora``: put each wrapped layer's original module back."""
    n = 0
    for name, m in _lora_modules(model):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, m._linear)
        n += 1
    return n


def extract_and_score(model, tok, train_ds, test_ds, *, hessian: str, proj_mode: str,
                      batch: int, n_test: int, project: str, log_root: Path,
                      dtype: torch.dtype, seed: int = 0, phase=None,
                      parallelism: str = "single", io: str = "log"):
    """One full LogIX pass: watch (+ LoRA), extract, score.  Returns (lx, score).

    ``io`` is ``log`` or ``memory`` (see the module docstring).

    ``phase(name, units)`` wraps the two timed sections when given; the warm-up
    passes ``None`` and is not timed.  Under torchrun, ``score`` is ``None`` on
    every rank but 0.  With ``parallelism="fsdp"`` the model is sharded after
    the LoRA modules are added, before the timed extraction (see the module
    docstring); the LoRA insertion is timed only in the other modes.
    """
    from logix import LogIX, LogIXScheduler
    from logix.utils import merge_logs
    from transformers import default_data_collator

    phase = phase or (lambda *_: contextlib.nullcontext())
    cfg = log_root / f"{project}.yaml"
    # LogIX writes its log through numpy, which has no bfloat16; for a
    # bfloat16 model the logged gradients are stored as float32 (LogIX's
    # ``logging.log_dtype`` option).  Model forward/backward and the LoRA
    # compression stay in the model's dtype.
    log_dtype = "float32" if dtype == torch.bfloat16 else "none"
    if _world()[0] == 0:
        cfg.write_text(f"root_dir: {log_root}/logs\nlora:\n  init: random\n  rank: {LORA_RANK}\n"
                       f"logging:\n  log_dtype: {log_dtype}\n")
    if _world()[1] > 1:
        import torch.distributed as dist

        dist.barrier()  # the config file is written before anyone reads it
    lx = LogIX(project, config=str(cfg))
    if _world()[1] > 1:
        import torch.distributed as dist

        dist.barrier()  # rank 0 creates the log directory the others write to
    lx.watch(model, name_filter=NAME_FILTER)
    sharded = parallelism == "fsdp"

    def add_lora() -> None:
        if proj_mode != "full":
            # LogIX draws the LoRA encoder/decoder (its random projection) from
            # torch's global RNG; seeding it makes the projection reproducible.
            torch.manual_seed(seed)
            lx.add_lora()  # rank-64 LoRA projection; full-dim skips this
            _cast_lora(model, dtype)

    if sharded:
        add_lora()
        model = _shard(model)

    # Multi-GPU: each rank extracts its own shard; LogIX all-reduces its
    # covariance state at finalize().
    rank, world = _world()
    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=False)
               if world > 1 else None)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=False, sampler=sampler,
                        collate_fn=default_data_collator)

    def backward(b, data_id=None):
        """One forward/backward of batch *b* under LogIX's logging context."""
        data_id = data_id or tok.batch_decode(b["input_ids"])
        tgt = b.pop("labels").cuda()
        b = {k: v.cuda() for k, v in b.items()}
        with lx(data_id=data_id, mask=b["attention_mask"]):
            model.zero_grad()
            shift_ce_sum(model(**b).logits, tgt).backward()

    def query_logs():
        qloader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=default_data_collator)
        lx.setup({"grad": ["log"]})
        lx.eval()
        logs = []
        for b in qloader:
            backward(b, tok.batch_decode(b["input_ids"], skip_special_tokens=True))
            logs.append(copy.deepcopy(lx.get_log()))
        return merge_logs(logs)

    if io == "memory":
        with phase("extract", len(train_ds)):  # the Kronecker statistics; GradDot has none
            if not sharded:
                add_lora()
            if hessian != "none":
                scheduler = LogIXScheduler(lx, lora="none", hessian=hessian, save="none")
                for _ in scheduler:
                    for b in loader:
                        backward(b)
                    lx.finalize()
        with phase("score", len(train_ds) + n_test):
            src = query_logs()
            if hessian != "none":
                src = lx.influence.precondition(src)
            ids, parts = [], []
            for b in loader:  # the training gradients, scored as they are computed
                backward(b)
                r = lx.compute_influence(src, lx.get_log(), precondition=False)
                ids += list(r["tgt_ids"])
                parts.append(r["influence"].detach().float().cpu())
            local = (torch.cat(parts, dim=1).T, ids)  # (n_local, n_test)
            if world > 1:
                import torch.distributed as dist

                gathered = [None] * world
                dist.all_gather_object(gathered, local)
                if rank != 0:
                    return lx, None
                score = torch.cat([g[0] for g in gathered])  # rank order, as the log store
            else:
                score = local[0]
        return lx, score

    with phase("extract", len(train_ds)):
        if not sharded:
            add_lora()
        scheduler = LogIXScheduler(lx, lora="none", hessian=hessian, save="grad")
        # LogIX writes per-rank log chunks.
        for _ in scheduler:
            for b in loader:
                backward(b)
            lx.finalize()

    if world > 1:
        import torch.distributed as dist

        dist.barrier()  # every rank's log chunks are on disk
        if rank != 0 and not sharded:
            return lx, None  # rank 0 scores from the merged logs

    # Rank 0 scores from the merged logs; under FSDP the other ranks take part
    # in the query forward/backward and return nothing.
    with phase("score", len(train_ds) + n_test):
        # ``initialize_from_log`` reloads the LoRA weights that are already in
        # the model, through ``model.state_dict()`` once per LoRA parameter; on
        # a sharded model each call gathers the whole model on every rank, so
        # the reload is skipped there (a ``lora_path`` without a ``lora/``
        # entry loads nothing).
        lx.initialize_from_log(lora_path=str(log_root / "no_lora") if sharded else None)
        log_loader = lx.build_log_dataloader(batch_size=64) if rank == 0 else None
        src = query_logs()
        if rank != 0:
            return lx, None
        # GradDot: dot products of the logged gradients.  With a Hessian,
        # LogIX preconditions with the K-FAC/EK-FAC statistics accumulated
        # during extraction.
        result = lx.influence.compute_influence_all(
            src, log_loader, precondition=(hessian != "none"))

    score = result["influence"].T.cpu().float()  # -> (n_train, n_test)
    return lx, score


def run(task: dict, out_root: Path) -> None:
    method = task["method"]
    if method not in _HESSIAN:
        msg = f"logix does not cover method {method!r}"
        raise ValueError(msg)
    hessian = _HESSIAN[method]
    n_train = task.get("n_train", 1024)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)
    proj_mode = task.get("proj_mode", "rank64")

    tag = f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}-{method}"
    run_dir = out_root / "runs" / f"logix-{tag}"
    # Multi-GPU under torchrun (``distributed_mode`` in the row): ``ddp`` is
    # one full model replica per rank, without a DDP wrapper (LogIX's hooks
    # read activations and output gradients, and its collectives merge the
    # per-rank covariance state); ``fsdp`` shards the model (see the module
    # docstring).  Rank 0 records and scores.
    rank, world = _world()
    parallelism = task.get("parallelism", "single") if world > 1 else "single"
    io = task.get("logix_io", "log")
    if io not in ("log", "memory"):
        msg = f"logix_io must be 'log' or 'memory', got {io!r}"
        raise ValueError(msg)
    if world > 1:
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
    # A model larger than one device is built on the CPU and moved shard by
    # shard by FSDP (``DATTRI_FSDP_CPU_INIT=1`` forces this at any size).
    cpu_init = parallelism == "fsdp" and (
        task["params_b"] > 60 or os.environ.get("DATTRI_FSDP_CPU_INIT", "0") == "1")
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch,
                      "lora_rank": LORA_RANK, "proj_mode": proj_mode,
                      "strategy": "logra", "world_size": world,
                      "distributed_mode": parallelism, "io": io},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB) if rank == 0 else None

    def phase(name: str, units: int | None = None):
        return bench.phase(name, units) if bench else contextlib.nullcontext()

    def record(**kv) -> None:
        if bench:
            bench.set(**kv)

    with phase("build_model"):
        model, tok, dtype_name = build_model(task["model"], task["params_b"],
                                             task.get("dtype"), cpu=cpu_init)
        record(dtype=dtype_name, proj_mode=proj_mode)
    with phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)
    dtype = getattr(torch, dtype_name)

    # Warm-up / measured split (see module docstring).  ``measure_train``
    # defaults to ``n_train - warmup_train``.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or (n_train - n_warm))
    if n_warm + n_meas > n_train:
        msg = f"n_train={n_train} < warmup_train + measure_train = {n_warm + n_meas}"
        raise ValueError(msg)
    record(warmup_train=n_warm, measure_train=n_meas)

    # One log root for every rank (LogIX merges the per-rank chunks from it).
    log_root = Path(tempfile.mkdtemp(prefix="logix_")) if rank == 0 else None
    if world > 1:
        import torch.distributed as dist

        holder = [str(log_root)]
        dist.broadcast_object_list(holder, src=0)
        log_root = Path(holder[0])
    common = dict(hessian=hessian, proj_mode=proj_mode, batch=batch, n_test=n_test,
                  log_root=log_root, dtype=dtype, seed=seed, parallelism=parallelism, io=io)
    if n_warm:
        lx_warm, _ = extract_and_score(model, tok, Subset(train_ds, range(n_warm)),
                                       test_ds, project=f"{tag}_warm", **common)
        lx_warm.clear()                     # remove the warm-up instance's hooks
        if parallelism != "fsdp":
            removed = _remove_lora(model)   # so the timed run adds LoRA itself
            record(warmup_lora_removed=removed)
        if world > 1:
            import torch.distributed as dist

            dist.barrier()  # rank 0 is done reading the warm-up logs
        if rank == 0:
            shutil.rmtree(log_root / "logs" / f"{tag}_warm", ignore_errors=True)
        del lx_warm
        if parallelism == "fsdp":
            # The sharded model keeps its LoRA modules: the timed run starts
            # from a fresh model (built outside the timed phases), once the
            # warm-up's model is released.
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model, tok, dtype_name = build_model(task["model"], task["params_b"],
                                                 task.get("dtype"), cpu=cpu_init)
        torch.cuda.empty_cache()

    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))
    project = f"{tag}_{hessian}"
    _lx, score = extract_and_score(model, tok, measured, test_ds, project=project,
                                   phase=phase, **common)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    bench.record_disk("log_store", log_root / "logs" / project)
    torch.save({"score": score}, run_dir / "score.pt")
    bench.set(score_shape=list(score.shape))
    bench.finish(status="ok")
    print(f"[done] logix {tag}: {dtype_name} warm {n_warm} score {tuple(score.shape)}",
          flush=True)


def main() -> None:
    # Requires the LogIX version pinned in versions.py.
    require("logix")
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
