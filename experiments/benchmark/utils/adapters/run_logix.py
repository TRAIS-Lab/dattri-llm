"""logix adapter for the universal benchmark (modal tree).

Runs logix's own LoGra pipeline -- rank-64 LoRA compression, extract
(gradients + optional Hessian statistics), then ``compute_influence_all`` --
on a universal (HF model, dataset) task and logs through ``log.BenchRun``.
This is the baseline whose projection is closest to ours: LoGra's random
Kronecker compression is a per-side rank-64 projection of each linear layer,
the same regime as our rank-64 capture.

    method -> logix hessian:  graddot -> "none",  kfac -> "kfac",  ekfac -> "ekfac"

GradDot is ``hessian="none"`` with ``precondition=False``: logged LoGra
gradients, dotted.  It is NOT ``hessian="raw"`` -- in logix "raw" schedules a
dense gradient *covariance* (a 4096 x 4096 Gram per layer at rank 64, ~11 GB
of state on a 0.5B model) and ``precondition_raw`` inverts it, i.e. a dense
Hessian influence function.  The parent-tree adapter mapped GradDot to "raw",
so its LogIX GradDot cells timed that heavier method; fixed here and there.

Warm-up / measured split with run_ours.py's semantics: ``warmup_train``
samples go through the full pipeline first (own ``LogIX`` instance, throwaway
project), then ``measure_train`` samples are timed.  logix wraps linear layers
in place when LoRA is added, so between the two runs the warm-up instance's
hooks are cleared and its LoRA wrappers removed (``LoraLinear._linear`` is the
original layer); the measured ``extract`` phase then contains exactly what the
A40 tables timed: ``add_lora`` + extraction.

dtype follows the task like every other adapter.  logix creates its LoRA
modules in float32 whatever the model's dtype, which is the "dtype-mismatch
matmul" that made the parent-tree adapter force float32; casting the LoRA
modules to the model's dtype after ``add_lora`` resolves it; the on-disk log
(numpy, no bf16) is written as float32 via logix's ``logging.log_dtype``.
Covariance eigendecompositions are done in double by logix itself, so bf16
statistics are fine.  Single process; the layer name filter ["att", "mlp"] matches attention
+ MLP linears across gpt2 / GPT-NeoX (pythia) / Qwen / Llama.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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

# logix predates torch 2.6's weights_only=True default; its saved state holds
# populated defaultdicts the weights-only unpickler cannot rebuild.  Every file
# loaded here is this run's own output, so restore the pre-2.6 behaviour.
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


def build_model(model_id: str, params_b: float, dtype_override: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = models.dtype_for(params_b, dtype_override)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=getattr(torch, dtype_name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.cuda().eval(), tok, dtype_name


def _lora_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if type(m).__name__ in ("LoraLinear", "LoraConv2d", "LoraEmbedding")]


def _cast_lora(model, dtype: torch.dtype) -> None:
    """logix builds LoRA modules in float32; match the model's dtype."""
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
                      dtype: torch.dtype, seed: int = 0, phase=None):
    """One full logix pass: watch (+ LoRA), extract, score.  Returns (lx, score).

    ``phase(name, units)`` wraps the two timed sections when given; the warm-up
    passes ``None`` and is not timed.
    """
    from logix import LogIX, LogIXScheduler
    from logix.utils import merge_logs
    from transformers import default_data_collator

    phase = phase or (lambda *_: contextlib.nullcontext())
    cfg = log_root / f"{project}.yaml"
    # logix writes its log through numpy, which has no bfloat16; for a bf16
    # model the logged tensors -- the rank-64 compressed gradients, tiny -- are
    # stored as float32 (logix's own ``logging.log_dtype`` option).  Model
    # forward/backward and the LoRA compression stay in the model's dtype.
    log_dtype = "float32" if dtype == torch.bfloat16 else "none"
    cfg.write_text(f"root_dir: {log_root}/logs\nlora:\n  init: random\n  rank: {LORA_RANK}\n"
                   f"logging:\n  log_dtype: {log_dtype}\n")
    lx = LogIX(project, config=str(cfg))
    lx.watch(model, name_filter=NAME_FILTER)

    with phase("extract", len(train_ds)):
        if proj_mode != "full":
            # logix draws the LoRA encoder/decoder (its random projection) from
            # torch's global RNG with no seed of its own; pin it so two runs of
            # the same cell project identically and their scores can be compared.
            torch.manual_seed(seed)
            lx.add_lora()  # rank-64 LoRA projection; full-dim skips this
            _cast_lora(model, dtype)
        scheduler = LogIXScheduler(lx, lora="none", hessian=hessian, save="grad")
        loader = DataLoader(train_ds, batch_size=batch, shuffle=False,
                            collate_fn=default_data_collator)
        for _ in scheduler:
            for b in loader:
                data_id = tok.batch_decode(b["input_ids"])
                tgt = b.pop("labels").cuda()
                b = {k: v.cuda() for k, v in b.items()}
                with lx(data_id=data_id, mask=b["attention_mask"]):
                    model.zero_grad()
                    loss = shift_ce_sum(model(**b).logits, tgt)
                    loss.backward()
            lx.finalize()

    with phase("score", len(train_ds) + n_test):
        lx.initialize_from_log()
        log_loader = lx.build_log_dataloader(batch_size=64)
        qloader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=default_data_collator)
        lx.setup({"grad": ["log"]})
        lx.eval()
        test_logs = []
        for b in qloader:
            data_id = tok.batch_decode(b["input_ids"], skip_special_tokens=True)
            tgt = b.pop("labels").cuda()
            b = {k: v.cuda() for k, v in b.items()}
            with lx(data_id=data_id, mask=b["attention_mask"]):
                model.zero_grad()
                loss = shift_ce_sum(model(**b).logits, tgt)
                loss.backward()
            test_logs.append(copy.deepcopy(lx.get_log()))
        # GradDot: plain dot products of the logged gradients.  With a Hessian,
        # logix's "auto" picks K-FAC/EK-FAC preconditioning from the statistics
        # the extract pass accumulated.
        result = lx.influence.compute_influence_all(
            merge_logs(test_logs), log_loader, precondition=(hessian != "none"))

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
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch,
                      "lora_rank": LORA_RANK, "proj_mode": proj_mode,
                      "strategy": "logra"},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB)

    with bench.phase("build_model"):
        model, tok, dtype_name = build_model(task["model"], task["params_b"],
                                             task.get("dtype"))
        bench.set(dtype=dtype_name, proj_mode=proj_mode)
    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)
    dtype = getattr(torch, dtype_name)

    # Warm-up / measured split (see module docstring).  Without ``measure_train``
    # the old meaning -- all ``n_train`` samples timed, no warm-up -- is kept.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or (n_train - n_warm))
    if n_warm + n_meas > n_train:
        msg = f"n_train={n_train} < warmup_train + measure_train = {n_warm + n_meas}"
        raise ValueError(msg)
    bench.set(warmup_train=n_warm, measure_train=n_meas)

    log_root = Path(tempfile.mkdtemp(prefix="logix_"))
    common = dict(hessian=hessian, proj_mode=proj_mode, batch=batch, n_test=n_test,
                  log_root=log_root, dtype=dtype, seed=seed)
    if n_warm:
        lx_warm, _ = extract_and_score(model, tok, Subset(train_ds, range(n_warm)),
                                       test_ds, project=f"{tag}_warm", **common)
        lx_warm.clear()                     # remove the warm-up instance's hooks
        removed = _remove_lora(model)       # so the timed run adds LoRA itself
        bench.set(warmup_lora_removed=removed)
        shutil.rmtree(log_root / "logs" / f"{tag}_warm", ignore_errors=True)
        del lx_warm
        torch.cuda.empty_cache()

    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))
    project = f"{tag}_{hessian}"
    _lx, score = extract_and_score(model, tok, measured, test_ds, project=project,
                                   phase=bench.phase, **common)

    bench.record_disk("log_store", log_root / "logs" / project)
    torch.save({"score": score}, run_dir / "score.pt")
    bench.set(score_shape=list(score.shape))
    bench.finish(status="ok")
    print(f"[done] logix {tag}: {dtype_name} warm {n_warm} score {tuple(score.shape)}",
          flush=True)


def main() -> None:
    # Refuse to benchmark against anything but the pinned baseline
    # (versions.py); the version is part of the result.
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
