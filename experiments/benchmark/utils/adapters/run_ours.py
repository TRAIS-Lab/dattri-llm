"""Single-GPU dattri_llm adapter for the benchmark.

Runs one of three methods on a (HF model, dataset) task, timing every phase
and recording peak memory (see ``log.BenchRun``):

    graddot   TracInAttributor (gradient dot; projected or full dimension)
    kfac      KFACAttributor  -- rank-64: materialized "logra" store +
              capture-time covariances; full dimension: live fit and score
    ekfac     EKFACAttributor -- the same two recipes, with the corrected spectrum

    python run_ours.py --task-file <plan.json> --out <dir>

Task fields (defaults in parentheses): ``model``, ``params_b``, ``dataset``,
``method``; ``n_train`` (1024), ``warmup_train`` (32) and ``measure_train``
(32) training samples for the untimed warm-up and the timed run, ``n_test``
(16), ``block_size`` (512), ``batch`` (8), ``eval_batch`` (``n_test``),
``seed`` (0), ``dtype`` (see ``models.dtype_for``), ``proj_mode``
("rank64" | "full") and ``route`` (one of ``ROUTES``).

Outputs under ``--out``: a row in ``results.jsonl``, ``runs/<tag>/record.json``
and ``runs/<tag>/score.pt`` (the ``[n_test, measure_train]`` score matrix), and
the gradient store of the store-based methods in ``store/<tag>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))  # data.py, models.py, log.py

import torch
import torch.nn.functional as F

import models
from data import load_task_data
from log import BenchRun

from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.task import AttributionTask

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64  # per factor side -> 64*64 = 4096 per-layer projected space
DAMPING = 1e-3
LIB = "dattri_llm"
ROUTES = ("auto", "factorized", "materialized")


def pin_route(route: str) -> None:
    """Make every factor x factor inner product take *route*.

    ``"auto"`` leaves the library's cost model in place.  ``"factorized"`` and
    ``"materialized"`` replace :func:`ops.maybe_use_materialized_gram` -- the
    rule behind the per-layer query routing in ``BaseInnerProductAttributor``
    and the train-side route in :func:`ops.cross_dot` -- with a constant.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if route == "auto":
        return
    import importlib

    import dattri_llm.gradient.ops as ops

    def rule(*_args, **_kwargs) -> bool:
        return route == "materialized"

    # ``ops.dot`` names a function in the ops namespace; the module of the
    # same name is imported by its full path.
    kernels = importlib.import_module("dattri_llm.gradient.ops.dot")
    ops.maybe_use_materialized_gram = rule
    kernels.maybe_use_materialized_gram = rule
    if kernels.cross_gram.__globals__["maybe_use_materialized_gram"] is not rule:
        raise RuntimeError("the route pin did not reach ops.dot.cross_gram")


def build_model(model_id: str, params_b: float, dtype_override: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = models.dtype_for(params_b, dtype_override)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.cuda().eval(), tok


def linear_layer_names(model) -> list[str]:
    """Name patterns of every nn.Linear in the transformer blocks (modules whose
    name contains "lm_head" or "embed" are excluded)."""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "embed" not in n:
            names.append(f"{n}$")
    return names


def loss_func(model, batch) -> torch.Tensor:
    """Token-summed next-token loss of *model* on *batch*."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1),
        labels[:, 1:].flatten(),
        reduction="sum",
        ignore_index=-100,
    )


def run(task: dict, out_root: Path) -> None:
    model_id = task["model"]
    params_b = task["params_b"]
    method = task["method"]
    n_train = task.get("n_train", 1024)
    # Warm up on the first n_warm training samples, then time the next n_meas.
    n_warm = task.get("warmup_train", 32)
    n_meas = task.get("measure_train", 32)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)

    # route: which representation the factor x factor inner products take
    # ("auto" = the cost model; the other two pin one route, see pin_route).
    route = task.get("route", "auto")
    pin_route(route)
    tag = (
        f"{task.get('family', '?')}-{task.get('scale', '?')}-{task['dataset']}-{method}"
        f"-T{block_size}" + ("" if route == "auto" else f"-{route}")
    )
    run_dir = out_root / "runs" / tag
    bench = BenchRun(
        {
            **task,
            "n_train": n_train,
            "n_test": n_test,
            "block_size": block_size,
            "batch": batch,
            "proj_dim": PROJ_DIM,
        },
        results_path=out_root / "results.jsonl",
        run_dir=run_dir,
        lib=LIB,
    )

    dtype_name = models.dtype_for(params_b, task.get("dtype"))
    bench.set(dtype=dtype_name)  # record the EFFECTIVE dtype, override or not
    with bench.phase("build_model"):
        model, tok = build_model(model_id, params_b, task.get("dtype"))
        layers = linear_layer_names(model)
        bench.set(n_linear_layers=len(layers))

    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(
            model_id, task["dataset"], block_size, n_train, n_test, seed
        )

    # The model's current parameters are the checkpoint; the model is not
    # updated during the run.
    task_obj = AttributionTask(loss_func, model)
    # eval_batch defaults to n_test: the test side is scored as one block.
    eval_batch = task.get("eval_batch", n_test)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="bench_"),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        dataloader_pin_memory=False,
    )
    bench.set(eval_batch=eval_batch)

    # proj_mode: "rank64" (LoGra projection to PROJ_DIM per factor side) or
    # "full" (no projection, full-dimension factors).
    proj_mode = task.get("proj_mode", "rank64")
    bench.set(proj_mode=proj_mode, route=route)
    # capture_style="auto": the library's capture-time cost model chooses the
    # representation per layer.  A pinned route pins the capture representation
    # to the same form.
    capture_style = "auto" if route == "auto" else route
    proj = (
        None
        if proj_mode == "full"
        else {
            "__default__": {
                "style": "logra",
                "proj_dim": PROJ_DIM,
                "proj_max_batch_size": 32,
                "proj_type": "rademacher",
                "proj_seed": 0,
            }
        }
    )
    bench.set(capture_style=capture_style)
    hook_config = HookManagerConfig(
        invasive_linear_io=layers, projection_kwargs=proj, capture_style=capture_style
    )

    cache_dir = out_root / "store" / tag
    shutil.rmtree(cache_dir, ignore_errors=True)

    if method == "graddot":
        attributor = TracInAttributor(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(tr, te, hook_config=hook_config),
            train_ds,
            test_ds,
            n_warm,
            n_meas,
        )
    elif method in ("kfac", "ekfac") and proj_mode == "full":
        # Full-dimension K-FAC/EK-FAC (no projection): ``attribute`` with the
        # default ``gradient_cache_residency=None`` fits and scores from live
        # model passes; no gradient store is written.
        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        attributor = cls(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(
                tr, te, hook_config=hook_config, damping=DAMPING
            ),
            train_ds,
            test_ds,
            n_warm,
            n_meas,
        )
    elif method in ("kfac", "ekfac"):
        # Rank-64 K-FAC/EK-FAC: ``attribute`` with a disk store.  The capture
        # stores the materialized logra block (the PROJ_DIM x PROJ_DIM
        # projected gradient per sequence) and collects the projected
        # covariances in the same pass; the fit and the scoring read the store.
        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        cap_style = "materialized"
        mat_config = HookManagerConfig(
            linear_io=layers, projection_kwargs=proj, capture_style=cap_style
        )
        bench.set(capture_style=cap_style, kfac_store_style=cap_style)
        # One store directory per call, so the warm-up and the measured run
        # share no state; the warm-up store is deleted after the timed phase.
        kfac_stores: list[Path] = []

        def _kfac_cached(tr_ds, te_ds):
            run_store = cache_dir / f"run{len(kfac_stores)}"
            kfac_stores.append(run_store)
            run_args = dataclasses.replace(args, output_dir=str(run_store))
            return cls(run_args, task=task_obj).attribute(
                tr_ds,
                te_ds,
                hook_config=mat_config,
                gradient_cache_residency="disk",
                damping=DAMPING,
            )

        score = _warm_then_time(bench, _kfac_cached, train_ds, test_ds, n_warm, n_meas)
        for spent in kfac_stores[:-1]:  # keep only the measured run's store
            shutil.rmtree(spent, ignore_errors=True)
    else:
        msg = f"unsupported method {method!r}"
        raise ValueError(msg)

    bench.record_disk("store", cache_dir)
    _, matrix = score.agnostic_matrix()
    matrix = matrix.cpu().float()
    torch.save({"score": matrix}, run_dir / "score.pt")
    bench.set(score_shape=list(matrix.shape))
    bench.finish(status="ok")
    print(f"[done] {tag}: score {tuple(matrix.shape)}", flush=True)


def _warm_then_time(bench, call, train_ds, test_ds, n_warm, n_meas):
    """Run ``call`` once as an untimed warm-up, then time it on a fresh slice.

    ``call(train_subset, test_ds)`` runs first on training samples
    ``[0, n_warm)`` outside any phase (skipped when ``n_warm`` is 0), then on
    samples ``[n_warm, n_warm + n_meas)`` inside the ``"attribute"`` phase with
    ``work_units=n_meas``.  Returns the score of the timed call.
    """
    from torch.utils.data import Subset

    if n_warm:
        call(Subset(train_ds, range(n_warm)), test_ds)
    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))
    with bench.phase("attribute", n_meas):
        return call(measured, test_ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="task JSON (single object)")
    g.add_argument(
        "--task-file",
        dest="task_file",
        help="path to a plan JSON whose 'task' field holds the task",
    )
    ap.add_argument("--out", default=str(BENCH / "out"), help="output root")
    a = ap.parse_args()
    if a.task_file:
        payload = json.loads(Path(a.task_file).read_text())
        task = payload.get("task", payload)
    else:
        task = json.loads(a.task)
    run(task, Path(a.out))


if __name__ == "__main__":
    main()
