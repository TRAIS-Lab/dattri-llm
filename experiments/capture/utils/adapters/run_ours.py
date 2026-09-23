"""dattri_llm adapter for the universal benchmark.

Runs one of the paper's three methods on a (HF model, dataset) task, timing
every phase and recording peak memory (see ``log.BenchRun``):

    graddot   TracInAttributor (gradient dot; projected or full dimension)
    kfac      KFACAttributor  -- rank-64: compact materialized "logra" store +
              capture-time covariances; full dimension: live fit and score
    ekfac     EKFACAttributor -- the same two recipes, with the corrected spectrum

The task's ``hook_family`` (``linear_io`` or ``invasive_linear_io``) sets the
capture path of every hooked layer; unset, each branch keeps the path of experiments/benchmark.
``shared_fit`` makes the invasive run of a pair score against the curvature fit
of its ``linear_io`` run.

    python run_ours.py --task-file <plan.json> --out <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "benchmark" / "utils"
sys.path.insert(0, str(BENCH))  # data.py, models.py, log.py of experiments/benchmark

import torch
import torch.nn.functional as F

import models
from data import load_task_data
from log import BenchRun

from dattri.task import AttributionTask
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64  # per factor side -> 64*64 = 4096 per-layer projected space
DAMPING = 1e-3
LIB = "dattri_llm"


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
    """Every nn.Linear in the transformer blocks (excludes lm_head / embeddings),
    family-agnostic -- attention + MLP projections carry the bulk of the params."""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "embed" not in n:
            names.append(n)
    return names


def loss_builder(model):
    def loss_func(params, batch):
        out = torch.func.functional_call(
            model,
            params,
            args=(),
            kwargs={
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
        )
        logits = out.logits if hasattr(out, "logits") else out
        labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
        return F.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            labels[:, 1:].flatten(),
            reduction="sum",
            ignore_index=-100,
        )

    return loss_func


def run(task: dict, out_root: Path) -> None:
    model_id = task["model"]
    params_b = task["params_b"]
    method = task["method"]
    n_train = task.get("n_train", 1024)
    # Steady-state timing: warm up on n_warm samples, then time the next n_meas.
    n_warm = task.get("warmup_train", 32)
    n_meas = task.get("measure_train", 32)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)
    proj_mode = task.get("proj_mode", "rank64")

    base = (
        f"{task.get('family', '?')}-{task.get('scale', '?')}-{task['dataset']}-{method}"
        f"-{proj_mode}-q{n_test}"
    )
    rep = f"-r{task['repeat']}" if "repeat" in task else ""
    tag = base + (f"-{task['hook_family']}" if "hook_family" in task else "") + rep
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
        bench.set(
            n_linear_layers=len(layers),
            layers_sha=hashlib.sha1("\n".join(layers).encode()).hexdigest()[:12],
            requires_grad=all(p.requires_grad for p in model.parameters()),
            tf32=torch.backends.cuda.matmul.allow_tf32,
        )

    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(
            model_id, task["dataset"], block_size, n_train, n_test, seed
        )

    loss_func = loss_builder(model)
    # The checkpoint handed to AttributionTask is the model's OWN state dict, not
    # a clone: a clone kept a second full copy of every weight on the device for
    # the whole run (a 29.5 GB model idled at 55 GB), which the memory column
    # then charged to the library.  The probe is frozen (no updates), so the
    # live tensors are the checkpoint; loading them is a self-copy.
    ckpt = model.state_dict()
    task_obj = AttributionTask(
        loss_func=loss_func, model=model, checkpoints=[ckpt], target_func=loss_func
    )
    # The test side is scored as ONE block: score_sources loops over cached test
    # blocks and each iteration re-materializes the train block, so an eval batch
    # smaller than n_test multiplies the train-side work by the block count for
    # no reason.  With eval_batch == n_test the query set becomes a single GEMM
    # dimension -- the same shape bergson's Scorer uses, where query gradients are
    # held as [dim_m, n_queries] and every train batch is consumed exactly once.
    eval_batch = task.get("eval_batch", n_test)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="bench_"),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        dataloader_pin_memory=False,
    )
    bench.set(eval_batch=eval_batch)

    # proj_mode: "rank64" (LoGra rank-64, the low-rank regime) or "full" (no
    # projection -- full-dimension factors, aligned with Bergson/Kronfluence).
    bench.set(proj_mode=proj_mode)
    cached_kfac = method in ("kfac", "ekfac") and proj_mode != "full"
    hook_family = task.get("hook_family") or (
        "linear_io" if cached_kfac else "invasive_linear_io"
    )
    # Explicit names, so both families hook exactly the same layers.
    hook_types = dict.fromkeys(layers, hook_family)
    bench.set(hook_family=hook_family)
    # capture_style="auto" lets the library's capture-time cost model choose
    # the representation per layer, the same rule scoring uses: materialize the
    # factors once S >= k_a*k_g/(k_a+k_g) on their (projected) widths.  Pinning
    # "factorized" for the rank-64 capture once made GradDot carry 16x the
    # payload of K-FAC/EK-FAC (at proj_dim 64 the crossover is S=32 and
    # sequences are 512), so the measured method ordering reflected the capture
    # style rather than the methods.  At full dimension the same rule now runs
    # on the raw widths, materializing the layers whose dense gradient is the
    # smaller form.
    capture_style = "auto"
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
        hook_types=hook_types, projection_kwargs=proj, capture_style=capture_style
    )

    cache_dir = out_root / "store" / tag
    shutil.rmtree(cache_dir, ignore_errors=True)

    fit_dir = None
    if task.get("shared_fit") and method in ("kfac", "ekfac"):
        fit_dir = out_root / "fits" / (base + rep)
        if hook_family == "linear_io":
            shutil.rmtree(fit_dir, ignore_errors=True)
        elif not fit_dir.is_dir():
            msg = f"shared fit {fit_dir} missing: run the linear_io cell first"
            raise FileNotFoundError(msg)
        bench.set(shared_fit=str(fit_dir.relative_to(out_root)))

    if method == "graddot":
        attributor = TracInAttributor(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te, _: attributor.attribute(tr, te, hook_config=hook_config),
            train_ds,
            test_ds,
            n_warm,
            n_meas,
        )
    elif method in ("kfac", "ekfac") and proj_mode == "full":
        # Full-dimension K-FAC/EK-FAC (no projection), aligned with bergson: do
        # NOT store.  ``gradient_cache_residency="disk"`` here wrote the
        # *factorized* (a, g) store -- ~0.81 GB per sample, so ~830 GB at
        # n=1024 -- to $TMPDIR and read it back for the fit and score passes,
        # against the ~12 GB bergson writes.  K-FAC's frozen probe is
        # re-runnable, so gradient_cache_residency=None (the attributor's own
        # default) re-runs the model instead, which is what
        # bergson's `score` pass does: recompute the train gradients on the fly
        # and consume each batch immediately.
        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        attributor = cls(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te, measured: attributor.attribute(
                tr,
                te,
                hook_config=hook_config,
                damping=DAMPING,
                fisher_dir=str(fit_dir) if measured and fit_dir else None,
            ),
            train_ds,
            test_ds,
            n_warm,
            n_meas,
        )
    elif cached_kfac:
        # Native-best (store-based) rank-64 K-FAC/EK-FAC: capture the token-summed
        # projected outer product once (one model pass), fit the Fisher from the
        # store (K-FAC: covariances *at capture* via KroneckerCovarianceCallback;
        # EK-FAC: eigenbases+Lambda via fit()), then score from cache with the
        # persisted factors -- no Fisher re-pass.  The fair analog to LogIX's
        # compact-LoRA store (the default attribute() re-fits, so it is slower).
        #
        # This branch runs under `_warm_then_time` like every other method.  It
        # used to time `n_train + n_test` samples inline with no warm-up, while
        # GradDot timed `n_meas` after one -- so K-FAC/EK-FAC were charged twice
        # the training samples AND the one-off CUDA-context/autotune costs that
        # GradDot's warm-up absorbs.  The measured method gap was then mostly an
        # artifact of the harness, the same way pinning the factorized capture once
        # made the gap an artifact of the capture style (see `style` above).
        from dattri_llm.attribution.utils import collect_gradients
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import GradientStreamer

        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        # Both methods capture the materialized logra block (the 64x64
        # projected gradient per sequence) and collect the projected
        # covariances (A, G) at capture with KroneckerCovarianceCallback, so
        # the store never holds per-token factors (24 GB at n_train=1024
        # against 1.6 GB).  K-FAC saves the covariances as its fit; EK-FAC
        # hands them to fit(covariances=...), which eigendecomposes them and
        # sweeps the compact store once for the corrected spectrum -- the same
        # recipe LogIX uses.  (Before fit(covariances=...) existed, EK-FAC had
        # to store per-token factors and re-derive the covariances from them;
        # and before that, a materialized EK-FAC store silently fell back to a
        # dense 4096x4096 Fisher per layer, a different algorithm.)
        cap_style = "materialized"
        mat_proj = {
            "__default__": {
                "style": "logra",
                "proj_dim": PROJ_DIM,
                "proj_max_batch_size": 32,
                "proj_type": "rademacher",
                "proj_seed": 0,
            }
        }
        mat_config = HookManagerConfig(
            hook_types=hook_types, projection_kwargs=mat_proj, capture_style=cap_style
        )
        # Record the style actually used: `capture_style` was set to "auto"
        # above, before this branch, so rows from this path all claimed "auto"
        # regardless -- which is exactly why pre-fix and post-fix EK-FAC rows
        # are indistinguishable in results.jsonl.
        bench.set(capture_style=cap_style, kfac_store_style=cap_style)
        # One store per invocation.  Reusing a single directory would force an
        # rmtree of the warm-up's shards *inside* the timed phase; a fresh path
        # costs nothing to create, and the warm-up store is deleted afterwards
        # (untimed) so `record_disk` still measures only the measured run.
        kfac_stores: list[Path] = []

        def _kfac_cached(tr_ds, te_ds, measured):
            """Capture -> fit -> score-from-cache for one training slice.

            Everything stateful is rebuilt per call so the warm-up cannot leak
            into the measured run: a reused KroneckerCovarianceCallback would
            accumulate the warm-up slice's covariances into the measured
            Fisher, and a store left in place would be re-read with the
            warm-up's shards still in it.
            """
            run_store = cache_dir / f"run{len(kfac_stores)}"
            kfac_stores.append(run_store)
            train_dir, test_dir = str(run_store / "train"), str(run_store / "test")
            attributor = cls(args, task=task_obj)
            cov = KroneckerCovarianceCallback()
            probe = attributor.load_checkpoint(0)
            splits: dict[str, float] = {}
            with _split(splits, "capture_train"):
                tr = GradientStreamer(
                    probe,
                    tr_ds,
                    args,
                    batch_size=batch,
                    loss_fn=attributor.train_loss_fn(),
                    config=mat_config,
                )
                tr.hook_manager.add_callback(cov)  # projected (A, G) at capture
                collect_gradients(tr, GradientStorageManager(train_dir))
            with _split(splits, "capture_test"):
                te = GradientStreamer(
                    probe,
                    te_ds,
                    args,
                    batch_size=batch,
                    loss_fn=attributor.test_loss_fn(),
                    config=mat_config,
                )
                collect_gradients(te, GradientStorageManager(test_dir))
            shared = str(fit_dir) if measured and fit_dir else None
            with _split(splits, "fit"):
                if shared and hook_family != "linear_io":
                    fisher_dir = shared
                elif method == "kfac":
                    fisher_dir = attributor.save_fisher(cov.result(), shared)
                else:
                    fisher_dir = attributor.fit(
                        train_dir, shared, covariances=cov.result()
                    )
            with _split(splits, "score"):
                result = attributor.attribute_from_cache(
                    train_dir, test_dir, damping=DAMPING, fisher_dir=fisher_dir
                )
            bench.set(attribute_splits=splits)  # the measured call writes last
            return result

        score = _warm_then_time(bench, _kfac_cached, train_ds, test_ds, n_warm, n_meas)
        for spent in kfac_stores[:-1]:  # untimed: keep only the measured store
            shutil.rmtree(spent, ignore_errors=True)
    else:
        msg = f"unsupported method {method!r}"
        raise ValueError(msg)

    bench.record_disk("store", cache_dir)
    train_ids, matrix = score.agnostic_matrix()
    matrix = matrix.cpu().float()
    # Hashes let a comparison align rows and columns across runs.
    score_file = run_dir / "score.pt"
    torch.save(
        {"score": matrix, "train_ids": train_ids, "test_ids": list(score.test_ids)},
        score_file,
    )
    bench.set(
        score_shape=list(matrix.shape),
        score_file=str(score_file.relative_to(out_root)),
    )
    bench.finish(status="ok")
    print(f"[done] {tag}: score {tuple(matrix.shape)}", flush=True)


def _warm_then_time(bench, call, train_ds, test_ds, n_warm, n_meas):
    """Run ``call`` once as an untimed warm-up, then time it on a fresh slice.

    The first invocation on a fresh process pays one-off costs -- CUDA context
    setup, autotuned/compiled kernels, allocator growth -- that are not part of
    steady-state attribution throughput.  Warming up on ``n_warm`` training
    samples and then timing the *next* ``n_meas`` gives a per-sample cost that is
    comparable across libraries and independent of the workload size.
    """
    from torch.utils.data import Subset

    if n_warm:
        call(Subset(train_ds, range(n_warm)), test_ds, False)
    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))
    with bench.phase("attribute", n_meas):
        return call(measured, test_ds, True)


@contextlib.contextmanager
def _split(splits: dict, name: str):
    """GPU-synced wall time of one step inside a phase (peak stats untouched)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.monotonic()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    splits[name] = round(time.monotonic() - t0, 3)


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
