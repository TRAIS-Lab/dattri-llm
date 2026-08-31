"""dattri_llm adapter for the universal benchmark.

Runs any supported attribution method on any (HF model, dataset) task from the
shared registry, with capture-time LoGra projection (proj_dim 64 per factor side)
via the invasive linear-io hook, and logs everything through ``log.BenchRun``.

    python adapters/dattri_llm.py --task '{"model":"EleutherAI/pythia-410m",
        "dataset":"wikitext103","method":"kfac","parallelism":"single",...}'

Method -> attributor:
    graddot | tracin   TracInAttributor (raw projected-gradient dot)
    gradcos            TracInAttributor (normalized_grad -> cosine)
    kfac               KFACAttributor
    ekfac              EKFACAttributor
    efim               KFACAttributor, dense empirical Fisher (non_kfac_strategy=direct)
    dvemb              DVEmbAttributor
"""

from __future__ import annotations

import argparse
import json
import os
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

from dattri.task import AttributionTask
from dattri_llm.attribution.algorithm.dvemb import DVEmbAttributor
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64      # per factor side -> 64*64 = 4096 per-layer projected space
DAMPING = 1e-3
LIB = "dattri_llm"


def build_model(model_id: str, params_b: float):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[models.dtype_for(params_b)]
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
            names.append(f"{n}$")
    return names


def loss_builder(model):
    def loss_func(params, batch):
        out = torch.func.functional_call(
            model, params, args=(),
            kwargs={"input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]})
        logits = out.logits if hasattr(out, "logits") else out
        labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
        return F.cross_entropy(
            logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(),
            reduction="sum", ignore_index=-100)
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


    tag = f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}-{method}"
    run_dir = out_root / "runs" / tag
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch, "proj_dim": PROJ_DIM},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB)

    with bench.phase("build_model"):
        model, tok = build_model(model_id, params_b)
        layers = linear_layer_names(model)
        bench.set(n_linear_layers=len(layers))

    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(model_id, task["dataset"], block_size,
                                           n_train, n_test, seed)

    loss_func = loss_builder(model)
    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task_obj = AttributionTask(loss_func=loss_func, model=model,
                               checkpoints=[ckpt], target_func=loss_func)
    # The test side is scored as ONE block: score_sources loops over cached test
    # blocks and each iteration re-materializes the train block, so an eval batch
    # smaller than n_test multiplies the train-side work by the block count for
    # no reason.  With eval_batch == n_test the query set becomes a single GEMM
    # dimension -- the same shape bergson's Scorer uses, where query gradients are
    # held as [dim_m, n_queries] and every train batch is consumed exactly once.
    eval_batch = task.get("eval_batch", n_test)
    # Prefetch depth: 1 keeps the next train block staged while the current one
    # is scored.  On the live-capture path the block is produced ON the device,
    # so there is no host->device copy to overlap -- the deque simply pulls the
    # next block early, holding two whole-model factorized blocks at once.  0
    # reverts to one block at a time.  Scores are identical either way.
    prefetch_depth = int(
        os.environ.get("DATTRI_PREFETCH_DEPTH", task.get("prefetch_depth", 1)),
    )
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="bench_"),
        per_device_train_batch_size=batch, per_device_eval_batch_size=eval_batch,
        device_prefetch_depth=prefetch_depth,
        dataloader_pin_memory=False)
    bench.set(eval_batch=eval_batch, prefetch_depth=prefetch_depth)

    # proj_mode: "rank64" (LoGra rank-64, the low-rank regime) or "full" (no
    # projection -- full-dimension factors, aligned with Bergson/Kronfluence).
    proj_mode = task.get("proj_mode", "rank64")
    bench.set(proj_mode=proj_mode)
    materialized = method == "efim"
    # "auto" lets the library's capture-time cost model choose, the same rule
    # scoring uses: materialize the projected factors once S >= k_a*k_g/(k_a+k_g).
    # Pinning "logra_factorized" here (the old default for everything but eFIM)
    # made GradDot carry 16x the payload of K-FAC/EK-FAC at r=64 -- at proj_dim
    # 64 the crossover is S=32 and sequences are 512 -- so the measured method
    # ordering reflected the capture style rather than the methods.  At full
    # dimension proj is None and this is moot; there the same rule keeps factors.
    style = "auto"
    proj = None if proj_mode == "full" else {"__default__": {
        "style": style, "proj_dim": PROJ_DIM, "proj_max_batch_size": 32,
        "proj_type": "rademacher", "proj_seed": 0}}
    bench.set(capture_style=style)
    hook_type = "linear_io" if materialized else "invasive_linear_io"
    hook_config = HookManagerConfig(**{hook_type: layers}, projection=proj)

    cache_dir = out_root / "store" / tag
    shutil.rmtree(cache_dir, ignore_errors=True)

    if method in ("graddot", "tracin", "gradcos"):
        attributor = TracInAttributor(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(
                tr, te, hook_config=hook_config,
                normalized_grad=(method == "gradcos")),
            train_ds, test_ds, n_warm, n_meas)
    elif method == "dvemb":
        attributor = DVEmbAttributor(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(tr, te, hook_config=hook_config),
            train_ds, test_ds, n_warm, n_meas)
    elif method in ("kfac", "ekfac") and proj_mode == "full":
        # Full-dimension K-FAC/EK-FAC (no projection), aligned with bergson: do
        # NOT store.  ``residency="disk"`` here wrote the *factorized* (a, g)
        # store -- ~0.81 GB per sample, so ~830 GB at n=1024 -- to $TMPDIR and
        # read it back for the fit and score passes, against the ~12 GB bergson
        # writes.  K-FAC's frozen probe is re-runnable, so residency=None (the
        # attributor's own default) re-runs the model instead, which is what
        # bergson's `score` pass does: recompute the train gradients on the fly
        # and consume each batch immediately.
        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        attributor = cls(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(tr, te, hook_config=hook_config,
                                                damping=DAMPING),
            train_ds, test_ds, n_warm, n_meas)
    elif method in ("kfac", "ekfac"):
        # Native-best (store-based) rank-64 K-FAC/EK-FAC: capture the token-summed
        # projected outer product once (one model pass), fit the Fisher from the
        # store (K-FAC: covariances *at capture* via KroneckerCovarianceCallback;
        # EK-FAC: eigenbases+Lambda via fit()), then score from cache with the
        # persisted factors -- no Fisher re-pass.  The fair analog to LogIX's
        # compact-LoRA store (the default attribute() re-fits, so it is slower).
        from dattri_llm.attribution.utils import collect_to_disk, task_loss_fn
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import GradientStreamer
        cls = EKFACAttributor if method == "ekfac" else KFACAttributor
        attributor = cls(args, task=task_obj)
        mat_proj = {"__default__": {"style": "logra_materialized", "proj_dim": PROJ_DIM,
                                    "proj_max_batch_size": 32, "proj_type": "rademacher",
                                    "proj_seed": 0}}
        mat_config = HookManagerConfig(linear_io=layers, projection=mat_proj)
        train_dir, test_dir = str(cache_dir / "train"), str(cache_dir / "test")
        cov = KroneckerCovarianceCallback() if method == "kfac" else None
        with bench.phase("attribute", n_train + n_test):
            task_obj._load_checkpoints(0)
            tr = GradientStreamer(task_obj.get_model(), train_ds, args, batch_size=batch,
                                  loss_fn=task_loss_fn(task_obj.original_loss_func),
                                  config=mat_config)
            if cov is not None:
                tr.hook_manager.add_callback(cov)  # projected (A, G) at capture
            collect_to_disk(tr, GradientStorageManager(train_dir))
            te = GradientStreamer(task_obj.get_model(), test_ds, args, batch_size=batch,
                                  loss_fn=task_loss_fn(task_obj.original_target_func),
                                  config=mat_config)
            collect_to_disk(te, GradientStorageManager(test_dir))
            fisher_dir = (attributor.save_fisher(cov.result()) if cov is not None
                          else attributor.fit(train_dir))
            score = attributor.attribute_from_cache(
                train_dir, test_dir, damping=DAMPING, fisher_dir=fisher_dir)
    elif method == "efim":
        attributor = KFACAttributor(args, task=task_obj)
        score = _warm_then_time(
            bench,
            lambda tr, te: attributor.attribute(tr, te, hook_config=hook_config,
                                                damping=DAMPING, residency="disk",
                                                non_kfac_strategy="direct"),
            train_ds, test_ds, n_warm, n_meas)
    else:
        msg = f"unsupported method {method!r}"
        raise ValueError(msg)

    # The residency="disk" paths write their gradient store to a tempfile.mkdtemp
    # inside the attributor, NOT to cache_dir, so recording only cache_dir made
    # those runs report disk=0.0 GB while they were in fact writing the whole
    # factorized store to $TMPDIR.  Measure both.
    bench.record_disk("store", cache_dir)
    for tmp in sorted(Path(tempfile.gettempdir()).glob("kron_*")):
        bench.record_disk(f"tmp/{tmp.name}", tmp)
    _, matrix = score.agnostic_matrix()
    matrix = matrix.cpu().float()
    torch.save({"score": matrix}, run_dir / "score.pt")
    bench.set(score_shape=list(matrix.shape))
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
        call(Subset(train_ds, range(n_warm)), test_ds)
    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))
    with bench.phase("attribute", n_meas):
        return call(measured, test_ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="task JSON (single object)")
    g.add_argument("--task-file", dest="task_file",
                   help="path to a plan JSON whose 'task' field holds the task")
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
