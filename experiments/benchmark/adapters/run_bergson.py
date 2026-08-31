"""bergson adapter for the universal benchmark.

bergson drives an on-disk gradient *index* through its own CLI / YAML pipeline
rather than an in-process ``attribute()`` call, so this adapter shells out to
``python -m bergson`` and times the end-to-end pipeline, sampling GPU memory via
NVML (bergson runs in a subprocess, so torch's allocator counter cannot see it).
It writes a record in the same schema as the other adapters (phases ``fit`` and
``score``; GPU peak from NVML) so ``summarize.py`` picks it up.

Native strategy (bergson's own): fixed ``chunk_length`` sequences, rank-64
projection for the grad-dot index; the K-FAC/EK-FAC pipeline scores with
full-dimension gradients (``projection_dim=0``), like Kronfluence.

    method -> bergson:
      graddot  build (rank-64 index) + programmatic grad-dot query
      kfac     ekfac pipeline, ev_correction=false  (full-dim factors)
      ekfac    ekfac pipeline, ev_correction=true
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

from log import device_details

LIB = "bergson"
PROJ_DIM = 64
CHUNK = 512
# bergson HF dataset spec per benchmark dataset name: (data_str, subset)
DATASETS = {
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
}
# Bergson materializes its gradient index on disk (>100 GB at full dimension), so
# this must point at a filesystem with room -- and must not be a hardcoded path
# from one machine.  ``BENCH_CACHE`` is set per-cluster by the launcher.
DAMPING = 0.1  # bergson --damping_factor (relative to mean eigenvalue)
# bergson takes a *row* split and re-chunks it to ``chunk_length`` itself, so a
# row count is not a workload: the query/train count it ends up with is
# ``total_tokens // CHUNK`` (a partial tail is dropped, not padded).
#
# The previous implementation converted a target chunk count with a fitted
# constant (~8.2 rows per chunk) measured from ONE observation.  WikiText token
# density is far from uniform -- its opening rows are blank lines and section
# headers -- so the constant mistranslated every request: 16 queries became 18,
# and 1 query became 0 (an empty dataset), then 6 once a MIN_ROWS floor was
# bolted on.  Every other library slices a pre-chunked pool and therefore hits
# n_test exactly; bergson can too.
#
# So: tokenize and accumulate until the token budget is reached.  Deterministic,
# dataset-agnostic, and exact -- no fitted constant, no floor.
_ROWS_CACHE: dict = {}


def _rows_for_chunks(model_id: str, data_str: str, subset: str | None,
                     chunks: int, split: str = "train") -> int:
    """Rows of *split* whose tokenization yields exactly ``chunks`` chunks.

    Returns the smallest row count whose cumulative token count reaches
    ``chunks * CHUNK`` -- i.e. the split bergson will re-chunk into exactly
    ``chunks`` blocks of ``CHUNK`` tokens.
    """
    key = (model_id, data_str, subset, chunks, split)
    if key in _ROWS_CACHE:
        return _ROWS_CACHE[key]

    from datasets import load_dataset
    from transformers import AutoTokenizer

    target = chunks * CHUNK
    tok = AutoTokenizer.from_pretrained(model_id)
    # Read in growing windows so a large target does not tokenize the corpus.
    window, cum, rows = max(4096, chunks * 64), 0, 0
    while True:
        spec = f"{split}[{rows}:{rows + window}]"
        ds = load_dataset(data_str, subset, split=spec) if subset \
            else load_dataset(data_str, split=spec)
        if len(ds) == 0:
            break
        for text in ds["text"]:
            cum += len(tok(text).input_ids)
            rows += 1
            if cum >= target:
                _ROWS_CACHE[key] = rows
                return rows
    msg = (f"{data_str} {split} has only {cum} tokens, short of the "
           f"{target} needed for {chunks} chunks of {CHUNK}")
    raise ValueError(msg)


def _batch_args(task: dict) -> list:
    """CLI flags pinning bergson's effective batch to the benchmark's.

    bergson has no per-sequence batch setting; the batch it ends up with is the
    minimum of two independent constraints in ``allocate_batches``:

        max(len in batch) * |batch| <= token_batch_size      (token budget)
        |batch|                     <= max_batch_size        (document cap)

    Both must be set or neither pins the batch.  ``max_batch_size`` alone only
    *caps*, so raising it above the token budget's implied batch is a no-op --
    which is why an earlier run passing ``--max_batch_size 8`` still ran at 4,
    the default 2048-token budget divided by the 512-token sequence length.  So:
    put the document cap at the target batch and give the token budget headroom
    (2x) so it never binds, making the effective batch exactly ``batch``.
    """
    batch = task.get("batch", 8)
    return ["--max_batch_size", str(batch),
            "--token_batch_size", str(batch * CHUNK * 2)]


def _index_counts(path: str) -> dict:
    """``num_rows`` / ``num_scores`` bergson recorded for an index, if present."""
    info = Path(path) / "info.json"
    if not info.exists():
        return {}
    try:
        d = json.loads(info.read_text())
    except (OSError, ValueError):
        return {}
    return {k: d[k] for k in ("num_items", "num_rows", "num_scores") if k in d}


def _run(cli_args: list) -> None:
    """``python -m bergson <args>``, surfacing stderr on failure."""
    proc = subprocess.run([sys.executable, "-m", "bergson", *cli_args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("=== bergson cmd ===\n" + " ".join(cli_args), flush=True)
        print("=== bergson STDERR (tail) ===\n" + (proc.stderr or "")[-3000:],
              flush=True)
        proc.check_returncode()
STORE_ROOT = os.environ.get(
    "BENCH_CACHE", str(Path.home() / "bergson_bench")) + "/bergson_bench"


class SmiPeak:
    """Poll nvidia-smi for device memory.used (MB) while a subprocess runs."""

    def __init__(self, gpu: str = "0") -> None:
        self.gpu, self.peak_mb = gpu, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False).stdout.strip()
            if out.isdigit():
                self.peak_mb = max(self.peak_mb, int(out))
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)

    @property
    def peak_gb(self) -> float:
        return round(self.peak_mb / 1024, 3)


def _dir_bytes(p: str) -> int:
    root = Path(p)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def run_graddot(task, data_str, subset, train_rows, query_rows, run_path):
    """bergson GradDot through the CLI ``score`` path.

    ``score`` drives :class:`bergson.score.scorer.Scorer`, which streams the index
    one module at a time out of the memmap and accumulates ``[batch, n_query]``.
    The previous implementation used :class:`bergson.query.attributor.Attributor`
    instead, whose constructor does

        self.grads = {name: numpy_to_tensor(mmap[:, lo:hi]).to(device=device) ...}

    i.e. it pulls the *entire* index onto the device.  That is fine at rank-64 and
    exhausts the card at full dimension, which is what the adapter was reporting
    as bergson OOM-ing -- it was our choice of entry point, not a library limit.

    Phases: ``fit`` builds the query index; ``score`` walks the train split, whose
    gradients are computed on the fly and never indexed.
    """
    proj = 0 if task.get("proj_mode", "rank64") == "full" else PROJ_DIM
    query_path = f"{run_path}/query"
    score_path = f"{run_path}/scores"
    common = ["--model", task["model"], "--dataset", data_str,
              "--chunk_length", str(CHUNK), "--projection_dim", str(proj)]
    if subset:
        common += ["--subset", subset]
    common += _batch_args(task)
    if task.get("bergson_max_batch"):
        common += ["--max_batch_size", str(task["bergson_max_batch"])]

    smi = SmiPeak()
    with smi:
        t0 = time.perf_counter()
        _run(["build", query_path, "--overwrite", "true",
              "--split", f"train[:{query_rows}]", *common])
        fit_wall = time.perf_counter() - t0

        t1 = time.perf_counter()
        _run(["score", score_path, "--query_path", query_path,
              "--split", f"train[:{train_rows}]", *common])
        score_wall = time.perf_counter() - t1

    counts = {"query": _index_counts(query_path), "scores": _index_counts(score_path)}
    return ({"wall": round(fit_wall, 3), "mem": smi.peak_gb},
            {"wall": round(score_wall, 3), "mem": smi.peak_gb},
            counts)


def run_hessian(task, method, data_str, subset, train_rows, query_rows, run_path):
    """bergson's K-FAC / EK-FAC influence pipeline (CLI >=0.26).

    Use bergson's own ``ekfac`` subcommand, which is the entry point to
    ``hessians/pipeline.py`` and runs all four steps in one process:

        1. build the QUERY gradients at FULL dimension (the pipeline forces
           ``query_cfg.projection_dim = 0``)
        2. fit the Kronecker factors on the training data
        3. apply the inverse Hessian to the query gradient, and *only then*
           project to ``projection_dim``
        4. score the training examples against the transformed query

    Step 3 is why the hand-rolled ``hessian``/``build``/``score`` sequence could
    not work with projection on: ``Preconditioner.apply`` reshapes each block to
    ``[n, O, I]``, so it needs gradients in full parameter space, but a
    ``build --projection_dim 64`` hands it 64-dim vectors and the reshape fails.
    bergson preconditions first and projects second; the order is not optional.

    The pipeline is a single command, so the whole run is reported as the fit
    phase (the cross-library table compares ``total_wall_s`` regardless).
    """
    ev = "true" if method == "ekfac" else "false"
    # EK-FAC still forbids projection outright -- hessian_approximations.py
    # raises when ev_correction is set and index_cfg.projection_dim != 0, and
    # step 2 above inherits index_cfg.  So EK-FAC is full-dimension only and its
    # cell belongs in the full-dim column whatever the preset asks for.
    proj = 0 if (task.get("proj_mode", "rank64") == "full" or method == "ekfac") \
        else PROJ_DIM

    args = ["ekfac", run_path, "--overwrite", "true",
            "--model", task["model"], "--method", "kfac",
            "--hessian_cfg.ev_correction", ev,
            "--projection_dim", str(proj),
            "--data.dataset", data_str, "--data.split", f"train[:{train_rows}]",
            "--data.chunk_length", str(CHUNK),
            "--query.dataset", data_str, "--query.split", f"train[:{query_rows}]",
            "--query.chunk_length", str(CHUNK),
            "--hessian_pipeline_cfg.inversion_cfg.damping_factor", str(DAMPING),
            # HessianPipelineConfig.query_aggregation defaults to "mean", which
            # collapses the whole query set into ONE mean gradient and emits a
            # single score column -- 1/16th of the query work every other library
            # in the table performs, and the reason bergson's hessian cells never
            # hit the memory wall.  "none" gives one column per query.
            "--query_aggregation", "none"]
    if subset:
        args += ["--data.subset", subset, "--query.subset", subset]
    args += _batch_args(task)
    if task.get("bergson_max_batch"):
        args += ["--max_batch_size", str(task["bergson_max_batch"])]

    smi = SmiPeak()
    with smi:
        t0 = time.perf_counter()
        _run(args)
        fit_wall = time.perf_counter() - t0
    score_wall = 0.0
    counts = {"query": _index_counts(f"{run_path}/query"),
              "scores": _index_counts(f"{run_path}/scores")}

    return ({"wall": round(fit_wall, 3), "mem": smi.peak_gb},
            {"wall": round(score_wall, 3), "mem": smi.peak_gb},
            counts)


def run(task: dict, out_root: Path) -> None:
    method = task["method"]
    proj_mode = task.get("proj_mode", "rank64")
    if method not in ("graddot", "kfac", "ekfac"):
        msg = f"bergson adapter covers graddot/kfac/ekfac, not {method!r}"
        raise ValueError(msg)
    data_str, subset = DATASETS[task["dataset"]]
    # bergson splits by ROW and re-chunks itself, so convert the benchmark's
    # chunk-count workload into row counts (see ROWS_PER_CHUNK).
    # Exact row splits: the benchmark asks for n_train / n_test blocks of CHUNK
    # tokens, and every other library gets exactly that from a pre-chunked pool.
    train_rows = task.get("bergson_docs") or _rows_for_chunks(
        task["model"], data_str, subset, task.get("n_train", 1024))
    query_rows = _rows_for_chunks(
        task["model"], data_str, subset, task.get("n_test", 16))
    # proj_mode MUST be in the tag: the r=64 and full arrays are separate SLURM
    # jobs and run concurrently, so a shared store dir means one run's `rm -rf`
    # races the other's build and leaves a stale `<path>.part` behind, which
    # validate_run_path then refuses.
    tag = (f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}"
           f"-{method}-{proj_mode}")
    store = f"{STORE_ROOT}/{tag}"
    subprocess.run(["rm", "-rf", store], check=False)

    t_start = time.perf_counter()
    if method == "graddot":
        fit, score, counts = run_graddot(
            task, data_str, subset, train_rows, query_rows, store)
    else:
        fit, score, counts = run_hessian(
            task, method, data_str, subset, train_rows, query_rows, store)
    # What bergson actually scored, so an unequal workload can never again hide
    # behind a row count that looked like a sample count.
    score_shape = [counts.get("scores", {}).get("num_rows"),
                   counts.get("scores", {}).get("num_scores")]

    record = {
        "lib": LIB,
        "task": {**task, "block_size": CHUNK, "proj_dim": (0 if proj_mode=="full" else PROJ_DIM), "proj_mode": proj_mode,
                 "bergson_train_rows": train_rows, "bergson_query_rows": query_rows,
                 "bergson_counts": counts, "strategy": "native"},
        "total_wall_s": round(time.perf_counter() - t_start, 3),
        "phases": [
            {"phase": "fit", "wall_s": round(fit["wall"], 3),
             "gpu_peak": [{"index": 0, "alloc_gb": fit["mem"]}], "cpu_rss_peak_gb": 0},
            {"phase": "score", "wall_s": round(score["wall"], 3),
             "gpu_peak": [{"index": 0, "alloc_gb": score["mem"]}], "cpu_rss_peak_gb": 0},
        ],
        "disk_gb": {"store": round(_dir_bytes(store) / 1024 ** 3, 4)},
        "disk_total_gb": round(_dir_bytes(store) / 1024 ** 3, 4),
        "mem_source": "nvml",
        "device": device_details(),
        "score_shape": score_shape,
        "status": "ok",
    }
    results = out_root / "results.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] bergson {tag}: fit {fit['wall']:.1f}s score {score['wall']:.1f}s "
          f"mem {max(fit['mem'], score['mem']):.1f}GB", flush=True)


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
