"""Bergson adapter for the benchmark (requires Bergson 0.26.1).

Runs one benchmark cell and appends a JSON row to ``<out>/results.jsonl``
with per-phase wall time and peak GPU memory.

Bergson drives an on-disk gradient *index* through its ``build`` / ``score`` /
``ekfac`` subcommands.  They run in this process through
``bergson.__main__.main``, i.e. through the argument parser of the
``python -m bergson`` CLI.  At ``nproc_per_node=1`` Bergson runs its worker in
the calling process; sharded runs (``parallelism: fsdp``, ``n_gpus > 1``)
spawn worker processes.

Phases of a row:

    build_model  Bergson's model loader (``setup_model_and_peft``); the model
                 is loaded once and reused by every later Bergson step
    load_data    Bergson's dataset load + tokenization
                 (``setup_data_pipeline``)
    fit, score   attribution; the time this process spends in the two setup
                 phases is subtracted (the ``ekfac`` pipeline is one command
                 and is reported as ``fit``)

GPU peak memory comes from torch's allocator.  Bergson's sharded steps run in
worker processes, so for those runs the peak is NVML's maximum over the cards
(``mem_source: "nvml"``), and the workers' model-load and start-up times are
reported in ``bergson_worker_overhead`` (see ``bergson_site/sitecustomize.py``).
An optional warm-up (``warmup_train`` chunks through the same pipeline into a
throwaway store) precedes the timed run, as in ``run_ours.py``.

Settings: fixed ``chunk_length`` sequences; ``proj_mode`` selects a rank-64
projection (``rank64``) or full-dimension gradients (``full``).

    method -> Bergson:
      graddot  ``build`` (query index) + ``score``
      kfac     ``ekfac`` pipeline, ev_correction=false
      ekfac    ``ekfac`` pipeline, ev_correction=true (full dimension)

Environment: ``BERGSON_STORE`` is the directory for Bergson's on-disk index;
``BENCH_CACHE`` is used when it is unset, then ``~/bergson_bench``.
"""

from __future__ import annotations

import argparse
import json
import contextlib
import functools
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

# TF32 matmuls for float32 runs, as in run_ours.py / run_logix.py.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

from log import device_details
from versions import require

LIB = "bergson"
PROJ_DIM = 64
CHUNK = 512  # default chunk length; a task's ``block_size`` overrides it
# Hugging Face dataset spec passed to Bergson's CLI, per benchmark dataset
# name: (data_str, subset).  The ids are the namespaced ones used in data.py.
DATASETS = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
}

# Task ``dtype`` -> Bergson precision name.  Bergson's precision is a
# per-subcommand flag: ``build`` takes ``--precision``; ``score`` and ``ekfac``
# carry both ``IndexConfig.precision`` (model parameters) and
# ``ScoreConfig.precision`` (dtype the gradients are converted to before
# scoring), set through ``--index_cfg.precision`` / ``--score_cfg.precision``.
_PRECISION = {"float32": "fp32", "bfloat16": "bf16"}


def _bergson_precision(task: dict) -> str:
    """Bergson's name for the task's ``dtype``; raises on an unmapped dtype."""
    wanted = task.get("dtype", "float32")
    if wanted not in _PRECISION:
        raise ValueError(f"run_bergson.py cannot set bergson to dtype={wanted!r}; "
                         f"supported: {sorted(_PRECISION)}")
    return _PRECISION[wanted]


def _precision_flags(task: dict, subcommand: str) -> list:
    """The precision flags *subcommand* accepts (see ``_PRECISION``)."""
    p = _bergson_precision(task)
    if subcommand == "build":
        return ["--precision", p]
    return ["--index_cfg.precision", p, "--score_cfg.precision", p]


DAMPING = 0.1  # Bergson --damping_factor (relative to mean eigenvalue)
# Bergson takes a *row* split and re-chunks it to ``chunk_length`` itself; the
# chunk count it ends up with is ``total_tokens // chunk_length`` (a partial
# tail is dropped).  ``_rows_for_chunks`` converts a chunk count into the row
# count that yields exactly that many chunks; results are memoized here.
_ROWS_CACHE: dict = {}


def _rows_for_chunks(model_id: str, data_str: str, subset: str | None,
                     chunks: int, split: str = "train", chunk: int = CHUNK) -> int:
    """Rows of *split* whose tokenization yields exactly ``chunks`` chunks.

    Returns the smallest row count whose cumulative token count reaches
    ``chunks * chunk`` -- i.e. the split bergson will re-chunk into exactly
    ``chunks`` blocks of ``chunk`` tokens.
    """
    key = (model_id, data_str, subset, chunks, split, chunk)
    if key in _ROWS_CACHE:
        return _ROWS_CACHE[key]

    from datasets import load_dataset
    from transformers import AutoTokenizer

    target = chunks * chunk
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
           f"{target} needed for {chunks} chunks of {chunk}")
    raise ValueError(msg)


def _parallel_args(task: dict) -> list:
    """Bergson's parallelism flags, taken from the task's topology.

    ``--nproc_per_node`` is passed explicitly as the task's ``n_gpus``
    (Bergson's default is one process per visible GPU).  A task with
    ``parallelism: fsdp`` and ``n_gpus > 1`` also gets ``--fsdp true``,
    Bergson's sharding switch.  A step whose dataset is smaller than the world
    size is run by Bergson in the calling process without a process group;
    ``_ModelCache`` loads the model unsharded for such a step.
    """
    n = int(task.get("n_gpus", 1) or 1)
    args = ["--nproc_per_node", str(n)]
    if task.get("parallelism") == "fsdp" and n > 1:
        args += ["--fsdp", "true"]
    return args


def _batch_args(task: dict) -> list:
    """CLI flags setting Bergson's effective batch to the task's ``batch``.

    Bergson's batch is the minimum of two constraints in ``allocate_batches``:

        max(len in batch) * |batch| <= token_batch_size      (token budget)
        |batch|                     <= max_batch_size        (document cap)

    ``--max_batch_size`` is set to ``batch`` and ``--token_batch_size`` to
    twice ``batch * block_size``, so the document cap is the binding
    constraint and the effective batch is ``batch`` sequences.
    """
    batch = task.get("batch", 8)
    chunk = int(task.get("block_size", CHUNK))
    return ["--max_batch_size", str(batch),
            "--token_batch_size", str(batch * chunk * 2)]


def _index_counts(path: str) -> dict:
    """Counts (``num_items`` / ``num_rows`` / ``num_scores``) from an index's
    ``info.json``, if present."""
    info = Path(path) / "info.json"
    if not info.exists():
        return {}
    try:
        d = json.loads(info.read_text())
    except (OSError, ValueError):
        return {}
    return {k: d[k] for k in ("num_items", "num_rows", "num_scores") if k in d}


def _run_inprocess(cli_args: list) -> None:
    """Run a Bergson subcommand in this process.

    ``bergson.__main__.main`` parses ``sys.argv`` with Bergson's CLI parser
    and calls ``<Command>.execute()``, so ``cli_args`` are the arguments of a
    ``python -m bergson`` invocation.  With ``--nproc_per_node 1`` every
    pipeline step runs in the calling process.
    """
    import bergson.__main__ as bergson_main

    _install_model_cache()
    saved = sys.argv
    sys.argv = ["bergson", *[str(a) for a in cli_args]]
    try:
        bergson_main.main()
    except BaseException:
        print("=== bergson cmd ===\n" + " ".join(map(str, cli_args)), flush=True)
        raise
    finally:
        sys.argv = saved


class _ModelCache:
    """Memoized, timed wrapper of Bergson's ``setup_model_and_peft``.

    Every Bergson worker (build, score, hessian fit) loads its model through
    that function.  The wrapper times each load (``load_s``, reported as the
    ``build_model`` phase) and memoizes the result per configuration, so later
    steps reuse the loaded model, as ``run_ours.py`` and ``run_kronfluence.py``
    reuse theirs.  Bergson's collectors register their hooks inside a context
    manager and remove them on exit.
    """

    def __init__(self, real):
        self.real = real
        self.models: dict = {}
        self.load_s = 0.0
        self.n_loads = 0
        self.n_calls = 0

    def __call__(self, cfg, *args, **kwargs):
        self.n_calls += 1
        if getattr(cfg, "fsdp", False) and not torch.distributed.is_initialized():
            # A step Bergson runs in the calling process, without a process
            # group: the loader gets a copy of the config with fsdp=False, so
            # it loads the model unsharded on this GPU.
            import copy
            cfg = copy.copy(cfg)
            cfg.fsdp = False
        key = (cfg.model, cfg.precision, getattr(cfg, "revision", None),
               getattr(cfg, "fsdp", False), getattr(cfg, "peft_init_kwargs", None),
               args, tuple(sorted(kwargs.items())), torch.distributed.is_initialized())
        if key not in self.models:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.models[key] = self.real(cfg, *args, **kwargs)
            torch.cuda.synchronize()
            self.load_s += time.perf_counter() - t0
            self.n_loads += 1
        return self.models[key]


class _CallTimer:
    """Times every call of one Bergson function; no memoization."""

    def __init__(self, real):
        self.real = real
        self.wall_s = 0.0
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return self.real(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            self.wall_s += time.perf_counter() - t0


_MODEL_CACHE: _ModelCache | None = None
_DATA_TIMER: _CallTimer | None = None


def _install_model_cache() -> _ModelCache:
    """Replace ``setup_model_and_peft`` / ``setup_data_pipeline`` with a
    ``_ModelCache`` / ``_CallTimer`` in every Bergson module that binds them by
    name.  Idempotent; returns the ``_ModelCache``."""
    global _MODEL_CACHE, _DATA_TIMER
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    import bergson.build
    import bergson.hessians.hessian_approximations
    import bergson.score.score
    import bergson.utils.worker_utils as wu

    cache = _ModelCache(wu.setup_model_and_peft)
    data = _CallTimer(wu.setup_data_pipeline)
    for mod in (wu, bergson.build, bergson.score.score,
                bergson.hessians.hessian_approximations):
        for name, repl in (("setup_model_and_peft", cache), ("setup_data_pipeline", data)):
            assert hasattr(mod, name), (mod.__name__, name)
            setattr(mod, name, repl)
    _MODEL_CACHE, _DATA_TIMER = cache, data
    return cache


_SHARDED = False   # set by run() for fsdp tasks: Bergson's workers are child processes
_WORLD = 1         # set by run(): workers per launch_distributed_run

# Timing of Bergson's spawned workers (see bergson_site/sitecustomize.py).  Each
# _timed call gets its own timing directory; _WORKER_OVERHEAD holds one
# _worker_overhead() summary per _timed call, in call order.
_SITE_DIR = str(Path(__file__).resolve().parent / "bergson_site")
_WORKER_OVERHEAD: list = []


def _worker_overhead(timing_dir: str, world: int) -> dict:
    """Model-load and start-up seconds of the worker groups one call spawned.

    Reads the ``<pid>.jsonl`` files written by ``bergson_site/sitecustomize.py``
    in ``timing_dir``.  A group is the ``world`` processes of one
    ``launch_distributed_run``; groups run one after another and the ranks of
    a group run concurrently, so ``load_s`` / ``startup_s`` are the sum over
    groups of the maximum over a group's ranks.  ``startup`` is interpreter
    start to entry into the model loader (imports, unpickling the dataset
    argument, CUDA context, NCCL rendezvous); ``load`` is
    ``setup_model_and_peft`` (weights + FSDP wrap).  ``groups`` lists the
    per-group values.
    """
    procs = []
    for f in sorted(Path(timing_dir).glob("*.jsonl")):
        recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        start = next((r["t"] for r in recs if r["ev"] == "proc_start"), None)
        loads = [r for r in recs if r["ev"] == "setup_model_and_peft"]
        if start is None or not loads:
            continue  # not a bergson worker (e.g. a datasets.map helper)
        procs.append({"start": start, "startup": loads[0]["t_start"] - start,
                      "load": sum(r["wall_s"] for r in loads)})
    procs.sort(key=lambda d: d["start"])
    if world > 0 and len(procs) % world == 0:
        groups = [procs[i:i + world] for i in range(0, len(procs), world)]
    else:  # unexpected process count: split where start times jump
        groups = []
        for d in procs:
            if groups and d["start"] - groups[-1][-1]["start"] < 2.0:
                groups[-1].append(d)
            else:
                groups.append([d])
    return {"n_groups": len(groups), "n_procs": len(procs),
            "load_s": round(sum(max(d["load"] for d in g) for g in groups), 3),
            "startup_s": round(sum(max(d["startup"] for d in g) for g in groups), 3),
            "groups": [{"load_s": round(max(d["load"] for d in g), 3),
                        "startup_s": round(max(d["startup"] for d in g), 3)} for g in groups]}



def _timed(fn) -> tuple[float, float]:
    """(wall_s, peak_gb) of ``fn()``.

    ``wall_s`` excludes the time this process spends inside Bergson's model
    loader and data pipeline (reported as ``build_model`` and ``load_data``).
    ``peak_gb`` comes from torch's allocator, or from NVML for a sharded run,
    whose workers are child processes.  For a sharded run the call also sets
    ``BERGSON_BENCH_TIMING_DIR`` and puts ``bergson_site`` on ``PYTHONPATH``,
    and appends the workers' load / start-up summary to ``_WORKER_OVERHEAD``.
    """
    cache = _install_model_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loads_before, data_before = cache.load_s, _DATA_TIMER.wall_s
    smi = SmiPeak() if _SHARDED else contextlib.nullcontext()
    timing_dir = None
    if _SHARDED:
        # Spawned workers inherit the environment; sitecustomize times their
        # model loads and start-up into timing_dir.
        timing_dir = tempfile.mkdtemp(prefix="bergson_workers_")
        os.environ["BERGSON_BENCH_TIMING_DIR"] = timing_dir
        path = os.environ.get("PYTHONPATH", "")
        if _SITE_DIR not in path.split(os.pathsep):
            os.environ["PYTHONPATH"] = _SITE_DIR + (os.pathsep + path if path else "")
    t0 = time.perf_counter()
    try:
        with smi:
            fn()
    finally:
        os.environ.pop("BERGSON_BENCH_TIMING_DIR", None)
    torch.cuda.synchronize()
    _WORKER_OVERHEAD.append(
        _worker_overhead(timing_dir, _WORLD) if timing_dir
        else {"n_groups": 0, "n_procs": 0, "load_s": 0.0, "startup_s": 0.0, "groups": []})
    if timing_dir:
        shutil.rmtree(timing_dir, ignore_errors=True)
    wall = (time.perf_counter() - t0 - (cache.load_s - loads_before)
            - (_DATA_TIMER.wall_s - data_before))
    if _SHARDED:
        # Workers are child processes, so the peak is NVML's max over cards.
        # Models held by this process are released before the next step.
        cache.models.clear()
        torch.cuda.empty_cache()
        return round(wall, 3), smi.peak_gb
    return round(wall, 3), round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)


def _phases(fit: dict, score: dict, n_meas: int) -> list:
    rows = [{"phase": "build_model", "wall_s": round(_MODEL_CACHE.load_s, 3),
             "gpu_peak": [], "cpu_rss_peak_gb": 0},
            {"phase": "load_data", "wall_s": round(_DATA_TIMER.wall_s, 3),
             "gpu_peak": [], "cpu_rss_peak_gb": 0}]
    for name, d in (("fit", fit), ("score", score)):
        rows.append({"phase": name, "wall_s": round(d["wall"], 3), "work_units": n_meas,
                     "gpu_peak": [{"index": 0, "alloc_gb": d["mem"]}], "cpu_rss_peak_gb": 0})
    return rows



# Root of Bergson's on-disk index: ``$BERGSON_STORE``, else ``$BENCH_CACHE``
# (the directory of the tokenized block pools), else ``~/bergson_bench``.  The
# index is large at full dimension; point BERGSON_STORE at local storage with
# enough room.
STORE_ROOT = (
    os.environ.get("BERGSON_STORE")
    or os.environ.get("BENCH_CACHE", str(Path.home() / "bergson_bench"))
) + "/bergson_bench"


class SmiPeak:
    """Poll ``nvidia-smi`` for device ``memory.used`` while sharded workers run.

    ``peak_gb`` is the maximum seen, over all visible GPUs when ``gpu`` is
    None or on the GPU with index ``gpu`` otherwise.
    """

    def __init__(self, gpu: str | None = None) -> None:
        self.gpu, self.peak_mb = gpu, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            if self.gpu is not None:
                cmd[1:1] = ["-i", self.gpu]
            out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            vals = [int(v) for v in out.split() if v.strip().isdigit()]
            if vals:
                self.peak_mb = max(self.peak_mb, max(vals))
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
    """Bergson GradDot through its ``build`` + ``score`` subcommands.

    ``fit`` builds the query index (``build``).  ``score`` runs
    :class:`bergson.score.scorer.Scorer` over the train split: train gradients
    are computed on the fly and scored against the query index, which is
    streamed one module at a time from its memmap.

    Returns ``(fit, score, counts)``: ``{"wall", "mem"}`` per phase and the
    ``_index_counts`` of the query index and the score store.
    """
    proj = 0 if task.get("proj_mode", "rank64") == "full" else PROJ_DIM
    chunk = int(task.get("block_size", CHUNK))
    query_path = f"{run_path}/query"
    score_path = f"{run_path}/scores"
    common = ["--model", task["model"], "--dataset", data_str,
              "--chunk_length", str(chunk), "--projection_dim", str(proj)]
    if subset:
        common += ["--subset", subset]
    common += _batch_args(task) + _parallel_args(task)
    if task.get("bergson_max_batch"):
        common += ["--max_batch_size", str(task["bergson_max_batch"])]

    fit_wall, fit_mem = _timed(lambda: _run_inprocess(
        ["build", query_path, "--overwrite", "true",
         "--split", f"train[:{query_rows}]", *common,
         *_precision_flags(task, "build")]))
    score_wall, score_mem = _timed(lambda: _run_inprocess(
        ["score", score_path, "--query_path", query_path,
         "--split", f"train[:{train_rows}]", *common,
         *_precision_flags(task, "score")]))

    counts = {"query": _index_counts(query_path), "scores": _index_counts(score_path)}
    return ({"wall": fit_wall, "mem": fit_mem},
            {"wall": score_wall, "mem": score_mem},
            counts)


def run_hessian(task, method, data_str, subset, train_rows, query_rows, run_path):
    """Bergson's K-FAC / EK-FAC influence pipeline.

    Runs Bergson's ``ekfac`` subcommand, the entry point to
    ``hessians/pipeline.py``, which performs four steps:

        1. build the query gradients at full dimension (the pipeline sets
           ``query_cfg.projection_dim = 0``)
        2. fit the Kronecker factors on the training data
        3. apply the inverse Hessian to the query gradients, then project to
           ``projection_dim``
        4. score the training examples against the transformed queries

    ``method`` selects ``--hessian_cfg.ev_correction`` (true for ``ekfac``).
    The pipeline is a single command: its wall time and peak memory are
    reported as the ``fit`` phase, and ``score`` has wall 0.  Returns
    ``(fit, score, counts)`` as ``run_graddot`` does.
    """
    ev = "true" if method == "ekfac" else "false"
    # Bergson requires projection_dim == 0 when ev_correction is set, so EK-FAC
    # always runs at full dimension.
    proj = 0 if (task.get("proj_mode", "rank64") == "full" or method == "ekfac") \
        else PROJ_DIM
    chunk = int(task.get("block_size", CHUNK))

    args = ["ekfac", run_path, "--overwrite", "true",
            "--model", task["model"], "--method", "kfac",
            "--hessian_cfg.ev_correction", ev,
            "--projection_dim", str(proj),
            "--data.dataset", data_str, "--data.split", f"train[:{train_rows}]",
            "--data.chunk_length", str(chunk),
            "--query.dataset", data_str, "--query.split", f"train[:{query_rows}]",
            "--query.chunk_length", str(chunk),
            "--hessian_pipeline_cfg.inversion_cfg.damping_factor", str(DAMPING),
            # "none": one score column per query (the default, "mean", scores
            # the mean query gradient).
            "--query_aggregation", "none"]
    if subset:
        args += ["--data.subset", subset, "--query.subset", subset]
    args += _batch_args(task) + _parallel_args(task)
    args += _precision_flags(task, "ekfac")
    if task.get("bergson_max_batch"):
        args += ["--max_batch_size", str(task["bergson_max_batch"])]

    fit_wall, fit_mem = _timed(lambda: _run_inprocess(args))
    score_wall, score_mem = 0.0, fit_mem
    counts = {"query": _index_counts(f"{run_path}/query"),
              "scores": _index_counts(f"{run_path}/scores")}

    return ({"wall": fit_wall, "mem": fit_mem},
            {"wall": score_wall, "mem": score_mem},
            counts)


def run(task: dict, out_root: Path) -> None:
    method = task["method"]
    proj_mode = task.get("proj_mode", "rank64")
    if method not in ("graddot", "kfac", "ekfac"):
        msg = f"bergson adapter covers graddot/kfac/ekfac, not {method!r}"
        raise ValueError(msg)
    data_str, subset = DATASETS[task["dataset"]]
    # Bergson splits by row and re-chunks itself, so the task's chunk counts
    # are converted into row counts (``_rows_for_chunks``).
    # Warm-up / measured split, as in run_ours.py: ``warmup_train`` chunks go
    # through the whole pipeline untimed (throwaway store), then the first
    # ``measure_train`` chunks of the train split are timed.  ``measure_train``
    # defaults to ``n_train``.  ``bergson_docs`` overrides the train row count.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or task.get("n_train", 1024))
    chunk = int(task.get("block_size", CHUNK))
    train_rows = task.get("bergson_docs") or _rows_for_chunks(
        task["model"], data_str, subset, n_meas, chunk=chunk)
    warm_rows = _rows_for_chunks(task["model"], data_str, subset, n_warm, chunk=chunk) \
        if n_warm else 0
    query_rows = _rows_for_chunks(
        task["model"], data_str, subset, task.get("n_test", 16), chunk=chunk)
    # One store directory per cell (the tag includes proj_mode), so cells can
    # run concurrently.  An existing store of the same cell is removed.
    tag = (f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}"
           f"-{method}-{proj_mode}-T{chunk}")
    _bergson_precision(task)  # validate up front, before any store is touched
    store = f"{STORE_ROOT}/{tag}"
    subprocess.run(["rm", "-rf", store], check=False)

    global _SHARDED, _WORLD
    _WORLD = int(task.get("n_gpus", 1) or 1)
    _SHARDED = task.get("parallelism") == "fsdp" and int(task.get("n_gpus", 1) or 1) > 1
    runner = run_graddot if method == "graddot" else functools.partial(
        run_hessian, method=method)
    t_start = time.perf_counter()
    if n_warm:
        # Same pipeline on ``warmup_train`` chunks into a throwaway store; the
        # (memoized, timed) model load happens here.
        warm_store = f"{store}_warm"
        subprocess.run(["rm", "-rf", warm_store], check=False)
        runner(task, data_str=data_str, subset=subset, train_rows=warm_rows,
               query_rows=query_rows, run_path=warm_store)
        subprocess.run(["rm", "-rf", warm_store], check=False)
        torch.cuda.empty_cache()
    measured_from = len(_WORKER_OVERHEAD)  # calls before this are the warm-up's
    fit, score, counts = runner(task, data_str=data_str, subset=subset,
                                train_rows=train_rows, query_rows=query_rows,
                                run_path=store)
    # [n_train, n_query] of the score store Bergson wrote.
    score_shape = [counts.get("scores", {}).get("num_rows"),
                   counts.get("scores", {}).get("num_scores")]

    record = {
        "dtype": task.get("dtype", "float32"),  # what _precision_flags passed
        "bergson_nproc_per_node": int(task.get("n_gpus", 1) or 1),
        "bergson_distributed_mode": "fsdp" if _SHARDED else "data-parallel",
        # True for a sharded run: Bergson's workers load the model in their own
        # processes, and the phases' ``wall_s`` includes those loads.
        "bergson_worker_model_loads_timed": _SHARDED,
        # One _worker_overhead() summary per measured fit / score call, in
        # order (timed inside the workers, see bergson_site/).  Subtract
        # ``load_s`` (and ``startup_s``) from a phase's ``wall_s`` to exclude them.
        "bergson_worker_overhead": _WORKER_OVERHEAD[measured_from:],
        "lib": LIB,
        "task": {**task, "block_size": chunk, "proj_dim": (0 if proj_mode=="full" else PROJ_DIM), "proj_mode": proj_mode,
                 "bergson_train_rows": train_rows, "bergson_query_rows": query_rows,
                 "warmup_train": n_warm, "measure_train": n_meas,
                 "bergson_warm_rows": warm_rows,
                 "bergson_counts": counts, "strategy": "native"},
        # number of model loads in this process; 1 means every step run here
        # reused one loaded model
        "bergson_model_loads": _MODEL_CACHE.n_loads if _MODEL_CACHE else None,
        "bergson_data_loads": _DATA_TIMER.n_calls if _DATA_TIMER else None,
        "total_wall_s": round(time.perf_counter() - t_start, 3),
        "phases": _phases(fit, score, n_meas),
        "disk_gb": {"store": round(_dir_bytes(store) / 1024 ** 3, 4)},
        "disk_total_gb": round(_dir_bytes(store) / 1024 ** 3, 4),
        "mem_source": "nvml" if _SHARDED else "torch",
        "device": device_details(),
        "score_shape": score_shape,
        "status": "ok",
    }
    results = out_root / "results.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(record) + "\n")
    load = (f"model {_MODEL_CACHE.load_s:.1f}s (x{_MODEL_CACHE.n_loads}) "
            f"data {_DATA_TIMER.wall_s:.1f}s (x{_DATA_TIMER.n_calls}) ") if _MODEL_CACHE else ""
    print(f"[done] bergson {tag}: {load}warm {n_warm} | "
          f"fit {fit['wall']:.1f}s score {score['wall']:.1f}s "
          f"mem {max(fit['mem'], score['mem']):.1f}GB", flush=True)


def main() -> None:
    # Requires the Bergson version pinned in versions.py.
    require("bergson")
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
