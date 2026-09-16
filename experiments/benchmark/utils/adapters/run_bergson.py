"""bergson adapter for the universal benchmark.

bergson drives an on-disk gradient *index* through its ``build`` / ``score`` /
``ekfac`` pipelines.  They run in this process through bergson's own Python
entry points (``bergson.build.build``, ``bergson.score.score.score_dataset``,
``hessians.pipeline.hessian_pipeline``), reached through the same argument
parser as its CLI so every setting is identical to a ``python -m bergson``
invocation.  At ``nproc_per_node=1`` bergson runs its worker in the calling
process (no spawn).  The model is loaded once through bergson's own loader
(``setup_model_and_peft``), timed and reported as the ``build_model`` phase
like every other adapter, and reused by every later bergson step; bergson's
dataset load + tokenization (``setup_data_pipeline``, which every step runs
before any gradient work) is likewise timed and reported as ``load_data``, the
phase in which the other adapters read their pre-chunked pools untimed -- it
was 64-72% of a timed GradDot cell on an H200 before it was excluded.  GPU
peak memory comes from torch's allocator, except for sharded runs, whose
workers are child processes: there the peak is NVML's maximum over the cards
(``mem_source: "nvml"``).  An optional warm-up (``warmup_train`` chunks through
the same pipeline into a throwaway store) precedes the timed run, matching
``run_ours.py``.  The record has the phases ``fit`` and ``score``.

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
import contextlib
import functools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

# Same matmul precision as run_ours.py / run_logix.py: TF32 tensor-core
# matmuls in the float32 tables (Ampere).  Set in this process, which is where
# the single-card library calls run; the bf16 scaling runs are unaffected.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

from log import device_details
from versions import require

LIB = "bergson"
PROJ_DIM = 64
CHUNK = 512
# bergson HF dataset spec per benchmark dataset name: (data_str, subset).
# Namespaced ids, matching data.py: the bare `wikitext` id is the legacy
# canonical-dataset form and current huggingface_hub rejects it -- parse_hf_uri
# requires `namespace/name` and raises HfUriError on a single segment.  This
# table is bergson's own and is passed straight to its CLI, so fixing data.py
# did not cover it.
DATASETS = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
}
# Bergson materializes its gradient index on disk (>100 GB at full dimension), so
# this must point at a filesystem with room -- and must not be a hardcoded path
# from one machine.  ``BENCH_CACHE`` is set per-cluster by the launcher.

# bergson's precision is a per-subcommand flag, not a global one.  ``build``
# has a single IndexConfig, so it takes a flat ``--precision``; ``score`` and
# ``ekfac`` carry BOTH ``IndexConfig.precision`` (model parameters) and
# ``ScoreConfig.precision`` (dtype the gradients are converted to before
# scoring), so there a flat ``--precision`` is ambiguous and the parser
# rejects it -- which is why an earlier version of this adapter concluded the
# precision could not be set at all and hardwired fp32.  Passing the
# fully-qualified names works on every subcommand used here.
_PRECISION = {"float32": "fp32", "bfloat16": "bf16"}


def _bergson_precision(task: dict) -> str:
    """Bergson's name for the experiment's dtype.

    Refuses anything unmapped rather than silently running at the wrong
    precision -- the failure mode the old hardwired fp32 was guarding against.
    """
    wanted = task.get("dtype", "float32")
    if wanted not in _PRECISION:
        raise ValueError(f"run_bergson.py cannot set bergson to dtype={wanted!r}; "
                         f"supported: {sorted(_PRECISION)}")
    return _PRECISION[wanted]


def _precision_flags(task: dict, subcommand: str) -> list:
    """The precision flags *this* subcommand accepts (see above)."""
    p = _bergson_precision(task)
    if subcommand == "build":
        return ["--precision", p]
    return ["--index_cfg.precision", p, "--score_cfg.precision", p]


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


def _parallel_args(task: dict) -> list:
    """bergson's own parallelism flags, taken from the task's topology.

    ``nproc_per_node`` MUST be passed explicitly.  Its default is
    ``max(1, torch.cuda.device_count())`` -- one process per *visible* GPU -- so
    left unset bergson silently scales with whatever the container happens to
    have, while every other adapter here is explicitly single-process.  On a
    multi-GPU box that hands bergson every card against everyone else's one, and
    nothing in the record would show it.

    ``--fsdp`` is bergson's own sharding switch, so a task marked
    ``parallelism: fsdp`` gets sharded capture rather than N replicas.
    """
    n = int(task.get("n_gpus", 1) or 1)
    # --fsdp is deliberately NOT passed.  bergson's hessian pipeline builds its
    # query index through launch_distributed_run, which for a small query set
    # calls worker(0, 0, 1, ...) -- world size 1, no launcher, no process group.
    # --fsdp then makes fully_shard() call init_process_group(), which dies on
    # "environment variable RANK expected, but not set" -- for the QUERY build
    # only: with a 1-chunk query set bergson caps that step's world size to 1
    # and runs it in the calling process.  Under the in-process entry that
    # step goes through our ``setup_model_and_peft`` wrapper, which skips the
    # sharding when no process group exists (the model is loaded whole on one
    # card; fits to 32B), while every train-side step (Fisher fit, apply,
    # score) has >= 4 chunks, spawns ``nproc_per_node`` workers and shards.
    # So sharded capture is only offered through this in-process path.
    args = ["--nproc_per_node", str(n)]
    if task.get("parallelism") == "fsdp" and n > 1:
        args += ["--fsdp", "true"]
    return args


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


def _run_inprocess(cli_args: list) -> None:
    """The same bergson subcommand, run in this process.

    ``bergson.__main__.main`` parses ``sys.argv`` with bergson's own
    simple-parsing dataclasses and calls ``<Command>.execute()``, so the flag
    lists mean exactly what they mean on the command line.  With
    ``--nproc_per_node 1`` every pipeline step runs in the calling process.
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
    """Memoized, timed stand-in for bergson's ``setup_model_and_peft``.

    Every bergson worker (build, score, hessian fit) loads its model through
    that one function.  Wrapping it does two things: the load is *timed* (so it
    can be reported as ``build_model``, outside attribution time, like the other
    adapters) and *memoized* (so the warm-up run pays it once and every later
    step reuses the same model object, as ``run_ours.py`` and
    ``run_kronfluence.py`` reuse theirs).  bergson's collectors register hooks
    inside a context manager and remove them on exit, so reuse is safe; GradDot
    scores from a reused model match a fresh subprocess to bf16 precision.
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
            # A step bergson runs in the calling process (world size capped to
            # 1 by a tiny dataset): no process group, so fully_shard() cannot
            # run -- load the model whole on this card instead.  Hand the
            # loader a copy with fsdp=False rather than apply_fsdp=False: with
            # cfg.fsdp set, bergson's loader picks device_map="cpu" (it expects
            # fully_shard to move the shards), so the step would run on the CPU
            # -- which is what happened to the query build of the first sharded
            # runs (72B "ran" its query build on the CPU in 441 s).
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
    """Times every call of one bergson function; no memoization."""

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
    """Patch ``setup_model_and_peft`` / ``setup_data_pipeline`` wherever bergson
    binds them by name (each worker module imports both into its namespace)."""
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


_SHARDED = False   # set by run() for fsdp tasks: bergson's workers are child processes


def _timed(fn) -> tuple[float, float]:
    """(wall_s, peak_gb) of ``fn()``: wall excluding time spent inside
    bergson's model loader and data pipeline (reported as ``build_model`` and
    ``load_data``; with a warm-up the loader never runs inside a timed call
    anyway), peak from torch -- or from NVML for a sharded run, whose workers
    are child processes.
    """
    cache = _install_model_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loads_before, data_before = cache.load_s, _DATA_TIMER.wall_s
    smi = SmiPeak() if _SHARDED else contextlib.nullcontext()
    t0 = time.perf_counter()
    with smi:
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0 - (cache.load_s - loads_before)
            - (_DATA_TIMER.wall_s - data_before))
    if _SHARDED:
        # Workers are child processes: torch's allocator here sees only the
        # parent's (unsharded query-build) model, so take NVML's max over cards.
        # Free the parent's model between steps so the workers get the card.
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
# BERGSON_STORE first, BENCH_CACHE only as a fallback.  These are different
# kinds of storage and conflating them is a measurement bug: BENCH_CACHE holds
# tokenized block pools, which are small, reused across every run and belong on
# shared//network storage; the bergson index is the artifact whose *local* write
# cost this benchmark reports, and it exceeds 100 GB at full dimension.  Writing
# it to a network volume measures the storage fabric and reports those bytes as
# the run's disk cost.
STORE_ROOT = (
    os.environ.get("BERGSON_STORE")
    or os.environ.get("BENCH_CACHE", str(Path.home() / "bergson_bench"))
) + "/bergson_bench"


class SmiPeak:
    """Poll nvidia-smi for device memory.used (MB) while sharded workers run."""

    def __init__(self, gpu: str | None = None) -> None:
        # None: every visible card (max over cards), for bergson's spawned workers
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
    """bergson GradDot through its ``build`` + ``score`` pipeline.

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
    # bergson splits by ROW and re-chunks itself, so convert the benchmark's
    # chunk-count workload into row counts (see ROWS_PER_CHUNK).
    # Exact row splits: the benchmark asks for n_train / n_test blocks of CHUNK
    # tokens, and every other library gets exactly that from a pre-chunked pool.
    # Warm-up / measured split with run_ours.py's semantics: ``warmup_train``
    # chunks go through the whole pipeline untimed (throwaway store), then
    # ``measure_train`` chunks are timed.  bergson re-chunks a row split
    # itself, so the measured set is the first ``measure_train`` chunks (its
    # content does not affect cost at fixed chunk length).  Without
    # ``measure_train`` the old meaning -- all ``n_train`` chunks timed, no
    # warm-up -- is kept, which is what reproduces earlier rows.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or task.get("n_train", 1024))
    train_rows = task.get("bergson_docs") or _rows_for_chunks(
        task["model"], data_str, subset, n_meas)
    warm_rows = _rows_for_chunks(task["model"], data_str, subset, n_warm) \
        if n_warm else 0
    query_rows = _rows_for_chunks(
        task["model"], data_str, subset, task.get("n_test", 16))
    # proj_mode MUST be in the tag: the r=64 and full arrays are separate SLURM
    # jobs and run concurrently, so a shared store dir means one run's `rm -rf`
    # races the other's build and leaves a stale `<path>.part` behind, which
    # validate_run_path then refuses.
    tag = (f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}"
           f"-{method}-{proj_mode}")
    _bergson_precision(task)  # validate up front, before any store is touched
    store = f"{STORE_ROOT}/{tag}"
    subprocess.run(["rm", "-rf", store], check=False)

    global _SHARDED
    _SHARDED = task.get("parallelism") == "fsdp" and int(task.get("n_gpus", 1) or 1) > 1
    runner = run_graddot if method == "graddot" else functools.partial(
        run_hessian, method=method)
    t_start = time.perf_counter()
    if n_warm:
        # Same pipeline on ``warmup_train`` chunks into a throwaway store; this
        # is also where the (memoized, timed) model load happens.
        warm_store = f"{store}_warm"
        subprocess.run(["rm", "-rf", warm_store], check=False)
        runner(task, data_str=data_str, subset=subset, train_rows=warm_rows,
               query_rows=query_rows, run_path=warm_store)
        subprocess.run(["rm", "-rf", warm_store], check=False)
        torch.cuda.empty_cache()
    fit, score, counts = runner(task, data_str=data_str, subset=subset,
                                train_rows=train_rows, query_rows=query_rows,
                                run_path=store)
    # What bergson actually scored, so an unequal workload can never again hide
    # behind a row count that looked like a sample count.
    score_shape = [counts.get("scores", {}).get("num_rows"),
                   counts.get("scores", {}).get("num_scores")]

    record = {
        "dtype": task.get("dtype", "float32"),  # what _precision_flags passed
        "bergson_nproc_per_node": int(task.get("n_gpus", 1) or 1),
        "bergson_distributed_mode": "fsdp" if _SHARDED else "data-parallel",
        # sharded: bergson's workers reload the model every step in their own
        # processes, which cannot be excluded from the timing from outside
        "bergson_worker_model_loads_timed": _SHARDED,
        "lib": LIB,
        "task": {**task, "block_size": CHUNK, "proj_dim": (0 if proj_mode=="full" else PROJ_DIM), "proj_mode": proj_mode,
                 "bergson_train_rows": train_rows, "bergson_query_rows": query_rows,
                 "warmup_train": n_warm, "measure_train": n_meas,
                 "bergson_warm_rows": warm_rows,
                 "bergson_counts": counts, "strategy": "native"},
        # 1 means every bergson step reused the one loaded model
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
    # Refuse to benchmark against anything but the pinned baseline
    # (versions.py); the version is part of the result.
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
