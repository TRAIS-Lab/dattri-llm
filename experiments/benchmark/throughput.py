"""Throughput on four H200s: every library at its largest batch, one
attribution workload up the Qwen ladder.

Each library runs on four H200s in its own multi-GPU mode, at the largest
per-GPU batch it completes the workload with (``BATCHES``).

Every library runs every method (GradDot, K-FAC, EK-FAC) with rank-64
projection wherever the library supports it and full dimension elsewhere
(``PROJECTION``).  Multi-GPU modes: dattri-llm FSDP (frozen capture, live
sharded scoring, ``run_ours_budget.py``); Bergson ``--fsdp``; Kronfluence
FSDP; LogIX data-parallel replicas, one per GPU.

Workload and timing: every cell attributes the same ``WORKLOAD`` training
samples, in ``WORKLOAD / (4 * batch)`` steps of its own batch, after 2 warm-up
steps through the full pipeline.  The time is the library's whole attribution
call (Kronecker-factor fit included); model build, data preparation and
process start-up are outside it for every library (Bergson's worker model
loads and start-up are measured inside its workers and subtracted; the row
keeps the raw time too).  Batch 0 means out of memory at batch 1; an attempt
is cut off after ``TIME_LIMIT_S``.  All libraries of a scale run on one
host, one after another.

    python throughput.py --run 0.5b,1b,3b,7b,14b,32b,72b,110b   # on a machine with four H200s
    python throughput.py --run 7b --libs bergson --methods graddot

Rows append to out/throughput/<scale>/results.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))


SCALES = ("0.5b", "1b", "3b", "7b", "14b", "32b", "72b", "110b")  # "1b" is Qwen2.5-1.5B
LIBS = ("dattri_llm", "bergson", "kronfluence", "logix")
METHODS = {lib: ("graddot", "kfac", "ekfac") for lib in LIBS}
#: What each library does with the gradient dimension: rank-64 projection
#: wherever the library supports it, full dimension where it does not.
PROJECTION = {
    ("dattri_llm", "graddot"): "rank-64",
    ("dattri_llm", "kfac"): "rank-64",
    ("dattri_llm", "ekfac"): "rank-64",
    ("bergson", "graddot"): "rank-64",
    ("bergson", "kfac"): "full-dim factors, rank-64 query",
    ("bergson", "ekfac"): "full-dim",
    ("kronfluence", "graddot"): "full-dim",
    ("kronfluence", "kfac"): "full-dim",
    ("kronfluence", "ekfac"): "full-dim",
    ("logix", "graddot"): "LoRA-64",
    ("logix", "kfac"): "LoRA-64",
    ("logix", "ekfac"): "LoRA-64",
}

#: Per-GPU batch of every cell: the largest power of two at which the library
#: completes the workload on four H200s (141 GB) within ``TIME_LIMIT_S``.
#: 0: the library runs out of memory at batch 1; the cell is recorded as
#: ``oom`` without running.
BATCHES = {
    "0.5b": {
        "dattri_llm": {"graddot": 128, "kfac": 128, "ekfac": 128},
        "bergson": {"graddot": 64, "kfac": 32, "ekfac": 32},
        "kronfluence": {"graddot": 64, "kfac": 64, "ekfac": 64},
        "logix": {"graddot": 128, "kfac": 128, "ekfac": 128},
    },
    "1b": {  # Qwen2.5-1.5B
        "dattri_llm": {"graddot": 64, "kfac": 64, "ekfac": 64},
        "bergson": {"graddot": 64, "kfac": 32, "ekfac": 16},
        "kronfluence": {"graddot": 32, "kfac": 32, "ekfac": 32},
        "logix": {"graddot": 64, "kfac": 64, "ekfac": 64},
    },
    "3b": {
        "dattri_llm": {"graddot": 32, "kfac": 32, "ekfac": 32},
        "bergson": {"graddot": 32, "kfac": 16, "ekfac": 8},
        "kronfluence": {"graddot": 16, "kfac": 16, "ekfac": 8},
        "logix": {"graddot": 32, "kfac": 32, "ekfac": 32},
    },
    "7b": {
        "dattri_llm": {"graddot": 32, "kfac": 32, "ekfac": 32},
        "bergson": {"graddot": 32, "kfac": 8, "ekfac": 4},
        "kronfluence": {"graddot": 16, "kfac": 0, "ekfac": 0},
        "logix": {"graddot": 32, "kfac": 32, "ekfac": 32},
    },
    "14b": {
        "dattri_llm": {"graddot": 16, "kfac": 16, "ekfac": 16},
        "bergson": {"graddot": 16, "kfac": 0, "ekfac": 0},
        "kronfluence": {"graddot": 0, "kfac": 0, "ekfac": 0},
        "logix": {"graddot": 16, "kfac": 16, "ekfac": 16},
    },
    "32b": {
        "dattri_llm": {"graddot": 8, "kfac": 8, "ekfac": 8},
        "bergson": {"graddot": 8, "kfac": 0, "ekfac": 0},
        "kronfluence": {"graddot": 0, "kfac": 0, "ekfac": 0},
        "logix": {"graddot": 8, "kfac": 8, "ekfac": 8},
    },
    "72b": {
        "dattri_llm": {"graddot": 8, "kfac": 8, "ekfac": 8},
        "bergson": {"graddot": 8, "kfac": 0, "ekfac": 0},
        "kronfluence": {"graddot": 0, "kfac": 0, "ekfac": 0},
        "logix": {"graddot": 0, "kfac": 0, "ekfac": 0},
    },
    "110b": {  # Qwen1.5-110B
        "dattri_llm": {"graddot": 4, "kfac": 4, "ekfac": 4},
        "bergson": {"graddot": 4, "kfac": 0, "ekfac": 0},
        "kronfluence": {"graddot": 0, "kfac": 0, "ekfac": 0},
        "logix": {"graddot": 0, "kfac": 0, "ekfac": 0},
    },
}
WARM = 2  # warm-up steps through the full pipeline, untimed
# GPUs per cell; ``THROUGHPUT_N_GPUS`` overrides it.
N_GPUS = int(__import__("os").environ.get("THROUGHPUT_N_GPUS", 4))
FAMILY = "qwen"
BASE = dict(
    dataset="wikitext103",
    block_size=512,
    seed=0,
    dtype="bfloat16",
    proj_mode="rank64",
)
#: The attribution phases summed into a library's time (build/load excluded).
PHASES = {
    "bergson": ("fit", "score"),
    "kronfluence": ("fit_factors", "pairwise_scores"),
    "logix": ("extract", "score"),
}
#: Seconds after which a cell is cut off and recorded as ``timeout``.
TIME_LIMIT_S = 3600


def batch_of(lib: str, method: str, scale: str):
    """The per-GPU batch of a cell (see ``BATCHES``)."""
    return BATCHES.get(scale, {}).get(lib, {}).get(method)


def n_test(scale: str) -> int:
    """Queries scored by every library at *scale*: one through 32B, four at
    72B and 110B -- the same for all libraries of a scale.

    Bergson caps the process count of its query build to the number of
    queries: a one-query build runs in one process with the whole model on
    one card, and four queries make the build shard over the four cards.  The
    count is recorded in every row.
    """
    return 4 if scale in ("72b", "110b") else 1


#: The fixed workload, in training samples, the same at every scale and for
#: every library.  A power of two equal to one step of the largest batch in
#: ``BATCHES`` (4 GPUs x 128), so it divides into whole steps of every
#: power-of-two batch.
WORKLOAD = 512


def workload(scale: str) -> int:  # noqa: ARG001 -- one workload for every scale
    """Samples every library attributes in a fixed-workload cell, in
    ``WORKLOAD // (N_GPUS * batch)`` steps of its own batch."""
    return WORKLOAD


def steps_at(b: int, scale: str) -> int:
    """Measured steps of the fixed workload at per-GPU batch *b*."""
    return workload(scale) // (N_GPUS * b)


def cell(lib: str, scale: str, method: str, batch: int) -> dict:
    """The fixed workload for a baseline: ``WARM`` warm-up steps, then the
    measured steps, at *batch* per GPU."""
    import runner

    step, k = batch * N_GPUS, steps_at(batch, scale)
    return runner.task(
        lib,
        FAMILY,
        scale,
        method,
        parallelism="ddp" if lib == "logix" else "fsdp",
        n_gpus=N_GPUS,
        batch=batch,
        n_test=n_test(scale),
        n_train=(WARM + k) * step,
        warmup_train=WARM * step,
        measure_train=k * step,
        time_limit_s=TIME_LIMIT_S,
        **BASE,
    )


def measurement(r: dict, lib: str, k: int) -> dict:
    """Samples and seconds of one adapter row *r* (*k* steps)."""
    ph = {p["phase"]: p["wall_s"] for p in r.get("phases", [])}
    raw = sum(ph.get(n, 0.0) for n in PHASES[lib])
    # Bergson loads the model in every worker group it spawns; the adapter
    # measures that inside the workers (model load, and the process start-up
    # before it), and both are subtracted here.
    over = r.get("bergson_worker_overhead") or []
    load = sum(o["load_s"] for o in over)
    startup = sum(o["startup_s"] for o in over)
    seconds = raw - load - startup
    n = k * r["task"]["batch"] * N_GPUS
    extra = (
        {
            "time_raw_s": raw,
            "time_no_worker_load_s": raw - load,
            "worker_load_s": load,
            "worker_startup_s": startup,
        }
        if over
        else {}
    )
    return {
        **extra,
        "status": "ok",
        "throughput": n / seconds,
        "time_s": seconds,
        "samples": n,
        "steps": k,
    }


# --------------------------------------------------------------------------- #
# One attempt of a cell at a given batch                                       #
# --------------------------------------------------------------------------- #


def attempt_baseline(
    lib: str, scale: str, method: str, b: int, out: Path
) -> tuple[bool, dict]:
    """Run the fixed workload of a baseline at *b*; ``(ok, result)``."""
    import shutil

    import runner

    task = cell(lib, scale, method, b)
    d = out / f"{lib}-{scale}-{method}-b{b}"
    runner.execute(runner.runs_for([task]), d, d.name)
    r = json.loads((d / "results.jsonl").read_text().splitlines()[-1])
    # Keep the row (it records the store's size) and delete the store.
    shutil.rmtree(d, ignore_errors=True)
    if r.get("status") != "ok":
        return False, {"status": r.get("status", "error")}
    k = steps_at(b, scale)
    return True, {
        **measurement(r, lib, k),
        "cell": r,
        "mode": task["parallelism"],
    }


def attempt_ours(
    scale: str, b: int, out: Path, methods=METHODS["dattri_llm"]
) -> tuple[bool, dict]:
    """Run dattri-llm's live sharded path (every method) at *b*; ``(ok, result)``."""
    import os
    import shutil
    import subprocess

    import models
    from runner import OOM_MARKERS

    model_id, params_b = models.resolve(FAMILY, scale)
    d = out / f"dattri_llm-{scale}-b{b}"
    d.mkdir(parents=True, exist_ok=True)
    plan = d / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "task": {
                    **BASE,
                    "family": FAMILY,
                    "lib": "dattri_llm",
                    "scale": scale,
                    "model": model_id,
                    "params_b": params_b,
                    "n_test": n_test(scale),
                    "batch": b,
                    "steps": steps_at(b, scale),
                    "warmup_steps": WARM,
                    "methods": list(methods),
                    "parallelism": "fsdp",
                    "n_gpus": N_GPUS,
                }
            }
        )
    )
    env = {**os.environ, "DATTRI_FSDP_CPU_INIT": "1" if params_b > 60 else "0"}
    proc = subprocess.run(  # noqa: S603
        [
            "torchrun",
            f"--nproc_per_node={N_GPUS}",
            str(HERE / "utils/adapters/run_ours_budget.py"),
            "--task-file",
            str(plan),
            "--out",
            str(d),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=len(methods) * TIME_LIMIT_S + 1800,
    )
    print(
        "\n".join(line for line in proc.stdout.splitlines() if "budget]" in line),
        flush=True,
    )
    if proc.returncode == 0 and (d / "results.jsonl").exists():
        rows = [
            json.loads(line) for line in (d / "results.jsonl").read_text().splitlines()
        ]
        shutil.rmtree(d, ignore_errors=True)  # scores and plan; the rows are kept
        return True, {"status": "ok", "rows": rows}
    shutil.rmtree(d, ignore_errors=True)
    oom = any(m in proc.stderr for m in OOM_MARKERS)
    if not oom:
        print(proc.stdout[-2000:], proc.stderr[-4000:], flush=True)
    return False, {"status": "oom" if oom else "error"}


# --------------------------------------------------------------------------- #
# One scale, at the batches of ``BATCHES``                                     #
# --------------------------------------------------------------------------- #


def run_scale(scale: str, libs, out: Path, methods=None) -> list[dict]:
    """Every (lib, method) of one scale at its ``BATCHES`` batch, in one process
    (one host).  A cell whose batch is 0 is recorded as out of memory without
    running; a cell that fails is recorded with its status.  The rows append
    to ``out/results.jsonl``.
    """
    import os

    os.environ.setdefault("BERGSON_STORE", str(out / "bergson"))
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def row(lib, m, result):
        return {
            "lib": lib,
            "scale": scale,
            "method": m,
            "n_gpus": N_GPUS,
            "n_test": n_test(scale),
            "batch_per_gpu": batch_of(lib, m, scale),
            "projection": PROJECTION[(lib, m)],
            "mode": "fsdp-live"
            if lib == "dattri_llm"
            else "ddp"
            if lib == "logix"
            else "fsdp",
            **result,
        }

    for lib in libs:
        wanted = [m for m in METHODS[lib] if methods is None or m in methods]
        new = [
            row(lib, m, {"status": "oom"})
            for m in wanted
            if batch_of(lib, m, scale) == 0
        ]
        runnable = [m for m in wanted if batch_of(lib, m, scale) > 0]
        if lib == "dattri_llm" and runnable:
            # one process for the three methods, which share one batch
            b = batch_of(lib, runnable[0], scale)
            ok, res = attempt_ours(scale, b, out, methods=runnable)
            print(
                f"########## dattri_llm {scale}: batch {b} {res['status']}", flush=True
            )
            if not ok:
                new += [row(lib, m, res) for m in runnable]
            else:
                keep = ("throughput", "time_s", "samples", "steps")
                new += [
                    row(
                        lib,
                        r["task"]["method"],
                        {"status": "ok"} | {k: r[k] for k in keep},
                    )
                    for r in res["rows"]
                ]
        else:
            for m in runnable:
                b = batch_of(lib, m, scale)
                _ok, res = attempt_baseline(lib, scale, m, b, out)
                print(
                    f"########## {lib} {scale} {m}: batch {b} {res['status']}",
                    flush=True,
                )
                new.append(row(lib, m, res))
        rows += new
        with (out / "results.jsonl").open("a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="scales to run here, comma-separated")
    ap.add_argument("--libs", default=",".join(LIBS))
    ap.add_argument(
        "--methods", default="", help="comma-separated (default: all three)"
    )
    ap.add_argument("--out_dir", default=str(HERE / "out" / "throughput"))
    a = ap.parse_args()
    for sc in a.run.split(","):
        run_scale(
            sc,
            tuple(a.libs.split(",")),
            Path(a.out_dir) / sc,
            tuple(a.methods.split(",")) if a.methods else None,
        )
