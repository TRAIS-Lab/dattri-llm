"""Plan and run benchmark cells: one (task, library) cell at a time.

A *cell* is a task dict -- model, dataset, method, workload, library -- that one
adapter runs end to end and records as one line of ``results.jsonl`` (see
``log.BenchRun``).  The two launchers (``benchmark.py`` for the tables,
``scaling.py`` for the figure) only differ in which cells they define; the
expansion, plan files and execution live here.

Cells run **sequentially on purpose**: two runs sharing a GPU contaminate each
other's timing and peak-memory numbers.  A failed cell (CUDA OOM, adapter
error) still leaves a row, with ``status: "oom" | "error"``, so a library that
stops climbing a ladder is recorded rather than silently absent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import models

UTILS = Path(__file__).resolve().parent
ADAPTERS = UTILS / "adapters"

# library -> adapter script.  Our own library has a second adapter for sharded
# runs: the single-process one drives ``torch.func.functional_call``, which
# bypasses FSDP's all-gather, so ``run_ours_fsdp.py`` captures through the
# HookManager on an FSDP-wrapped model instead.
ADAPTER = {
    "dattri_llm": "run_ours.py",
    "logix": "run_logix.py",
    "bergson": "run_bergson.py",
    "kronfluence": "run_kronfluence.py",
}

# How a sharded task must be launched, per adapter.  ``torchrun`` adapters are
# rank-aware; ``self`` adapters spawn their own workers (bergson takes
# ``--nproc_per_node`` itself) and must be started as ONE process; ``none``
# adapters have no sharded path and are refused rather than launched N times.
LAUNCH = {
    "run_ours.py": "none",
    "run_ours_fsdp.py": "torchrun",
    "run_bergson.py": "self",
    "run_logix.py": "none",
    "run_kronfluence.py": "none",
}


def task(lib: str, family: str, scale: str, method: str, *, parallelism: str = "single",
         n_gpus: int = 1, **workload) -> dict:
    """One cell: the task dict an adapter consumes, with the model resolved."""
    hf_id, params_b = models.resolve(family, scale)
    return {"lib": lib, "family": family, "scale": scale, "model": hf_id,
            "params_b": params_b, "method": method, "parallelism": parallelism,
            "n_gpus": n_gpus, **workload}


def grid(libs, methods, scales, *, family: str, skip=(), **kw) -> list[dict]:
    """Every (lib, method, scale) cell, minus ``skip`` pairs of (lib, method)."""
    skip = set(skip)
    return [task(lib, family, sc, m, **kw)
            for lib in libs for m in methods for sc in scales
            if (lib, m) not in skip]


def runs_for(cells: list[dict]) -> list[dict]:
    """Attach the adapter to each cell."""
    out = []
    for t in cells:
        adapter = ADAPTER[t["lib"]]
        if t["parallelism"] == "fsdp" and t["lib"] == "dattri_llm":
            adapter = "run_ours_fsdp.py"
        out.append({"adapter": adapter, "lib": t["lib"], "task": t})
    return out


def write_plan(runs: list[dict], out: Path, name: str) -> Path:
    plan_dir = out / "plans" / name
    plan_dir.mkdir(parents=True, exist_ok=True)
    for old in plan_dir.glob("*.json"):
        old.unlink()
    for i, r in enumerate(runs):
        (plan_dir / f"{i}.json").write_text(json.dumps(r, indent=2))
    return plan_dir


def describe(runs: list[dict]) -> None:
    for i, r in enumerate(runs):
        t = r["task"]
        print(f"  {i:3d}  {r['lib']:12s} {t['family']}/{t['scale']:5s} {t['method']:8s} "
              f"{t.get('proj_mode', '-'):7s} batch {t.get('batch', '-'):<3} "
              f"n_test {t.get('n_test', '-'):<3} {t['parallelism']}({t['n_gpus']} gpu)")


def execute(runs: list[dict], out: Path, name: str) -> tuple[int, int]:
    """Run the cells in order; append a failure row for any that die."""
    plan_dir = write_plan(runs, out, name)
    env = dict(os.environ)
    # bergson writes its gradient index here; it must be local disk, not the
    # shared block-pool cache, because its write cost is part of the measurement.
    env.setdefault("BERGSON_STORE", str(out / "bergson"))
    ok = fail = 0
    for i, r in enumerate(runs):
        t = r["task"]
        label = f"{r['lib']}/{t['family']}-{t['scale']}/{t['method']}"
        adapter = ADAPTERS / r["adapter"]
        how = LAUNCH[r["adapter"]]
        if t["parallelism"] == "fsdp" and how == "none":
            print(f"########## SKIP  {label}: {r['adapter']} has no sharded path")
            fail += 1
            continue
        if t["parallelism"] == "fsdp" and how == "torchrun":
            cmd = ["torchrun", f"--nproc_per_node={t['n_gpus']}",
                   f"--master_port={20000 + os.getpid() % 10000}", str(adapter)]
            # Let FSDP move each unit to its device as it shards instead of
            # materializing the whole model on one card first.
            env["DATTRI_FSDP_CPU_INIT"] = "1"
        else:
            cmd = [sys.executable, "-u", str(adapter)]
        cmd += ["--task-file", str(plan_dir / f"{i}.json"), "--out", str(out)]
        print(f"\n########## [{i + 1}/{len(runs)}] {label}  ({time.strftime('%H:%M:%S')})")
        print("  " + " ".join(cmd), flush=True)
        # Own session so a time limit can take down a library's worker
        # processes together with the adapter.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # noqa: S603
                                text=True, bufsize=1, env=env, start_new_session=True)
        oom = timed_out = False
        limit = t.get("time_limit_s")
        deadline = time.monotonic() + limit if limit else None
        assert proc.stdout is not None
        # A reader thread keeps the pipe drained while the parent watches the clock.
        import threading

        def _pump() -> None:
            nonlocal oom
            for line in proc.stdout:
                sys.stdout.write(line)
                if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                    oom = True

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        while True:
            try:
                rc = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() > deadline:
                    timed_out = True
                    print(f"########## TIME LIMIT {limit}s reached for {label}: killing")
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                    break
        pump.join(timeout=10)
        sys.stdout.flush()
        if rc == 0:
            ok += 1
            print(f"########## DONE  {label}")
            continue
        fail += 1
        status = "oom" if oom else "timeout" if timed_out else "error"
        print(f"########## FAIL  {label} (exit {rc}, {status})")
        with (out / "results.jsonl").open("a") as fh:
            fh.write(json.dumps({"lib": r["lib"], "task": t, "status": status,
                                 "exit_code": rc, "phases": []}) + "\n")
    print(f"\n=== {name}: {ok} ok, {fail} failed -> {out / 'results.jsonl'} ===")
    return ok, fail


def main(experiments: dict[str, list[dict]], argv: list[str] | None = None) -> None:
    """Shared command line: ``--experiment NAME [--run | --dry-run] [--out_dir]``."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(experiments))
    ap.add_argument("--out_dir", default=None,
                    help="plans, logs and results.jsonl (default: out/<experiment>)")
    ap.add_argument("--run", action="store_true", help="execute the cells in order")
    ap.add_argument("--dry-run", action="store_true", help="list the cells only")
    ap.add_argument("--libs", default=None,
                    help="comma-separated libraries to keep, e.g. dattri_llm (default: all)")
    a = ap.parse_args(argv)
    cells = experiments[a.experiment]
    if a.libs:
        keep = set(a.libs.split(","))
        cells = [t for t in cells if t["lib"] in keep]
    runs = runs_for(cells)
    out = Path(a.out_dir) if a.out_dir else UTILS.parent / "out" / a.experiment
    print(f"=== {a.experiment}: {len(runs)} cells ===")
    describe(runs)
    if not a.run:
        write_plan(runs, out, a.experiment)
        print(f"\n(plans in {out / 'plans' / a.experiment}; add --run to execute)")
        return
    execute(runs, out, a.experiment)
