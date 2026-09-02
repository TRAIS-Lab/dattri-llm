"""Orchestrator for the efficiency and scaling benchmark.

Expands a named experiment into one plan file per (task, library) cell, then
runs them **one at a time**: concurrent runs sharing a GPU would corrupt both
the timing and the peak-memory numbers.

    python run.py --experiment scaling-single-gpu --dry-run   # print the plan
    python run.py --experiment scaling-single-gpu --run       # execute in order

No scheduler is required. To use one, wrap the same command in a job script.
Sharded experiments are self-describing: a plan whose task carries
``parallelism: fsdp`` names the FSDP adapter and is launched under ``torchrun``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path

import tasks as T

HERE = Path(__file__).resolve().parent

# library -> (conda env, adapter script).  Only adapters that exist are runnable.
# modal/-only: dattri_llm alone.  The logix / kronfluence / bergson adapters
# are not carried here -- their packages are not installable in the image (this
# repo records no install command for any of them), so an entry pointing at a
# missing file could only fail at launch.  They remain in the parent tree.
BACKENDS: dict[str, tuple[str, str]] = {
    "dattri_llm": ("dattri", "run_ours.py"),
}

# One workload everywhere, so every number in the paper is directly comparable:
# 64 training samples, a single query, batch 1, 512-token sequences.
WORKLOAD = {"n_train": 64, "n_test": 1, "block_size": 512, "batch": 1, "seed": 0}

# Scale ranges are empirical, measured at this workload:
#   * Qwen-32B exhausts one H200 at sequence length 512 (139.8 GB used, 786 MiB
#     short), so the single-device ladder stops at 14B and the sharded one
#     starts at 32B.
#   * Pythia tops out at 6.9B, which fits on one A40, so both Pythia ladders
#     span the same models and differ only in the device they are run on.
#
# Every experiment states its ``dtype`` outright instead of inheriting
# ``models.dtype_for``'s size-based rule (fp32 below 1.0B, bf16 at or above).
# That rule crosses its threshold between the ladder's first and second rung, so
# the 0.5B point of every scaling curve ran in fp32 and the rest in bf16 -- both
# time and peak memory then FALL from 0.5B to 1B, which reads as a scaling
# property but is a precision change.  The split here:
#   * scaling-*    bf16 across the ladder.  Not a preference: Pythia-6.9B is
#                  ~40 GB of peak at bf16 against a 46 GB A40, so fp32 would not
#                  reach the top rung at all.  bf16 is the only constant that
#                  spans 0.41B..6.9B on one card.
#   * efficiency-* fp32 for every library.  run_logix.py hardcodes fp32 (its
#                  LoGra covariance machinery is fp32-only and a bf16 model
#                  raises a dtype-mismatch matmul), so bf16 here would compare a
#                  bf16 ours/kronfluence against an fp32 logix.  These are all
#                  single-scale (Pythia-0.5B), so no dip is possible either way.
# Note run_bergson.py shells out to bergson's own CLI and sets no dtype, so
# bergson picks its own regardless of what this says.
EXPERIMENTS = {
    # -- cross-library efficiency (Pythia-0.5B) ---------------------------
    "efficiency-dattri-llm-rank64": dict(
        families=("pythia",), scales=("0.5b",), datasets=("wikitext103",),
        methods=("graddot", "kfac", "ekfac"), libs=("dattri_llm",),
        workload={"proj_mode": "rank64", "dtype": "float32"}),
    "efficiency-dattri-llm-full": dict(
        families=("pythia",), scales=("0.5b",), datasets=("wikitext103",),
        methods=("graddot", "kfac", "ekfac"), libs=("dattri_llm",),
        workload={"proj_mode": "full", "dtype": "float32"}),
    # -- scaling: one ladder per (family, device, parallelism) ------------
    # Qwen on one H200: 0.5B up to the largest that fits (14B).
    "scaling-qwen-H200": dict(
        families=("qwen",), scales=("0.5b", "1b", "3b", "7b", "14b"),
        datasets=("wikitext103",), methods=("graddot", "kfac", "ekfac"),
        libs=("dattri_llm",), parallelisms=("single",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),
    # Qwen on 4x H200 with FSDP: from the smallest model that does not fit on
    # one H200 (32B) to the largest that fits on four (72B).
    "scaling-qwen-H200-fsdp4": dict(
        families=("qwen",), scales=("32b", "72b"),
        datasets=("wikitext103",), methods=("graddot", "kfac", "ekfac"),
        libs=("dattri_llm",), parallelisms=("fsdp",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),
    # Both families on ONE H200: the controlled cross-family comparison.
    # modal/-only.  Every other experiment is single-family, so comparing Qwen
    # against Pythia meant joining two results.jsonl produced on two devices
    # (Pythia/A40 vs Qwen/H200) -- family and device varying together.  Here the
    # device, dtype, workload and methods are all held fixed and only the model
    # family and its scale move, which is what a family comparison has to mean.
    # 9 models (Pythia 0.41-6.9B, Qwen 0.49-14.8B) x 3 methods = 27 cells; the
    # same cells as scaling-pythia-H200 + scaling-qwen-H200, in one run and one
    # results.jsonl.  Pythia has no 14b entry, so grid() skips it for that
    # family rather than erroring.
    "scaling-families-H200": dict(
        families=("pythia", "qwen"), scales=("0.5b", "1b", "3b", "7b", "14b"),
        datasets=("wikitext103",), methods=("graddot", "kfac", "ekfac"),
        libs=("dattri_llm",), parallelisms=("single",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),

    # EK-FAC only, both families, one H200 -- the redo lane.  modal/-only.
    # The 9 EK-FAC cells of scaling-families-H200 were captured materialized and
    # silently fell back to a dense Fisher; the other 18 cells are unaffected.
    # This re-runs just those 9 into their OWN results.jsonl, because the
    # harness appends and never overwrites: pointing a redo at the original
    # directory would leave two sets of EK-FAC rows with nothing to tell them
    # apart. Weights and token pools are already cached by then, so this is far
    # cheaper than repeating all 27.
    "scaling-families-H200-ekfac": dict(
        families=("pythia", "qwen"), scales=("0.5b", "1b", "3b", "7b", "14b"),
        datasets=("wikitext103",), methods=("ekfac",),
        libs=("dattri_llm",), parallelisms=("single",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),

    # Pythia on one A40; 6.9B is the largest Pythia released and it fits.
    "scaling-pythia-A40": dict(
        families=("pythia",), scales=("0.5b", "1b", "3b", "7b"),
        datasets=("wikitext103",), methods=("graddot", "kfac", "ekfac"),
        libs=("dattri_llm",), parallelisms=("single",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),
    # The same Pythia ladder on one H200, to separate the device from the model.
    "scaling-pythia-H200": dict(
        families=("pythia",), scales=("0.5b", "1b", "3b", "7b"),
        datasets=("wikitext103",), methods=("graddot", "kfac", "ekfac"),
        libs=("dattri_llm",), parallelisms=("single",),
        workload={"proj_mode": "rank64", "dtype": "bfloat16"}),
}


def build_plan(spec: dict) -> list[dict]:
    libs_filter = set(spec.get("libs") or BACKENDS)
    runs: list[dict] = []
    for t in T.grid(families=spec["families"], scales=spec["scales"],
                    datasets=spec["datasets"], methods=spec["methods"],
                    parallelisms=spec.get("parallelisms")):
        for lib in t["libs"]:
            if lib not in libs_filter or lib not in BACKENDS:
                continue
            if (lib, t["method"]) in set(spec.get("skip", ())):
                continue
            env, adapter = BACKENDS[lib]
            # Sharded runs need the FSDP adapter: torch.func.functional_call
            # bypasses FSDP's all-gather, so run_ours.py cannot be used.
            # Record it in the plan so nothing downstream has to infer it.
            if t.get('parallelism') == 'fsdp' and lib == 'dattri_llm':
                adapter = 'run_ours_fsdp.py'
            task = {**t, **WORKLOAD, **spec.get("workload", {}), "lib": lib}
            runs.append({"env": env, "adapter": adapter,
                         "lib": lib, "task": task})
    return runs


def write_plan(runs: list[dict], out: Path, name: str) -> Path:
    # Per-submission plan dir so a pending job keeps reading its own plans even
    # after a later submission is prepared.
    plan_dir = out / "plans" / name
    plan_dir.mkdir(parents=True, exist_ok=True)
    for old in plan_dir.glob("*.json"):
        old.unlink()
    for i, r in enumerate(runs):
        (plan_dir / f"{i}.json").write_text(json.dumps(r, indent=2))
    (out / "plan_index.jsonl").write_text(
        "".join(json.dumps({"i": i, "lib": r["lib"], "adapter": r["adapter"],
                            "task": {k: r["task"][k] for k in
                                     ("family", "scale", "dataset", "method",
                                      "parallelism", "n_gpus")}}) + "\n"
                for i, r in enumerate(runs)))
    return plan_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS),
                    help="which paper experiment to run (see README)")
    ap.add_argument("--out_dir", default=str(HERE / "out"),
                    help="where plans, logs and results.jsonl are written")
    ap.add_argument("--run", action="store_true",
                    help="execute the cells sequentially (no scheduler needed)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args()

    spec = EXPERIMENTS[a.experiment]
    runs = build_plan(spec)
    out = Path(a.out_dir)
    plan_dir = write_plan(runs, out, a.experiment)
    n = len(runs)
    print(f"=== {a.experiment}: {n} (task, library) runs ===")
    for i, r in enumerate(runs):
        t = r["task"]
        print(f"  {i:3d}  {r['lib']:12s} {t['family']}/{t['scale']:4s} "
              f"{t['dataset']:11s} {t['method']:7s} "
              f"{t['parallelism']}({t['n_gpus']}gpu)")
    print(f"\nplans written to {plan_dir}")

    if a.dry_run or not a.run:
        print("\n(dry run -- re-run with --run to execute these cells in order)")
        return

    # Execute one cell at a time, in order.  Sequential on purpose: two runs
    # sharing a GPU would contaminate every timing and memory number.
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, r in enumerate(runs):
        t = r["task"]
        label = f"{r['lib']}/{t['family']}-{t['scale']}/{t['method']}"
        adapter = HERE / "adapters" / r["adapter"]
        if t.get("parallelism") == "fsdp":
            nproc = t.get("n_gpus") or 1
            cmd = ["torchrun", f"--nproc_per_node={nproc}",
                   f"--master_port={20000 + os.getpid() % 10000}", str(adapter)]
        else:
            cmd = [sys.executable, "-u", str(adapter)]
        cmd += ["--task-file", str(plan_dir / f"{i}.json"), "--out", str(out)]
        print(f"\n########## [{i + 1}/{n}] {label}  ({time.strftime('%H:%M:%S')})")
        print("  " + " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc == 0:
            ok += 1
            print(f"########## DONE  {label}")
        else:
            fail += 1
            print(f"########## FAIL  {label} (exit {rc})")
    print(f"\n=== {a.experiment}: {ok} ok, {fail} failed -> {out / 'results.jsonl'} ===")
    return

    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
