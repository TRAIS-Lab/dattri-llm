"""Single entry point for the pairwise efficiency benchmarks.

    python run.py logix                 # run one pair here, serially
    python run.py logix --slurm         # submit it as one SLURM job
    python run.py all --slurm           # all pairs, chained (never concurrent)
    python run.py --list                # show the pairs and their steps

Each pair is one declarative entry below: the files to clear first, and the
commands to run in order inside the pair's directory.  Everything runs
**serially** -- concurrent pairs contend for the GPU and silently corrupt the
walltimes, so the SLURM path chains jobs with a dependency rather than
submitting them side by side.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# SLURM resources for one pair (one GPU, single node).
SBATCH = [
    "--partition=standard",
    "--nodes=1",
    "--ntasks-per-node=1",
    "--cpus-per-task=8",
    "--mem=120G",
    "--gres=gpu:1",
    "--time=8:00:00",
    "--export=ALL",
]


@dataclass
class Pair:
    """One head-to-head benchmark: what to clear, then what to run in order."""

    dir: str
    steps: list[str]
    clean: list[str] = field(default_factory=list)


PAIRS: dict[str, Pair] = {
    "dattri": Pair(
        dir="vs_dattri",
        # "ours" first: its exact score matrix is the fidelity reference.
        clean=[
            "results.jsonl",
            "score_ours.pt",
            "score_dattri.pt",
            "score_dattri-cuda.pt",
        ],
        steps=[
            "run_pair.py --side ours",
            "run_pair.py --side dattri",
            "run_pair.py --side dattri-cuda",
        ],
    ),
    "logix": Pair(
        dir="vs_logix",
        clean=[
            "results.jsonl",
            "score_ours_disk.pt",
            "score_ours_otf.pt",
            "score_ours_refit.pt",
            "score_ours_compact.pt",
            "score_logix_raw.pt",
            "score_logix_kfac.pt",
            "logix_logs",
            "ours_grads",
            "ours_grads_compact",
        ],
        steps=[
            "run_logix.py --stage extract --hessian raw",
            "run_logix.py --stage score   --hessian raw",
            "run_logix.py --stage extract --hessian kfac",
            "run_logix.py --stage score   --hessian kfac",
            "run_dattri_llm.py --workflow disk",
            "run_dattri_llm.py --workflow otf",
            "run_dattri_llm.py --workflow refit",
            "run_dattri_llm.py --workflow compact",
        ],
    ),
    "kronfluence": Pair(
        dir="vs_kronfluence",
        # Needs ./checkpoints/model.pth first (their train.py -- see README).
        clean=[
            "results.jsonl",
            "score_ours.pt",
            "score_kronfluence.pt",
            "kronfluence_store",
        ],
        steps=["run_kronfluence.py", "run_dattri_llm.py"],
    ),
    "ghostsuite": Pair(
        dir="vs_ghostsuite",
        clean=["results.jsonl"],
        steps=[
            *(
                f"run_pair.py --side {s}"
                for s in (
                    "ghost-regular",
                    "ghost-dotprod",
                    "ghost-dotprod-eager",
                    "ours-baseline",
                    "ours-callback",
                )
            ),
        ],
    ),
    "bergson": Pair(
        dir="vs_bergson",
        clean=[
            "results.jsonl",
            "score_bergson.pt",
            "score_ours_logra.pt",
            "score_ours_logramat.pt",
            "score_ours_logramat-mmap.pt",
            "bergson_index",
            "bergson_index.part",
            "ours_index",
            "ours_index_logra",
            "ours_index_logramat",
            "ours_index_logramat-mmap",
        ],
        steps=[
            "run_bergson.py --stage build --gpu 0",
            "run_bergson.py --stage query --gpu 0",
            "run_dattri_llm.py --style logra_materialized --disk-format memmap",
        ],
    ),
}


def clean(pair: Pair) -> None:
    """Delete the pair's previous results, score matrices, and gradient stores."""
    root = HERE / pair.dir
    for name in pair.clean:
        target = root / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def run_local(name: str) -> int:
    """Run one pair's steps in order; returns the number that failed."""
    pair = PAIRS[name]
    root = HERE / pair.dir
    clean(pair)
    failed = 0
    for step in pair.steps:
        print(f"\n=== [{name}] {step} ===", flush=True)
        t0 = time.monotonic()
        rc = subprocess.run(
            [sys.executable, "-u", *step.split()],
            cwd=root,
            check=False,
        ).returncode
        tag = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"=== [{name}] {step} -> {tag} in {time.monotonic() - t0:.1f}s ===")
        failed += rc != 0
    results = root / "results.jsonl"
    if results.exists():
        print(f"\n== {name} results ==\n{results.read_text()}")
    return failed


def submit_slurm(names: list[str]) -> None:
    """Submit each pair as its own job, chained so they never run concurrently."""
    (HERE / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
    dep: str | None = None
    for name in names:
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name=eff-{name}",
            "--output=slurm/logs/%x-%j.out",
            *SBATCH,
        ]
        if dep is not None:
            cmd.append(f"--dependency=afterany:{dep}")  # strictly serial
        cmd += ["slurm/pair.sbatch", name]
        job = subprocess.run(
            cmd,
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"submitted {name:12s} job {job}" + (f" (after {dep})" if dep else ""))
        dep = job


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pair", nargs="?", help=f"one of: {', '.join(PAIRS)}, or 'all'")
    p.add_argument("--slurm", action="store_true", help="submit instead of running")
    p.add_argument("--list", action="store_true", help="show pairs and steps")
    a = p.parse_args()

    if a.list or not a.pair:
        for name, pair in PAIRS.items():
            print(f"\n{name}  ({pair.dir})")
            for step in pair.steps:
                print(f"    python {step}")
        return 0

    names = list(PAIRS) if a.pair == "all" else [a.pair]
    unknown = [n for n in names if n not in PAIRS]
    if unknown:
        p.error(f"unknown pair(s) {unknown}; choose from {list(PAIRS)} or 'all'")

    if a.slurm:
        submit_slurm(names)
        return 0
    return min(sum(run_local(n) for n in names), 1)


if __name__ == "__main__":
    sys.exit(main())
