"""Fidelity of optimizer-aware attribution (the paper's fidelity section and appendix).

Every run trains one AdamW trajectory, scores it with the methods and
compares them with trajectory-specific leave-one-out retraining (TSLOO):

    mlp             MNIST + MLP, three learning rates x three seeds:
                    ours (DVEmb, AdamW-influence) + ground truth, MAGIC, SOURCE
    gpt2            WikiText-2 + GPT-2, the same grid with the k=512 mask
    gpt2-full       AdamW-influence and DVEmb over every coordinate
    gpt2-protocols  the other ground-truth conventions (every / first / each
                    occurrence) at lr 1e-5, seed 0
    gpt2-mask       the k=1024 mask at lr 1e-5, seed 0
    mlp-ablations   curvature term (lr 1e-3) and coverage (lr 1e-5), seed 0
    timing          wall-clock of every method on GPT-2, lr 1e-5, seed 0
                    (one exclusive A40 in the paper)

    python fidelity.py --experiment gpt2 --dry-run   # list the runs
    python fidelity.py --experiment gpt2 --run       # run them in order
    python fidelity.py --collect                     # run dirs -> results/fidelity.jsonl
    python fidelity.py --table                       # print the tables

Runs write to results/<setting>/lr<lr>_seed<seed>[_variant]/ and are skipped
when their result.json exists; a run that needs another's ground truth
(``--truth-dir`` / ``--tsloo-from``) comes after it in the list.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UTILS = HERE / "utils"
RESULTS = HERE / "results"
sys.path.insert(0, str(UTILS))

LRS = {"mlp": ("1e-3", "1e-4", "1e-5"), "gpt2": ("1e-4", "5e-5", "1e-5")}
SEEDS = (0, 1, 2)


def name(setting: str, lr: str, seed: int, variant: str = "") -> str:
    return f"lr{float(lr):g}_seed{seed}{variant}"


def cell(setting: str, lr: str, seed: int, variant: str, script: str, *args: str) -> dict:
    """One run: its result directory (for skipping) and its command line."""
    run = name(setting, lr, seed, variant)
    return {"dir": f"{setting}/{run}",
            "cmd": [script, "--setting", setting, "--lr", lr, "--seed", str(seed), *args]}


def ours(setting, lr, seed, variant="", *args):
    extra = ["--tag", variant] if variant else []
    return cell(setting, lr, seed, variant, "ours.py", *extra, *args)


def truth(setting, lr, seed):
    return f"results/{setting}/{name(setting, lr, seed)}"


def grid(setting: str) -> list[dict]:
    at = ["--tsloo-at", "last"] if setting == "gpt2" else []
    n_q = "64" if setting == "gpt2" else "500"
    source = "source.py" if setting == "gpt2" else "source_mlp.py"
    cells = []
    for lr in LRS[setting]:
        for seed in SEEDS:
            cells.append(ours(setting, lr, seed, "", *at))
            cells.append(cell(setting, lr, seed, "_magic", "magic.py",
                              "--n-queries", n_q, "--truth-dir", truth(setting, lr, seed)))
            cells.append(cell(setting, lr, seed, "_source", source,
                              *(["--n-queries", n_q, "--truth-dir", truth(setting, lr, seed)]
                                if setting == "gpt2" else [])))
    return cells


FULL = ["--n-masks", "0", "--recompute", "--n-val", "64", "--val-batch-size", "8", "--tsloo-at", "last"]

EXPERIMENTS = {
    "mlp": grid("mlp"),
    "gpt2": grid("gpt2"),
    "gpt2-full": [ours("gpt2", lr, seed, "_full", *FULL, "--tsloo-from", truth("gpt2", lr, seed))
                  for lr, seed in (("1e-5", 0), ("1e-5", 1), ("1e-5", 2), ("1e-4", 0), ("5e-5", 0))],
    "gpt2-protocols": [ours("gpt2", "1e-5", 0, "_all"),
                       ours("gpt2", "1e-5", 0, "_first", "--tsloo-at", "first"),
                       ours("gpt2", "1e-5", 0, "_each", "--tsloo-at", "each")],
    "gpt2-mask": [ours("gpt2", "1e-5", 0, "_k1024", "--mask-dim", "1024", "--tsloo-at", "last",
                       "--tsloo-from", truth("gpt2", "1e-5", 0))],
    "mlp-ablations": [
        cell("mlp", "1e-3", 0, "_curvature", "curvature.py"),
        ours("mlp", "1e-5", 0, "_k64x10", "--n-masks", "10", "--mask-dim", "64",
             "--tsloo-from", truth("mlp", "1e-5", 0)),
        ours("mlp", "1e-5", 0, "_k64x1", "--n-masks", "1", "--mask-dim", "64",
             "--tsloo-from", truth("mlp", "1e-5", 0)),
    ],
    "timing": [
        cell("gpt2", "1e-5", 0, "_timing_baseline", "timing.py"),
        ours("gpt2", "1e-5", 0, "_timing_dvemb", "--skip-tsloo", "--methods", "dvemb"),
        ours("gpt2", "1e-5", 0, "_timing_adamw", "--skip-tsloo", "--methods", "adamw_influence"),
        ours("gpt2", "1e-5", 0, "_timing_full", *FULL, "--methods", "adamw_influence",
             "--tsloo-from", truth("gpt2", "1e-5", 0)),
        cell("gpt2", "1e-5", 0, "_timing_magic", "magic.py", "--tag", "_timing",
             "--n-queries", "64"),
        cell("gpt2", "1e-5", 0, "_timing_source", "source.py", "--tag", "_timing",
             "--n-queries", "64"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--run", action="store_true", help="execute the runs in order")
    ap.add_argument("--dry-run", action="store_true", help="list the runs only")
    ap.add_argument("--collect", action="store_true", help="run directories -> results/fidelity.jsonl")
    ap.add_argument("--table", action="store_true", help="print the tables from results/fidelity.jsonl")
    a = ap.parse_args()
    if a.collect or a.table:
        import tables

        path = RESULTS / "fidelity.jsonl"
        if a.collect:
            rows = tables.collect(RESULTS)
            tables.write_rows(rows, path)
            print(f"{len(rows)} rows -> {path}")
        if a.table:
            tables.print_tables(tables.read_rows(path))
        return
    if a.experiment is None:
        ap.error("--experiment is required (or --collect / --table)")
    cells = EXPERIMENTS[a.experiment]
    for i, c in enumerate(cells):
        done = (RESULTS / c["dir"] / "result.json").exists()
        print(f"{i:3d}  {'done' if done else '    '}  {c['dir']:36} python utils/{' '.join(c['cmd'])}")
    if not a.run:
        print("\n(add --run to execute; runs with a result.json are skipped)")
        return
    for c in cells:
        if (RESULTS / c["dir"] / "result.json").exists():
            continue
        print(f"\n##### {c['dir']}", flush=True)
        subprocess.run([sys.executable, "-u", str(UTILS / c["cmd"][0]), *c["cmd"][1:]],
                       cwd=HERE, check=True)


if __name__ == "__main__":
    main()
