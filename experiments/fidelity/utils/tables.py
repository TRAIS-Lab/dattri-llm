"""Collect the fidelity rows from the run directories and print the tables.

``collect`` turns every ``results/<setting>/<run>/`` into rows of
``results/fidelity.jsonl`` (one per setting, learning rate, seed, variant
and method): the Spearman correlation with the run's ground truth on the
queries every method shares (all 500 test images for the MLP; the first 64
validation blocks for GPT-2, the ones MAGIC and SOURCE were run on), plus
the wall-clock rows of the timing runs.  ``print_tables`` prints the
paper's tables from those rows.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import torch

from protocol import SIGN, make_batches, occurrences, spearman_per_column

LRS = {"mlp": (1e-3, 1e-4, 1e-5), "gpt2": (1e-4, 5e-5, 1e-5)}
SEEDS = (0, 1, 2)
QUERIES = {"mlp": 500, "gpt2": 64}
BATCHES = {"mlp": (6000, 64, 1), "gpt2": (512, 32, 3)}  # n_train, batch, epochs
RUN = re.compile(r"lr(?P<lr>[^_]+)_seed(?P<seed>\d+)(?P<variant>_.*)?$")
OURS = ("adamw_influence", "dvemb")


def _load(path: Path):
    return torch.load(path, weights_only=False)


def _mean_std(vals: list[float]) -> str:
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.3f}±{sd:.3f}"


# --------------------------------------------------------------------------- #
# Rows from run directories                                                    #
# --------------------------------------------------------------------------- #


def _rho(pred: torch.Tensor, truth: torch.Tensor, n_q: int) -> float:
    return float(spearman_per_column(pred[:, :n_q], truth[:, :n_q]).mean())


def _magic_pred(scores: torch.Tensor, batches, pairs) -> torch.Tensor:
    """MAGIC's per-(sample, step) weights summed over the pairs' steps."""
    pred = torch.zeros(len(pairs), scores.shape[0])
    for k, (i, t) in enumerate(pairs):
        for tt in [t] if t is not None else occurrences(batches, i):
            pos = int((batches[tt] == i).nonzero()[0])
            pred[k] += scores[:, tt, pos]
    return pred


def _timing_row(setting: str, lr: float, seed: int, method: str, run: Path) -> dict | None:
    """Phase times and peaks parsed from a timing run's log."""
    log = (run / "log.txt").read_text() if (run / "log.txt").exists() else ""
    secs = lambda pat: [float(m) for m in re.findall(pat + r"[^\n]*?(\d+)s", log)]  # noqa: E731
    peaks = [float(p) for p in re.findall(r"peak (\d+\.\d+) GB", log)]
    if method == "baseline":
        r = json.loads((run / "result.json").read_text())
        train, attr = r["train_s"], 0.0
        peaks = [float(re.search(r"(\d+\.\d+)", r["peak"]).group(1))]
    elif method == "magic":
        train, attr = secs(r"forward:")[0], secs(r"query \d+/\d+ ")[-1]
    elif method == "source":
        train = secs(r"trajectory \+ checkpoints:")[0]
        attr = secs(r"factors \(steps 1-4\):")[0] + secs(r"queries \+ walk \+ scoring")[0]
    else:
        train, attr = secs(r"reference \+ capture:")[0], secs(r"attribution total:")[0]
    return {"setting": setting, "lr": lr, "seed": seed, "variant": "timing", "method": method,
            "train_s": train, "attribution_s": attr, "total_s": train + attr,
            "peak_gb": max(peaks) if peaks else None}


def collect(results: Path) -> list[dict]:
    rows: list[dict] = []
    for setting in LRS:
        base_dir = results / setting
        if not base_dir.is_dir():
            continue
        n_q = QUERIES[setting]
        for run in sorted(base_dir.iterdir()):
            m = RUN.match(run.name)
            if not run.is_dir() or m is None:
                continue
            lr, seed, variant = float(m["lr"]), int(m["seed"]), (m["variant"] or "").lstrip("_")
            key = {"setting": setting, "lr": lr, "seed": seed}
            base = base_dir / f"lr{lr:g}_seed{seed}"
            if variant.startswith("timing_"):
                rows.append(_timing_row(setting, lr, seed, variant[len("timing_"):], run))
                continue
            if variant == "curvature":
                r = json.loads((run / "result.json").read_text())
                rows += [{**key, "variant": "curvature", "method": "adamw_influence",
                          "curvature": mode, "rho": r[mode]}
                         for mode in ("none", "empirical_fisher", "exact_ggn")]
                continue
            if not (run / "matrices.pt").exists():
                continue
            saved = _load(run / "matrices.pt")
            if variant in ("magic", "source"):
                # A baseline run scored against the every-occurrence ground
                # truth (one row per (sample, step)) is reduced to its
                # last-occurrence rows, which are the protocol's pairs.
                pred, pairs = saved["pred"], saved["pairs"]
                if all(t is not None for _, t in pairs):
                    batches = make_batches(*BATCHES[setting], seed)
                    pred = pred[[k for k, (i, t) in enumerate(pairs)
                                 if t == occurrences(batches, i)[-1]]]
                truth = _load(base / "matrices.pt")["tsloo"]
                rows.append({**key, "variant": "", "method": variant, "n_queries": n_q,
                             "rho": _rho(pred, truth, n_q)})
                if variant == "magic" and (run / "magic_scores.pt").exists():
                    # The same per-(sample, step) weights under the other
                    # ground-truth conventions (every / first / each occurrence).
                    ms = _load(run / "magic_scores.pt")
                    for conv in ("all", "first", "each"):
                        other = base_dir / f"lr{lr:g}_seed{seed}_{conv}"
                        if (other / "matrices.pt").exists():
                            t = _load(other / "matrices.pt")
                            pairs = t.get("pairs") or [(int(i), None) for i in t["selected"].tolist()]
                            pred = _magic_pred(ms["scores"], ms["batches"], pairs)
                            rows.append({**key, "variant": conv, "method": "magic",
                                         "n_queries": n_q, "rho": _rho(pred, t["tsloo"], n_q)})
                continue
            # Our methods, scored against the run's own ground truth.
            truth = saved["tsloo"]
            for method in OURS:
                if method in saved:
                    rows.append({**key, "variant": variant, "method": method, "n_queries": n_q,
                                 "rho": _rho(SIGN[method] * saved[method], truth, n_q)})
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()] if path.is_file() else []


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #


def _cell(rows, **where) -> str:
    vals = [r["rho"] for r in rows if all(r.get(k) == v for k, v in where.items())]
    return _mean_std(vals)


def _time(rows, method: str) -> str:
    r = [r for r in rows if r["variant"] == "timing" and r["method"] == method]
    return f"{r[0]['total_s']:.0f}s {r[0]['peak_gb']:.0f}GB" if r else "-"


def print_tables(rows: list[dict]) -> None:
    print("Fidelity on GPT-2 at lr 1e-5 (Spearman with TSLOO, mean±std over seeds; "
          "wall-clock and peak on one A40, 64 queries)")
    for label, where, timing in (
        ("DVEmb (k=512)", dict(variant="", method="dvemb"), "dvemb"),
        ("AdamW-influence (k=512)", dict(variant="", method="adamw_influence"), "adamw"),
        ("AdamW-influence (full)", dict(variant="full", method="adamw_influence"), "full"),
        ("MAGIC", dict(variant="", method="magic"), "magic"),
    ):
        print(f"  {label:26} {_cell(rows, setting='gpt2', lr=1e-5, **where):>13}  {_time(rows, timing):>12}")
    print(f"  {'plain training':26} {'':>13}  {_time(rows, 'baseline'):>12}")

    print("\nBoth settings, three learning rates (mean±std over seeds)")
    cols = [("dvemb", "DVEmb"), ("source", "SOURCE"), ("magic", "MAGIC"), ("adamw_influence", "AdamW-inf.")]
    print(f"  {'setting':8} {'lr':>6} " + "".join(f"{lab:>14}" for _, lab in cols))
    for setting, lrs in LRS.items():
        for lr in lrs:
            print(f"  {setting:8} {lr:>6g} " + "".join(
                f"{_cell(rows, setting=setting, lr=lr, variant='', method=m):>14}" for m, _ in cols))
    print(f"  {'GPT-2 wall-clock (s)':15} " + "".join(f"{_time(rows, t):>14}" for t in ("dvemb", "source", "magic", "adamw")))

    print("\nGround-truth conventions, GPT-2 lr 1e-5 seed 0 (last = the protocol above)")
    for conv in ("", "all", "first", "each"):
        print(f"  {conv or 'last':6} AdamW-influence {_cell(rows, setting='gpt2', lr=1e-5, seed=0, variant=conv, method='adamw_influence'):>13}"
              f"   MAGIC {_cell(rows, setting='gpt2', lr=1e-5, seed=0, variant=conv, method='magic'):>13}")

    print("\nEvery coordinate (full) on GPT-2, seed 0, and the k=1024 mask at lr 1e-5")
    for lr in LRS["gpt2"]:
        print(f"  lr {lr:<7g} AdamW-influence {_cell(rows, setting='gpt2', lr=lr, seed=0, variant='full', method='adamw_influence'):>13}"
              f"   DVEmb {_cell(rows, setting='gpt2', lr=lr, seed=0, variant='full', method='dvemb'):>13}")
    print(f"  k=1024     AdamW-influence {_cell(rows, setting='gpt2', lr=1e-5, seed=0, variant='k1024', method='adamw_influence'):>13}")

    print("\nMLP ablations, seed 0: curvature term at lr 1e-3; coverage at lr 1e-5")
    for mode in ("none", "empirical_fisher", "exact_ggn"):
        print(f"  curvature {mode:17} {_cell(rows, setting='mlp', lr=1e-3, seed=0, variant='curvature', curvature=mode):>13}")
    for variant, label in (("", "every coordinate"), ("k64x10", "10 masks x 64/layer"), ("k64x1", "1 mask x 64/layer")):
        print(f"  {label:27} {_cell(rows, setting='mlp', lr=1e-5, seed=0, variant=variant, method='adamw_influence'):>13}")
