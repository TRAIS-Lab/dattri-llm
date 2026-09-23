"""Report the ordinary-versus-invasive capture pairs of ``capture.py``.

Time and memory follow ``tables.py``: every phase except setup, and the peak
over those phases.  Agreement compares the two score matrices of each
repetition after aligning rows and columns by sample hash.  The floor is the
same comparison between repetitions of the ordinary path alone (run-to-run
nondeterminism of the reference); a pair agrees when its relative Frobenius
error is within ``max(TOL[precision], FLOOR_FACTOR * floor)``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from tables import SETUP, attribution_time

FAMILIES = ("linear_io", "invasive_linear_io")
METHODS = (("graddot", "GradDot"), ("kfac", "K-FAC"), ("ekfac", "EK-FAC"))
# Relative Frobenius tolerance by arithmetic precision.
TOL = {"float32": 1e-5, "tf32": 1e-3, "bfloat16": 1e-2}
FLOOR_FACTOR = 10
TOPK = 10


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def precision(rec: dict) -> str:
    if rec.get("dtype") == "bfloat16":
        return "bfloat16"
    return "tf32" if rec.get("tf32") else "float32"


def peaks(rec: dict) -> tuple[float, float]:
    """Peak allocated and reserved GiB over the non-setup phases."""
    devs = [d for p in rec["phases"] if p["phase"] not in SETUP for d in p.get("gpu_peak", [])]
    return (max((d.get("alloc_gb") or 0.0 for d in devs), default=0.0),
            max((d.get("reserved_gb") or 0.0 for d in devs), default=0.0))


def scores(root: Path, rec: dict) -> tuple[torch.Tensor, list[str], list[str]]:
    obj = torch.load(root / rec["score_file"], weights_only=False)
    return obj["score"].double(), obj["train_ids"], obj["test_ids"]


def aligned(a, b) -> tuple[torch.Tensor, torch.Tensor]:
    """``b``'s matrix reordered to ``a``'s train and test hashes."""
    (x, tr_a, te_a), (y, tr_b, te_b) = a, b
    if set(tr_a) != set(tr_b) or set(te_a) != set(te_b):
        msg = "the two runs scored different samples"
        raise ValueError(msg)
    rows = {h: i for i, h in enumerate(tr_b)}
    cols = {h: j for j, h in enumerate(te_b)}
    return x, y[[rows[h] for h in tr_a]][:, [cols[h] for h in te_a]]


def agreement(a, b) -> dict:
    x, y = aligned(a, b)  # (n_train, n_test)
    diff = x - y
    rx = x.argsort(0).argsort(0).double()
    ry = y.argsort(0).argsort(0).double()
    rx, ry = rx - rx.mean(0), ry - ry.mean(0)
    rho = (rx * ry).sum(0) / (rx.norm(dim=0) * ry.norm(dim=0))
    k = min(TOPK, x.shape[0])
    tx, ty = x.topk(k, dim=0).indices.T, y.topk(k, dim=0).indices.T
    top = torch.tensor([len(set(p.tolist()) & set(q.tolist())) / k for p, q in zip(tx, ty, strict=True)])
    return {"max_abs": diff.abs().max().item(),
            "rel_fro": (diff.norm() / x.norm().clamp_min(1e-300)).item(),
            "rho_mean": rho.mean().item(), "rho_min": rho.min().item(),
            "top_mean": top.mean().item(), "top_min": top.min().item()}


def pairs(rows: list[dict]) -> dict[tuple, dict[int, dict[str, dict]]]:
    """(method, proj_mode) -> repeat -> hook family -> row (the last ok row)."""
    out: dict = {}
    for r in rows:
        t = r["task"]
        if r.get("status") != "ok" or "hook_family" not in t:
            continue
        out.setdefault((t["method"], t["proj_mode"]), {}).setdefault(
            t.get("repeat", 0), {})[t["hook_family"]] = r
    return out


def floor(root: Path, reps: dict[int, dict[str, dict]]) -> float | None:
    """Largest relative error between consecutive repetitions of the ordinary path."""
    runs = [reps[r]["linear_io"] for r in sorted(reps)]
    errs = [agreement(scores(root, p), scores(root, q))["rel_fro"]
            for p, q in zip(runs, runs[1:], strict=False)]
    return max(errs) if errs else None


def fmt_range(vals: list[float]) -> str:
    med = statistics.median(vals)
    return f"{med:.2f}x" if len(vals) == 1 else f"{med:.2f}x [{min(vals):.2f}-{max(vals):.2f}]"


def report(name: str, root: Path, timed: bool) -> None:
    by = pairs(load(root / "results.jsonl"))
    if not by:
        print(f"\n{name}: no completed pairs in {root}")
        return
    print(f"\n=== {name} ({root}) ===")
    if timed:
        head = (f"{'method':8} {'proj':7} {'T ordinary':>11} {'T invasive':>11} "
                f"{'speedup':>20} {'alloc GiB o/i':>15} {'reserved GiB o/i':>17}  n")
        print(head)
        print("-" * len(head))
    splits_lines, agree_lines = [], []
    for method, label in METHODS:
        for proj in ("full", "rank64"):
            reps = {r: fams for r, fams in by.get((method, proj), {}).items()
                    if all(f in fams for f in FAMILIES)}
            if not reps:
                continue
            ordered = [reps[r] for r in sorted(reps)]
            t = {f: [attribution_time(p[f]) for p in ordered] for f in FAMILIES}
            if timed:
                mem = {f: [peaks(p[f]) for p in ordered] for f in FAMILIES}
                alloc = "/".join(f"{statistics.median(m[0] for m in mem[f]):.1f}" for f in FAMILIES)
                res = "/".join(f"{statistics.median(m[1] for m in mem[f]):.1f}" for f in FAMILIES)
                ratio = [o / i for o, i in zip(t["linear_io"], t["invasive_linear_io"], strict=True)]
                print(f"{label:8} {proj:7} {statistics.median(t['linear_io']):>9.1f} s "
                      f"{statistics.median(t['invasive_linear_io']):>9.1f} s "
                      f"{fmt_range(ratio):>20} {alloc:>15} {res:>17}  {len(ordered)}")
                for f in FAMILIES:
                    split = [p[f].get("attribute_splits") for p in ordered]
                    if all(split):
                        parts = ", ".join(f"{k} {statistics.median(s[k] for s in split):.1f}"
                                          for k in split[0])
                        splits_lines.append(f"  {label} {proj} {f}: {parts} s")
            agree = [agreement(scores(root, p["linear_io"]), scores(root, p["invasive_linear_io"]))
                     for p in ordered]
            fl = floor(root, reps)
            prec = precision(ordered[0]["linear_io"])
            tol = max(TOL[prec], FLOOR_FACTOR * fl) if fl is not None else TOL[prec]
            worst = max(a["rel_fro"] for a in agree)
            agree_lines.append(
                f"{label:8} {proj:7} {max(a['max_abs'] for a in agree):>10.2e} {worst:>10.2e} "
                f"{'-' if fl is None else f'{fl:.2e}':>9} {tol:>9.1e} {prec:>8} "
                f"{statistics.mean(a['rho_mean'] for a in agree):>8.5f}/"
                f"{min(a['rho_min'] for a in agree):.5f} "
                f"{statistics.mean(a['top_mean'] for a in agree):>6.2f}/"
                f"{min(a['top_min'] for a in agree):.2f}  "
                f"{'agree' if worst <= tol else 'DIFFER'}")
    for line in splits_lines:
        print(line)
    head = (f"{'method':8} {'proj':7} {'max |d|':>10} {'rel Fro':>10} {'floor':>9} {'tol':>9} "
            f"{'prec':>8} {'Spearman mean/min':>17} {'top-10 mean/min':>15}")
    print("\n" + head)
    print("-" * len(head))
    for line in agree_lines:
        print(line)


def main(here: Path, experiments: dict) -> None:
    for name in experiments:
        root = here / "results" / name
        if not (root / "results.jsonl").is_file():
            root = here / "out" / name
        # The invasive shared-fit run loads its fit, so its time is not comparable.
        report(name, root, timed="shared-fit" not in name)
