"""Print the cross-library tables from results/query1.jsonl and query16.jsonl.

Attribution time T is every recorded phase except ``build_model`` and
``load_data`` (the phase names differ per library, so setup is excluded rather
than attribution named); M is the peak GPU memory over those phases.  When a
cell was measured more than once the row with the median T is used.
"""

from __future__ import annotations

import json
from pathlib import Path

SETUP = {"build_model", "load_data"}
LIBS = [("dattri_llm", "dattri-llm"), ("logix", "LogIX"), ("bergson", "Bergson"),
        ("kronfluence", "Kronfluence")]
METHODS = [("graddot", "GradDot"), ("kfac", "K-FAC"), ("ekfac", "EK-FAC")]


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def attribution_time(rec: dict) -> float | None:
    secs = [p.get("wall_s") or 0.0 for p in rec.get("phases", []) if p["phase"] not in SETUP]
    return sum(secs) if secs else None


def peak_gb(rec: dict) -> float:
    vals = [d.get("alloc_gb") or 0.0 for p in rec.get("phases", [])
            if p["phase"] not in SETUP for d in p.get("gpu_peak", []) if isinstance(d, dict)]
    return max(vals) if vals else 0.0


def median_cell(rows: list[dict]) -> dict | None:
    """The measured row with the median attribution time (an OOM row if none ran)."""
    ok = sorted((r for r in rows if r.get("status") == "ok" and attribution_time(r) is not None),
                key=attribution_time)
    if ok:
        return ok[(len(ok) - 1) // 2]
    return rows[0] if rows else None


def cells(rows: list[dict]) -> dict[tuple, list[dict]]:
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        t = r["task"]
        by.setdefault((r["lib"], t["method"], t.get("proj_mode")), []).append(r)
    return by


def print_table(title: str, rows: list[dict], not_expressible: set) -> None:
    by = cells(rows)
    print(f"\n{title}")
    head = f"{'method':8} {'proj':7}" + "".join(f"{lab:>22}" for _, lab in LIBS)
    print(head)
    print("-" * len(head))
    notes = []
    for method, m_label in METHODS:
        for proj in ("rank64", "full"):
            line = f"{m_label:8} {proj:7}"
            for lib, _ in LIBS:
                if (lib, method, proj) in not_expressible:
                    line += f"{'n/a':>22}"
                    continue
                r = median_cell(by.get((lib, method, proj), []))
                if r is None:
                    line += f"{'--':>22}"
                elif r.get("status") != "ok":
                    line += f"{r['status'].upper():>22}"
                else:
                    line += f"{attribution_time(r):>12.0f} s {peak_gb(r):>5.1f} GB"
                    n = len([x for x in by[(lib, method, proj)] if x.get("status") == "ok"])
                    batch = r["task"].get("batch")
                    if batch != 8 or n > 1:
                        notes.append(f"{lib} {method} {proj}: batch {batch}, {n} run(s)")
            print(line)
    for note in notes:
        print("  note:", note)


def main(results: Path, not_expressible: set) -> None:
    for name, title in (("query1", "One query (Table 1)"), ("query16", "Sixteen queries (Table 6)")):
        rows = load(results / f"{name}.jsonl")
        if not rows:
            print(f"\n{title}: no results/{name}.jsonl")
            continue
        print_table(title + "  (T = attribution seconds, M = peak GPU GB)", rows, not_expressible)
