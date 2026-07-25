"""Aggregate every pair's results.jsonl into one readable summary.

    python summarize.py            # prints a per-pair table
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    for pair_dir in sorted(HERE.glob("vs_*")):
        f = pair_dir / "results.jsonl"
        if not f.exists():
            continue
        print(f"\n== {pair_dir.name} ==")
        rows = [json.loads(x) for x in f.open()]
        # phases
        for r in rows:
            if r.get("phase") == "summary":
                extras = {k: v for k, v in r.items()
                          if k in ("store_bytes", "log_store_bytes",
                                   "pearson_vs_exact", "score_shape",
                                   "step_time_median_s", "step_time_first_s")}
                if not extras:
                    continue
                if "store_bytes" in extras:
                    extras["store"] = fmt_bytes(extras.pop("store_bytes"))
                if "log_store_bytes" in extras:
                    extras["store"] = fmt_bytes(extras.pop("log_store_bytes"))
                print(f"  {r['lib']:<38} summary   {extras}")
            elif r.get("phase") == "agreement":
                print(f"  {'':<38} agreement {r.get('pair')}: "
                      f"pearson={r.get('pearson')} spearman={r.get('spearman')}")
            else:
                mem = r.get("peak_mem_gb", r.get("peak_mem_gb_smi"))
                print(f"  {r['lib']:<38} {r['phase']:<9} "
                      f"{r.get('wall_s', '?'):>8}s  "
                      f"{r.get('throughput', '?'):>8} {r.get('unit', '')}  "
                      f"peak {mem} GB")


if __name__ == "__main__":
    main()
