"""Collect the run directories into the rows of ``results/fidelity.jsonl``.

``collect`` reads every ``results/<setting>/lr<lr>_seed<seed>_<run>/result.json``
into one row per (scale, method): the Spearman correlation with TSLOO, the
attribution time, and the peak GPU memory (or the status of a run that did
not complete within the budget).
"""

from __future__ import annotations

import json
from pathlib import Path

SCALES = ("gpt2", "qwen0.5b", "qwen1.5b", "qwen3b", "olmo2-1b", "olmo3-7b")
BASE = "lr1e-05_seed0"
# run-directory suffix -> key of the correlation in result.json
METHODS = {
    "masked": "adamw_influence",
    "masked2048": "adamw_influence",
    "masked4096": "adamw_influence",
    "masked8192": "adamw_influence",
    "full": "adamw_influence",
    "ekfac_k64": "ekfac_k64",
    "ekfac": "ekfac",
    "magic": "magic",
    "source": "source",
    "trackstar": "trackstar",
    "bergson_ekfac": "bergson_ekfac",
}


def collect(results: Path) -> list[dict]:
    rows = []
    for scale in SCALES:
        truth = results / scale / BASE / "result.json"
        if truth.exists():
            r = json.loads(truth.read_text())
            rows.append({"scale": scale, "run": "tsloo", "tsloo_s": r["tsloo_s"],
                         "abs_mean": r["tsloo_abs_mean"]})
        for run, key in METHODS.items():
            path = results / scale / f"{BASE}_{run}" / "result.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            if r.get("status") in ("oom", "failed"):  # a run recorded as not completed
                rows.append({"scale": scale, "run": run, "status": r["status"]})
            else:
                rows.append({"scale": scale, "run": run, "rho": r[key], "attribute_s": r["attribute_s"],
                             "peak_gb": r.get("peak_gb", r.get("capture_peak_gb"))})
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
