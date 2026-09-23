"""Cross-library scaling measurements: four libraries up the Qwen ladder on H200s.

Qwen2.5 0.5B to 110B, WikiText-103, bf16, batch 1, 512-token sequences,
rank-64 projection wherever the library supports it, one query per device.  Every library
runs 8 warm-up samples through its full pipeline untimed, then 64 measured
samples; model build and dataset load are recorded as separate phases.  One
experiment per method and hardware tier:

    scaling-<method>         one H200, 0.5B-32B: dattri-llm, LogIX and
                             Kronfluence; Bergson (K-FAC/EK-FAC up to 3B)
    scaling-<method>-fsdp4   four H200s: dattri-llm at 72B and 110B (FSDP);
                             Bergson's own sharding for K-FAC/EK-FAC at
                             7B-32B and for GradDot at 72B/110B;
                             Kronfluence (FSDP) from the first scale that
                             does not fit one H200 (GradDot 32B, K-FAC and
                             EK-FAC 7B)

Run on a machine with the right GPUs:

    python scaling.py --experiment scaling-graddot --dry-run      # list the cells
    python scaling.py --experiment scaling-graddot --run          # one H200
    python scaling.py --experiment scaling-graddot-fsdp4 --run    # four H200s

Results append to out/<experiment>/results.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

METHODS = ("graddot", "kfac", "ekfac")
SINGLE = ("0.5b", "1b", "3b", "7b", "14b", "32b")

BASE = dict(family="qwen", dataset="wikitext103", block_size=512, seed=0,
            dtype="bfloat16", proj_mode="rank64", batch=1,
            n_train=72, warmup_train=8, measure_train=64, n_test=1)
SHARDED = dict(parallelism="fsdp", n_gpus=4)


def single(method: str) -> list[dict]:
    cells = runner.grid(["dattri_llm", "logix"], [method], SINGLE, **BASE)
    # Bergson's K-FAC / EK-FAC keep full-dimension factors; their 7B-32B
    # cells are in the sharded experiment.
    scales = SINGLE if method == "graddot" else ("0.5b", "1b", "3b")
    cells += runner.grid(["bergson"], [method], scales, **BASE)
    # Kronfluence runs every method at full dimension; a cell that reaches
    # ``time_limit_s`` is recorded as a timeout.
    return cells + runner.grid(["kronfluence"], [method], SINGLE,
                               **{**BASE, "time_limit_s": 1800})


def sharded(method: str) -> list[dict]:
    cells = runner.grid(["dattri_llm"], [method], ["72b", "110b"], **BASE, **SHARDED)
    # Kronfluence under FSDP, at the scales that do not fit one H200.
    scales = ["32b"] if method == "graddot" else ["7b", "14b", "32b"]
    cells += runner.grid(["kronfluence"], [method], scales,
                         **{**BASE, "time_limit_s": 1800}, **SHARDED)
    if method == "graddot":
        # One query per device for both libraries: dattri-llm captures its one
        # query on every rank, and Bergson shards a query set across its ranks
        # (it shards the build only from four queries), so it is given four.
        return cells + runner.grid(["bergson"], [method], ["72b", "110b"],
                                   **{**BASE, "n_test": 4}, **SHARDED)
    return cells + runner.grid(["bergson"], [method], ["7b", "14b", "32b"], **BASE, **SHARDED)


EXPERIMENTS = {}
for _m in METHODS:
    EXPERIMENTS[f"scaling-{_m}"] = single(_m)
    EXPERIMENTS[f"scaling-{_m}-fsdp4"] = sharded(_m)


def n_gpus(experiment: str) -> int:
    return max(t["n_gpus"] for t in EXPERIMENTS[experiment])


if __name__ == "__main__":
    runner.main(EXPERIMENTS)
