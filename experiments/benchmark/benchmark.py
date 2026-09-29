"""Cross-library efficiency measurements on one model.

Pythia-0.5B, one A40, WikiText-103, n_train = 1024 sequences of 512 tokens,
fp32, batch 8, every library in both projection regimes: rank-64 and full
dimension.  ``query1`` scores one query (n_test = 1); ``query16`` scores sixteen.

    python benchmark.py --experiment query1 --dry-run   # list the cells
    python benchmark.py --experiment query1 --run       # run them in order

Cells run one at a time.  Results append to out/<experiment>/results.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

LIBS = ("dattri_llm", "logix", "bergson", "kronfluence")
METHODS = ("graddot", "kfac", "ekfac")

# (library, method, projection) combinations that are not listed as cells:
# Kronfluence runs at full dimension only, and Bergson's EK-FAC accepts no
# projection.
NOT_EXPRESSIBLE = {("kronfluence", "graddot", "rank64"), ("kronfluence", "kfac", "rank64"),
                   ("kronfluence", "ekfac", "rank64"), ("bergson", "ekfac", "rank64")}

BASE = dict(family="pythia", dataset="wikitext103", block_size=512, seed=0,
            dtype="float32", n_train=1024, warmup_train=0, measure_train=1024)


def table_cells(n_test: int) -> list[dict]:
    cells = []
    for proj_mode in ("rank64", "full"):
        for lib in LIBS:
            for method in METHODS:
                if (lib, method, proj_mode) in NOT_EXPRESSIBLE:
                    continue
                cells += runner.grid([lib], [method], ["0.5b"], proj_mode=proj_mode,
                                     n_test=n_test, batch=8, **BASE)
    return cells


EXPERIMENTS = {
    "query1": table_cells(n_test=1),
    "query16": table_cells(n_test=16),
}


if __name__ == "__main__":
    runner.main(EXPERIMENTS)
