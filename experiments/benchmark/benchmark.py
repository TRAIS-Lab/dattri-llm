"""Cross-library efficiency tables (Tables 1 and 6 in the paper).

Pythia-0.5B, one A40, WikiText-103, n_train = 1024 sequences of 512 tokens,
fp32, every library in both projection regimes: rank-64 (LoGra) and full
dimension.  Table 1 scores one query (n_test = 1); Table 6 scores sixteen.

    python benchmark.py --experiment query1 --dry-run   # list the cells
    python benchmark.py --experiment query1 --run       # run them in order
    python benchmark.py --table                         # print both tables

Cells run one at a time; wrap the same command in a job script to use a
scheduler (see README.md).  Results append to out/<experiment>/results.jsonl;
copy the file to results/<experiment>.jsonl to make it the table's source.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

LIBS = ("dattri_llm", "logix", "bergson", "kronfluence")
METHODS = ("graddot", "kfac", "ekfac")

# Cells a library cannot express, reported as n/a rather than measured:
# kronfluence has no projected mode (its rank-64 run would repeat the
# full-dimension computation), and bergson's EK-FAC refuses any projection.
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
    "query1": table_cells(n_test=1),  # Table 1: one query
    "query16": table_cells(n_test=16),  # Table 6: sixteen queries
}


if __name__ == "__main__":
    if "--table" in sys.argv:
        import tables

        tables.main(HERE / "results", NOT_EXPRESSIBLE)
    else:
        runner.main(EXPERIMENTS)
