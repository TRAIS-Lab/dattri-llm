"""dattri-llm's AdamW-influence on the fidelity trajectory, or its ground truth.

One run: the reference trajectory under hooks, the query capture, the
attribution, and the Spearman correlation with TSLOO; writes ``result.json``
and ``matrices.pt`` under ``results/qwen<scale>/lr<lr>_seed<seed><tag>``.

The ground truth, then the masked run scored against it (run from the
directory of ``fidelity.py``; ``TRUTH`` is the ground-truth run directory):

    TRUTH=results/qwen0.5b/lr1e-05_seed0
    python utils/attribution/adamw_influence.py --scale 0.5b --lr 1e-5 --seed 0 --tsloo-only
    python utils/attribution/adamw_influence.py --scale 0.5b --lr 1e-5 --seed 0 --tag _masked --n-val 64 --tsloo-from $TRUTH
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]

from protocol import run  # noqa: E402
from settings import build, parser  # noqa: E402

if __name__ == "__main__":
    run(build(parser().parse_args()))
