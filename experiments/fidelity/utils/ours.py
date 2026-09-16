"""DVEmb and AdamW-influence on one trajectory, with the TSLOO ground truth
(``protocol.run``): the reference run under hooks, the query capture, the
attribution sweeps, and the leave-one-out reruns unless ``--tsloo-from``
reuses another run's.  Writes ``result.json``, ``rows.pt`` and
``matrices.pt`` under ``results/<setting>/lr<lr>_seed<seed><tag>``.

    python utils/ours.py --setting mlp  --lr 1e-3 --seed 0
    python utils/ours.py --setting gpt2 --lr 1e-5 --seed 0 --tsloo-at last
"""

from __future__ import annotations

from protocol import run
from settings import build, parser

if __name__ == "__main__":
    run(build(parser().parse_args()))
