"""The ground truth of GPT-2 and Qwen2.5: the reference run and the leave-one-out reruns.

Trains the trajectory once, records the validation losses, then repeats the
run without each of the 50 selected blocks and records the change
(``protocol.run_tsloo_only``).  Writes ``result.json``, ``log.txt`` and
``matrices.pt`` (``tsloo``, ``selected``, ``pairs``, ``ref_losses``) under
``results/<setting>/lr<lr>_seed<seed>``, which the attribution scripts read
through ``--truth-dir`` / ``--tsloo-from``.

    python utils/truth/tsloo.py --scale gpt2 --lr 1e-5 --seed 0
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]

from protocol import run  # noqa: E402
from settings import build, parser  # noqa: E402

if __name__ == "__main__":
    a = parser().parse_args()
    a.tsloo_only = True
    run(build(a))
