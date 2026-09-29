"""The ground truth of the OLMo models under OLMo-core's trainer.

``prepare`` writes the token files and the initial checkpoint, ``train`` trains
once (the reference run, or with one block left out), ``compare`` prints the
difference between two runs, ``selected`` lists the 50 blocks left out, and
``truth`` collects the runs into ``results/<scale>/lr<lr>_seed<seed>/matrices.pt``
(see ``utils/olmocore.py``).  Multi-GPU runs are launched with ``torchrun``.

    torchrun --nproc-per-node 4 utils/truth/olmo.py prepare --scale olmo2-1b
    torchrun --nproc-per-node 4 utils/truth/olmo.py train --scale olmo2-1b --left-out none --tag ref
    python utils/truth/olmo.py truth --scale olmo2-1b
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]

import olmocore  # noqa: E402

if __name__ == "__main__":
    olmocore.main(olmocore.TRUTH_MODES, __doc__.split("\n\n")[0])
