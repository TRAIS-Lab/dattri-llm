"""dattri-llm's methods on the OLMo models trained by OLMo-core's trainer.

``export`` trains once and saves the trained model as a Hugging Face
checkpoint for the scripts that take ``--final-model``; ``adamw`` captures the
trajectory under the hooks with the optimizer's moments and ``adamw-score``
scores it (masked AdamW-influence); ``capture`` stores the factorized gradients
on the trained model and ``ekfac`` scores them (see ``utils/olmocore.py``).
Multi-GPU runs are launched with ``torchrun``.

    python utils/attribution/olmo.py export --scale olmo3-7b --microbatch 32
    python utils/attribution/olmo.py adamw --scale olmo3-7b --microbatch 32 --hook-layers all
    python utils/attribution/olmo.py adamw-score --scale olmo3-7b
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]

import olmocore  # noqa: E402

if __name__ == "__main__":
    olmocore.main(olmocore.ATTRIBUTION_MODES, __doc__.split("\n\n")[0])
