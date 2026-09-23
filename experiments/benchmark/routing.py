"""Routing measurements: attribution cost against sequence length at full dimension.

dattri-llm's cost model picks, per layer, between the factorized inner
product (quadratic in the sequence length T) and materializing the per-sample
gradient first (linear in T, with a one-time construction cost).  This
experiment sweeps T and runs dattri-llm with the cost model on ("auto") and
with one route pinned for every layer ("factorized", "materialized"), together
with Bergson and Kronfluence.

Setting: that of benchmark.py with T varied.  Pythia-0.5B, fp32, WikiText-103, one
A40, GradDot at full dimension, one query, T from 32 to 2048 (Pythia's
context) and, at one sequence per step, on to 16384, the longest sequence
whose step fits the A40.

``routing-steps-b<B>`` is the per-step protocol: one fixed batch size B for
every T, 128 timed steps after 8 warm-up steps, runtime reported per step.
``routing-steps-b1`` continues to T = 16384.  ``routing`` is the fixed-token
protocol (524,288 training tokens at every T, batch 8 up to T = 512 and 4096
tokens per batch beyond it), and ``routing-batch1`` runs that protocol at one
sequence per step on an eighth of the tokens.

    python routing.py --experiment routing-steps-b8 --dry-run   # list the cells
    python routing.py --experiment routing-steps-b8 --run       # run them in order

Cells run one at a time.  Results append to out/<experiment>/results.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

SEQ_LENS = (32, 64, 128, 256, 512, 1024, 2048)  # Pythia's context is 2048
# At one sequence per step: on to the longest sequence whose step fits an A40
# (positions past the context are rotary extrapolation; only the cost is read).
LONG_SEQ_LENS = (*SEQ_LENS, 4096, 8192, 16384)
TRAIN_TOKENS = 1024 * 512  # the training set of benchmark.py, in tokens
ROUTES = ("auto", "factorized", "materialized")
BASE = dict(
    family="pythia",
    dataset="wikitext103",
    seed=0,
    dtype="float32",
    proj_mode="full",
    n_test=1,
    warmup_train=0,
)


def cells(batch: int | None = None, train_tokens: int = TRAIN_TOKENS) -> list[dict]:
    """Five cells per sequence length over *train_tokens* training tokens.
    ``batch=None`` is batch 8 up to T = 512, then 4096 tokens per batch; an
    integer fixes the batch."""
    out = []
    for seq_len in SEQ_LENS:
        n_train = train_tokens // seq_len
        workload = dict(BASE, block_size=seq_len, n_train=n_train, measure_train=n_train,
                        batch=batch if batch is not None else min(8, 4096 // seq_len))
        for route in ROUTES:
            out += runner.grid(["dattri_llm"], ["graddot"], ["0.5b"], route=route, **workload)
        out += runner.grid(["bergson", "kronfluence"], ["graddot"], ["0.5b"], **workload)
    return out


STEPS, WARMUP_STEPS = 128, 8
BATCHES = (1, 2, 4, 8, 16)


def step_cells(batch: int) -> list[dict]:
    """The sweep at one fixed batch size for every T, timed over a fixed
    number of steps (after a short warm-up) and reported per step."""
    out = []
    for seq_len in LONG_SEQ_LENS if batch == 1 else SEQ_LENS:
        workload = dict(BASE, block_size=seq_len, batch=batch, steps=STEPS,
                        warmup_train=WARMUP_STEPS * batch, measure_train=STEPS * batch,
                        n_train=(WARMUP_STEPS + STEPS) * batch)
        for route in ROUTES:
            out += runner.grid(["dattri_llm"], ["graddot"], ["0.5b"], route=route, **workload)
        out += runner.grid(["bergson", "kronfluence"], ["graddot"], ["0.5b"], **workload)
    return out


EXPERIMENTS = {
    "routing": cells(),
    # One sequence per step: an eighth of the tokens, so every cell takes the
    # same number of steps as its batch-8 counterpart.
    "routing-batch1": cells(batch=1, train_tokens=TRAIN_TOKENS // 8),
    # Fixed steps and a fixed batch for every T; one experiment per batch size.
    **{f"routing-steps-b{b}": step_cells(b) for b in BATCHES},
}


if __name__ == "__main__":
    runner.main(EXPERIMENTS)
