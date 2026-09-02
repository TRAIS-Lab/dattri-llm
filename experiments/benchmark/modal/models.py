"""Model registry for the universal attribution benchmark: family x scale -> a
concrete HF model id + metadata.

Scale labels are the *nominal* ladder (0.5b/1b/3b/7b/14b/32b); the mapped model
is the closest real release (e.g. "1b" -> Qwen2.5-1.5B), with its true parameter
count recorded so the x-axis is honest.  Feasibility is for a single A40 (46 GB)
under per-sample gradient capture (memory-heavier than plain inference).
"""

from __future__ import annotations

# scale -> (hf_id, params_in_billions)
MODELS: dict[str, dict[str, tuple[str, float]]] = {
    # Qwen2.5 -- open, the only family covering the full 0.5b..32b ladder.
    "qwen": {
        "0.5b": ("Qwen/Qwen2.5-0.5B", 0.49),
        "1b": ("Qwen/Qwen2.5-1.5B", 1.54),
        "3b": ("Qwen/Qwen2.5-3B", 3.09),
        "7b": ("Qwen/Qwen2.5-7B", 7.62),
        "14b": ("Qwen/Qwen2.5-14B", 14.77),
        "32b": ("Qwen/Qwen2.5-32B", 32.5),
        # Beyond a single H200: sharded capture only.
        "72b": ("Qwen/Qwen2.5-72B", 72.7),
    },
    # Llama-3 -- gated (needs an HF token with Llama-3 access).
    # Pythia -- open; same architecture & training data across scales, so the
    # cleanest family for a pure model-scale study.
    "pythia": {
        "0.5b": ("EleutherAI/pythia-410m", 0.41),
        "1b": ("EleutherAI/pythia-1b", 1.01),
        "3b": ("EleutherAI/pythia-2.8b", 2.78),
        "7b": ("EleutherAI/pythia-6.9b", 6.86),
    },
}


def resolve(family: str, scale: str) -> tuple[str, float]:
    """(hf_id, params_b) for a (family, scale); raises KeyError if unavailable."""
    return MODELS[family][scale]


def dtype_for(params_b: float, override: str | None = None) -> str:
    """Model dtype for a run; ``override`` (the experiment's ``dtype``) wins.

    The size-based fallback below couples precision to the x-axis of the
    scaling plots: the ladder crosses the 1.0B threshold between its first and
    second rung, so the 0.5B point of every curve runs in fp32 and the rest in
    bf16.  That shows up as a *drop* in both time and peak memory from 0.5B to
    1B -- fewer activation/gradient-capture bytes and bf16 tensor cores -- which
    reads as a scaling property but is a dtype change.  Every experiment in
    ``run.py`` therefore states its dtype outright; the fallback is kept only
    for ad-hoc task dicts that do not.
    """
    if override is not None:
        return override
    return "float32" if params_b < 1.0 else "bfloat16"


# --- parallelism, coupled to scale -------------------------------------------
# Single-A40 (46 GB) ceiling for per-sample gradient capture under capture-time
# projection: the whole nominal "7b" tier (pythia-6.9B, Qwen-7.6B, Llama-8.0B)
# fits on one card this way; 14B+ must be sharded with FSDP.  (Raw, unprojected
# capture of the 7b tier is tighter -- prefer FSDP there.)
_SINGLE_CARD_CEILING_B = 8.5


def natural_parallelism(params_b: float) -> str:
    """The strategy a scale is *naturally* run with: one card while it fits,
    else FSDP (sharded)."""
    return "single" if params_b <= _SINGLE_CARD_CEILING_B else "fsdp"


def parallelism_options(params_b: float) -> list[str]:
    """Every parallelism worth benchmarking at this scale.  Below the single-card
    ceiling we also offer DDP (replicated -- throughput + multi-replica capture
    correctness) and FSDP (sharded, to exercise the sharded path); above it only
    FSDP fits."""
    return ["single", "ddp", "fsdp"] if params_b <= _SINGLE_CARD_CEILING_B else ["fsdp"]


def n_gpus(params_b: float, parallelism: str) -> int:
    """GPUs required for a (scale, parallelism) on A40 (46 GB)."""
    if parallelism == "single":
        return 1
    if parallelism == "ddp":
        return 2  # replicated -- 2 is the minimal meaningful DDP
    # FSDP shards ~2 B/param (bf16) weights + optimizer/grad-capture headroom.
    # The (80, 4) rung is a modal/-only addition.  Upstream this table stops at
    # 32 and everything larger falls through to `return 8`, so Qwen-32B and -72B
    # both requested 8 ranks -- contradicting the `-fsdp4` experiment name and
    # the parent README's "4x H200".  The table is documented as an A40 (46 GB)
    # calculation; on 141 GB H200s, 72.7B in bf16 is ~145 GB of weights, so 4
    # ranks hold ~36 GB of shard each and have ample headroom.  Four also halves
    # the CPU-side init cost, which is the binding constraint at 72B: every rank
    # calls from_pretrained independently, so host RAM scales with world size.
    for ceiling, g in ((8, 2), (16, 2), (32, 4), (80, 4)):
        if params_b <= ceiling:
            return g
    return 8
