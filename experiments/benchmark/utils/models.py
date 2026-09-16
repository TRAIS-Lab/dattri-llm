"""Model registry: family x scale -> HF model id and true parameter count.

Scale labels are the nominal ladder; the mapped model is the closest real
release (e.g. "1b" -> Qwen2.5-1.5B) and its true size is recorded so every
x-axis is honest.
"""

from __future__ import annotations

# scale -> (hf_id, params_in_billions)
MODELS: dict[str, dict[str, tuple[str, float]]] = {
    "qwen": {
        "0.5b": ("Qwen/Qwen2.5-0.5B", 0.49),
        "1b": ("Qwen/Qwen2.5-1.5B", 1.54),
        "3b": ("Qwen/Qwen2.5-3B", 3.09),
        "7b": ("Qwen/Qwen2.5-7B", 7.62),
        "14b": ("Qwen/Qwen2.5-14B", 14.77),
        "32b": ("Qwen/Qwen2.5-32B", 32.5),
        "72b": ("Qwen/Qwen2.5-72B", 72.7),
        # Qwen1.5: the 2.5 line stops at 72B, and 110B is the only dense Qwen
        # release between there and what four H200s hold.
        "110b": ("Qwen/Qwen1.5-110B", 111.2),
    },
    "pythia": {
        "0.5b": ("EleutherAI/pythia-410m", 0.41),
        "1b": ("EleutherAI/pythia-1b", 1.01),
        "3b": ("EleutherAI/pythia-2.8b", 2.78),
        "7b": ("EleutherAI/pythia-6.9b", 6.86),
    },
}


def resolve(family: str, scale: str) -> tuple[str, float]:
    """(hf_id, params_b) for a (family, scale); KeyError if unavailable."""
    return MODELS[family][scale]


def dtype_for(params_b: float, override: str | None = None) -> str:
    """Model dtype for a run.  Every experiment states its dtype; the size rule
    is a fallback for ad-hoc task dicts only."""
    if override is not None:
        return override
    return "float32" if params_b < 1.0 else "bfloat16"
