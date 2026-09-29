"""Model registry: family x scale -> HF model id and parameter count.

A scale label is a nominal size; it maps to a released model of that family
(e.g. "1b" -> Qwen2.5-1.5B) together with that model's parameter count in
billions, which is stored in the task as ``params_b``.
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
        # The 110B scale is a Qwen1.5 model.
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
    """Model dtype for a run: *override* (the task's ``dtype`` field) if given,
    else float32 below 1B parameters and bfloat16 from 1B up."""
    if override is not None:
        return override
    return "float32" if params_b < 1.0 else "bfloat16"
