"""dattri-llm: Training data attribution for large language models.

The advertised surface is importable from the top level::

    from dattri_llm import HookManager, HookManagerConfig, OffloadCallback

    with HookManager(model, callbacks=[...]).collect():
        trainer.train()

The gradient-capture core (hooks, callbacks, data model, storage) is imported
eagerly and needs only ``torch``.  The attribution layer (attributors, score,
arguments) and the live streamer are resolved lazily on first attribute
access, so ``import dattri_llm`` stays light and does not require their
optional dependencies (``dattri``, ``tqdm``) until they are actually used.
"""

from importlib import import_module

from dattri_llm.gradient import (
    REGISTER_ALL,
    CaptureCallback,
    DataSelectionCallback,
    GradientFileManager,
    GradientRecord,
    HookManager,
    HookManagerCallback,
    HookManagerConfig,
    OffloadCallback,
    default_hook_assignment,
)
from dattri_llm.gradient.gradient import Factorized, Gradient
from dattri_llm.utils.hashing import hash_batch, hash_sample

__version__ = "0.1.0"

# Lazily resolved exports: {name: module that provides it}.  Kept out of the
# eager imports so the capture core stays importable with torch alone.
_LAZY_EXPORTS = {
    "GradientStreamer": "dattri_llm.gradient.streaming",
    "DiskGradientSource": "dattri_llm.gradient.streaming",
    "AttributionArguments": "dattri_llm.attribution.arguments",
    "AttributionScore": "dattri_llm.attribution.score",
    "TracInAttributor": "dattri_llm.attribution",
    "KFACAttributor": "dattri_llm.attribution",
    "EKFACAttributor": "dattri_llm.attribution",
    "DVEmbAttributor": "dattri_llm.attribution",
}

__all__ = [
    "REGISTER_ALL",
    # lazily resolved (see _LAZY_EXPORTS):
    "AttributionArguments",
    "AttributionScore",
    "CaptureCallback",
    "DVEmbAttributor",
    "DataSelectionCallback",
    "DiskGradientSource",
    "EKFACAttributor",
    "Factorized",
    "Gradient",
    "GradientFileManager",
    "GradientRecord",
    "GradientStreamer",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "KFACAttributor",
    "OffloadCallback",
    "TracInAttributor",
    "default_hook_assignment",
    "hash_batch",
    "hash_sample",
]


def __getattr__(name: str) -> object:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
