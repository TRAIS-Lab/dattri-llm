"""Gradient collection utilities for training data attribution."""

from dattri_llm.gradient import ops
from dattri_llm.gradient.async_writer import AsyncGradientWriter
from dattri_llm.gradient.callbacks import (
    CaptureCallback,
    DataSelectionCallback,
    HookManagerCallback,
    OffloadCallback,
)
from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient.hooks import (
    REGISTER_ALL,
    HookManager,
    HookManagerConfig,
    default_hook_assignment,
    register_linear_io_hooks,
    register_linear_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.ops import (
    FisherAccumulator,
    KroneckerAccumulator,
    LayerFisherAccumulator,
    LayerKroneckerAccumulator,
    canonical_class_name,
    dot,
    fim,
    grad_norm_sq,
    kfac,
    materialize,
    pairwise_dot,
)
from dattri_llm.gradient.prefetch import prefetch_to_device
from dattri_llm.gradient.storage_manager import GradientStorageManager

__all__ = [
    "REGISTER_ALL",
    "AsyncGradientWriter",
    "CaptureCallback",
    "DataSelectionCallback",
    "FisherAccumulator",
    "GradientRecord",
    "GradientStorageManager",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "KroneckerAccumulator",
    "LayerFisherAccumulator",
    "LayerKroneckerAccumulator",
    "OffloadCallback",
    "canonical_class_name",
    "default_hook_assignment",
    "dot",
    "fim",
    "grad_norm_sq",
    "kfac",
    "materialize",
    "ops",
    "pairwise_dot",
    "prefetch_to_device",
    "register_linear_io_hooks",
    "register_linear_param_hooks",
    "register_param_grad_hooks",
    "remove_hooks",
]
