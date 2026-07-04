"""Gradient collection utilities for training data attribution."""

from dattri_llm.gradient.callbacks import (
    CaptureCallback,
    DataSelectionCallback,
    HookManagerCallback,
    OffloadCallback,
)
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
from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient import ops
from dattri_llm.gradient.ops import (
    FisherAccumulator,
    KroneckerAccumulator,
    LayerFisherAccumulator,
    LayerKroneckerAccumulator,
    canonical_class_name,
    materialize,
    dot,
    pairwise_dot,
    grad_norm_sq,
    kfac,
    fim,
)

__all__ = [
    "CaptureCallback",
    "DataSelectionCallback",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "REGISTER_ALL",
    "default_hook_assignment",
    "OffloadCallback",
    "register_linear_io_hooks",
    "register_linear_param_hooks",
    "register_param_grad_hooks",
    "remove_hooks",
    "GradientRecord",
    "ops",
    "FisherAccumulator",
    "KroneckerAccumulator",
    "LayerFisherAccumulator",
    "LayerKroneckerAccumulator",
    "canonical_class_name",
    "materialize",
    "dot",
    "pairwise_dot",
    "grad_norm_sq",
    "kfac",
    "fim",
]
