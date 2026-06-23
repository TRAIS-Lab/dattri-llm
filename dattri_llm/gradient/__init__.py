"""Gradient collection utilities for training data attribution."""

from dattri_llm.gradient.callbacks import (
    DataSelectionCallback,
    HookManagerCallback,
    OffloadCallback,
)
from dattri_llm.gradient.hooks import (
    REGISTER_ALL,
    HookManager,
    HookManagerConfig,
    register_linear_io_hooks,
    register_linear_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient.utils import hash_sample
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient import ops
from dattri_llm.gradient.ops import (
    KFACAccumulator,
    FIMAccumulator,
    canonical_class_name,
    materialize,
    dot,
    pairwise_dot,
    grad_norm_sq,
    kfac,
    fim,
)

__all__ = [
    "DataSelectionCallback",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "REGISTER_ALL",
    "OffloadCallback",
    "register_linear_io_hooks",
    "register_linear_param_hooks",
    "register_param_grad_hooks",
    "remove_hooks",
    "GradientRecord",
    "hash_sample",
    "GradientFileManager",
    "ops",
    "KFACAccumulator",
    "FIMAccumulator",
    "canonical_class_name",
    "materialize",
    "dot",
    "pairwise_dot",
    "grad_norm_sq",
    "kfac",
    "fim",
]
