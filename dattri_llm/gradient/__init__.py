"""Gradient collection utilities for training data attribution."""

from dattri_llm.gradient.callbacks import (
    DataSelectionCallback,
    HookManagerCallback,
    OffloadCallback,
)
from dattri_llm.gradient.hooks import (
    HookManager,
    HookManagerConfig,
    register_mlp_hooks,
    register_mlp_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.gradient import GradientRecord, hash_sample
from dattri_llm.gradient.file_manager import GradientFileManager

__all__ = [
    "DataSelectionCallback",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "OffloadCallback",
    "register_mlp_hooks",
    "register_mlp_param_hooks",
    "register_param_grad_hooks",
    "remove_hooks",
    "GradientRecord",
    "hash_sample",
    "GradientFileManager",
]
