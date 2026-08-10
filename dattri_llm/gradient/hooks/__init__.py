"""Core PyTorch hook utilities, HookManager, and related classes.

This package re-exports the full surface of its submodules so existing
``dattri_llm.gradient.hooks`` imports keep working unchanged:

* :mod:`~dattri_llm.gradient.hooks.hooks` -- low-level hook registration
  (the ``linear_io`` and ``param_grad`` hook families).
* :mod:`~dattri_llm.gradient.hooks.config` -- :class:`HookManagerConfig`,
  selectors, and hook-assignment resolution.
* :mod:`~dattri_llm.gradient.hooks.manager` -- :class:`HookManager`.
"""

from dattri_llm.gradient.callbacks import HookManagerCallback, OffloadCallback
from dattri_llm.gradient.hooks.config import (
    INVASIVE_LINEAR_IO,
    LINEAR_IO,
    PARAM_GRAD,
    REGISTER_ALL,
    HookManagerConfig,
    Selector,
    _derive_scalar_loss,
    _invoke_model,
    _resolve_projector,
    _selector_matches,
    _warn_zero_layers,
    default_hook_assignment,
    resolve_hook_assignments,
)
from dattri_llm.gradient.hooks.hooks import (
    HF_Conv1D,
    LayerBuffer,
    ParamGradBuffer,
    _capture_projected,
    _has_trainable_params,
    _is_invasive_capable,
    _is_linear_io_capable,
    _make_layer_buffer,
    install_invasive_forward,
    register_linear_io_hooks,
    register_linear_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.hooks.manager import HookManager

__all__ = [
    "INVASIVE_LINEAR_IO",
    "LINEAR_IO",
    "PARAM_GRAD",
    "REGISTER_ALL",
    "HF_Conv1D",
    "HookManager",
    "HookManagerCallback",
    "HookManagerConfig",
    "LayerBuffer",
    "OffloadCallback",
    "ParamGradBuffer",
    "Selector",
    "_capture_projected",
    "_derive_scalar_loss",
    "_has_trainable_params",
    "_invoke_model",
    "_is_invasive_capable",
    "_is_linear_io_capable",
    "_make_layer_buffer",
    "_resolve_projector",
    "_selector_matches",
    "_warn_zero_layers",
    "default_hook_assignment",
    "install_invasive_forward",
    "register_linear_io_hooks",
    "register_linear_param_hooks",
    "register_param_grad_hooks",
    "remove_hooks",
    "resolve_hook_assignments",
]
