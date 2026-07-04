"""Gradient collection callbacks.

Three built-in callbacks are provided:

``HookManagerCallback``
    Base class.  All methods are no-ops; subclass and override only what you
    need.

``OffloadCallback``
    Periodically flushes :class:`GradientRecord` objects to disk via
    :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.
    Supports both per-batch and per-sample recording granularity.

``DataSelectionCallback``
    Online data selection: computes per-sample influence scores at the end of
    each step, then removes low-influence samples' contributions from
    ``param.grad`` before ``optimizer.step()``.

    Two scoring modes are available via ``score_mode``:

    * ``"ghost"`` (default) — gram-matrix form, no weight-gradient
      materialisation. Cost O((B*T)^2 * (out + in)) per layer.
    * ``"materialized"`` — builds the explicit per-sample weight gradient
      and dots it against the batch gradient. Easier to verify but uses
      more memory.

    Both modes produce identical scores.

This package holds one module per callback; everything is re-exported here so
existing ``dattri_llm.gradient.callbacks`` imports keep working unchanged.
"""

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.callbacks.capture_callback import CaptureCallback
from dattri_llm.gradient.callbacks.data_selection_callback import (
    _SCORE_MODES,
    _TARGET_MODES,
    _THRESHOLD_MODES,
    DataSelectionCallback,
    _ShardSpec,
)
from dattri_llm.gradient.callbacks.offload_callback import OffloadCallback
