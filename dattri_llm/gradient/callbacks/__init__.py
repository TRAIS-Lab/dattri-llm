"""Gradient collection callbacks.

This package holds one module per callback; every callback is re-exported
here under ``dattri_llm.gradient.callbacks``:

``HookManagerCallback``
    Base class.  All methods are no-ops; subclass and override only what you
    need.

``CaptureCallback``
    Holds the most recent step's
    :class:`~dattri_llm.gradient.gradient.GradientRecord` in memory.

``OffloadCallback``
    Periodically flushes :class:`~dattri_llm.gradient.gradient.GradientRecord`
    objects to disk via
    :class:`~dattri_llm.gradient.storage_manager.GradientStorageManager`.
    Supports both per-batch and per-sample recording granularity.

``DataSelectionCallback``
    Online data selection: computes per-sample influence scores at the end of
    each step, then removes low-influence samples' contributions from
    ``param.grad`` before ``optimizer.step()``.

    Two scoring modes are available via ``scoring_kwargs["score_mode"]``:

    * ``"ghost"`` (default) -- gram-matrix form, no weight-gradient
      materialization.  Cost O((B*T)^2 * (out + in)) per layer.
    * ``"materialized"`` -- builds the explicit per-sample weight gradient
      and dots it against the batch gradient.  Uses more memory.

    Both modes produce identical scores.

``KroneckerCovarianceCallback``
    Accumulates the per-layer K-FAC covariances ``(A, G)`` during collection.

``OptimizerStateCallback``
    Records an Adam-family optimizer's moments on both sides of every step.

``ParameterSnapshotCallback``
    Stores the parameters every capture step runs from.
"""

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.callbacks.capture_callback import CaptureCallback
from dattri_llm.gradient.callbacks.data_selection_callback import (
    DataSelectionCallback,
)
from dattri_llm.gradient.callbacks.kronecker_covariance_callback import (
    KroneckerCovarianceCallback,
)
from dattri_llm.gradient.callbacks.offload_callback import OffloadCallback
from dattri_llm.gradient.callbacks.optimizer_state_callback import (
    OptimizerStateCallback,
)
from dattri_llm.gradient.callbacks.parameter_snapshot_callback import (
    ParameterSnapshotCallback,
)

__all__ = [
    "CaptureCallback",
    "DataSelectionCallback",
    "HookManagerCallback",
    "KroneckerCovarianceCallback",
    "OffloadCallback",
    "OptimizerStateCallback",
    "ParameterSnapshotCallback",
]
