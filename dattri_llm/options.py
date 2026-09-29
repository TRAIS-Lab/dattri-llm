"""``Literal`` aliases for the library's enumerated string options.

Each alias spells out the values one option accepts, so type checkers and
IDEs can show and check them.  They are annotations only: every option is
still validated at runtime where it is consumed (for example against
:data:`~dattri_llm.gradient.ops.CAPTURE_STYLES`).  The module holds nothing
but these aliases, so importing it has no side effects and
``typing.get_type_hints`` resolves them.  Import them for your own
annotations::

    from dattri_llm.options import AttributionGranularity, CaptureStyle
"""

from __future__ import annotations

from typing import Literal

AttributionGranularity = Literal["instance", "token"]
"""One score row per training sample, or one per training token position."""

LossReduction = Literal["mean", "sum"]
"""How the training loss was reduced over each batch."""

Propagation = Literal["train", "test"]
"""Which side of a trajectory sweep carries the propagation."""

HessianMode = Literal["full", "diagonal"]
"""DVEmb's per-step Hessian (Fisher) approximation."""

CaptureStyle = Literal["factorized", "materialized", "auto"]
"""The representation a captured layer is buffered in (``CAPTURE_STYLES``)."""

RoutingMode = Literal["factorized", "materialized", "auto"]
"""How an inner-product kernel forms its result; ``"auto"`` uses the cost rule."""

DotReduce = Literal["sum", "none"]
"""Sum the layerwise cross-grams over layers, or keep them per layer."""

CacheResidency = Literal["disk", "memory", "tiered"]
"""Where a cache or gradient store keeps its entries (``CACHE_RESIDENCIES``)."""

DiskFormat = Literal["pickle", "memmap"]
"""The on-disk format of a gradient store (``DISK_FORMATS``)."""

RecordingType = Literal["per_sample", "per_batch"]
"""The granularity an ``OffloadCallback`` records at."""

ProjectionStyle = Literal["logra", "dense", "mask"]
"""How a layer is projected (``PROJECTION_STYLES``)."""

ScoreTrajectory = Literal["aware", "agnostic"]
"""One row per training sample, or one per ``(sample, step)`` pair."""

EKFACMode = Literal["exact", "approx"]
"""EK-FAC's eigenvalue estimate (``EKFACAttributor.EKFAC_MODES``)."""
