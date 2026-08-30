"""Caching budget for the attribution layer.

Attributors speed up scoring by materializing a block's per-sample gradients once
and reusing the dense form across the blocks it is scored against.  In the
projected regime that costs a few MB and is close to free; at full dimension the
same dense form is ~1 GB *per sample*, so caching it silently converts a working
run into an out-of-memory failure.

The rule here is that a cache must never be the reason a run dies: caching is an
optimization, so it is taken only when the dense form demonstrably fits a budget,
and otherwise the caller falls back to the factorized path, which materializes at
most one layer at a time inside the similarity kernel.

The budget is derived from free device memory (a fraction of it, so the cache
cannot crowd out the activations and workspace scoring itself needs), and can be
pinned explicitly with ``DATTRI_CACHE_BUDGET_GB``.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.ops.types import is_embedding, is_norm

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Gradient

# Fraction of *free* device memory a materialization cache may occupy.  Scoring
# still needs room for the model, activations and the GEMM workspace, so the
# cache is deliberately given a minority share.
DEFAULT_CACHE_FRACTION = 0.35


def dense_nbytes(gradient: Gradient) -> int:
    """Conservative byte count for ``gradient.materialize()``.

    Estimated from the factor shapes rather than by materializing anything, so
    the check itself is free.  Per layer the dense weight gradient is
    ``B x K x D`` for the linear/conv family, ``B x d`` for normalization layers
    (whose token axis is contracted), and ``B x vocab x E`` for embeddings.
    """
    from dattri_llm.gradient.gradient import Factorized

    total = 0
    for name, val in gradient.data.items():
        if not isinstance(val, Factorized):
            total += val.numel() * val.element_size()
            continue
        bf = val.as_batch_first()
        a, g = bf.activation, bf.pre_activation_grad
        b = a.shape[0]
        itemsize = g.element_size()
        layer_type = gradient.layer_types.get(name, "nn.Linear")
        if is_norm(layer_type):
            total += b * a.shape[-1] * itemsize
        elif is_embedding(layer_type):
            mk = bf.module_kwargs or {}
            vocab = mk.get("num_embeddings") or int(a.max().item()) + 1
            total += b * vocab * g.shape[-1] * itemsize
        else:
            total += b * a.shape[-1] * g.shape[-1] * itemsize
    return total


def cache_budget_bytes(device: object, fraction: float = DEFAULT_CACHE_FRACTION) -> int:
    """Bytes a materialization cache may use on ``device``.

    ``DATTRI_CACHE_BUDGET_GB`` pins the budget explicitly (``0`` disables
    materialization caching entirely, forcing the factorized path).  Otherwise a
    fraction of currently-free device memory is used; on CPU the budget is
    effectively unbounded because host memory is not the failure mode here.
    """
    override = os.environ.get("DATTRI_CACHE_BUDGET_GB")
    if override is not None:
        return int(float(override) * (1 << 30))
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return 1 << 62
    free, _total = torch.cuda.mem_get_info(dev)
    return int(free * fraction)


def can_cache(gradient: Gradient, device: object, *, reserve: int = 0) -> bool:
    """Whether ``gradient`` may be materialized and cached within the budget.

    ``reserve`` accounts for dense copies the caller already holds (e.g. a cached
    test side) so that the *sum* of live caches respects the budget rather than
    each one individually.
    """
    return dense_nbytes(gradient) + reserve <= cache_budget_bytes(device)


def cache_fits(cache: dict, new_value: torch.Tensor, device: object) -> bool:
    """Whether ``new_value`` may join a per-layer dense ``cache``.

    The K-FAC family caches materialized gradients one layer at a time, so the
    quantity to bound is the *accumulated* cache rather than any single entry.
    """
    held = sum(v.numel() * v.element_size() for v in cache.values())
    incoming = new_value.numel() * new_value.element_size()
    return held + incoming <= cache_budget_bytes(device)


# Fraction of *free* device memory the in-flight scoring blocks may occupy.
# Larger than the cache share: these blocks are the scoring input, not an
# optimization, so they get the majority of the room.
DEFAULT_PREFETCH_FRACTION = 0.5


def payload_nbytes(gradient: Gradient) -> int:
    """Bytes ``gradient`` occupies **as stored**, factorized or dense.

    Unlike :func:`dense_nbytes` this measures the block itself rather than what
    materializing it would cost, so it is the right quantity for deciding how
    many blocks may be resident on the device at once.
    """
    from dattri_llm.gradient.gradient import Factorized

    total = 0
    for value in gradient.data.values():
        if isinstance(value, Factorized):
            for t in (value.activation, value.pre_activation_grad):
                total += t.numel() * t.element_size()
        else:
            total += value.numel() * value.element_size()
    return total


class DenseCacheAccountant:
    """Running total of the dense bytes the attribution caches currently hold.

    ``can_cache`` bounds one materialization against the budget, but a scoring
    pass holds several at once -- every prepared test block plus the train block
    being scored.  Checked one at a time each fits and the sum still does not,
    which is exactly how a full-dimension run dies with a *small* failing
    allocation on an already-full heap.  Registering each dense copy here lets
    the next check see what is already outstanding.

    Entries are released by a :mod:`weakref` finalizer, so a block dropped by the
    caller (``loop_over_test`` re-preparing the test side, a weakly-cached train
    block evicted) stops counting without any explicit bookkeeping.
    """

    def __init__(self) -> None:
        self._bytes = 0

    def held(self) -> int:
        """Dense bytes currently outstanding."""
        return self._bytes

    def register(self, obj: object, nbytes: int) -> None:
        """Count ``nbytes`` against the budget until ``obj`` is collected."""
        self._bytes += nbytes
        weakref.finalize(obj, self._release, nbytes)

    def _release(self, nbytes: int) -> None:
        self._bytes = max(0, self._bytes - nbytes)


# Process-wide accountant shared by the attributors' materialization caches.
DENSE_CACHE = DenseCacheAccountant()


def safe_prefetch_depth(
    block: Gradient,
    device: object,
    requested: int = 1,
) -> int:
    """Prefetch depth that keeps the in-flight blocks inside the budget.

    Prefetching keeps ``depth + 1`` blocks device-resident (the one being scored
    plus those staged ahead).  In the projected regime a block is a few MB and
    the requested depth always fits; at full dimension a single block is a
    sizeable fraction of the card, and double buffering alone can exhaust it.

    Returns ``requested`` unchanged off CUDA or when the blocks are small, and
    ``0`` (synchronous, one block resident) when even double buffering does not
    fit.  Depth affects only speed and memory -- never the scores.
    """
    if requested <= 0:
        return 0
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return requested
    per_block = payload_nbytes(block)
    if per_block <= 0:
        return requested
    free, _total = torch.cuda.mem_get_info(dev)
    affordable = int(free * DEFAULT_PREFETCH_FRACTION) // per_block
    return max(0, min(requested, affordable - 1))
