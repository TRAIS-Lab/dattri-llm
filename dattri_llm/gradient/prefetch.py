"""Device prefetching for gradient block streams.

Overlaps host->device copies with compute via classic double buffering: while
block *k* is consumed on the compute stream, block *k+1*'s copy is in flight
on a dedicated copy stream (pinned memory + ``non_blocking=True``).  Hand-off
is safe without a host sync: the consumer's stream ``wait_event``'s the copy,
yielded tensors are ``record_stream``-ed, and the pinned source block stays
referenced until its event is consumed.

On a non-CUDA target or build, the iterator degrades to a synchronous
``block.to(device)`` passthrough, so callers can wrap sources unconditionally.

Memory note: ``depth`` extra blocks are device-resident beyond the one being
consumed; size the scoring batch accordingly.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, TypeVar

import torch

from dattri_llm.gradient.gradient import Factorized, Gradient

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

M = TypeVar("M")
M2 = TypeVar("M2")

# A prefetchable item: only the middle Gradient is moved; the flanking
# metadata (canonically ``step`` and ``hashes``) passes through untouched.
Block = tuple[M, Gradient, M2]

# One copy stream per CUDA device, shared by all prefetchers targeting it.
_copy_streams: dict[int, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    stream = _copy_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _copy_streams[index] = stream
    return stream


def _payload_tensors(gradient: Gradient) -> Iterator[torch.Tensor]:
    """Yield every raw tensor held by *gradient* (both factors of a
    :class:`~dattri_llm.gradient.gradient.Factorized` layer).
    """
    for value in gradient.data.values():
        if isinstance(value, Factorized):
            yield value.activation
            yield value.pre_activation_grad
        else:
            yield value


def prefetch_to_device(
    blocks: Iterable[Block],
    device: torch.device | str,
    *,
    depth: int = 1,
    pin: bool = True,
) -> Iterator[Block]:
    """Yield *blocks* with their :class:`Gradient` payloads device-resident,
    overlapping the host->device copies with the consumer's compute.

    Args:
        blocks: Iterable of ``(meta, Gradient, meta)`` triples, e.g. a
            ``GradientSource``'s ``(step, Gradient, hashes)`` stream.
        device: Target device.  Overlap needs CUDA; anything else falls back
            to a synchronous ``block.to(device)`` with identical semantics.
        depth: Blocks kept in flight ahead of the consumer (each is
            device-resident memory).  ``1`` = double buffering; ``0``
            disables prefetching.
        pin: Pin unpinned CPU payloads before the async copy (unpinned
            sources would serialize the transfer).  A no-op when the
            upstream loader already pins.

    Yields:
        The input triples with each ``Gradient`` on *device*, order
        preserved; already-resident blocks pass through unchanged.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}.")

    device = torch.device(device)
    if depth == 0 or device.type != "cuda" or not torch.cuda.is_available():
        # Nothing to overlap on: plain synchronous move, same semantics.
        for meta, gradient, meta2 in blocks:
            yield meta, gradient.to(device), meta2
        return

    stream = _copy_stream(device)
    # (meta, device_gradient, meta2, copy_event | None, host_ref | None)
    pending: deque[tuple] = deque()
    it = iter(blocks)

    def stage(item: Block) -> tuple:
        meta, gradient, meta2 = item
        if gradient.device == device:
            # Already resident: nothing to copy, nothing to synchronize.
            return (meta, gradient, meta2, None, None)
        host_g = gradient
        if pin and host_g.device.type == "cpu":
            host_g = host_g.pin_memory()
        with torch.cuda.stream(stream):
            dev_g = host_g.to(device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        # host_g is kept in the entry so its memory outlives the in-flight DMA.
        return (meta, dev_g, meta2, event, host_g)

    def finalize(entry: tuple) -> Block:
        meta, dev_g, meta2, event, _host_ref = entry
        if event is not None:
            current = torch.cuda.current_stream(device)
            current.wait_event(event)
            # Tensors were allocated on the copy stream; record the consumer's
            # stream so the caching allocator doesn't reuse them early.
            for t in _payload_tensors(dev_g):
                t.record_stream(current)
        return meta, dev_g, meta2

    exhausted = False
    while True:
        while not exhausted and len(pending) <= depth:
            try:
                pending.append(stage(next(it)))
            except StopIteration:
                exhausted = True
        if not pending:
            return
        yield finalize(pending.popleft())
