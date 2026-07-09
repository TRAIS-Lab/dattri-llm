"""Guarded ``torch.autograd`` engine helpers.

Both functions schedule work at the *end of the in-flight backward pass* --
the point at which every ``param.grad`` is fully written, including writes the
parallelism wrappers perform late (FSDP's sharded write-back, DDP's averaged
bucket write-back).  They are safe to call on builds where the engine does not
expose ``queue_callback``: they return ``False`` instead of raising, and the
caller decides on a fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def queue_backward_end_callback(fn: Callable[[], None]) -> bool:
    """Schedule ``fn`` to run once the in-flight backward pass fully completes.

    Must be called from *within* a backward pass (e.g. a module full-backward
    hook).  The callback fires after the autograd engine has finished -- the
    point at which FSDP has written every (sharded) ``param.grad`` back to its
    original parameter, so reading ``param.grad`` inside ``fn`` is safe.

    **Ordering caveat**: the engine runs queued callbacks in FIFO order, but
    the order in which *other* actors queued theirs during the same backward
    is not observable -- DDP's reducer, for example, queues its averaged-
    gradient write-back from its own autograd hook, and ``fn`` may run before
    it.  When ``fn`` must run after everything queued during the backward, use
    :func:`queue_after_backward_finalization` instead.

    Returns ``True`` if the callback was successfully queued, ``False`` if the
    autograd engine does not expose ``queue_callback`` on this build (in which
    case the caller should fall back to its existing behaviour).
    """
    try:
        torch.autograd.Variable._execution_engine.queue_callback(fn)
    except Exception:  # noqa: BLE001 - guarded probe of autograd internals
        return False
    return True


def queue_after_backward_finalization(fn: Callable[[], None]) -> bool:
    """Schedule ``fn`` to run after **everything** queued during the backward.

    A single :func:`queue_backward_end_callback` gives no ordering guarantee
    relative to callbacks other actors queued during the same backward (see
    its ordering caveat).  This variant queues a callback whose only job is to
    queue ``fn``: the engine's finalisation loop re-reads the callback list as
    it grows, so a callback appended *while the loop is executing* runs after
    every callback that was queued during the backward -- e.g. after DDP's
    averaged-gradient write-back, whichever order the first-level callbacks
    landed in.  This is what
    :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback` uses to edit
    ``param.grad`` under DDP without the reducer overwriting the edit.

    Must be called from within a backward pass.  Returns ``True`` if queued;
    ``False`` when the engine exposes no queue (caller falls back).
    """

    def _requeue() -> None:
        if not queue_backward_end_callback(fn):
            fn()

    return queue_backward_end_callback(_requeue)
