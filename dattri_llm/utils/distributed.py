"""Guarded ``torch.distributed`` helpers.

Every function is safe to call whether or not a process group is initialised
(and even when the distributed backend is unavailable), so call sites need no
``dist.is_available() and dist.is_initialized()`` boilerplate.
"""

from __future__ import annotations


def is_dist_initialized() -> bool:
    """Return ``True`` when a ``torch.distributed`` process group is active."""
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:  # noqa: BLE001 - guarded probe; backend may raise anything
        return False


def dist_rank() -> int | None:
    """Return the current distributed rank, or ``None`` outside a distributed
    context.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001, S110 - guarded probe; backend may raise anything
        pass
    return None


def dist_world_size() -> int:
    """Return the process-group size, or ``1`` outside a distributed context."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:  # noqa: BLE001, S110 - guarded probe; backend may raise anything
        pass
    return 1
