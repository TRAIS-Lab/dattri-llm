"""Guarded ``torch.distributed`` helpers.

Every function is safe to call whether or not a process group is initialised
(and even when the distributed backend is unavailable), so call sites need no
``dist.is_available() and dist.is_initialized()`` boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


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


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum *tensor* over every rank in place and return it.

    A no-op outside a distributed context.  The reduction runs on the
    process group's device (NCCL reduces CUDA tensors only; gloo works on
    CPU), so a tensor living elsewhere is moved there and back.

    Args:
        tensor: The per-rank partial sum.

    Returns:
        *tensor*, now holding the sum across ranks.
    """
    if dist_world_size() == 1:
        return tensor
    import torch
    import torch.distributed as dist

    if "nccl" in str(dist.get_backend()).lower():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    buf = tensor.to(device)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    if buf is not tensor:
        tensor.copy_(buf)
    return tensor
