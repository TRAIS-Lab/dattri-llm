"""Compute-dtype policy for the gradient operations.

Captured factors arrive in whatever dtype the training loop produced: pure
``bfloat16`` when the model itself is bf16, and a **mix** under the usual
mixed-precision recipe (fp32 master weights + ``torch.autocast``), where the
output gradient is bf16 while activations coming off an fp32 normalization stay
fp32.  The capture layer is deliberately transparent about this -- the hooks
store ``inp[0]`` and ``grad_output[0]`` unchanged.

The ops then have to choose what to compute in, and the choice is worth real
time: upcasting bf16 factors to fp32 gives up the tensor cores, which measured
2.4x on a cross-gram and 7.4x on a materialization at Llama-3.2-1B's layer
shapes.  This module makes that choice explicit and global instead of hardcoded
per kernel.

``"auto"`` (the default) computes in the **promoted dtype of the operands**: it
never widens beyond what it was given, so bf16 in stays bf16, while a genuinely
mixed fp32/bf16 pair still resolves to fp32 rather than producing an ill-typed
matmul.  Passing an explicit dtype forces it everywhere instead::

    from dattri_llm.gradient import ops

    ops.set_compute_dtype(torch.float32)          # process-wide
    with ops.compute_dtype("float32"):            # or scoped
        scores = attributor.attribute(...)

Reductions whose accuracy does not survive low precision -- covariance
accumulation, eigendecomposition, matrix inverse -- pin fp32 regardless via
``minimum=torch.float32``; they run once per fit rather than once per step, so
the precision costs nothing that matters.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

_POLICY: str | torch.dtype = "auto"

_NAMED = {
    "auto": "auto",
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _as_policy(policy: str | torch.dtype) -> str | torch.dtype:
    if isinstance(policy, torch.dtype):
        return policy
    if isinstance(policy, str) and policy in _NAMED:
        return _NAMED[policy]
    raise ValueError(
        f"compute dtype policy must be 'auto', a torch.dtype, or one of "
        f"{sorted(k for k in _NAMED if k != 'auto')}; got {policy!r}",
    )


def set_compute_dtype(policy: str | torch.dtype) -> str | torch.dtype:
    """Set the process-wide compute dtype; returns the previous policy."""
    global _POLICY  # noqa: PLW0603
    previous, _POLICY = _POLICY, _as_policy(policy)
    return previous


def get_compute_dtype() -> str | torch.dtype:
    """The active policy -- ``"auto"`` or an explicit :class:`torch.dtype`."""
    return _POLICY


@contextmanager
def compute_dtype(policy: str | torch.dtype) -> Iterator[None]:
    """Scope a compute-dtype policy to a block."""
    previous = set_compute_dtype(policy)
    try:
        yield
    finally:
        set_compute_dtype(previous)


def resolve(
    *tensors: torch.Tensor | None,
    minimum: torch.dtype | None = None,
) -> torch.dtype:
    """The dtype these operands should be computed in.

    Under ``"auto"`` this is the promotion of the floating operands' dtypes --
    the narrowest type that holds all of them, so nothing is widened
    gratuitously and nothing is left mixed.  Integer operands (embedding token
    ids) are ignored.  *minimum* raises the floor for kernels that need it.
    """
    if _POLICY != "auto":
        dtype = _POLICY
    else:
        dtype = None
        for t in tensors:
            if t is None or not t.is_floating_point():
                continue
            dtype = t.dtype if dtype is None else torch.promote_types(dtype, t.dtype)
        if dtype is None:
            dtype = torch.float32
    if minimum is not None:
        dtype = torch.promote_types(dtype, minimum)
    return dtype


def align(
    *tensors: torch.Tensor | None,
    minimum: torch.dtype | None = None,
) -> tuple[torch.Tensor | None, ...]:
    """Cast every floating operand to :func:`resolve`'s dtype.

    Non-floating tensors (and ``None``) pass through untouched, so embedding
    token-id factors survive.  Tensors already in the target dtype are returned
    as-is, so the common case costs nothing.
    """
    dtype = resolve(*tensors, minimum=minimum)
    return tuple(
        t
        if (t is None or not t.is_floating_point() or t.dtype == dtype)
        else t.to(dtype)
        for t in tensors
    )


def as_float(
    *tensors: torch.Tensor | None,
    minimum: torch.dtype | None = None,
) -> tuple[torch.Tensor | None, ...]:
    """Like :func:`align`, but integer operands are converted too.

    :func:`align` deliberately leaves non-floating tensors alone so an
    embedding's token-id factor survives the ghost path intact.  A few kernels
    genuinely require a float -- the random projection, which multiplies its
    input by a Gaussian/Rademacher matrix, and the one-hot expansion feeding it
    -- and this is the explicit way to say so.  With only integer operands the
    resolved dtype falls back to fp32.
    """
    dtype = resolve(*tensors, minimum=minimum)
    return tuple(t if (t is None or t.dtype == dtype) else t.to(dtype) for t in tensors)
