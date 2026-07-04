"""Random projection -- TRAK-style (materialized) and LoGRA-style (factorized)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from dattri_llm.gradient.ops.materialize import _materialize
from dattri_llm.gradient.ops.preprocess import _preprocess_factorized, _to_3d
from dattri_llm.gradient.ops.types import is_embedding, is_norm


def _apply_projector(
    projector: Callable,
    x: torch.Tensor,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Random-project the last axis of *x* from ``D`` to ``proj_dim``.

    *projector* follows dattri's ``random_project`` protocol:
    ``projector(feature, batch_size, proj_dim=..., proj_seed=..., **kw)`` returns
    a callable mapping a ``(N, D)`` feature to ``(N, proj_dim)``.  Any leading
    axes of *x* (the batch, plus the token axis when projecting a factor) are
    folded into ``N`` and restored afterward.  ``device`` defaults to *x*'s.
    """
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).float()                  # (N, D)
    proj_kwargs.setdefault("device", flat.device)
    out = projector(
        flat, flat.shape[0], proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs
    )(flat)                                                     # (N, proj_dim)
    return out.reshape(*lead, proj_dim)


def _project_materialized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """TRAK-style: materialize the per-sample weight gradient, then project it.

    Returns a dense ``(B, proj_dim)`` tensor -- the full gradient is reduced to a
    single random-projected vector per sample.
    """
    mat = _materialize(a, g, layer_type, module_kwargs, include_bias)   # (B, D)
    return _apply_projector(
        projector, mat, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs
    )


def _project_factorized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LoGRA-style: project the two factorized factors, keeping the structure.

    Each factor is independently projected to width ``proj_dim`` -- the output
    factor with ``proj_seed`` and the input factor with ``proj_seed + 1`` (dattri's
    LoGRA convention) -- so the per-sample gradient stays the outer product of two
    ``(B, T, proj_dim)`` factors.  Returns ``(a_p, g_p)``.

    Only layer types whose gradient *is* an outer product of the factors (linear,
    conv, transposed conv) are supported; norm layers (diagonal gradient) and
    embeddings (integer index factors) must use materialized projection instead.
    """
    if is_norm(layer_type) or is_embedding(layer_type):
        raise ValueError(
            f"factorized projection is undefined for {layer_type!r}: its gradient "
            "is not an outer product of the factors -- use materialized projection"
        )
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    a_f = _to_3d(a.float())   # (B, T, d_in)
    g_f = _to_3d(g.float())   # (B, T, d_out)
    g_p = _apply_projector(projector, g_f, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs)
    a_p = _apply_projector(projector, a_f, proj_dim=proj_dim, proj_seed=proj_seed + 1, **proj_kwargs)
    return a_p, g_p


def project_materialized(
    f, layer_type: str, projector: Callable, *,
    proj_dim: int, include_bias: bool = True, proj_seed: int = 0, **proj_kwargs,
) -> torch.Tensor:
    """:func:`_project_materialized` on a :class:`Factorized` (batch-first-safe).

    Also accepts an already-dense ``(B, D)`` tensor, which is projected directly.
    """
    if isinstance(f, torch.Tensor):
        return _apply_projector(
            projector, f, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs
        )
    bf = f.as_batch_first()
    return _project_materialized(
        bf.activation, bf.pre_activation_grad, layer_type, projector,
        bf.module_kwargs, include_bias, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs,
    )


def project_factorized(
    f, layer_type: str, projector: Callable, *,
    proj_dim: int, include_bias: bool = True, proj_seed: int = 0, **proj_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_project_factorized` on a :class:`Factorized` (batch-first-safe).

    Returns the projected ``(a_p, g_p)`` factor tuple; the caller rewraps it into
    a :class:`Factorized` with ``module_kwargs=None`` (the factors are final).
    """
    bf = f.as_batch_first()
    return _project_factorized(
        bf.activation, bf.pre_activation_grad, layer_type, projector,
        bf.module_kwargs, include_bias, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs,
    )


def project_layer(
    f, layer_type: str, projector: Callable, *, factorize: bool = True, **proj_kwargs,
) -> Tuple[object, bool]:
    """Route one layer to factorized (LoGRA) or materialized (TRAK) projection.

    Returns ``(payload, is_factorized)``: the payload is the ``(a_p, g_p)`` factor
    tuple when factorized (the caller rewraps it into a :class:`Factorized` with
    ``module_kwargs=None``), else a dense ``(B, proj_dim)`` tensor.  This is the
    per-layer routing shared by :meth:`Gradient.project`.
    """
    if factorize and not isinstance(f, torch.Tensor):
        return project_factorized(f, layer_type, projector, **proj_kwargs), True
    return project_materialized(f, layer_type, projector, **proj_kwargs), False
