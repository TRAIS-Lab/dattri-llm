"""Gradient dot products: cross/pairwise grams, per-sample dots and norms,
plus the factorized-vs-materialized representation routing heuristic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from dattri_llm.gradient.ops.preprocess import _preprocess_factorized, _to_3d
from dattri_llm.gradient.ops.types import (
    is_conv,
    is_conv_transpose,
    is_embedding,
    is_norm,
)

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Factorized


# ---------------------------------------------------------------------------
# cross_dot / pairwise_dot
# ---------------------------------------------------------------------------

def _cross_gram(
    a1: torch.Tensor, g1: torch.Tensor,
    a2: torch.Tensor, g2: torch.Tensor,
    layer_type: str,
    mode: str = "auto",
) -> torch.Tensor:
    """Cross-gram ``K[i, j] = ⟨∇W1_i, ∇W2_j⟩`` on *already-preprocessed* factors.

    Shared kernel behind :func:`_cross_dot`, :func:`_pairwise_dot`, and
    :func:`_kfac_cross` (which whitens side 1 first).  Inputs must already be in
    the form returned by :func:`_preprocess_factorized`.

    For the linear/conv family the result is computed either **factorized**
    (``Σ_{t,s}`` ghost contraction, the ``(B1,T1,B2,S2)`` kernel) or
    **materialized** (contract each side's tokens into the per-sample weight
    gradient ``(B, K, D)`` then GEMM — no ``S²`` intermediate).  Both are exact
    (a reassociation of the same sum); ``mode="auto"`` picks the cheaper per the
    ``H=DK/(D+K)`` rule (:func:`maybe_use_materialized_gram`).  Embedding/norm
    layers have their own path and ignore ``mode``.
    """
    if is_embedding(layer_type):
        # K[i,j] = sum_t g1_i[t] · G2_sum_j[tok1_i[t]]
        # where G2_sum_j[k] = sum_{s: tok2_j[s]==k} g2_j[s]
        tok1, tok2 = a1, a2          # (B1, T1), (B2, T2) int
        g1_f, g2_f = g1.float(), g2.float()
        B1, T1 = tok1.shape
        B2, T2 = tok2.shape
        E = g1_f.shape[-1]
        vocab = int(max(tok1.max().item(), tok2.max().item())) + 1
        flat1 = tok1.reshape(-1)
        K = torch.zeros(B1, B2, dtype=g1_f.dtype, device=g1_f.device)
        for j in range(B2):
            G2_sum = torch.zeros(vocab, E, dtype=g2_f.dtype, device=g2_f.device)
            G2_sum.scatter_add_(0, tok2[j].unsqueeze(-1).expand(T2, E), g2_f[j])
            gathered = G2_sum[flat1].reshape(B1, T1, E)   # (B1, T1, E)
            K[:, j] = (g1_f * gathered).sum((1, 2))
        return K

    a1_f = _to_3d(a1.float()); g1_f = _to_3d(g1.float())
    a2_f = _to_3d(a2.float()); g2_f = _to_3d(g2.float())

    if is_norm(layer_type):
        grad1 = (a1_f * g1_f).flatten(1)   # (B1, T*d)
        grad2 = (a2_f * g2_f).flatten(1)   # (B2, T*d)
        return grad1 @ grad2.T             # (B1, B2)

    # Linear, Conv, ConvTranspose — route factorized (ghost) vs materialized.
    B1, S, K = a1_f.shape          # K = input width, S = token/patch count
    B2 = a2_f.shape[0]
    D = g1_f.shape[-1]             # output width
    if mode == "auto":
        mode = "materialized" if maybe_use_materialized_gram(B1, B2, S, K, D) else "factorized"
    if mode == "materialized":
        # Contract tokens into per-sample weight grads, then GEMM — no S² tensor.
        M1 = torch.einsum("btk,btd->bkd", a1_f, g1_f).reshape(B1, -1)   # (B1, K·D)
        M2 = torch.einsum("csk,csd->ckd", a2_f, g2_f).reshape(B2, -1)   # (B2, K·D)
        return M1 @ M2.T                                                # (B1, B2)
    K_a = torch.einsum("btk,csk->btcs", a1_f, a2_f)
    K_g = torch.einsum("btd,csd->btcs", g1_f, g2_f)
    return torch.einsum("btcs,btcs->bc", K_a, K_g)   # (B1, B2)


def _cross_dot(
    a1: torch.Tensor, g1: torch.Tensor,
    a2: torch.Tensor, g2: torch.Tensor,
    layer_type: str,
    module_kwargs1: Optional[dict] = None,
    module_kwargs2: Optional[dict] = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return the (B1, B2) cross-gram ``K[i, j] = ⟨∇W1_i, ∇W2_j⟩``.

    Generalises :func:`_pairwise_dot` (which is the self case
    ``_cross_dot(a, g, a, g, ...)``) to two distinct sets of factorized
    gradients — e.g. a training batch against a fixed target gradient.  Each
    side is preprocessed independently via :func:`_preprocess_factorized`.
    ``mode`` (``"auto"``/``"factorized"``/``"materialized"``) selects the cross-gram
    path; see :func:`_cross_gram`.

    For norm layers the per-position (diagonal) convention is used, so the two
    sides must share the same flattened ``T * d`` dimension (equal token/spatial
    count); this holds whenever both gradients come from the same model run at
    the same sequence length.
    """
    a1, g1 = _preprocess_factorized(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = _preprocess_factorized(a2, g2, layer_type, module_kwargs2, include_bias)
    return _cross_gram(a1, g1, a2, g2, layer_type, mode)


def _pairwise_dot(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return (B, B) pairwise dot product matrix of per-sample gradients.

    *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
    Equivalent to the self case of :func:`_cross_dot`.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    return _cross_gram(a, g, a, g, layer_type, mode)


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------

def _dot(
    a1: torch.Tensor, g1: torch.Tensor,
    a2: torch.Tensor, g2: torch.Tensor,
    layer_type: str,
    module_kwargs1: Optional[dict] = None,
    module_kwargs2: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Return (B,) per-sample dot products ⟨∇W1_i, ∇W2_i⟩.

    *module_kwargs1* and *module_kwargs2* are passed to
    :func:`_preprocess_factorized` for the respective tensor pairs when provided.
    """
    a1, g1 = _preprocess_factorized(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = _preprocess_factorized(a2, g2, layer_type, module_kwargs2, include_bias)

    if is_embedding(layer_type):
        B = g1.shape[0]
        T = a1.shape[1]
        E = g1.shape[-1]
        vocab = int(max(a1.max().item(), a2.max().item())) + 1
        g1_f = g1.float()
        g2_f = g2.float()
        result = torch.zeros(B, dtype=g1_f.dtype, device=g1_f.device)
        for i in range(B):
            G2_sum = torch.zeros(vocab, E, dtype=g2_f.dtype, device=g2_f.device)
            G2_sum.scatter_add_(0, a2[i].unsqueeze(-1).expand(T, E), g2_f[i])
            gathered = G2_sum[a1[i]]             # (T, E)
            result[i] = (g1_f[i] * gathered).sum()
        return result

    a1_f = _to_3d(a1.float()); g1_f = _to_3d(g1.float())
    a2_f = _to_3d(a2.float()); g2_f = _to_3d(g2.float())

    if is_norm(layer_type):
        grad1 = (a1_f * g1_f).flatten(1)         # (B, T*d)
        grad2 = (a2_f * g2_f).flatten(1)         # (B, T*d)
        return (grad1 * grad2).sum(-1)            # (B,)

    # Linear, Conv, ConvTranspose
    K_a = torch.einsum("btd,bsd->bts", a1_f, a2_f)   # (B, T, T)
    K_g = torch.einsum("bte,bse->bts", g1_f, g2_f)   # (B, T, T)
    return (K_a * K_g).sum((1, 2))                    # (B,)


# ---------------------------------------------------------------------------
# grad_norm_sq
# ---------------------------------------------------------------------------

def _grad_norm_sq(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return (B,) per-sample squared Frobenius norms of weight gradients.

    *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
    For the linear/conv family the norm is computed either **factorized** (the
    ``S²`` ghost contraction) or **materialized** (token-contract to the per-sample
    weight gradient, then sum of squares); both are exact, and ``mode="auto"``
    picks the cheaper via :func:`maybe_use_materialized_norm`.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)

    if is_embedding(layer_type):
        return _pairwise_dot(a, g, layer_type).diagonal()

    a_f = _to_3d(a.float())   # (B, T, d_in)
    g_f = _to_3d(g.float())   # (B, T, d_out)

    if is_norm(layer_type):
        return (a_f * g_f).square().sum((1, 2))  # (B,)

    # Linear, Conv, ConvTranspose — route factorized (ghost) vs materialized.
    _, S, K = a_f.shape
    D = g_f.shape[-1]
    if mode == "auto":
        mode = "materialized" if maybe_use_materialized_norm(S, K, D) else "factorized"
    if mode == "materialized":
        M = torch.einsum("btk,btd->bkd", a_f, g_f).flatten(1)   # (B, K·D)
        return (M * M).sum(-1)                                   # (B,)
    K_a = torch.einsum("btk,bsk->bts", a_f, a_f)   # (B, T, T)
    K_g = torch.einsum("btd,bsd->bts", g_f, g_f)   # (B, T, T)
    return (K_a * K_g).sum((1, 2))                  # (B,)


def grad_norm_sq(
    f: "Factorized", layer_type: str, include_bias: bool = True, mode: str = "auto"
) -> torch.Tensor:
    """:func:`_grad_norm_sq` on a :class:`Factorized` (batch-first-safe).

    ``mode`` routes factorized vs materialized per :func:`_grad_norm_sq`."""
    bf = f.as_batch_first()
    return _grad_norm_sq(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs,
        include_bias, mode,
    )


def pairwise_dot(
    f: "Factorized", layer_type: str, include_bias: bool = True, mode: str = "auto"
) -> torch.Tensor:
    """:func:`_pairwise_dot` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _pairwise_dot(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs,
        include_bias, mode,
    )


def dot(
    f1: "Factorized", f2: "Factorized", layer_type: str, include_bias: bool = True
) -> torch.Tensor:
    """:func:`_dot` on two :class:`Factorized` (batch-first-safe)."""
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return _dot(
        b1.activation, b1.pre_activation_grad, b2.activation, b2.pre_activation_grad,
        layer_type, b1.module_kwargs, b2.module_kwargs, include_bias,
    )


def cross_dot(
    f1: "Factorized", f2: "Factorized", layer_type: str, include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """:func:`_cross_dot` on two :class:`Factorized` (batch-first-safe).

    ``mode`` (``"auto"``/``"factorized"``/``"materialized"``) routes the cross-gram
    per :func:`_cross_gram`; ``"auto"`` is the cost-optimal choice."""
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return _cross_dot(
        b1.activation, b1.pre_activation_grad, b2.activation, b2.pre_activation_grad,
        layer_type, b1.module_kwargs, b2.module_kwargs, include_bias, mode,
    )


# --------------------------------------------------------------------------- #
# Representation routing heuristic (factorized vs materialized)               #
#                                                                             #
# The per-sample weight gradient G = gᵀa (Dx K, summed over S token/patch     #
# positions) can be dotted/normed either factorized ("ghost") or materialized.#
# Which is cheaper is governed by S relative to H = DK/(D+K); see             #
# docs/gradient_representation_complexity.md.  These predicates are consumed   #
# *here at the bottom* — _cross_gram / _grad_norm_sq route on them — so every  #
# caller (Gradient.similarity, K-FAC's _kfac_cross, …) shares one routed       #
# implementation; ``mode="auto"`` triggers the heuristic, and the explicit     #
# "factorized"/"materialized" modes override it.                              #
# --------------------------------------------------------------------------- #


def effective_dims(f: "Factorized", layer_type: str) -> Tuple[int, int, int, int]:
    """Cheap ``(B, S, K, D)`` for the cost heuristic: batch, token/patch count,
    input width, output width — the *post-preprocess* dims, read straight from the
    raw factor shapes (no im2col / materialization).  Bias's ``+1`` on ``K`` is
    ignored (it is a heuristic)."""
    bf = f.as_batch_first()
    a, g = bf.activation, bf.pre_activation_grad
    mk = bf.module_kwargs or {}
    if is_conv(layer_type):
        # a=(B,C_in,*sp_in), g=(B,C_out,*sp_out): S = output positions,
        # K = C_in·∏kernel, D = C_out
        kprod = math.prod(mk["kernel_size"]) if "kernel_size" in mk else 1
        return a.shape[0], math.prod(g.shape[2:]), a.shape[1] * kprod, g.shape[1]
    if is_conv_transpose(layer_type):
        # roles reversed: a flattened over spatial (K=C_in), g unfolded (D=C_out·∏K)
        kprod = math.prod(mk["kernel_size"]) if "kernel_size" in mk else 1
        return a.shape[0], math.prod(a.shape[2:]), a.shape[1], g.shape[1] * kprod
    # linear-family (and norm layers): a=(B, *T, K), g=(B, *T, D)
    return a.shape[0], math.prod(a.shape[1:-1]), a.shape[-1], g.shape[-1]


def maybe_use_materialized_gram(
    B1: int, B2: int, S: int, K: int, D: int, kappa: float = 1.0
) -> bool:
    """``True`` when materialize-then-GEMM is the cheaper way to form the
    ``(B1, B2)`` cross-gram (§3.2):

        cost_F = B1·B2·S²·(D+K)            cost_M = (B1+B2)·S·D·K + B1·B2·D·K

    Materialize iff ``κ·cost_F ≥ cost_M`` (``κ=1`` is the pure-flop rule).
    """
    cost_f = B1 * B2 * S * S * (D + K)
    cost_m = (B1 + B2) * S * D * K + B1 * B2 * D * K
    return kappa * cost_f >= cost_m


def maybe_use_materialized_norm(S: int, K: int, D: int) -> bool:
    """``True`` when materializing is cheaper for per-sample norms (§3.1).  Here
    ``cost_F = S²(D+K)`` and ``cost_M = S·D·K`` (per sample, batch cancels), so
    materialize iff ``S·(D+K) ≥ DK``, i.e. ``S ≥ H = DK/(D+K)``."""
    return S * (D + K) >= D * K
