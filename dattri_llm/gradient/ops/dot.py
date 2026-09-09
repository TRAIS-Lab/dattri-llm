"""Gradient dot products: cross/pairwise grams, per-sample dots and norms,
plus the factorized-vs-materialized representation routing heuristic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.ops import dtypes
from dattri_llm.gradient.ops.materialize import materialize
from dattri_llm.gradient.ops.preprocess import preprocess_factors, to_3d
from dattri_llm.gradient.ops.types import (
    is_conv,
    is_conv_transpose,
    is_embedding,
    is_norm,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dattri_llm.gradient.gradient import Factorized, Gradient
    from dattri_llm.utils.cache import TensorCache


# ---------------------------------------------------------------------------
# cross_dot / pairwise_dot
# ---------------------------------------------------------------------------


def cross_gram(
    a1: torch.Tensor,
    g1: torch.Tensor,
    a2: torch.Tensor,
    g2: torch.Tensor,
    layer_type: str,
    mode: str = "auto",
) -> torch.Tensor:
    """Cross-gram ``K[i, j] = <dW1_i, dW2_j>`` on *already-preprocessed* factors.

    Shared kernel behind :func:`cross_dot_factors`, :func:`pairwise_dot_factors`, and
    :func:`kfac_cross_factors` (which whitens side 1 first).  Inputs must already be in
    the form returned by :func:`preprocess_factors`.

    For the linear/conv family the result is computed either **factorized**
    (``sum_{t,s}`` ghost contraction, the ``(B1,T1,B2,S2)`` kernel) or
    **materialized** (contract each side's tokens into the per-sample weight
    gradient ``(B, K, D)`` then GEMM -- no ``S^2`` intermediate).  Both are exact
    (a reassociation of the same sum); ``mode="auto"`` picks the cheaper per the
    ``H=DK/(D+K)`` rule (:func:`maybe_use_materialized_gram`).  Embedding/norm
    layers have their own path and ignore ``mode``.
    """
    if is_embedding(layer_type):
        # K[i,j] = sum_t g1_i[t] * G2_sum_j[tok1_i[t]]
        # where G2_sum_j[k] = sum_{s: tok2_j[s]==k} g2_j[s]
        tok1, tok2 = a1, a2  # (B1, T1), (B2, T2) int
        g1_f, g2_f = dtypes.align(g1, g2)
        B1, T1 = tok1.shape
        B2, T2 = tok2.shape
        E = g1_f.shape[-1]
        vocab = int(max(tok1.max().item(), tok2.max().item())) + 1
        flat1 = tok1.reshape(-1)
        K = torch.zeros(B1, B2, dtype=g1_f.dtype, device=g1_f.device)
        for j in range(B2):
            G2_sum = torch.zeros(vocab, E, dtype=g2_f.dtype, device=g2_f.device)
            G2_sum.scatter_add_(0, tok2[j].unsqueeze(-1).expand(T2, E), g2_f[j])
            gathered = G2_sum[flat1].reshape(B1, T1, E)  # (B1, T1, E)
            K[:, j] = (g1_f * gathered).sum((1, 2))
        return K

    a1, g1, a2, g2 = dtypes.align(a1, g1, a2, g2)
    a1_f, g1_f = to_3d(a1), to_3d(g1)
    a2_f, g2_f = to_3d(a2), to_3d(g2)

    if is_norm(layer_type):
        # dW_i = sum_t x_hat_it * g_it: contract positions first so the dot is
        # the true weight-gradient inner product (cross-position terms
        # included) and the two sides may have different position counts.
        grad1 = (a1_f * g1_f).sum(1)  # (B1, d)
        grad2 = (a2_f * g2_f).sum(1)  # (B2, d)
        return grad1 @ grad2.T  # (B1, B2)

    # Linear, Conv, ConvTranspose -- route factorized (ghost) vs materialized.
    B1, S, K = a1_f.shape  # K = input width, S = token/patch count
    B2 = a2_f.shape[0]
    D = g1_f.shape[-1]  # output width
    if mode == "auto":
        mode = (
            "materialized"
            if maybe_use_materialized_gram(B1, B2, S, K, D)
            else "factorized"
        )
    if mode == "materialized":
        # Contract tokens into per-sample weight grads, then GEMM -- no S^2 tensor.
        M1 = torch.einsum("btk,btd->bkd", a1_f, g1_f).reshape(B1, -1)  # (B1, K*D)
        M2 = torch.einsum("csk,csd->ckd", a2_f, g2_f).reshape(B2, -1)  # (B2, K*D)
        return M1 @ M2.T  # (B1, B2)
    K_a = torch.einsum("btk,csk->btcs", a1_f, a2_f)
    K_g = torch.einsum("btd,csd->btcs", g1_f, g2_f)
    return torch.einsum("btcs,btcs->bc", K_a, K_g)  # (B1, B2)


def cross_dot_factors(
    a1: torch.Tensor,
    g1: torch.Tensor,
    a2: torch.Tensor,
    g2: torch.Tensor,
    layer_type: str,
    module_kwargs1: dict | None = None,
    module_kwargs2: dict | None = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return the (B1, B2) cross-gram ``K[i, j] = <dW1_i, dW2_j>``.

    Generalises :func:`pairwise_dot_factors` (which is the self case
    ``cross_dot_factors(a, g, a, g, ...)``) to two distinct sets of factorized
    gradients -- e.g. a training batch against a fixed target gradient.  Each
    side is preprocessed independently via :func:`preprocess_factors`.
    ``mode`` (``"auto"``/``"factorized"``/``"materialized"``) selects the cross-gram
    path; see :func:`cross_gram`.

    For norm layers the per-position (diagonal) convention is used, so the two
    sides must share the same flattened ``T * d`` dimension (equal token/spatial
    count); this holds whenever both gradients come from the same model run at
    the same sequence length.
    """
    a1, g1 = preprocess_factors(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = preprocess_factors(a2, g2, layer_type, module_kwargs2, include_bias)
    return cross_gram(a1, g1, a2, g2, layer_type, mode)


def pairwise_dot_factors(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return (B, B) pairwise dot product matrix of per-sample gradients.

    *module_kwargs* is passed to :func:`preprocess_factors` when provided.
    Equivalent to the self case of :func:`cross_dot_factors`.
    """
    a, g = preprocess_factors(a, g, layer_type, module_kwargs, include_bias)
    return cross_gram(a, g, a, g, layer_type, mode)


# ---------------------------------------------------------------------------
# per-token-position cross-gram
# ---------------------------------------------------------------------------


def cross_gram_per_token(
    a1: torch.Tensor,
    g1: torch.Tensor,
    a2: torch.Tensor,
    g2: torch.Tensor,
    layer_type: str,
) -> torch.Tensor:
    """Per-token-position cross-gram ``K[i, t, j]`` on *preprocessed* factors:
    the contribution of side-1 sample ``i``'s **token position ``t``** to the
    weight-gradient inner product ``<dW1_i, dW2_j>``.

    This keeps side 1's token axis and sums out side 2's; summing the result over
    ``t`` recovers the ordinary ``(B1, B2)`` cross-gram exactly (``sum_t K[i,t,j] =
    <dW1_i, dW2_j>``), so a token heatmap is a *decomposition* of the score, not a
    different quantity.  Only the factorized (ghost) representation admits this --
    the materialized path has already contracted the token axis away.  Returns a
    ``(B1, T1, B2)`` tensor.
    """
    if is_embedding(layer_type):
        tok1, tok2 = a1, a2  # (B1, T1), (B2, T2) int
        g1_f, g2_f = dtypes.align(g1, g2)
        B1, T1 = tok1.shape
        B2, T2 = tok2.shape
        E = g1_f.shape[-1]
        vocab = int(max(tok1.max().item(), tok2.max().item())) + 1
        flat1 = tok1.reshape(-1)
        out = torch.zeros(B1, T1, B2, dtype=g1_f.dtype, device=g1_f.device)
        for j in range(B2):
            g2_sum = torch.zeros(vocab, E, dtype=g2_f.dtype, device=g2_f.device)
            g2_sum.scatter_add_(0, tok2[j].unsqueeze(-1).expand(T2, E), g2_f[j])
            gathered = g2_sum[flat1].reshape(B1, T1, E)  # (B1, T1, E)
            out[:, :, j] = (g1_f * gathered).sum(-1)
        return out

    a1, g1, a2, g2 = dtypes.align(a1, g1, a2, g2)
    a1_f, g1_f = to_3d(a1), to_3d(g1)
    a2_f, g2_f = to_3d(a2), to_3d(g2)

    if is_norm(layer_type):
        # dW2_j = sum_s x_hat2_js * g2_js (d-vector, diagonal). Query token t's
        # contribution is (x_hat1_it * g1_it) dotted into it.
        grad2 = (a2_f * g2_f).sum(1)  # (B2, d)
        contrib = a1_f * g1_f  # (B1, T1, d)
        return torch.einsum("btd,cd->btc", contrib, grad2)

    # Linear/Conv family: materialize side 2's per-sample weight grad (K,D) once,
    # then contract each side-1 token's rank-1 (a1_it (x) g1_it) against it.
    m2 = torch.einsum("csk,csd->ckd", a2_f, g2_f)  # (B2, K, D)
    return torch.einsum("btk,btd,ckd->btc", a1_f, g1_f, m2)  # (B1, T1, B2)


def cross_dot_per_token(
    f1: Factorized,
    f2: Factorized,
    layer_type: str,
    include_bias: bool = True,
) -> torch.Tensor:
    """Per-token-position cross-gram on two :class:`Factorized` (batch-first-safe).

    Returns ``(B1, T1, B2)`` -- side-1 token positions preserved; see
    :func:`cross_gram_per_token`.
    """
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    a1, g1 = preprocess_factors(
        b1.activation,
        b1.pre_activation_grad,
        layer_type,
        b1.module_kwargs,
        include_bias,
    )
    a2, g2 = preprocess_factors(
        b2.activation,
        b2.pre_activation_grad,
        layer_type,
        b2.module_kwargs,
        include_bias,
    )
    return cross_gram_per_token(a1, g1, a2, g2, layer_type)


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------


def dot_factors(
    a1: torch.Tensor,
    g1: torch.Tensor,
    a2: torch.Tensor,
    g2: torch.Tensor,
    layer_type: str,
    module_kwargs1: dict | None = None,
    module_kwargs2: dict | None = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Return (B,) per-sample dot products <dW1_i, dW2_i>.

    *module_kwargs1* and *module_kwargs2* are passed to
    :func:`preprocess_factors` for the respective tensor pairs when provided.
    """
    a1, g1 = preprocess_factors(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = preprocess_factors(a2, g2, layer_type, module_kwargs2, include_bias)

    if is_embedding(layer_type):
        B = g1.shape[0]
        T2 = a2.shape[1]  # the two sides may have different token counts
        E = g1.shape[-1]
        vocab = int(max(a1.max().item(), a2.max().item())) + 1
        g1_f, g2_f = dtypes.align(g1, g2)
        result = torch.zeros(B, dtype=g1_f.dtype, device=g1_f.device)
        for i in range(B):
            G2_sum = torch.zeros(vocab, E, dtype=g2_f.dtype, device=g2_f.device)
            G2_sum.scatter_add_(0, a2[i].unsqueeze(-1).expand(T2, E), g2_f[i])
            gathered = G2_sum[a1[i]]  # (T1, E)
            result[i] = (g1_f[i] * gathered).sum()
        return result

    a1, g1, a2, g2 = dtypes.align(a1, g1, a2, g2)
    a1_f, g1_f = to_3d(a1), to_3d(g1)
    a2_f, g2_f = to_3d(a2), to_3d(g2)

    if is_norm(layer_type):
        # Contract positions into per-sample weight grads before the dot
        # (see the matching branch in cross_gram).
        grad1 = (a1_f * g1_f).sum(1)  # (B, d)
        grad2 = (a2_f * g2_f).sum(1)  # (B, d)
        return (grad1 * grad2).sum(-1)  # (B,)

    # Linear, Conv, ConvTranspose
    K_a = torch.einsum("btd,bsd->bts", a1_f, a2_f)  # (B, T, T)
    K_g = torch.einsum("bte,bse->bts", g1_f, g2_f)  # (B, T, T)
    return (K_a * K_g).sum((1, 2))  # (B,)


# ---------------------------------------------------------------------------
# grad_norm_sq
# ---------------------------------------------------------------------------


def grad_norm_sq_factors(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Return (B,) per-sample squared Frobenius norms of weight gradients.

    *module_kwargs* is passed to :func:`preprocess_factors` when provided.
    For the linear/conv family the norm is computed either **factorized** (the
    ``S^2`` ghost contraction) or **materialized** (token-contract to the per-sample
    weight gradient, then sum of squares); both are exact, and ``mode="auto"``
    picks the cheaper via :func:`maybe_use_materialized_norm`.
    """
    a, g = preprocess_factors(a, g, layer_type, module_kwargs, include_bias)

    if is_embedding(layer_type):
        return pairwise_dot_factors(a, g, layer_type).diagonal()

    a, g = dtypes.align(a, g)
    a_f = to_3d(a)  # (B, T, d_in)
    g_f = to_3d(g)  # (B, T, d_out)

    if is_norm(layer_type):
        # ||sum_t x_hat_it * g_it||^2 -- positions contracted first.
        return (a_f * g_f).sum(1).square().sum(-1)  # (B,)

    # Linear, Conv, ConvTranspose -- route factorized (ghost) vs materialized.
    _, S, K = a_f.shape
    D = g_f.shape[-1]
    if mode == "auto":
        mode = "materialized" if maybe_use_materialized_norm(S, K, D) else "factorized"
    if mode == "materialized":
        M = torch.einsum("btk,btd->bkd", a_f, g_f).flatten(1)  # (B, K*D)
        return (M * M).sum(-1)  # (B,)
    K_a = torch.einsum("btk,bsk->bts", a_f, a_f)  # (B, T, T)
    K_g = torch.einsum("btd,bsd->bts", g_f, g_f)  # (B, T, T)
    return (K_a * K_g).sum((1, 2))  # (B,)


def grad_norm_sq(
    f: Factorized | torch.Tensor,
    layer_type: str,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """Per-sample squared gradient norms ``(B,)`` of one layer, whatever its form.

    A :class:`Factorized` layer goes through :func:`grad_norm_sq_factors`
    (batch-first-safe; ``mode`` routes factorized vs materialized).  A dense
    ``(B, d)`` layer is squared and summed directly.
    """
    if isinstance(f, torch.Tensor):
        (flat,) = dtypes.align(f.reshape(f.shape[0], -1))
        return (flat * flat).sum(-1)
    bf = f.as_batch_first()
    return grad_norm_sq_factors(
        bf.activation,
        bf.pre_activation_grad,
        layer_type,
        bf.module_kwargs,
        include_bias,
        mode,
    )


def pairwise_dot(
    f: Factorized,
    layer_type: str,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """:func:`pairwise_dot_factors` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return pairwise_dot_factors(
        bf.activation,
        bf.pre_activation_grad,
        layer_type,
        bf.module_kwargs,
        include_bias,
        mode,
    )


def dot(
    f1: Factorized,
    f2: Factorized,
    layer_type: str,
    include_bias: bool = True,
) -> torch.Tensor:
    """:func:`dot_factors` on two :class:`Factorized` (batch-first-safe)."""
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return dot_factors(
        b1.activation,
        b1.pre_activation_grad,
        b2.activation,
        b2.pre_activation_grad,
        layer_type,
        b1.module_kwargs,
        b2.module_kwargs,
        include_bias,
    )


def cross_dot(
    f1: Factorized | torch.Tensor,
    f2: Factorized | torch.Tensor,
    layer_type: str,
    include_bias: bool = True,
    mode: str = "auto",
) -> torch.Tensor:
    """``(B1, B2)`` cross-gram ``K[i, j] = <dW1_i, dW2_j>`` of one layer, in
    whatever form each side holds.

    This is the single per-layer product entry point.  Each side may be

    * a raw :class:`Factorized` capture (preprocessed here),
    * *final* factors -- a :class:`Factorized` with ``module_kwargs=None``,
      e.g. projected or K-FAC-whitened factors, used as they are, or
    * a dense ``(B, d)`` tensor (a materialized or projected layer).

    Two factorized sides go through :func:`cross_dot_factors`, where ``mode``
    (``"auto"``/``"factorized"``/``"materialized"``) routes the ghost vs
    materialized contraction (see :func:`cross_gram`).  When either side is
    dense the other is materialized (:func:`materialize`) and the result is
    one GEMM.
    """
    if isinstance(f1, torch.Tensor) or isinstance(f2, torch.Tensor):
        m1 = materialize(f1, layer_type, include_bias)
        m2 = materialize(f2, layer_type, include_bias)
        m1, m2 = dtypes.align(m1, m2)
        return m1 @ m2.T
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return cross_dot_factors(
        b1.activation,
        b1.pre_activation_grad,
        b2.activation,
        b2.pre_activation_grad,
        layer_type,
        b1.module_kwargs,
        b2.module_kwargs,
        include_bias,
        mode,
    )


# ---------------------------------------------------------------------------
# layerwise cross-gram over whole gradient blocks
# ---------------------------------------------------------------------------


def _expand_broadcast(m: torch.Tensor, b1: int, b2: int) -> torch.Tensor:
    """Expand a broadcast (batch-1) layer's cross matrix to ``(b1, b2)``."""
    return m.expand(
        b1 if m.shape[0] == 1 else m.shape[0], b2 if m.shape[1] == 1 else m.shape[1]
    )


def layerwise_cross_dot(
    train: Gradient,
    test: Gradient,
    *,
    layers: Iterable[str] | None = None,
    mode: str = "auto",
    dense_cache: TensorCache | None = None,
    reduce: str = "sum",
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Layer-by-layer cross-gram of two gradient blocks, summed over layers.

    For every layer shared by *train* and *test* (or the given *layers*) this
    forms the ``(B_train, B_test)`` cross matrix with :func:`cross_dot`, so each
    side may hold the layer raw-factorized, as final factors, or dense -- and
    the representation may differ between the two sides and between layers.
    Only one layer is in flight at a time, so no whole-block materialization
    ever happens: this is the kernel an attributor's ``inner_product`` should
    call, and the reason it needs no cache logic of its own.

    Args:
        train: The row-side block.
        test: The column-side block.  Typically the (transformed / preconditioned)
            test representation.
        layers: Restrict to these layer names; ``None`` scores every shared
            layer.  Layers absent from either side are skipped.
        mode: Factorized-vs-materialized routing for factor x factor layers
            (see :func:`cross_gram`); ``"auto"`` picks the cheaper per layer.
        dense_cache: A :class:`~dattri_llm.utils.cache.TensorCache` **scoped to
            this train block**.  When a train layer has to be materialized
            (because the test side of that layer is dense), the dense copy is
            fetched from or stored in it under the layer name -- so a block
            scored against several test blocks is materialized once, and the
            cache's budget bounds how much dense state is retained.  ``None``
            recomputes per call.
        reduce: ``"sum"`` returns the ``(B_train, B_test)`` whole-model
            cross-gram (per-layer matrices summed; a broadcast batch-1 layer's
            shared row is expanded to every sample first).  ``"none"`` returns
            ``{layer: matrix}``.

    Raises:
        ValueError: If ``reduce="sum"`` and no layer is shared.
    """
    if reduce not in {"sum", "none"}:
        raise ValueError("reduce must be 'sum' or 'none'")
    names = list(train.data) if layers is None else list(layers)
    per_layer: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in train.data or name not in test.data:
            continue
        layer_type = train.layer_types[name]
        tr, te = train.data[name], test.data[name]
        if isinstance(te, torch.Tensor) and not isinstance(tr, torch.Tensor):
            tr = (
                dense_cache.get_or_compute(
                    name,
                    lambda tr=tr, lt=layer_type: materialize(tr, lt),
                )
                if dense_cache is not None
                else materialize(tr, layer_type)
            )
        per_layer[name] = cross_dot(tr, te, layer_type, mode=mode)
    if reduce == "none":
        return per_layer
    if not per_layer:
        raise ValueError("No shared layers between the two gradient blocks.")
    b_tr, b_te = train.batch_size, test.batch_size
    total = None
    for matrix in per_layer.values():
        expanded = _expand_broadcast(matrix, b_tr, b_te)
        total = expanded.clone() if total is None else total + expanded
    return total


# --------------------------------------------------------------------------- #
# Representation routing heuristic (factorized vs materialized)               #
#                                                                             #
# The per-sample weight gradient G = g^Ta (Dx K, summed over S token/patch     #
# positions) can be dotted/normed either factorized ("ghost") or materialized.#
# Which is cheaper is governed by S relative to H = DK/(D+K); see             #
# docs/gradient_representation_complexity.md.  These predicates are consumed   #
# *here at the bottom* -- cross_gram / grad_norm_sq_factors route on them -- so   #
# every caller (Gradient.similarity, K-FAC's kfac_cross_factors, ...) shares     #
# one routed implementation; ``mode="auto"`` triggers the heuristic, and the     #
# explicit "factorized"/"materialized" modes override it.                        #
# --------------------------------------------------------------------------- #


def effective_dims(f: Factorized, layer_type: str) -> tuple[int, int, int, int]:
    """Cheap ``(B, S, K, D)`` for the cost heuristic: batch, token/patch count,
    input width, output width -- the *post-preprocess* dims, read straight from the
    raw factor shapes (no im2col / materialization).  Bias's ``+1`` on ``K`` is
    ignored (it is a heuristic).
    """
    bf = f.as_batch_first()
    a, g = bf.activation, bf.pre_activation_grad
    mk = bf.module_kwargs or {}
    if is_conv(layer_type):
        # a=(B,C_in,*sp_in), g=(B,C_out,*sp_out): S = output positions,
        # K = C_in*prodkernel, D = C_out
        kprod = math.prod(mk["kernel_size"]) if "kernel_size" in mk else 1
        return a.shape[0], math.prod(g.shape[2:]), a.shape[1] * kprod, g.shape[1]
    if is_conv_transpose(layer_type):
        # roles reversed: a flattened over spatial (K=C_in), g unfolded
        # (D=C_out*prod(K))
        kprod = math.prod(mk["kernel_size"]) if "kernel_size" in mk else 1
        return a.shape[0], math.prod(a.shape[2:]), a.shape[1], g.shape[1] * kprod
    # linear-family (and norm layers): a=(B, *T, K), g=(B, *T, D)
    return a.shape[0], math.prod(a.shape[1:-1]), a.shape[-1], g.shape[-1]


def maybe_use_materialized_gram(
    B1: int,
    B2: int,
    S: int,
    K: int,
    D: int,
    kappa: float = 1.0,
) -> bool:
    """``True`` when materialize-then-GEMM is the cheaper way to form the
    ``(B1, B2)`` cross-gram (Sec. 3.2):

        cost_F = B1*B2*S^2*(D+K)            cost_M = (B1+B2)*S*D*K + B1*B2*D*K

    Materialize iff ``kappa*cost_F >= cost_M`` (``kappa=1`` is the pure-flop rule).
    """
    cost_f = B1 * B2 * S * S * (D + K)
    cost_m = (B1 + B2) * S * D * K + B1 * B2 * D * K
    return kappa * cost_f >= cost_m


def maybe_use_materialized_norm(S: int, K: int, D: int) -> bool:
    """``True`` when materializing is cheaper for per-sample norms (Sec. 3.1).  Here
    ``cost_F = S^2(D+K)`` and ``cost_M = S*D*K`` (per sample, batch cancels), so
    materialize iff ``S*(D+K) >= DK``, i.e. ``S >= H = DK/(D+K)``.
    """
    return S * (D + K) >= D * K
