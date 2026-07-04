"""Materialization of factorized captures into per-sample weight gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from dattri_llm.gradient.ops.preprocess import _preprocess_factorized, _to_3d
from dattri_llm.gradient.ops.types import is_conv_transpose, is_embedding, is_norm

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Factorized


# ---------------------------------------------------------------------------
# Embedding materialization helper
# ---------------------------------------------------------------------------

def _materialize_embedding(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Scatter-add materialization for embedding weight gradients."""
    token_ids = a                          # (B, T) int
    grad = g.float()                       # (B, T, embed_dim)
    B, T = token_ids.shape
    embed_dim = grad.shape[-1]
    vocab_size = int(token_ids.max().item()) + 1

    idx = token_ids.unsqueeze(-1).expand(B, T, embed_dim)  # (B, T, embed_dim)
    result = torch.zeros(B, vocab_size, embed_dim, dtype=grad.dtype, device=grad.device)
    result.scatter_add_(1, idx, grad)      # (B, vocab_size, embed_dim)
    return result.reshape(B, -1)           # (B, vocab_size * embed_dim)


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------

def _materialize(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
    per_token: bool = False,
) -> torch.Tensor:
    """Compute the per-sample weight gradient, returning shape (B, d).

    When *module_kwargs* is provided the raw hook captures are preprocessed
    first via :func:`_preprocess_factorized` (im2col for Conv, x̂ for
    LayerNorm, bias augmentation for Linear).  Pass ``module_kwargs=None``
    when the tensors are already in the preprocessed form.

    A norm layer's gradient is per-position (elementwise).  By default
    (``per_token=False``) the token axis is summed, giving the actual weight
    gradient ``(B, d)`` — a fixed dimension independent of sequence length, the
    granularity an empirical Fisher or influence score operates on.  Pass
    ``per_token=True`` to keep the un-summed per-position products ``(B, T*d)``.
    For every other layer type the token/spatial axis is already contracted by
    the gradient's structure, so ``per_token`` has no effect.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)

    if is_embedding(layer_type):
        return _materialize_embedding(a, g)

    a_f = _to_3d(a.float())   # (B, T, d_in)
    g_f = _to_3d(g.float())   # (B, T, d_out)

    if is_norm(layer_type):
        prod = a_f * g_f                       # (B, T, d) per-position gradient
        return prod.flatten(1) if per_token else prod.sum(1)  # (B, T*d) or (B, d)

    if is_conv_transpose(layer_type):
        # a=(B,L,C_in), g=(B,L,P): ∇W_i = Σ_l a_il g_il^T
        result = torch.einsum("blc,blp->bcp", a_f, g_f)
        return result.flatten(1)               # (B, C_in*P)

    # Linear and Conv: ∇W_i = Σ_t g_it ⊗ a_it
    result = torch.einsum("bto,bti->boi", g_f, a_f)
    return result.flatten(1)                   # (B, out*in)


def materialize(
    f: "Factorized", layer_type: str, include_bias: bool = True, per_token: bool = False
) -> torch.Tensor:
    """:func:`_materialize` on a :class:`Factorized` (batch-first-safe).

    ``per_token=False`` (default) sums a norm layer's token axis to the actual
    weight gradient; ``per_token=True`` keeps the per-position products."""
    bf = f.as_batch_first()
    return _materialize(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs,
        include_bias, per_token,
    )
