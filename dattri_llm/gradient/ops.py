"""Layer-type-aware gradient operations for per-sample gradient computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Factorized, Gradient

# ---------------------------------------------------------------------------
# Layer type constants
# ---------------------------------------------------------------------------

LINEAR_TYPES = frozenset({
    "nn.Linear",
    "nn.Bilinear",
    "nn.NonDynamicallyQuantizableLinear",
    "transformers.pytorch_utils.Conv1D",
})
CONV_TYPES = frozenset({"nn.Conv1d", "nn.Conv2d", "nn.Conv3d"})
CONV_TRANSPOSE_TYPES = frozenset({
    "nn.ConvTranspose1d", "nn.ConvTranspose2d", "nn.ConvTranspose3d"
})
NORM_TYPES = frozenset({
    "nn.LayerNorm", "nn.RMSNorm", "nn.GroupNorm",
    "nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d",
})
EMBEDDING_TYPES = frozenset({"nn.Embedding", "nn.EmbeddingBag"})

# Special marker for aggregated param-level gradients (no batch dim).
PARAM_GRAD_TYPES = "param_grad_layers"

ALL_LAYER_TYPES = LINEAR_TYPES | CONV_TYPES | CONV_TRANSPOSE_TYPES | NORM_TYPES | EMBEDDING_TYPES


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def is_linear(layer_type: str) -> bool:
    """Return True if layer_type is a linear layer type."""
    return layer_type in LINEAR_TYPES


def is_conv(layer_type: str) -> bool:
    """Return True if layer_type is a convolution layer type."""
    return layer_type in CONV_TYPES


def is_conv_transpose(layer_type: str) -> bool:
    """Return True if layer_type is a transposed convolution layer type."""
    return layer_type in CONV_TRANSPOSE_TYPES


def is_norm(layer_type: str) -> bool:
    """Return True if layer_type is a normalization layer type."""
    return layer_type in NORM_TYPES


def is_embedding(layer_type: str) -> bool:
    """Return True if layer_type is an embedding layer type."""
    return layer_type in EMBEDDING_TYPES


# ---------------------------------------------------------------------------
# Canonical class name
# ---------------------------------------------------------------------------

def canonical_class_name(module: nn.Module) -> str:
    """Return canonical string for a module class, e.g. 'nn.Linear'."""
    mod = type(module).__module__
    name = type(module).__name__
    if mod.startswith("torch.nn"):
        return f"nn.{name}"
    return f"{mod}.{name}"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _compute_layer_norm_x_hat(
    x: torch.Tensor, normalized_shape: tuple, eps: float
) -> torch.Tensor:
    """Return x̂ = (x − μ) / √(σ² + ε) for a LayerNorm-style normalisation."""
    n_dims = len(normalized_shape)
    reduce_dims = tuple(range(-n_dims, 0))
    mean = x.mean(dim=reduce_dims, keepdim=True)
    var = x.var(dim=reduce_dims, unbiased=False, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


def _compute_rms_x_hat(
    x: torch.Tensor, normalized_shape: tuple, eps: Optional[float]
) -> torch.Tensor:
    """Return x̂ = x / √(mean(x²) + ε) for an RMSNorm normalisation.

    RMSNorm omits mean subtraction.  When *eps* is ``None`` (PyTorch's
    ``nn.RMSNorm`` default) the dtype's machine epsilon is used, matching
    :func:`torch.nn.functional.rms_norm`.
    """
    n_dims = len(normalized_shape)
    reduce_dims = tuple(range(-n_dims, 0))
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    mean_sq = x.pow(2).mean(dim=reduce_dims, keepdim=True)
    return x * torch.rsqrt(mean_sq + eps)


def _compute_group_norm_x_hat(
    x: torch.Tensor, num_groups: int, eps: float
) -> torch.Tensor:
    """Return x̂ for a GroupNorm-style normalisation.

    *x* has shape ``(N, C, *spatial)``.  Channels are split into ``num_groups``
    contiguous groups; each group is normalised jointly over its channels and
    all spatial locations.  Setting ``num_groups == C`` recovers InstanceNorm
    (per-channel normalisation over spatial dims only).
    """
    n, c = x.shape[:2]
    spatial = x.shape[2:]
    grouped = x.reshape(n, num_groups, -1)
    mean = grouped.mean(dim=2, keepdim=True)
    var = grouped.var(dim=2, unbiased=False, keepdim=True)
    x_hat = (grouped - mean) * torch.rsqrt(var + eps)
    return x_hat.reshape(n, c, *spatial)


def _conv1d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv1d input (N, C_in, L_in) → (N, L_out, C_in*k)."""
    N, C_in, _ = x.shape
    k, s, p, d = (kwargs["kernel_size"][0], kwargs["stride"][0],
                  kwargs["padding"][0], kwargs["dilation"][0])
    if p > 0:
        x = F.pad(x, (p, p))
    L_out = (x.shape[-1] - d * (k - 1) - 1) // s + 1
    l_idx = torch.arange(L_out, device=x.device) * s
    k_idx = torch.arange(k, device=x.device) * d
    idx = l_idx.unsqueeze(-1) + k_idx.unsqueeze(0)   # (L_out, k)
    patches = x[:, :, idx]                             # (N, C_in, L_out, k)
    return patches.permute(0, 2, 1, 3).contiguous().reshape(N, L_out, C_in * k)


def _conv2d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv2d input (N, C_in, H, W) → (N, L_out, C_in*kH*kW)."""
    unf = F.unfold(x, kwargs["kernel_size"], kwargs["dilation"],
                   kwargs["padding"], kwargs["stride"])
    return unf.permute(0, 2, 1).contiguous()   # (N, L, C_in*kH*kW)


def _conv3d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv3d input (N, C_in, D, H, W) → (N, L_out, C_in*kD*kH*kW)."""
    N, C_in = x.shape[:2]
    kD, kH, kW = kwargs["kernel_size"]
    sD, sH, sW = kwargs["stride"]
    pD, pH, pW = kwargs["padding"]
    dD, dH, dW = kwargs["dilation"]
    if any(p > 0 for p in (pD, pH, pW)):
        x = F.pad(x, (pW, pW, pH, pH, pD, pD))
    D_pad, H_pad, W_pad = x.shape[2:]
    D_out = (D_pad - dD * (kD - 1) - 1) // sD + 1
    H_out = (H_pad - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_pad - dW * (kW - 1) - 1) // sW + 1
    L_out = D_out * H_out * W_out
    d_idx = (torch.arange(D_out, device=x.device).unsqueeze(-1) * sD
             + torch.arange(kD, device=x.device).unsqueeze(0) * dD)
    h_idx = (torch.arange(H_out, device=x.device).unsqueeze(-1) * sH
             + torch.arange(kH, device=x.device).unsqueeze(0) * dH)
    w_idx = (torch.arange(W_out, device=x.device).unsqueeze(-1) * sW
             + torch.arange(kW, device=x.device).unsqueeze(0) * dW)
    patches = x[
        :, :,
        d_idx.view(D_out, kD, 1, 1, 1, 1),
        h_idx.view(1, 1, H_out, kH, 1, 1),
        w_idx.view(1, 1, 1, 1, W_out, kW),
    ]
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return patches.reshape(N, L_out, C_in * kD * kH * kW)


# Spatial-rank → im2col helper.  Keyed by the trailing digit of the layer-type
# string ("nn.Conv2d" / "nn.ConvTranspose2d" → 2).
_CONV_IM2COL = {1: _conv1d_im2col, 2: _conv2d_im2col, 3: _conv3d_im2col}


def _conv_spatial_rank(layer_type: str) -> int:
    """Return the spatial rank (1/2/3) encoded in a Conv layer-type string."""
    return int(layer_type[-2])


# ---------------------------------------------------------------------------
# Augmentation helpers (bias folding per the factorized-gradient formulation)
# ---------------------------------------------------------------------------

def _augment_token_norm(
    x_hat: torch.Tensor, g: torch.Tensor, has_bias: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias-augment a LayerNorm/RMSNorm pair (affine over the last dim).

    With bias, append a ones block to ``x̂`` and duplicate ``g`` so that the
    per-position elementwise product ``a ⊙ g`` yields ``[∇γ, ∇β]``.
    """
    if has_bias:
        return (
            torch.cat([x_hat, torch.ones_like(x_hat)], dim=-1),
            torch.cat([g, g], dim=-1),
        )
    return x_hat, g


def _augment_channel_norm(
    x_hat: torch.Tensor, g: torch.Tensor, has_bias: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out a GroupNorm/InstanceNorm pair as ``(N, S, C)`` and bias-augment.

    The affine parameters are per-channel ``(C,)`` while ``x̂`` and ``g`` have
    shape ``(N, C, *spatial)``.  Spatial locations play the role of the token
    dimension, so we permute to ``(N, S, C)``.  With bias, append a ones block
    in the channel dimension and duplicate ``g`` so ``a ⊙ g`` yields
    ``[∇γ, ∇β]`` per position.
    """
    n, c = x_hat.shape[:2]
    x_sc = x_hat.reshape(n, c, -1).permute(0, 2, 1).contiguous()  # (N, S, C)
    g_sc = g.reshape(n, c, -1).permute(0, 2, 1).contiguous()      # (N, S, C)
    if has_bias:
        return (
            torch.cat([x_sc, torch.ones_like(x_sc)], dim=-1),
            torch.cat([g_sc, g_sc], dim=-1),
        )
    return x_sc, g_sc


def _preprocess_embedding_bag(
    a: torch.Tensor, g: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-bag EmbeddingBag gradients to the per-token Embedding form.

    ``a`` are token indices ``(N, T)`` and ``g`` is the per-bag output gradient
    ``(N, E)``.  Each bag's gradient is broadcast to its ``T`` tokens (scaled by
    ``1/T`` for ``mode='mean'``), producing ``(N, T, E)`` so the standard
    embedding scatter-add materialisation applies.

    Only 2-D inputs (one bag per row, no ``offsets``) are supported.
    ``mode='max'`` is not supported.
    """
    if mode == "max":
        raise NotImplementedError(
            "EmbeddingBag mode='max' is not supported for factorized gradients."
        )
    if a.ndim != 2:
        raise NotImplementedError(
            "EmbeddingBag factorization supports only 2-D inputs (N, T) without "
            "offsets."
        )
    t = a.shape[1]
    g_exp = g.unsqueeze(1).expand(-1, t, -1)   # (N, T, E)
    if mode == "mean":
        g_exp = g_exp / t
    return a, g_exp.contiguous()


def _preprocess_conv_transpose(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: dict,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorize a transposed-convolution layer.

    The roles of activation and gradient unfolding are reversed relative to a
    regular convolution: the *output gradient* is unfolded into kernel patches
    while the input is merely flattened over spatial locations.  This yields

        a ∈ (N, L, C_in),   g ∈ (N, L, C_out·∏K),

    where ``L`` is the number of input spatial locations and ``∇W = Σ_ℓ aᵀg``.

    Bias cannot be folded by a single ones column because ``∇b`` has dimension
    ``C_out`` (not ``C_out·∏K``).  Following the factorized-gradient formulation
    we instead extend both factors with one extra location and feature block so
    the bias gradient is stored jointly:

        ã ∈ (N, L+1, C_in+1),   g̃ ∈ (N, L+1, C_out·∏K + C_out).
    """
    rank = _conv_spatial_rank(layer_type)
    n, c_in = a.shape[:2]
    a_flat = a.reshape(n, c_in, -1).permute(0, 2, 1).contiguous()   # (N, L, C_in)
    g_unf = _CONV_IM2COL[rank](g, module_kwargs)                    # (N, L, C_out*∏K)
    if not has_bias:
        return a_flat, g_unf

    c_out = g.shape[1]
    L = a_flat.shape[1]
    spatial_dims = tuple(range(2, g.ndim))
    bias_grad = g.reshape(n, c_out, -1).sum(dim=2)                  # (N, C_out)

    # Activation: zero-pad the feature dim, then append a bias location whose
    # only non-zero feature is the appended one.
    a_pad = torch.cat([a_flat, torch.zeros_like(a_flat[..., :1])], dim=-1)
    a_bias = torch.zeros(n, 1, a_pad.shape[-1], dtype=a_pad.dtype, device=a_pad.device)
    a_bias[..., -1] = 1.0
    a_aug = torch.cat([a_pad, a_bias], dim=1)                       # (N, L+1, C_in+1)

    # Gradient: zero-pad the feature dim with C_out columns, then append a bias
    # location carrying the bias gradient in those columns.
    g_pad = torch.cat(
        [g_unf, torch.zeros(n, L, c_out, dtype=g_unf.dtype, device=g_unf.device)],
        dim=-1,
    )
    g_bias = torch.zeros(n, 1, g_pad.shape[-1], dtype=g_pad.dtype, device=g_pad.device)
    g_bias[..., -c_out:] = bias_grad.unsqueeze(1)
    g_aug = torch.cat([g_pad, g_bias], dim=1)                       # (N, L+1, P+C_out)
    return a_aug, g_aug


def extract_module_kwargs(module: nn.Module, layer_type: str) -> dict:
    """Extract the minimal hyperparameters from *module* needed by :func:`_preprocess_factorized`.

    Returns a plain serialisable dict — no reference to the module object is
    retained, so the result can be pickled cheaply alongside gradient tensors.

    Args:
        module: The hooked ``nn.Module`` instance.
        layer_type: Canonical class name (e.g. ``"nn.Conv2d"``).

    Returns:
        A dict with at least ``"has_bias": bool``.  Conv and ConvTranspose
        layers additionally include ``"kernel_size"``, ``"stride"``,
        ``"padding"``, and ``"dilation"``.  LayerNorm and RMSNorm include
        ``"normalized_shape"`` and ``"eps"``.  GroupNorm includes
        ``"num_groups"``, ``"num_channels"``, and ``"eps"``.  InstanceNorm
        includes ``"num_features"`` and ``"eps"``.  EmbeddingBag includes
        ``"mode"``.
    """
    kwargs: dict = {"has_bias": getattr(module, "bias", None) is not None}
    if layer_type in CONV_TYPES or layer_type in CONV_TRANSPOSE_TYPES:
        kwargs["kernel_size"] = module.kernel_size
        kwargs["stride"] = module.stride
        kwargs["padding"] = module.padding
        kwargs["dilation"] = module.dilation
    elif layer_type in ("nn.LayerNorm", "nn.RMSNorm"):
        kwargs["normalized_shape"] = tuple(module.normalized_shape)
        kwargs["eps"] = module.eps
    elif layer_type == "nn.GroupNorm":
        kwargs["num_groups"] = module.num_groups
        kwargs["num_channels"] = module.num_channels
        kwargs["eps"] = module.eps
    elif layer_type in ("nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"):
        kwargs["num_features"] = module.num_features
        kwargs["eps"] = module.eps
    elif layer_type == "nn.EmbeddingBag":
        kwargs["mode"] = module.mode
    return kwargs


def _preprocess_factorized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw hook captures into the (a, g) form expected by ops.

    Hooks store exactly what PyTorch delivers (raw ``input[0]`` and
    ``grad_output[0]``).  Before the factorised data can be passed to the ops
    functions in this module, layer-specific transforms are applied:

    * **LayerNorm / RMSNorm**: normalise ``a`` → ``x̂`` (RMSNorm omits the mean
      subtraction).  With bias, append a ones block to ``a`` and duplicate ``g``
      so ``a ⊙ g = [∇γ, ∇β]`` per position.
    * **GroupNorm / InstanceNorm**: normalise per group/channel → ``x̂``, then
      lay out spatial locations as the token dimension ``(N, S, C)``.  Bias is
      folded as for LayerNorm but in the channel dimension.
    * **Linear types with bias**: append a ones column to ``a`` so the last
      column of ``∇W̃ = g ãᵀ`` recovers ``∇b``.
    * **Conv1d / Conv2d / Conv3d**: run im2col on ``a`` to get the unfolded
      ``(N, L, patch_size)`` form; reshape ``g`` to ``(N, L, C_out)``; append a
      ones column to ``a`` if the layer has a bias.
    * **ConvTranspose1d / 2d / 3d**: flatten ``a`` over spatial locations and
      unfold ``g`` into kernel patches (roles reversed vs. Conv).  Bias is
      folded via an extra location/feature block (see
      :func:`_preprocess_conv_transpose`).
    * **EmbeddingBag**: broadcast per-bag gradients to per-token gradients so
      the standard embedding scatter-add applies.

    Plain ``nn.Embedding`` is passed through unchanged.

    Args:
        a: Raw activation captured by the forward hook.
        g: Raw gradient captured by the backward hook.
        layer_type: Canonical class-name string (e.g. ``"nn.Conv2d"``).
        module_kwargs: Serialisable dict produced by :func:`extract_module_kwargs`.
            When ``None`` no preprocessing is applied.
        include_bias: When ``False``, bias augmentation (ones column / g
            duplication) is skipped even if the layer has a bias.  Defaults
            to ``True``.

    Returns:
        ``(a_processed, g_processed)`` ready for :func:`_materialize` and
        related functions.
    """
    if module_kwargs is None:
        return a, g

    has_bias = module_kwargs.get("has_bias", False) and include_bias

    # ── Normalisation layers ────────────────────────────────────────────────
    if layer_type == "nn.LayerNorm":
        x_hat = _compute_layer_norm_x_hat(
            a,
            module_kwargs.get("normalized_shape", (a.shape[-1],)),
            module_kwargs.get("eps", 1e-5),
        )
        return _augment_token_norm(x_hat, g, has_bias)

    if layer_type == "nn.RMSNorm":
        x_hat = _compute_rms_x_hat(
            a,
            module_kwargs.get("normalized_shape", (a.shape[-1],)),
            module_kwargs.get("eps"),
        )
        return _augment_token_norm(x_hat, g, has_bias)

    if layer_type == "nn.GroupNorm":
        x_hat = _compute_group_norm_x_hat(
            a, module_kwargs["num_groups"], module_kwargs.get("eps", 1e-5)
        )
        return _augment_channel_norm(x_hat, g, has_bias)

    if layer_type in ("nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"):
        # InstanceNorm = GroupNorm with one group per channel.
        x_hat = _compute_group_norm_x_hat(
            a, module_kwargs["num_features"], module_kwargs.get("eps", 1e-5)
        )
        return _augment_channel_norm(x_hat, g, has_bias)

    # ── Embedding bag ───────────────────────────────────────────────────────
    if layer_type == "nn.EmbeddingBag":
        return _preprocess_embedding_bag(a, g, module_kwargs.get("mode", "mean"))

    # ── Convolution ─────────────────────────────────────────────────────────
    if layer_type in CONV_TYPES:
        rank = _conv_spatial_rank(layer_type)
        a_unf = _CONV_IM2COL[rank](a, module_kwargs)            # (N, L, C_in*∏K)
        n, c_out = g.shape[0], g.shape[1]
        g_out = g.reshape(n, c_out, -1).permute(0, 2, 1).contiguous()  # (N, L, C_out)
        if has_bias:
            a_unf = torch.cat([a_unf, torch.ones_like(a_unf[..., :1])], dim=-1)
        return a_unf, g_out

    if layer_type in CONV_TRANSPOSE_TYPES:
        return _preprocess_conv_transpose(a, g, layer_type, module_kwargs, has_bias)

    # ── Linear with bias ────────────────────────────────────────────────────
    if is_linear(layer_type) and has_bias:
        return torch.cat([a, torch.ones_like(a[..., :1])], dim=-1), g

    return a, g


# ---------------------------------------------------------------------------
# Shape helper
# ---------------------------------------------------------------------------

def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """Expand a (B, D) tensor to (B, 1, D); leave (B, T, D) unchanged.

    All ops work on 3-D (B, T, D) tensors internally.  This helper lets
    callers pass raw 2-D activations without a separate code path.
    """
    return x.unsqueeze(1) if x.ndim == 2 else x


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


# ---------------------------------------------------------------------------
# kfac
# ---------------------------------------------------------------------------

def _flatten_for_kfac(
    a: torch.Tensor, g: torch.Tensor, layer_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten spatial/token dims for K-FAC; raise for norm/embedding types."""
    if is_norm(layer_type):
        raise NotImplementedError("K-FAC is not defined for normalization layers")
    if is_embedding(layer_type):
        raise NotImplementedError("K-FAC is not defined for embedding layers")
    a_f = a.float().reshape(-1, a.shape[-1])
    g_f = g.float().reshape(-1, g.shape[-1])
    return a_f, g_f


def _kfac(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (A, G) K-FAC covariance factor matrices.

    *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    a_f, g_f = _flatten_for_kfac(a, g, layer_type)
    N = a_f.shape[0]
    A = a_f.T @ a_f / N
    G = g_f.T @ g_f / N
    return A, G


# ---------------------------------------------------------------------------
# fim
# ---------------------------------------------------------------------------

def _fim(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Return (d, d) empirical Fisher information matrix.

    Built from the per-sample :func:`_materialize` (token-summed for norm layers,
    so the Fisher is over the layer's actual parameters).  *module_kwargs* is
    passed to :func:`_preprocess_factorized` when provided.
    """
    grad = _materialize(a, g, layer_type, module_kwargs, include_bias)   # (B, d)
    B = grad.shape[0]
    return grad.T @ grad / B


# ---------------------------------------------------------------------------
# K-FAC / EK-FAC attribution kernels
# ---------------------------------------------------------------------------

def sym_inverse(matrix: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    """Damped symmetric inverse ``(matrix + damping·I)^{-1}``.

    Computed via eigendecomposition so the result stays symmetric and a small
    *damping* keeps a rank-deficient K-FAC covariance factor invertible.
    """
    evals, evecs = torch.linalg.eigh(matrix.float())
    return (evecs / (evals + damping)) @ evecs.T


def _kfac_cross(
    a1: torch.Tensor, g1: torch.Tensor,
    a2: torch.Tensor, g2: torch.Tensor,
    layer_type: str,
    A_inv: torch.Tensor,
    G_inv: torch.Tensor,
    module_kwargs1: Optional[dict] = None,
    module_kwargs2: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """K-FAC preconditioned cross-gram between two factorized gradient sets.

    Returns ``K[i, j] = vec(∇W1_i)ᵀ (A⁻¹ ⊗ G⁻¹) vec(∇W2_j)`` — i.e.
    :func:`_cross_dot` with the side-1 factors whitened by the inverse K-FAC
    covariances (``A_inv`` over the input dim, ``G_inv`` over the output dim).
    Both inverses are symmetric, so whitening either side gives the same value.
    Defined for linear and convolution layers.
    """
    a1, g1 = _preprocess_factorized(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = _preprocess_factorized(a2, g2, layer_type, module_kwargs2, include_bias)
    a1 = a1.float() @ A_inv.float()
    g1 = g1.float() @ G_inv.float()
    return _cross_gram(a1, g1, a2, g2, layer_type)


def kfac_eigh(
    A: torch.Tensor, G: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigendecompose both K-FAC factors: returns ``(s_A, U_A, s_G, U_G)``."""
    s_A, U_A = torch.linalg.eigh(A.float())
    s_G, U_G = torch.linalg.eigh(G.float())
    return s_A, U_A, s_G, U_G


def _ekfac_materialize(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    U_A: torch.Tensor,
    U_G: torch.Tensor,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Per-sample weight gradient rotated into the K-FAC eigenbasis.

    Returns ``(B, d_out·d_in)`` flattened ``U_Gᵀ ∇W U_A`` — the eigenbasis
    coordinates whose empirical second moments are the EK-FAC corrected
    eigenvalues, and against which test/train gradients are scored.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    a = a.float() @ U_A.float()
    g = g.float() @ U_G.float()
    return _materialize(a, g, layer_type)


# ---------------------------------------------------------------------------
# Public Factorized-input API
# ---------------------------------------------------------------------------
#
# The private ``_``-prefixed functions above operate on raw
# ``(activation, pre_activation_grad)`` tensors and **always assume batch-first**
# ``(B, T, ...)`` input.  The public functions below take the
# :class:`~dattri_llm.gradient.gradient.Factorized` container instead: they call
# :meth:`Factorized.as_batch_first` to normalise a sequence-first capture, unpack
# ``module_kwargs``, and delegate to the private version.  These are the entry
# points to prefer — the call is both shorter and automatically correct for
# non-batch-first layers, so nothing downstream needs to reason about tensor
# layout.  Reach for the raw ``_``-prefixed kernels only when you already hold
# bare, batch-first factor tensors (e.g. inside another kernel).


def preprocess_factorized(
    f: "Factorized", layer_type: str, include_bias: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_preprocess_factorized` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _preprocess_factorized(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs, include_bias
    )


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


def kfac(
    f: "Factorized", layer_type: str, include_bias: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_kfac` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _kfac(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs, include_bias
    )


def fim(
    f: "Factorized", layer_type: str, include_bias: bool = True
) -> torch.Tensor:
    """:func:`_fim` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _fim(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs, include_bias
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


def kfac_cross(
    f1: "Factorized", f2: "Factorized", layer_type: str,
    A_inv: torch.Tensor, G_inv: torch.Tensor, include_bias: bool = True,
) -> torch.Tensor:
    """:func:`_kfac_cross` on two :class:`Factorized` (batch-first-safe)."""
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return _kfac_cross(
        b1.activation, b1.pre_activation_grad, b2.activation, b2.pre_activation_grad,
        layer_type, A_inv, G_inv, b1.module_kwargs, b2.module_kwargs, include_bias,
    )


def ekfac_materialize(
    f: "Factorized", layer_type: str,
    U_A: torch.Tensor, U_G: torch.Tensor, include_bias: bool = True,
) -> torch.Tensor:
    """:func:`_ekfac_materialize` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _ekfac_materialize(
        bf.activation, bf.pre_activation_grad, layer_type, U_A, U_G,
        bf.module_kwargs, include_bias,
    )


# ---------------------------------------------------------------------------
# Per-layer streaming accumulators
# ---------------------------------------------------------------------------

@dataclass
class LayerKroneckerAccumulator:
    """Streaming K-FAC covariance accumulator for a *single* layer.

    Accumulates the Kronecker factors ``(A, G)`` (input-activation and
    output-gradient covariances).  For the across-layers version see
    :class:`KroneckerAccumulator`.
    """

    _A: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _G: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def update(
        self,
        a: torch.Tensor,
        g: torch.Tensor,
        layer_type: str,
        module_kwargs: Optional[dict] = None,
        include_bias: bool = True,
    ) -> None:
        """Accumulate one batch of factorized gradient data.

        *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
        """
        a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
        a_f, g_f = _flatten_for_kfac(a, g, layer_type)
        N = a_f.shape[0]
        if self._A is None:
            self._A = torch.zeros(
                a_f.shape[-1], a_f.shape[-1], dtype=a_f.dtype, device=a_f.device
            )
            self._G = torch.zeros(
                g_f.shape[-1], g_f.shape[-1], dtype=g_f.dtype, device=g_f.device
            )
        self._A += a_f.T @ a_f
        self._G += g_f.T @ g_f
        self._n += N

    def result(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A, G) normalized covariance matrices."""
        if self._A is None:
            raise RuntimeError("No data has been accumulated")
        return self._A / self._n, self._G / self._n

    def reset(self) -> None:
        """Reset accumulator state."""
        self._A = None
        self._G = None
        self._n = 0


@dataclass
class LayerFisherAccumulator:
    """Streaming empirical Fisher accumulator for a *single* layer.

    Accumulates ``Σ_i g_i g_iᵀ`` over the per-sample :func:`_materialize` ``g_i``.
    For the across-layers version see :class:`FisherAccumulator`.
    """

    _F: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def update(
        self,
        a: torch.Tensor,
        g: torch.Tensor,
        layer_type: str,
        module_kwargs: Optional[dict] = None,
        include_bias: bool = True,
    ) -> None:
        """Accumulate one batch from its factorized factors.

        *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
        """
        self.update_from_grad(_materialize(a, g, layer_type, module_kwargs, include_bias))

    def update_from_grad(self, grad: torch.Tensor) -> None:
        """Accumulate one batch from already-materialized per-sample gradients.

        *grad* is ``(B, d)``.  Use this when the caller has materialized the
        per-sample weight gradient itself (e.g. with a non-default token
        reduction) and only needs the streaming outer-product accumulation.
        """
        B = grad.shape[0]
        if self._F is None:
            self._F = torch.zeros(
                grad.shape[-1], grad.shape[-1], dtype=grad.dtype, device=grad.device
            )
        self._F += grad.T @ grad
        self._n += B

    def result(self) -> torch.Tensor:
        """Return normalized empirical Fisher matrix."""
        if self._F is None:
            raise RuntimeError("No data has been accumulated")
        return self._F / self._n

    def reset(self) -> None:
        """Reset accumulator state."""
        self._F = None
        self._n = 0


# ---------------------------------------------------------------------------
# Multi-layer streaming accumulators
# ---------------------------------------------------------------------------

class KroneckerAccumulator:
    """Streaming K-FAC covariance accumulator across a model's layers.

    Holds one :class:`LayerKroneckerAccumulator` per layer and fans each streamed
    :class:`~dattri_llm.gradient.gradient.Gradient` block out to the requested
    layers, so a fit loop is a single ``update`` call per block::

        acc = KroneckerAccumulator()
        for block in blocks:
            acc.update(block, kfac_eligible_layers)
        factors = acc.result()          # {layer: (A, G)}
    """

    def __init__(self) -> None:
        self._layers: Dict[str, LayerKroneckerAccumulator] = {}

    def update(self, gradient: "Gradient", layers: Iterable[str]) -> None:
        """Accumulate the factors for *layers* from one gradient block."""
        for name in layers:
            bf = gradient.data[name].as_batch_first()
            self._layers.setdefault(name, LayerKroneckerAccumulator()).update(
                bf.activation, bf.pre_activation_grad,
                gradient.layer_types[name], bf.module_kwargs,
            )

    def result(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return ``{layer: (A, G)}`` for every accumulated layer."""
        return {name: acc.result() for name, acc in self._layers.items()}


class FisherAccumulator:
    """Streaming empirical-Fisher accumulator across a model's layers.

    Holds one :class:`LayerFisherAccumulator` per layer, accumulating the dense
    Fisher from each layer's per-sample :func:`_materialize`.  When *max_params*
    is given, layers whose parameter count exceeds it are skipped (and recorded
    in :attr:`skipped`) to bound the dense ``O(d²)`` Fisher::

        acc = FisherAccumulator(max_params=4096)
        for block in blocks:
            acc.update(block, fisher_eligible_layers)
        fishers = acc.result()          # {layer: F}
        dropped = acc.skipped           # {layer: param_count}
    """

    def __init__(self, max_params: Optional[int] = None) -> None:
        self._max_params = max_params
        self._layers: Dict[str, LayerFisherAccumulator] = {}
        self._skipped: Dict[str, int] = {}

    def update(self, gradient: "Gradient", layers: Iterable[str]) -> None:
        """Accumulate the Fisher for *layers* from one gradient block."""
        for name in layers:
            if name in self._skipped:
                continue
            g = materialize(gradient.data[name], gradient.layer_types[name])  # (B, P)
            if self._max_params is not None and g.shape[-1] > self._max_params:
                self._skipped[name] = g.shape[-1]
                self._layers.pop(name, None)
                continue
            self._layers.setdefault(name, LayerFisherAccumulator()).update_from_grad(g)

    def result(self) -> Dict[str, torch.Tensor]:
        """Return ``{layer: F}`` for every accumulated (non-skipped) layer."""
        return {name: acc.result() for name, acc in self._layers.items()}

    @property
    def skipped(self) -> Dict[str, int]:
        """``{layer: param_count}`` for layers dropped by the ``max_params`` cap."""
        return dict(self._skipped)
