"""Normalization-layer x_hat computation and bias-augmentation helpers."""

from __future__ import annotations

import math

import torch


def _compute_layer_norm_x_hat(
    x: torch.Tensor,
    normalized_shape: tuple,
    eps: float,
) -> torch.Tensor:
    """Return x_hat = (x - mu) / sqrt(sigma^2 + eps) for a LayerNorm-style
    normalisation.
    """
    n_dims = len(normalized_shape)
    reduce_dims = tuple(range(-n_dims, 0))
    mean = x.mean(dim=reduce_dims, keepdim=True)
    var = x.var(dim=reduce_dims, unbiased=False, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


def _compute_rms_x_hat(
    x: torch.Tensor,
    normalized_shape: tuple,
    eps: float | None,
) -> torch.Tensor:
    """Return x_hat = x / sqrt(mean(x^2) + eps) for an RMSNorm normalisation.

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
    x: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Return x_hat for a GroupNorm-style normalisation.

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


def _fold_broadcast_axes(
    x_hat: torch.Tensor,
    g: torch.Tensor,
    normalized_shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold every axis between batch and *normalized_shape* into one position
    axis, restoring the ``(batch, positions, features)`` contract.

    A token norm broadcasts over all leading axes, so its weight gradient sums
    ``g * x_hat`` over every non-normalized axis.  Inputs with more than one
    broadcast axis -- e.g. Qwen3's per-head QK-norm, whose input is
    ``(B, T, heads, head_dim)`` with a ``(head_dim,)`` weight -- are reshaped
    to ``(B, T*heads, head_dim)`` so the extra axes become position axes for
    every downstream op.  Inputs already in canonical layout pass through
    unchanged.
    """
    if x_hat.ndim <= 2 + len(normalized_shape):
        return x_hat, g
    d = math.prod(normalized_shape)
    return (
        x_hat.reshape(x_hat.shape[0], -1, d),
        g.reshape(g.shape[0], -1, d),
    )


# ---------------------------------------------------------------------------
# Augmentation helpers (bias folding per the factorized-gradient formulation)
# ---------------------------------------------------------------------------


def _augment_token_norm(
    x_hat: torch.Tensor,
    g: torch.Tensor,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias-augment a LayerNorm/RMSNorm pair (affine over the last dim).

    With bias, append a ones block to ``x_hat`` and duplicate ``g`` so that the
    per-position elementwise product ``a * g`` yields ``[dgamma, dbeta]``.
    """
    if has_bias:
        return (
            torch.cat([x_hat, torch.ones_like(x_hat)], dim=-1),
            torch.cat([g, g], dim=-1),
        )
    return x_hat, g


def _augment_channel_norm(
    x_hat: torch.Tensor,
    g: torch.Tensor,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out a GroupNorm/InstanceNorm pair as ``(N, S, C)`` and bias-augment.

    The affine parameters are per-channel ``(C,)`` while ``x_hat`` and ``g`` have
    shape ``(N, C, *spatial)``.  Spatial locations play the role of the token
    dimension, so we permute to ``(N, S, C)``.  With bias, append a ones block
    in the channel dimension and duplicate ``g`` so ``a * g`` yields
    ``[dgamma, dbeta]`` per position.
    """
    n, c = x_hat.shape[:2]
    x_sc = x_hat.reshape(n, c, -1).permute(0, 2, 1).contiguous()  # (N, S, C)
    g_sc = g.reshape(n, c, -1).permute(0, 2, 1).contiguous()  # (N, S, C)
    if has_bias:
        return (
            torch.cat([x_sc, torch.ones_like(x_sc)], dim=-1),
            torch.cat([g_sc, g_sc], dim=-1),
        )
    return x_sc, g_sc
