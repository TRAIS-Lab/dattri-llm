"""Normalization-layer x_hat computation and bias-augmentation helpers."""

from __future__ import annotations

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
