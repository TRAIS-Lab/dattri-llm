"""Preprocessing of raw hook captures into the (a, g) form expected by ops.

Includes the conv im2col unfolding (a special case of preprocessing), the
per-layer-type dispatch, and hyperparameter extraction from hooked modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from dattri_llm.gradient.ops.norm import (
    _augment_channel_norm,
    _augment_token_norm,
    _compute_group_norm_x_hat,
    _compute_layer_norm_x_hat,
    _compute_rms_x_hat,
    _fold_broadcast_axes,
)
from dattri_llm.gradient.ops.types import (
    CONV_TRANSPOSE_TYPES,
    CONV_TYPES,
    is_linear,
)

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Factorized


def _conv1d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv1d input (N, C_in, L_in) -> (N, L_out, C_in*k)."""
    N, C_in, _ = x.shape
    k, s, p, d = (
        kwargs["kernel_size"][0],
        kwargs["stride"][0],
        kwargs["padding"][0],
        kwargs["dilation"][0],
    )
    if p > 0:
        x = F.pad(x, (p, p))
    L_out = (x.shape[-1] - d * (k - 1) - 1) // s + 1
    l_idx = torch.arange(L_out, device=x.device) * s
    k_idx = torch.arange(k, device=x.device) * d
    idx = l_idx.unsqueeze(-1) + k_idx.unsqueeze(0)  # (L_out, k)
    patches = x[:, :, idx]  # (N, C_in, L_out, k)
    return patches.permute(0, 2, 1, 3).contiguous().reshape(N, L_out, C_in * k)


def _conv2d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv2d input (N, C_in, H, W) -> (N, L_out, C_in*kH*kW)."""
    unf = F.unfold(
        x,
        kwargs["kernel_size"],
        kwargs["dilation"],
        kwargs["padding"],
        kwargs["stride"],
    )
    return unf.permute(0, 2, 1).contiguous()  # (N, L, C_in*kH*kW)


def _conv3d_im2col(x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Unfold a raw Conv3d input (N, C_in, D, H, W) -> (N, L_out, C_in*kD*kH*kW)."""
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
    d_idx = (
        torch.arange(D_out, device=x.device).unsqueeze(-1) * sD
        + torch.arange(kD, device=x.device).unsqueeze(0) * dD
    )
    h_idx = (
        torch.arange(H_out, device=x.device).unsqueeze(-1) * sH
        + torch.arange(kH, device=x.device).unsqueeze(0) * dH
    )
    w_idx = (
        torch.arange(W_out, device=x.device).unsqueeze(-1) * sW
        + torch.arange(kW, device=x.device).unsqueeze(0) * dW
    )
    patches = x[
        :,
        :,
        d_idx.view(D_out, kD, 1, 1, 1, 1),
        h_idx.view(1, 1, H_out, kH, 1, 1),
        w_idx.view(1, 1, 1, 1, W_out, kW),
    ]
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return patches.reshape(N, L_out, C_in * kD * kH * kW)


# Spatial-rank -> im2col helper.  Keyed by the trailing digit of the layer-type
# string ("nn.Conv2d" / "nn.ConvTranspose2d" -> 2).
_CONV_IM2COL = {1: _conv1d_im2col, 2: _conv2d_im2col, 3: _conv3d_im2col}


def _conv_spatial_rank(layer_type: str) -> int:
    """Return the spatial rank (1/2/3) encoded in a Conv layer-type string."""
    return int(layer_type[-2])


def _preprocess_embedding_bag(
    a: torch.Tensor,
    g: torch.Tensor,
    mode: str,
    padding_idx: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-bag EmbeddingBag gradients to the per-token Embedding form.

    ``a`` are token indices ``(N, T)`` and ``g`` is the per-bag output gradient
    ``(N, E)``.  Each bag's gradient is broadcast to its ``T`` tokens (scaled by
    ``1/T`` for ``mode='mean'``), producing ``(N, T, E)`` so the standard
    embedding scatter-add materialisation applies.

    With a *padding_idx*, pad tokens are excluded from the bag reduction, so
    their positions receive zero gradient and the ``mode='mean'`` divisor is
    each bag's **non-pad** token count.

    Only 2-D inputs (one bag per row, no ``offsets``) are supported.
    ``mode='max'`` is not supported.
    """
    if mode == "max":
        raise NotImplementedError(
            "EmbeddingBag mode='max' is not supported for factorized gradients.",
        )
    if a.ndim != 2:
        raise NotImplementedError(
            "EmbeddingBag factorization supports only 2-D inputs (N, T) without "
            "offsets.",
        )
    t = a.shape[1]
    g_exp = g.unsqueeze(1).expand(-1, t, -1).contiguous()  # (N, T, E)
    if padding_idx is not None:
        pad = a == padding_idx  # (N, T)
        g_exp = g_exp.masked_fill(pad.unsqueeze(-1), 0)
        if mode == "mean":
            # Per-bag divisor over non-pad tokens; clamp keeps an all-pad
            # bag's 0/0 at zero gradient (its output is zero in autograd).
            counts = (~pad).sum(dim=1).clamp_min(1)  # (N,)
            g_exp /= counts.view(-1, 1, 1)
    elif mode == "mean":
        g_exp /= t
    return a, g_exp


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

        a in (N, L, C_in),   g in (N, L, C_out*prod(K)),

    where ``L`` is the number of input spatial locations and ``dW = sum_l a^Tg``.

    Bias cannot be folded by a single ones column because ``db`` has dimension
    ``C_out`` (not ``C_out*prod(K)``).  Following the factorized-gradient formulation
    we instead extend both factors with one extra location and feature block so
    the bias gradient is stored jointly:

        a_tilde in (N, L+1, C_in+1),   g_tilde in (N, L+1, C_out*prod(K) + C_out).
    """
    rank = _conv_spatial_rank(layer_type)
    n, c_in = a.shape[:2]
    a_flat = a.reshape(n, c_in, -1).permute(0, 2, 1).contiguous()  # (N, L, C_in)
    g_unf = _CONV_IM2COL[rank](g, module_kwargs)  # (N, L, C_out*prod(K))
    if not has_bias:
        return a_flat, g_unf

    c_out = g.shape[1]
    L = a_flat.shape[1]
    bias_grad = g.reshape(n, c_out, -1).sum(dim=2)  # (N, C_out)

    # Activation: zero-pad the feature dim, then append a bias location whose
    # only non-zero feature is the appended one.
    a_pad = torch.cat([a_flat, torch.zeros_like(a_flat[..., :1])], dim=-1)
    a_bias = torch.zeros(n, 1, a_pad.shape[-1], dtype=a_pad.dtype, device=a_pad.device)
    a_bias[..., -1] = 1.0
    a_aug = torch.cat([a_pad, a_bias], dim=1)  # (N, L+1, C_in+1)

    # Gradient: zero-pad the feature dim with C_out columns, then append a bias
    # location carrying the bias gradient in those columns.
    g_pad = torch.cat(
        [g_unf, torch.zeros(n, L, c_out, dtype=g_unf.dtype, device=g_unf.device)],
        dim=-1,
    )
    g_bias = torch.zeros(n, 1, g_pad.shape[-1], dtype=g_pad.dtype, device=g_pad.device)
    g_bias[..., -c_out:] = bias_grad.unsqueeze(1)
    g_aug = torch.cat([g_pad, g_bias], dim=1)  # (N, L+1, P+C_out)
    return a_aug, g_aug


def extract_module_kwargs(module: nn.Module, layer_type: str) -> dict:
    """Extract the minimal hyperparameters from *module* needed by
    :func:`_preprocess_factorized`.

    Returns a plain serialisable dict -- no reference to the module object is
    retained, so the result can be pickled cheaply alongside gradient tensors.

    Args:
        module: The hooked ``nn.Module`` instance.
        layer_type: Canonical class name (e.g. ``"nn.Conv2d"``).

    Returns:
        dict: A dict with at least ``"has_bias": bool``.  Conv and ConvTranspose
        layers additionally include ``"kernel_size"``, ``"stride"``,
        ``"padding"``, and ``"dilation"``.  LayerNorm and RMSNorm include
        ``"normalized_shape"`` and ``"eps"``.  GroupNorm includes
        ``"num_groups"``, ``"num_channels"``, and ``"eps"``.  InstanceNorm
        includes ``"num_features"`` and ``"eps"``.  Embedding includes
        ``"padding_idx"``.  EmbeddingBag includes ``"mode"`` and
        ``"padding_idx"``.

    Raises:
        NotImplementedError: If an embedding layer sets
            ``scale_grad_by_freq=True`` (its inverse-frequency gradient
            scaling is a whole-batch statistic the factorized capture does
            not model).
    """
    kwargs: dict = {"has_bias": getattr(module, "bias", None) is not None}
    if layer_type in ("nn.Embedding", "nn.EmbeddingBag") and getattr(
        module,
        "scale_grad_by_freq",
        False,
    ):
        # TODO: supportable in principle -- the inverse-frequency scaling is a
        # whole-forward-call statistic, so it must be applied to ``g`` at
        # capture/assembly time (per buffered part, while the call's full id
        # tensor is intact); preprocess-time counting is wrong once records
        # are sliced or concatenated.  No modern architecture sets the flag
        # (0 of 665 nn.Embedding sites in transformers 4.55), so refuse until
        # someone needs it.
        raise NotImplementedError(
            f"{layer_type} with scale_grad_by_freq=True is not supported: the "
            "inverse-frequency gradient scaling is a whole-batch statistic "
            "that the factorized capture does not model. Disable the flag or "
            "exclude this layer from hooking.",
        )
    if layer_type == "nn.Embedding":
        kwargs["num_embeddings"] = module.num_embeddings
        # Already canonical: nn.Embedding.__init__ normalises a negative
        # padding_idx to its non-negative form before storing the attribute.
        kwargs["padding_idx"] = module.padding_idx
    elif layer_type in CONV_TYPES or layer_type in CONV_TRANSPOSE_TYPES:
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
        kwargs["num_embeddings"] = module.num_embeddings
        kwargs["mode"] = module.mode
        kwargs["padding_idx"] = module.padding_idx
    return kwargs


def _preprocess_factorized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw hook captures into the (a, g) form expected by ops.

    Hooks store exactly what PyTorch delivers (raw ``input[0]`` and
    ``grad_output[0]``).  Before the factorised data can be passed to the ops
    functions in this module, layer-specific transforms are applied:

    * **LayerNorm / RMSNorm**: normalise ``a`` -> ``x_hat`` (RMSNorm omits the mean
      subtraction).  With bias, append a ones block to ``a`` and duplicate ``g``
      so ``a * g = [dgamma, dbeta]`` per position.
    * **GroupNorm / InstanceNorm**: normalise per group/channel -> ``x_hat``, then
      lay out spatial locations as the token dimension ``(N, S, C)``.  Bias is
      folded as for LayerNorm but in the channel dimension.
    * **Linear types with bias**: append a ones column to ``a`` so the last
      column of ``dW_tilde = g a_tilde^T`` recovers ``db``.
    * **Conv1d / Conv2d / Conv3d**: run im2col on ``a`` to get the unfolded
      ``(N, L, patch_size)`` form; reshape ``g`` to ``(N, L, C_out)``; append a
      ones column to ``a`` if the layer has a bias.
    * **ConvTranspose1d / 2d / 3d**: flatten ``a`` over spatial locations and
      unfold ``g`` into kernel patches (roles reversed vs. Conv).  Bias is
      folded via an extra location/feature block (see
      :func:`_preprocess_conv_transpose`).
    * **EmbeddingBag**: broadcast per-bag gradients to per-token gradients so
      the standard embedding scatter-add applies; with a ``padding_idx``, pad
      positions get zero gradient and the ``mean`` divisor is the per-bag
      non-pad token count.
    * **Embedding**: when the layer has a ``padding_idx``, zero ``g`` at the
      padded positions -- autograd's embedding backward skips those positions
      when scattering into ``weight.grad`` (the pad row's gradient is always
      zero), and the reconstruction must match.  Without a ``padding_idx``
      the factors pass through unchanged.

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

    has_bias = module_kwargs["has_bias"] and include_bias

    # -- Normalisation layers ------------------------------------------------
    if layer_type == "nn.LayerNorm":
        normalized_shape = module_kwargs.get("normalized_shape", (a.shape[-1],))
        x_hat = _compute_layer_norm_x_hat(
            a,
            normalized_shape,
            module_kwargs["eps"],
        )
        x_hat, g = _fold_broadcast_axes(x_hat, g, normalized_shape)
        return _augment_token_norm(x_hat, g, has_bias)

    if layer_type == "nn.RMSNorm":
        normalized_shape = module_kwargs.get("normalized_shape", (a.shape[-1],))
        x_hat = _compute_rms_x_hat(
            a,
            normalized_shape,
            module_kwargs["eps"],  # explicit None selects the machine epsilon
        )
        x_hat, g = _fold_broadcast_axes(x_hat, g, normalized_shape)
        return _augment_token_norm(x_hat, g, has_bias)

    if layer_type == "nn.GroupNorm":
        x_hat = _compute_group_norm_x_hat(
            a,
            module_kwargs["num_groups"],
            module_kwargs["eps"],
        )
        return _augment_channel_norm(x_hat, g, has_bias)

    if layer_type in ("nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"):
        # InstanceNorm = GroupNorm with one group per channel.
        x_hat = _compute_group_norm_x_hat(
            a,
            module_kwargs["num_features"],
            module_kwargs["eps"],
        )
        return _augment_channel_norm(x_hat, g, has_bias)

    # -- Embedding -------------------------------------------------------------
    if layer_type == "nn.Embedding":
        padding_idx = module_kwargs["padding_idx"]
        if padding_idx is not None:
            g = g.masked_fill((a == padding_idx).unsqueeze(-1), 0)
        return a, g

    # -- Embedding bag -------------------------------------------------------
    if layer_type == "nn.EmbeddingBag":
        return _preprocess_embedding_bag(
            a,
            g,
            module_kwargs["mode"],
            module_kwargs["padding_idx"],
        )

    # -- Convolution ---------------------------------------------------------
    if layer_type in CONV_TYPES:
        rank = _conv_spatial_rank(layer_type)
        a_unf = _CONV_IM2COL[rank](a, module_kwargs)  # (N, L, C_in*prod(K))
        n, c_out = g.shape[0], g.shape[1]
        g_out = g.reshape(n, c_out, -1).permute(0, 2, 1).contiguous()  # (N, L, C_out)
        if has_bias:
            a_unf = torch.cat([a_unf, torch.ones_like(a_unf[..., :1])], dim=-1)
        return a_unf, g_out

    if layer_type in CONV_TRANSPOSE_TYPES:
        return _preprocess_conv_transpose(a, g, layer_type, module_kwargs, has_bias)

    # -- Linear with bias ----------------------------------------------------
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
# Public Factorized-input API
# ---------------------------------------------------------------------------
#
# The private ``_``-prefixed functions above operate on raw
# ``(activation, pre_activation_grad)`` tensors and **always assume batch-first**
# ``(B, T, ...)`` input.  The public functions below take the
# :class:`~dattri_llm.gradient.gradient.Factorized` container instead: they call
# :meth:`Factorized.as_batch_first` to normalise a sequence-first capture, unpack
# ``module_kwargs``, and delegate to the private version.  These are the entry
# points to prefer -- the call is both shorter and automatically correct for
# non-batch-first layers, so nothing downstream needs to reason about tensor
# layout.  Reach for the raw ``_``-prefixed kernels only when you already hold
# bare, batch-first factor tensors (e.g. inside another kernel).


def preprocess_factorized(
    f: Factorized,
    layer_type: str,
    include_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_preprocess_factorized` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _preprocess_factorized(
        bf.activation,
        bf.pre_activation_grad,
        layer_type,
        bf.module_kwargs,
        include_bias,
    )
