"""Builders for the per-layer ``module_kwargs`` hyperparameter dicts.

``extract_module_kwargs`` (gradient layer) normally reads a hooked layer's
hyperparameters straight off the module.  When a layer's class stores them
under non-standard attribute names -- e.g. an HF ``LlamaRMSNorm``, whose
epsilon lives in ``variance_epsilon`` -- the user supplies the dict manually
through ``HookManagerConfig(module_kwargs={...})`` together with a
``layer_types`` declaration.  The helpers here build those dicts with explicit,
validated arguments so the expected keys never have to be remembered::

    from dattri_llm.utils.module import rms_norm_module_kwargs

    HookManagerConfig(
        hook_types={"model.norm": "linear_io"},
        layer_types={"model.norm": "nn.RMSNorm"},
        module_kwargs={"model.norm": rms_norm_module_kwargs(
            normalized_shape=4096, eps=1e-6,
        )},
    )

One helper exists per accepted layer type.  ``linear_module_kwargs`` also
covers ``nn.NonDynamicallyQuantizableLinear`` and
``transformers.pytorch_utils.Conv1D``, whose dicts are identical to
``nn.Linear``'s.  Convolution arguments accept an int or a per-dimension
sequence, mirroring the corresponding ``torch.nn`` constructors.

Every argument is keyword-only and **required**: these helpers describe
non-standard layers, exactly the situation where silently assumed defaults
(a bias that is not there, a different epsilon) would corrupt the captured
gradients without any error.  Forgetting a field fails loudly at config-build
time instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_EMBEDDING_BAG_MODES = ("sum", "mean")


def _to_tuple(value: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    """Normalise an int or per-dimension sequence to a length-*rank* tuple."""
    if isinstance(value, int):
        return (value,) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise ValueError(
            f"{name} must be an int or a length-{rank} sequence, got {value!r}.",
        )
    return result


def _conv_module_kwargs(
    rank: int,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    return {
        "has_bias": has_bias,
        "kernel_size": _to_tuple(kernel_size, rank, "kernel_size"),
        "stride": _to_tuple(stride, rank, "stride"),
        "padding": _to_tuple(padding, rank, "padding"),
        "dilation": _to_tuple(dilation, rank, "dilation"),
    }


# ---------------------------------------------------------------------------
# Linear family and embeddings
# ---------------------------------------------------------------------------


def linear_module_kwargs(*, has_bias: bool) -> dict:
    """Module kwargs for ``nn.Linear`` (also
    ``nn.NonDynamicallyQuantizableLinear`` and
    ``transformers.pytorch_utils.Conv1D``).

    Args:
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return {"has_bias": has_bias}


def bilinear_module_kwargs(*, has_bias: bool) -> dict:
    """Module kwargs for ``nn.Bilinear``.

    Args:
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return {"has_bias": has_bias}


def embedding_module_kwargs() -> dict:
    """Module kwargs for ``nn.Embedding`` (embeddings never carry a bias)."""
    return {"has_bias": False}


def embedding_bag_module_kwargs(*, mode: str) -> dict:
    """Module kwargs for ``nn.EmbeddingBag``.

    Args:
        mode: The bag reduction, ``"sum"`` or ``"mean"`` (``"max"`` is not
            supported for factorized gradients).

    Returns:
        The ``module_kwargs`` dict for the layer.

    Raises:
        ValueError: If *mode* is not a supported reduction.
    """
    if mode not in _EMBEDDING_BAG_MODES:
        raise ValueError(
            f"mode must be one of {_EMBEDDING_BAG_MODES}, got {mode!r}.",
        )
    return {"has_bias": False, "mode": mode}


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------


def conv1d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.Conv1d``.

    Args:
        kernel_size: Kernel size (int or length-1 sequence).
        stride: Stride (int or length-1 sequence).
        padding: Zero-padding (int or length-1 sequence).
        dilation: Dilation (int or length-1 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(1, kernel_size, stride, padding, dilation, has_bias)


def conv2d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.Conv2d``.

    Args:
        kernel_size: Kernel size (int or length-2 sequence).
        stride: Stride (int or length-2 sequence).
        padding: Zero-padding (int or length-2 sequence).
        dilation: Dilation (int or length-2 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(2, kernel_size, stride, padding, dilation, has_bias)


def conv3d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.Conv3d``.

    Args:
        kernel_size: Kernel size (int or length-3 sequence).
        stride: Stride (int or length-3 sequence).
        padding: Zero-padding (int or length-3 sequence).
        dilation: Dilation (int or length-3 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(3, kernel_size, stride, padding, dilation, has_bias)


def conv_transpose1d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.ConvTranspose1d``.

    Args:
        kernel_size: Kernel size (int or length-1 sequence).
        stride: Stride (int or length-1 sequence).
        padding: Zero-padding (int or length-1 sequence).
        dilation: Dilation (int or length-1 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(1, kernel_size, stride, padding, dilation, has_bias)


def conv_transpose2d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.ConvTranspose2d``.

    Args:
        kernel_size: Kernel size (int or length-2 sequence).
        stride: Stride (int or length-2 sequence).
        padding: Zero-padding (int or length-2 sequence).
        dilation: Dilation (int or length-2 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(2, kernel_size, stride, padding, dilation, has_bias)


def conv_transpose3d_module_kwargs(
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.ConvTranspose3d``.

    Args:
        kernel_size: Kernel size (int or length-3 sequence).
        stride: Stride (int or length-3 sequence).
        padding: Zero-padding (int or length-3 sequence).
        dilation: Dilation (int or length-3 sequence).
        has_bias: Whether the layer has a bias parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _conv_module_kwargs(3, kernel_size, stride, padding, dilation, has_bias)


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------


def layer_norm_module_kwargs(
    *,
    normalized_shape: int | Sequence[int],
    eps: float,
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.LayerNorm``.

    Args:
        normalized_shape: The trailing shape normalised over (int or sequence),
            as in the ``nn.LayerNorm`` constructor.
        eps: Numerical-stability epsilon.
        has_bias: Whether the layer has a bias (beta) parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    return {
        "has_bias": has_bias,
        "normalized_shape": tuple(normalized_shape),
        "eps": eps,
    }


def rms_norm_module_kwargs(
    *,
    normalized_shape: int | Sequence[int],
    eps: float | None,
) -> dict:
    """Module kwargs for ``nn.RMSNorm`` (and hand-rolled RMSNorm classes
    declared as such, e.g. HF ``LlamaRMSNorm`` -- pass its
    ``variance_epsilon`` as *eps*).

    Args:
        normalized_shape: The trailing shape normalised over (int or sequence),
            as in the ``nn.RMSNorm`` constructor.
        eps: Numerical-stability epsilon; pass ``None`` explicitly to use the
            dtype's machine epsilon, matching ``nn.RMSNorm``'s default.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    return {
        "has_bias": False,
        "normalized_shape": tuple(normalized_shape),
        "eps": eps,
    }


def group_norm_module_kwargs(
    *,
    num_groups: int,
    num_channels: int,
    eps: float,
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.GroupNorm``.

    Args:
        num_groups: Number of channel groups normalised jointly.
        num_channels: Total number of channels.
        eps: Numerical-stability epsilon.
        has_bias: Whether the layer has a bias (beta) parameter.

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return {
        "has_bias": has_bias,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
    }


def _instance_norm_module_kwargs(num_features: int, eps: float, has_bias: bool) -> dict:
    return {"has_bias": has_bias, "num_features": num_features, "eps": eps}


def instance_norm1d_module_kwargs(
    *,
    num_features: int,
    eps: float,
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.InstanceNorm1d``.

    Args:
        num_features: Number of channels.
        eps: Numerical-stability epsilon.
        has_bias: Whether the layer has a bias (beta) parameter (requires
            ``affine=True`` on the module).

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _instance_norm_module_kwargs(num_features, eps, has_bias)


def instance_norm2d_module_kwargs(
    *,
    num_features: int,
    eps: float,
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.InstanceNorm2d``.

    Args:
        num_features: Number of channels.
        eps: Numerical-stability epsilon.
        has_bias: Whether the layer has a bias (beta) parameter (requires
            ``affine=True`` on the module).

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _instance_norm_module_kwargs(num_features, eps, has_bias)


def instance_norm3d_module_kwargs(
    *,
    num_features: int,
    eps: float,
    has_bias: bool,
) -> dict:
    """Module kwargs for ``nn.InstanceNorm3d``.

    Args:
        num_features: Number of channels.
        eps: Numerical-stability epsilon.
        has_bias: Whether the layer has a bias (beta) parameter (requires
            ``affine=True`` on the module).

    Returns:
        The ``module_kwargs`` dict for the layer.
    """
    return _instance_norm_module_kwargs(num_features, eps, has_bias)
