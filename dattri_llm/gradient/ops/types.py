"""Layer-type constants, predicates, and canonical class naming."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

# ---------------------------------------------------------------------------
# Layer type constants
# ---------------------------------------------------------------------------

LINEAR_TYPES = frozenset(
    {
        "nn.Linear",
        "nn.Bilinear",
        "nn.NonDynamicallyQuantizableLinear",
        "transformers.pytorch_utils.Conv1D",
    },
)
CONV_TYPES = frozenset({"nn.Conv1d", "nn.Conv2d", "nn.Conv3d"})
CONV_TRANSPOSE_TYPES = frozenset(
    {
        "nn.ConvTranspose1d",
        "nn.ConvTranspose2d",
        "nn.ConvTranspose3d",
    },
)
NORM_TYPES = frozenset(
    {
        "nn.LayerNorm",
        "nn.RMSNorm",
        "nn.GroupNorm",
        "nn.InstanceNorm1d",
        "nn.InstanceNorm2d",
        "nn.InstanceNorm3d",
    },
)
EMBEDDING_TYPES = frozenset({"nn.Embedding", "nn.EmbeddingBag"})

# Special marker for aggregated param-level gradients (no batch dim).
PARAM_GRAD_TYPES = "param_grad_layers"

ALL_LAYER_TYPES = (
    LINEAR_TYPES | CONV_TYPES | CONV_TRANSPOSE_TYPES | NORM_TYPES | EMBEDDING_TYPES
)


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
