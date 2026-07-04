"""Base class for HookManager callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from dattri_llm.gradient.gradient import GradientRecord


class HookManagerCallback:
    """Base class for :class:`~dattri_llm.gradient.hooks.HookManager` callbacks.

    All methods are no-ops by default; subclasses override only what they need.
    ``on_layer_forward`` and ``on_layer_backward`` are wired directly into
    PyTorch's hook system and fire regardless of trainer callback support.
    """

    def on_layer_forward(self, layer_name: str, activation: torch.Tensor) -> None:
        """Called after each layer's forward hook captures the activation.

        Fires once per layer per forward pass.  ``activation`` is on CPU.

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            activation: Captured input activation, shape ``(B, *, in_features)``.
        """

    def on_layer_backward(self, layer_name: str, grad_output: torch.Tensor) -> None:
        """Called after each layer's backward hook captures the gradient.

        ``grad_output`` is on CPU.  Fires once per layer per backward pass
        (once per replica under DataParallel).

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            grad_output: Captured output gradient, shape ``(B, *, out_features)``.
        """

    def on_step_end(self, record: GradientRecord) -> None:
        """Called exactly once per batch step after the :class:`GradientRecord`
        is assembled.

        The record always contains the full-batch gradient (``input_hash`` is a
        list of B hashes).  Per-sample slicing is the callback's responsibility
        -- see :class:`OffloadCallback` for an example.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
        """

    def on_context_end(self) -> None:
        """Called when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect`
        context closes.

        Use this to flush any remaining staged records to disk.
        """
