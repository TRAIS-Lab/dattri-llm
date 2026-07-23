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

    def on_layer_forward(
        self,
        layer_name: str,
        activation: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,
    ) -> None:
        """Called after each layer's forward hook captures the activation.

        Fires once per layer per forward pass.  ``activation`` is on the
        capture device -- the training device by default, CPU when the
        manager was built with ``offload_to_cpu=True`` -- and is the **raw**
        (un-projected) input, whatever projection style the store uses.

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            activation: Captured input activation, shape ``(B, *, in_features)``.
            layer_type: Canonical class name of the layer (e.g. ``"nn.Linear"``),
                as used by :mod:`dattri_llm.gradient.ops`.
            module_kwargs: The layer's serializable hyperparameters (from
                :func:`~dattri_llm.gradient.ops.extract_module_kwargs` -- e.g.
                ``has_bias``, conv stride/padding).  Together with ``layer_type``
                this is everything needed to build the layer's K-FAC covariance
                without re-inspecting the module.
        """

    def on_layer_backward(
        self,
        layer_name: str,
        grad_output: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,
    ) -> None:
        """Called after each layer's backward hook captures the gradient.

        ``grad_output`` is on the capture device (see :meth:`on_layer_forward`)
        and is the **raw** (un-projected) output gradient.  Fires once per layer
        per backward pass (once per replica under DataParallel).

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            grad_output: Captured output gradient, shape ``(B, *, out_features)``.
            layer_type: Canonical class name of the layer (see
                :meth:`on_layer_forward`).
            module_kwargs: The layer's serializable hyperparameters (see
                :meth:`on_layer_forward`).
        """

    def on_step_end(self, record: GradientRecord) -> None:
        """Called exactly once per batch step after the :class:`GradientRecord`
        is assembled.

        The record always contains the full-batch gradient (``input_hash`` is a
        list of B hashes).  Per-sample slicing is the callback's responsibility
        -- see :class:`OffloadCallback` for an example.

        Dispatch happens with the manager's per-step state already reset and
        its internal lock released, so re-entering the manager from here is
        supported -- including running a **secondary backward pass** through
        the hooked model.  The contract for doing so:

        * This method executes inside an autograd hook, where gradient mode
          is disabled -- wrap tracked work in ``torch.enable_grad()``.
        * The secondary pass completes a capture step of its own: every
          attached callback receives its record, so guard against re-entering
          on your own secondary record.
        * To leave the manager's training-facing state (step counter, last
          gradient/inputs) untouched, bracket the pass with
          :meth:`HookManager.save_state` / :meth:`HookManager.clear_state` /
          :meth:`HookManager.load_state`.
        * Under FSDP, step completion relies on an end-of-backward engine
          callback whose ordering for *nested* backwards is not guaranteed --
          reentrancy there is untested territory.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
        """

    def on_context_end(self) -> None:
        """Called when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect`
        context closes.

        Use this to flush any remaining staged records to disk.
        """
