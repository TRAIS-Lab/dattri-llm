"""GradientCollector: stateful wrapper for per-sample gradient collection.

The collector registers MLP hooks on a model, then provides a context manager
that brackets a single forward+backward pass.  After the pass, callers can
retrieve raw buffers or fully-expanded per-sample gradients.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn

from dattri_llm.gradient.hooks import register_mlp_hooks, remove_hooks


class GradientCollector:
    """Collect per-sample gradients from MLP layers via forward/backward hooks.

    Usage::

        collector = GradientCollector(model)
        with collector:
            loss = model(**batch).loss
            loss.backward()

        # Raw buffers: {layer_name: {"activation": Tensor, "grad_output": Tensor}}
        buffers = collector.get_gradients()

        # Per-sample outer-product gradients: {layer_name: Tensor(B, out, in)}
        per_sample = collector.get_per_sample_gradients()

        collector.remove()  # clean up all hooks

    The collector can also be used as a plain context manager multiple times;
    buffers are reset at the start of each ``__enter__``.

    DataParallel support
    --------------------
    When the model is wrapped in ``nn.DataParallel``, each replica fires the
    hooks in its own thread.  The internal accumulator lists (``_act_parts`` /
    ``_grad_parts``) capture all replica outputs; they are sorted by device
    index (matching DataParallel's scatter order) and concatenated along the
    batch dimension before the outer-product is computed.

    Args:
        model: The PyTorch model (plain, ``DataParallel``, or
            ``DistributedDataParallel`` wrapped).
        name_patterns: Optional list of regex strings to restrict which
            layers are hooked.  Defaults to the MLP-keyword heuristic in
            :mod:`dattri_llm.gradient.hooks`.
    """

    def __init__(
        self,
        model: nn.Module,
        name_patterns: list[str] | None = None,
    ) -> None:
        self._model = model
        self._buffers, self._handles = register_mlp_hooks(model, name_patterns)

    # ---------------------------------------------------------------------- #
    # Context manager interface                                                #
    # ---------------------------------------------------------------------- #

    def __enter__(self) -> "GradientCollector":
        # Reset every buffer slot so stale data is not returned.
        for buf in self._buffers.values():
            buf["activation"] = None
            buf["grad_output"] = None
            buf["_act_parts"] = []
            buf["_grad_parts"] = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Hooks stay registered; the collector is reusable across many steps.
        pass

    @contextmanager
    def collect(self) -> Generator["GradientCollector", None, None]:
        """Convenience context manager (alias for ``with collector:``)."""
        self.__enter__()
        try:
            yield self
        finally:
            self.__exit__(None, None, None)

    # ---------------------------------------------------------------------- #
    # Data access                                                              #
    # ---------------------------------------------------------------------- #

    def get_gradients(self) -> dict[str, dict]:
        """Return raw buffer dicts for all hooked layers.

        The ``"activation"`` and ``"grad_output"`` keys hold the
        *last-written* tensor (suitable for single-GPU / DDP use).  For
        DataParallel, use :meth:`get_per_sample_gradients` which correctly
        assembles all replica outputs.

        Returns:
            A dict mapping layer name to the layer buffer dict.
        """
        return self._buffers

    def get_per_sample_gradients(self) -> dict[str, torch.Tensor]:
        """Compute per-sample weight gradients via the outer-product identity.

        For each hooked layer the method assembles activation and grad-output
        tensors from all hook calls (one per DataParallel replica or one for
        single-GPU / DDP), then computes:

        .. code-block:: text

            grad[b] = g[b].T @ a[b]     →  shape (out_features, in_features)

        where the sequence-length dimension is summed over, matching the
        behaviour of ``loss.backward()`` which reduces over tokens.

        In DataParallel mode the replica outputs are sorted by source device
        index (= scatter order) and concatenated along the batch dimension,
        so the result always has shape ``(total_batch, out_features, in_features)``.

        Returns:
            A dict mapping layer name to a tensor of shape
            ``(batch_size, out_features, in_features)``.  All tensors are on
            CPU.

        Raises:
            RuntimeError: If a forward+backward pass has not been executed.
        """
        result: dict[str, torch.Tensor] = {}
        for layer_name, (a, g) in self._assemble_ag().items():
            B = a.shape[0]
            a_2d = a.reshape(B, -1, a.shape[-1])    # (B, T, in)
            g_2d = g.reshape(B, -1, g.shape[-1])    # (B, T, out)
            # Per-sample outer product summed over T: "bto,bti->boi"  →  (B, out, in)
            result[layer_name] = torch.einsum("bto,bti->boi", g_2d, a_2d)
        return result

    def get_activations_and_grad_outputs(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return assembled activations and output gradients without forming the outer product.

        This is more memory-efficient than :meth:`get_per_sample_gradients` and
        sufficient for comparing gradient signals across training setups.  The
        returned tensors can be used directly for downstream TDA computation or
        saved to disk for later processing.

        Returns:
            A dict mapping layer name to ``{"activation": Tensor, "grad_output": Tensor}``
            where both tensors have shape ``(batch_size, seq_len, features)`` and
            live on CPU.

        Raises:
            RuntimeError: If a forward+backward pass has not been executed.
        """
        return {
            name: {"activation": a, "grad_output": g}
            for name, (a, g) in self._assemble_ag().items()
        }

    def _assemble_ag(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Assemble and return ``(activation, grad_output)`` tensors per layer.

        Parts accumulated from DataParallel replicas are sorted by device index
        (= scatter order) then concatenated along the batch dimension.

        Returns:
            Dict mapping layer name to ``(a, g)`` where both tensors are on CPU
            with shape ``(B, *, features)``.

        Raises:
            RuntimeError: If no buffered data exists for a layer.
        """
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, buf in self._buffers.items():
            act_parts = buf["_act_parts"]
            grad_parts = buf["_grad_parts"]

            if not act_parts or not grad_parts:
                raise RuntimeError(
                    f"Layer '{layer_name}' has no buffered data. "
                    "Run a forward+backward pass inside the collector context first."
                )

            # Sort by device index so DP replicas appear in scatter order.
            a = torch.cat([t for _, t in sorted(act_parts,  key=lambda x: x[0])], dim=0)
            g = torch.cat([t for _, t in sorted(grad_parts, key=lambda x: x[0])], dim=0)
            result[layer_name] = (a, g)
        return result

    # ---------------------------------------------------------------------- #
    # Lifecycle                                                                #
    # ---------------------------------------------------------------------- #

    def remove(self) -> None:
        """Remove all registered hooks and clear the buffer dict."""
        remove_hooks(self._handles)
        self._buffers.clear()

    @property
    def layer_names(self) -> list[str]:
        """Names of all hooked layers."""
        return list(self._buffers.keys())
