"""HuggingFace Transformers Trainer integration for gradient collection.

Subclasses ``transformers.Trainer`` and injects a :class:`GradientCollector`
into ``training_step()`` so that MLP activations and output gradients are
captured and persisted after each optimiser step.

Gradient files are written to ``<output_dir>/gradients/step_{global_step}.pt``
as a ``dict[layer_name, {"activation": Tensor(B, T, in), "grad_output": Tensor(B, T, out)}]``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
import torch.nn as nn

try:
    from transformers import Trainer
    from transformers.trainer_utils import EvalLoopOutput
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for this module.  "
        "Install with: pip install transformers"
    ) from exc

from dattri_llm.gradient.collector import GradientCollector

GradientCollectionMode = Literal["mlp_io", "full"]


class GradientCollectingTrainer(Trainer):
    """A ``transformers.Trainer`` subclass that collects per-sample gradients.

    During each training step the forward+backward pass is bracketed by a
    :class:`~dattri_llm.gradient.collector.GradientCollector` context.
    Collected gradients are saved to ``<output_dir>/gradients/`` after every
    step.

    Args:
        gradient_collection_mode: ``"mlp_io"`` (default) collects MLP
            activations + output gradients (memory-efficient approximation).
            ``"full"`` saves the full ``.grad`` tensors for every parameter.
        gradient_save_dir: Override the save directory.  Defaults to
            ``<output_dir>/gradients``.
        mlp_name_patterns: Optional list of regex strings forwarded to
            :class:`~dattri_llm.gradient.collector.GradientCollector` to
            restrict which layers are hooked.
        **kwargs: Forwarded verbatim to ``transformers.Trainer.__init__``.
    """

    def __init__(
        self,
        *args: Any,
        gradient_collection_mode: GradientCollectionMode = "mlp_io",
        gradient_save_dir: str | None = None,
        mlp_name_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_collection_mode: GradientCollectionMode = gradient_collection_mode
        self._gradient_save_dir = gradient_save_dir
        self._mlp_name_patterns = mlp_name_patterns
        self._collector: GradientCollector | None = None

    # ---------------------------------------------------------------------- #
    # Collector lifecycle                                                       #
    # ---------------------------------------------------------------------- #

    def _ensure_collector(self) -> None:
        """Lazily initialise the collector once the model is on device."""
        if self._collector is None and self.gradient_collection_mode == "mlp_io":
            self._collector = GradientCollector(
                self.model,
                name_patterns=self._mlp_name_patterns,
            )

    @property
    def gradient_save_dir(self) -> str:
        """Resolved path to the directory where gradient files are written."""
        if self._gradient_save_dir is not None:
            return self._gradient_save_dir
        return os.path.join(self.args.output_dir, "gradients")

    # ---------------------------------------------------------------------- #
    # Override: training_step                                                   #
    # ---------------------------------------------------------------------- #

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Run one forward+backward step and collect per-sample gradients.

        In ``"mlp_io"`` mode the pass is bracketed by a
        :class:`~dattri_llm.gradient.collector.GradientCollector` context so
        that MLP activations and output gradients are buffered.  Gradients are
        then saved to disk before returning.

        In ``"full"`` mode the standard Trainer ``training_step`` runs
        unmodified, and afterwards the ``.grad`` attribute of every parameter
        is snapshot and saved.

        Args:
            model: The model (already moved to device by Trainer).
            inputs: Batch dict returned by the data collator.
            num_items_in_batch: Passed through to the base class when present.

        Returns:
            The scalar loss tensor (detached) as returned by the base class.
        """
        self._ensure_collector()

        if self.gradient_collection_mode == "mlp_io":
            return self._training_step_mlp_io(model, inputs, num_items_in_batch)
        else:
            return self._training_step_full(model, inputs, num_items_in_batch)

    def _training_step_mlp_io(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        """Training step with MLP activation + grad-output collection."""
        assert self._collector is not None

        with self._collector:
            if num_items_in_batch is not None:
                loss = super().training_step(model, inputs, num_items_in_batch)
            else:
                loss = super().training_step(model, inputs)

        # Persist collected activations and output gradients.
        try:
            ag = self._collector.get_activations_and_grad_outputs()
            self._save_gradients(ag, inputs=inputs)
        except RuntimeError:
            # Buffers empty — backward may not have been called (e.g. eval).
            pass

        return loss

    def _training_step_full(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        """Training step that snapshots full parameter .grad tensors."""
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        full_grads: dict[str, torch.Tensor] = {}
        root: nn.Module = getattr(model, "module", model)
        for name, param in root.named_parameters():
            if param.grad is not None:
                full_grads[name] = param.grad.detach().clone()

        self._save_gradients(full_grads, inputs=inputs)
        return loss

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def _save_gradients(self, gradients: dict, inputs: dict[str, Any] | None = None) -> None:
        """Write a gradient snapshot to disk.

        Files are named ``step_{global_step}.pt`` inside
        :attr:`gradient_save_dir`.

        The saved dict always contains the gradient payload (keyed by layer
        name or parameter name).  When *inputs* contains ``"input_ids"``, a
        ``"_sample_ids"`` key is added holding the token IDs on CPU — a
        per-sample identifier that can be used to reconstruct exactly which
        dataset rows were seen at this step.

        In ``"mlp_io"`` mode the payload is a nested dict::

            {layer_name: {"activation": Tensor(B, T, in), "grad_output": Tensor(B, T, out)}}

        In ``"full"`` mode it is a flat dict of parameter-name → grad tensor.

        Args:
            gradients: Snapshot to persist.
            inputs: The raw batch dict passed to ``training_step``.  When
                provided and ``"input_ids"`` is present, token IDs are saved
                under ``"_sample_ids"`` for downstream traceability.
        """
        payload: dict[str, Any] = dict(gradients)
        if inputs is not None and "input_ids" in inputs:
            payload["_sample_ids"] = inputs["input_ids"].detach().cpu()

        os.makedirs(self.gradient_save_dir, exist_ok=True)
        step = self.state.global_step
        path = os.path.join(self.gradient_save_dir, f"step_{step}.pt")
        torch.save(payload, path)

    # ---------------------------------------------------------------------- #
    # Clean teardown                                                           #
    # ---------------------------------------------------------------------- #

    def __del__(self) -> None:
        if self._collector is not None:
            try:
                self._collector.remove()
            except Exception:
                pass
