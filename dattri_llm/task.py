"""The attribution task: a model, the loss to attribute, and the checkpoints.

:class:`AttributionTask` is what the live attribution methods take (the
``task=`` of every attributor).  It follows the shape of dattri's task --
``AttributionTask(loss_func, model, checkpoints, target_func)`` -- with one
difference that matters at LLM scale: the loss runs the **live model**,

.. code-block:: python

    def loss_func(model, batch):
        return model(**batch).loss

    task = AttributionTask(loss_func, model)                 # this checkpoint
    task = AttributionTask(loss_func, model, checkpoints=[ckpt_a, ckpt_b])

rather than a ``torch.func`` functional forward over a parameter dict.  A
plain ``model(...)`` call is what the capture hooks need (they read each
layer's inputs and output gradients, never the parameter gradient), it costs
nothing per step, and it is the only forward that works through a DDP or
FSDP wrapper: ``functional_call`` swaps the parameters underneath the
wrapper, so it cannot run a sharded model.  ``model`` may therefore be the
wrapper itself -- the task keeps it as :attr:`forward_model` and exposes the
wrapped module as :attr:`model` for the hooks.

A dattri task still works everywhere a task is accepted:
:meth:`AttributionTask.from_dattri` adapts it (its ``(params, data)`` loss
runs through ``functional_call`` on the live parameters, its checkpoint
loader is kept), and the attributors call it for you.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


def default_loss_func(model: nn.Module, batch: object) -> torch.Tensor:
    """The Hugging Face convention, ``model(**batch).loss``."""
    if not isinstance(batch, dict):
        raise TypeError(
            "default_loss_func expects a dict batch (model(**batch).loss); "
            "pass a loss_func for other batch formats.",
        )
    out = model(**batch)
    loss = getattr(out, "loss", None)
    if loss is None:
        raise ValueError(
            "model(**batch) returned no .loss; pass a loss_func that computes "
            "the loss from the model output (e.g. with labels).",
        )
    return loss


def _unwrap(model: nn.Module) -> nn.Module:
    """The module under a DDP/FSDP wrapper (``model`` itself when unwrapped)."""
    from torch.nn.parallel import DistributedDataParallel

    wrappers: tuple[type, ...] = (DistributedDataParallel,)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        wrappers = (*wrappers, FullyShardedDataParallel)
    except ImportError:  # pragma: no cover - FSDP ships with every CUDA torch
        pass
    return model.module if isinstance(model, wrappers) else model


def default_checkpoint_load_func(model: nn.Module, checkpoint: object) -> nn.Module:
    """Load *checkpoint* -- a state dict, or a path ``torch.load`` reads --
    into *model* with ``load_state_dict``; ``None`` leaves the model as it is.
    """
    if checkpoint is None:
        return model
    if isinstance(checkpoint, (str, Path)):
        device = next(model.parameters()).device
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    return model


class AttributionTask:
    """A model, the loss to attribute (and optionally a separate target), and
    the checkpoints to attribute at.

    Args:
        loss_func: ``(model, batch) -> scalar``: the training loss, evaluated
            by calling the live model on a loader batch.  Its gradient is the
            **train-side** representation.  ``None`` uses
            :func:`default_loss_func` (``model(**batch).loss``).
        model: The model, as it is -- unwrapped, or already wrapped in DDP or
            FSDP by the caller (a trainer, or one wrapper built for several
            passes).  The hooks go on the wrapped module; forward and backward
            run on the wrapper.
        checkpoints: The parameters to attribute at: one checkpoint or a list
            (an ensemble), each a state dict, a path, or ``None`` for the
            parameters the model holds now.  ``None`` (default) is that single
            checkpoint, which needs no loading -- the usual case for a model
            that arrived trained, and the only one a wrapped model supports
            with the default loader.
        target_func: ``(model, batch) -> scalar`` for the **test side** (the
            quantity to explain); ``None`` uses ``loss_func``.
        checkpoints_load_func: ``(model, checkpoint) -> model`` replacing
            :func:`default_checkpoint_load_func` -- e.g. one that loads a
            sharded checkpoint into an FSDP model, or rebuilds a Hugging Face
            model from a directory.  It receives :attr:`model` (the wrapped
            module) and its return value replaces it.
    """

    def __init__(
        self,
        loss_func: Callable[[nn.Module, object], torch.Tensor] | None,
        model: nn.Module,
        checkpoints: object = None,
        target_func: Callable[[nn.Module, object], torch.Tensor] | None = None,
        checkpoints_load_func: Callable[[nn.Module, object], nn.Module] | None = None,
    ) -> None:
        self.loss_func = loss_func if loss_func is not None else default_loss_func
        self.target_func = target_func if target_func is not None else self.loss_func
        self._forward_model = model
        self._model = _unwrap(model)
        self.checkpoints: list = (
            list(checkpoints)
            if isinstance(checkpoints, (list, tuple))
            else [checkpoints]
        )
        if not self.checkpoints:
            raise ValueError("checkpoints must hold at least one checkpoint.")
        self._load = (
            checkpoints_load_func
            if checkpoints_load_func is not None
            else default_checkpoint_load_func
        )
        if (
            self._load is default_checkpoint_load_func
            and self.is_wrapped
            and any(c is not None for c in self.checkpoints)
        ):
            raise ValueError(
                "The default checkpoint loader cannot load a state dict "
                "into a DDP/FSDP-wrapped model; pass checkpoints=None to "
                "attribute at the parameters it holds, or a "
                "checkpoints_load_func that loads through the wrapper.",
            )
        self.current_checkpoint_idx: int | None = None

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> nn.Module:
        """The (unwrapped) module the capture hooks are registered on."""
        return self._model

    @property
    def forward_model(self) -> nn.Module | None:
        """The DDP/FSDP wrapper forward/backward run on, when ``model`` came
        wrapped; ``None`` otherwise (the streamer then wraps per its
        arguments, or runs the model directly).
        """
        return self._forward_model if self.is_wrapped else None

    @property
    def is_wrapped(self) -> bool:
        """Whether the model was handed over inside a DDP/FSDP wrapper."""
        return self._forward_model is not self._model

    def get_model(self) -> nn.Module:
        """The (unwrapped) model; see :attr:`model`."""
        return self._model

    def get_checkpoints(self) -> list:
        """The checkpoints, in ensemble order."""
        return self.checkpoints

    def num_checkpoints(self) -> int:
        """How many checkpoints the ensemble has."""
        return len(self.checkpoints)

    def load_checkpoint(self, index: int) -> nn.Module:
        """Load the *index*-th checkpoint (a no-op when it is already loaded,
        or is ``None``) and return :attr:`model`.
        """
        if self.current_checkpoint_idx != index:
            checkpoint = self.checkpoints[index]
            loaded = self._load(self._model, checkpoint)
            if loaded is not self._model:
                # A loader that rebuilt the model (e.g. from_pretrained).
                if self.is_wrapped:
                    raise ValueError(
                        "checkpoints_load_func returned a new module for a "
                        "wrapped model; it must load in place through the "
                        "wrapper and return the same module.",
                    )
                self._model = self._forward_model = loaded
            self.current_checkpoint_idx = index
        return self._model

    # ------------------------------------------------------------------ #
    # dattri interoperability                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dattri(cls, task: object) -> AttributionTask:
        """Adapt a ``dattri.task.AttributionTask``.

        Its ``(params, data)`` loss and target are called with the live
        model's ``named_parameters()`` -- they run the ``functional_call``
        forward they define -- and its checkpoint loader is kept.  Such a
        task cannot drive a wrapped model (see the module docstring).
        """
        if isinstance(task, cls):
            return task
        for attr in (
            "original_loss_func",
            "original_target_func",
            "model",
            "checkpoints",
        ):
            if not hasattr(task, attr):
                raise TypeError(
                    f"from_dattri expects a dattri AttributionTask (missing {attr!r}).",
                )
        return cls(
            _functional(task.original_loss_func),  # type: ignore[attr-defined]
            task.model,  # type: ignore[attr-defined]
            checkpoints=list(task.checkpoints),  # type: ignore[attr-defined]
            target_func=_functional(task.original_target_func),  # type: ignore[attr-defined]
            checkpoints_load_func=task.checkpoints_load_func,  # type: ignore[attr-defined]
        )


def _functional(func: Callable) -> Callable[[nn.Module, object], torch.Tensor]:
    """``(params, data) -> loss`` (functorch style) as ``(model, batch) -> loss``."""

    def loss_fn(model: nn.Module, batch: object) -> torch.Tensor:
        return func(dict(model.named_parameters()), batch)

    return loss_fn


def as_task(task: object) -> AttributionTask:
    """*task* as an :class:`AttributionTask`: itself, or a dattri task adapted
    through :meth:`AttributionTask.from_dattri`.
    """
    return AttributionTask.from_dattri(task)
