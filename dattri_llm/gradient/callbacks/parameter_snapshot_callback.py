"""Callback that snapshots the parameters every capture step runs from."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dattri_llm.gradient.callbacks.base import HookManagerCallback

if TYPE_CHECKING:
    from torch import nn

    from dattri_llm.gradient.gradient import GradientRecord
    from dattri_llm.gradient.snapshots import TrajectorySnapshots


class ParameterSnapshotCallback(HookManagerCallback):
    """Store the trainable parameters at every capture step.

    ``on_step_end`` fires inside the backward pass, before the training loop
    calls ``optimizer.step()``, so what it stores are the parameters the
    step's gradients were taken at.  Together with the step's batch this is
    what :class:`~dattri_llm.gradient.streaming.ReplayGradientSource` needs
    to recompute the step's per-sample gradients later, instead of storing
    them now.

    The batch is not part of the record, so a training loop the library
    does not drive stores it itself, with
    :meth:`TrajectorySnapshots.save_batch` and the same step index the
    manager stamps on the record (its per-collection step counter).  The
    :class:`~dattri_llm.gradient.streaming.GradientStreamer` does both
    when handed a snapshot store.

    Args:
        model: The model the hooks are registered on.
        snapshots: The snapshot store to write to.
    """

    def __init__(self, model: nn.Module, snapshots: TrajectorySnapshots) -> None:
        self._model = model
        self.snapshots = snapshots

    def on_step_end(self, record: GradientRecord) -> None:
        """Store the parameters the record's step ran from."""
        self.snapshots.save_parameters(record.step, self._model)
