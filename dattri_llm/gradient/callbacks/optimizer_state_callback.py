"""Callback that records the optimizer's moments at every capture step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.optimizer_state import OptimizerSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from dattri_llm.gradient.gradient import GradientRecord


class OptimizerStateCallback(HookManagerCallback):
    """Snapshot an Adam-family optimizer's moments on each layer's coordinates.

    ``on_step_end`` fires inside the backward pass, *before* the training loop
    calls ``optimizer.step()``, so what it reads are the moments the step is
    about to update from; :meth:`record_post` reads the same coordinates
    again once the step has run.  Trajectory methods (AdamW-influence) need
    both sides of every update; the consumed batch gradient is recovered
    exactly from the two.

    Args:
        model: The model the optimizer trains (the hooked module tree).
        optimizer: The optimizer to read, or a zero-argument callable
            returning it (for a streamer that builds its optimizer lazily).
        projection: The projection config the gradients are captured with,
            to regenerate each layer's ``"subset_materialized"`` coordinates;
            ``None`` reads every coordinate of every layer.
        layers: Restrict to these layer names (default: every layer in the
            record).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | Callable[[], torch.optim.Optimizer],
        *,
        projection: dict[str, dict] | None = None,
        layers: Iterable[str] | None = None,
    ) -> None:
        self._model = model
        self._optimizer_src = optimizer
        self._snapshot: OptimizerSnapshot | None = None
        self._projection = projection
        self._layers = None if layers is None else set(layers)
        self._projector = ops.DattriProjector()
        self._coords: dict[str, torch.Tensor | None] = {}
        self.pre: dict[int, dict] = {}
        self.post: dict[int, dict] = {}

    @property
    def snapshot(self) -> OptimizerSnapshot:
        """The optimizer snapshot being read (built on first access)."""
        if self._snapshot is None:
            src = self._optimizer_src
            snapshot = OptimizerSnapshot(self._model, src() if callable(src) else src)
            if snapshot.optimizer_type not in ("Adam", "AdamW"):
                raise NotImplementedError(
                    "OptimizerStateCallback records Adam-family moments; got a "
                    f"{snapshot.optimizer_type} optimizer.",
                )
            self._snapshot = snapshot
        return self._snapshot

    def coordinates(self, layer_name: str) -> torch.Tensor | None:
        """Coordinates read for *layer_name* (``None`` = all)."""
        if layer_name not in self._coords:
            self._coords[layer_name] = ops.subset_coordinates(
                self._projection,
                layer_name,
                self.snapshot.width(layer_name),
                self._projector,
            )
        return self._coords[layer_name]

    def _read(self, layers: Iterable[str]) -> dict:
        snap = self.snapshot
        out: dict = {"layers": {}}
        for name in layers:
            if self._layers is not None and name not in self._layers:
                continue
            idx = self.coordinates(name)
            m = snap.state(name, "exp_avg", idx)
            v = snap.state(name, "exp_avg_sq", idx)
            width = snap.width(name) if idx is None else idx.numel()
            zeros = torch.zeros(width)
            out["layers"][name] = (
                zeros if m is None else m.detach().to("cpu", torch.float32),
                zeros.clone() if v is None else v.detach().to("cpu", torch.float32),
            )
            if "step" not in out:
                out["step"] = snap.step_count(name)
                hp = snap.hyperparameters(name)
                out.update({k: hp[k] for k in ("lr", "betas", "eps", "weight_decay")})
        return out

    def on_step_end(self, record: GradientRecord) -> None:
        """Pre-step moments of every hooked layer, keyed by the record's step."""
        self.pre[record.step] = self._read(record.gradient.data.keys())

    def record_post(self, step: int) -> None:
        """Post-step moments of the layers seen at *step* (call after ``step()``)."""
        if step not in self.pre:
            raise KeyError(f"no pre-step snapshot recorded for step {step}.")
        self.post[step] = self._read(self.pre[step]["layers"].keys())

    def dynamics(self) -> dict[int, dict]:
        """``{step: {"pre", "post", "step", "lr", "betas", "eps", "weight_decay"}}``
        for every step with both snapshots.
        """
        out = {}
        for step, pre in self.pre.items():
            if step not in self.post:
                continue
            post = self.post[step]
            out[step] = {
                "pre": pre["layers"],
                "post": post["layers"],
                "step": post["step"],
                "lr": pre["lr"],
                "betas": tuple(pre["betas"]),
                "eps": pre["eps"],
                "weight_decay": pre["weight_decay"],
            }
        return out
