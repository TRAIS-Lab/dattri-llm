"""On-disk snapshots of a training trajectory, for replaying it later.

A trajectory attributor needs the per-sample gradient of every training step
at the parameters that step ran from.  Storing those gradients costs
``steps x batch x parameters`` values; storing what is needed to *recompute*
them costs ``steps x parameters`` (the parameters before each update) plus
the step's batch, which is small.  :class:`TrajectorySnapshots` is that
store: one file per step for the parameters, one for the batch, and -- for
optimizer-aware methods -- one per side of the update for the optimizer
moments.  :class:`~dattri_llm.gradient.streaming.ReplayGradientSource`
reads it back, step by step, running the same forward and backward under the
same hooks the trajectory was captured with.

Layout under ``root``::

    snapshots.json            manifest (marks the directory as a snapshot store)
    theta_<step>.pt           {name: tensor} of the trainable parameters
    theta_<step>.ref          "<other step>" when the parameters equal that
                              step's (micro-batches of one accumulation window)
    batch_<step>.pt           the collated batch the step consumed, on CPU
    dynamics_<step>_pre.pt    optimizer moments before the update (optional)
    dynamics_<step>_post.pt   optimizer moments after the update (optional)
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

MANIFEST_FILE = "snapshots.json"


def _to_cpu(obj: object) -> object:
    """*obj* with every tensor in it detached and moved to the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class TrajectorySnapshots:
    """Per-step parameters, batches and optimizer moments of a trajectory.

    Args:
        root: Directory of the store; created (with its manifest) if absent.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / MANIFEST_FILE
        if not manifest.exists():
            manifest.write_text(json.dumps({"format": 1}))

    @staticmethod
    def is_snapshot_dir(path: object) -> bool:
        """Whether *path* names a directory holding a snapshot manifest."""
        return (
            isinstance(path, (str, pathlib.Path))
            and (pathlib.Path(path) / MANIFEST_FILE).is_file()
        )

    def __repr__(self) -> str:
        return f"TrajectorySnapshots({str(self.root)!r}, steps={len(self.steps())})"

    # ------------------------------------------------------------------ #
    # Parameters                                                           #
    # ------------------------------------------------------------------ #
    def save_parameters(
        self,
        step: int,
        model: nn.Module,
        *,
        same_as: int | None = None,
    ) -> None:
        """Store the trainable parameters the step *step* runs from.

        Args:
            step: The step index.
            model: The model; only ``requires_grad`` parameters are stored,
                in their own dtype.
            same_as: Record that the parameters equal those already stored
                for this other step (a later micro-batch of one accumulation
                window) instead of writing them again.
        """
        if same_as is not None:
            (self.root / f"theta_{step}.ref").write_text(str(int(same_as)))
            return
        state = {
            name: p.detach().to("cpu")
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        torch.save(state, self.root / f"theta_{step}.pt")

    def _parameter_file(self, step: int) -> pathlib.Path:
        ref = self.root / f"theta_{step}.ref"
        if ref.exists():
            return self._parameter_file(int(ref.read_text()))
        path = self.root / f"theta_{step}.pt"
        if not path.exists():
            raise KeyError(f"no parameter snapshot for step {step} in {self.root}.")
        return path

    def load_parameters(self, step: int, model: nn.Module) -> None:
        """Copy the parameters stored for *step* into *model* (in place)."""
        state = torch.load(self._parameter_file(step), weights_only=True)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in state:
                    raise KeyError(
                        f"parameter {name!r} has no snapshot at step {step}."
                    )
                p.copy_(state[name].to(p.device, p.dtype))

    # ------------------------------------------------------------------ #
    # Batches                                                             #
    # ------------------------------------------------------------------ #
    def save_batch(self, step: int, batch: object) -> None:
        """Store the collated batch the step *step* consumed (moved to CPU)."""
        torch.save(_to_cpu(batch), self.root / f"batch_{step}.pt")

    def load_batch(self, step: int) -> object:
        """The batch stored for *step*, on CPU."""
        path = self.root / f"batch_{step}.pt"
        if not path.exists():
            raise KeyError(f"no batch snapshot for step {step} in {self.root}.")
        return torch.load(path, weights_only=False)

    # ------------------------------------------------------------------ #
    # Optimizer moments                                                    #
    # ------------------------------------------------------------------ #
    def save_dynamics(self, step: int, side: str, entry: dict) -> None:
        """Store one side (``"pre"`` or ``"post"``) of the step's moments."""
        if side not in ("pre", "post"):
            raise ValueError(f"side must be 'pre' or 'post', got {side!r}.")
        torch.save(_to_cpu(entry), self.root / f"dynamics_{step}_{side}.pt")

    def load_dynamics(self, step: int, side: str) -> dict | None:
        """One side of the step's moments, or ``None`` if not stored."""
        path = self.root / f"dynamics_{step}_{side}.pt"
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)

    def dynamics_steps(self) -> list[int]:
        """Steps with both sides of the moments stored."""
        pre = {int(p.stem.split("_")[1]) for p in self.root.glob("dynamics_*_pre.pt")}
        post = {int(p.stem.split("_")[1]) for p in self.root.glob("dynamics_*_post.pt")}
        return sorted(pre & post)

    # ------------------------------------------------------------------ #
    # Steps                                                               #
    # ------------------------------------------------------------------ #
    def steps(self) -> list[int]:
        """Steps with both a parameter snapshot and a batch, ascending."""
        params = {int(p.stem.split("_")[1]) for p in self.root.glob("theta_*.pt")}
        params |= {int(p.stem.split("_")[1]) for p in self.root.glob("theta_*.ref")}
        batches = {int(p.stem.split("_")[1]) for p in self.root.glob("batch_*.pt")}
        return sorted(params & batches)


class LazyDynamics(Mapping):
    """``{step: dynamics}`` read from a :class:`TrajectorySnapshots` on access.

    Each entry has the layout ``OptimizerStateCallback.dynamics`` (see
    :mod:`dattri_llm.gradient.callbacks`) produces, assembled from the stored
    ``pre`` and ``post`` sides by *assemble*.
    """

    def __init__(
        self,
        snapshots: TrajectorySnapshots,
        assemble: object,
    ) -> None:
        self._snapshots = snapshots
        self._assemble = assemble
        self._steps = snapshots.dynamics_steps()

    def __getitem__(self, step: int) -> dict:
        pre = self._snapshots.load_dynamics(step, "pre")
        post = self._snapshots.load_dynamics(step, "post")
        if pre is None or post is None:
            raise KeyError(step)
        return self._assemble(pre, post)  # type: ignore[operator]

    def __iter__(self) -> Iterator[int]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, step: object) -> bool:
        return step in self._steps
