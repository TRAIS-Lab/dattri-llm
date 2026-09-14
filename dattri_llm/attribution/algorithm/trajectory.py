"""Shared machinery of the trajectory-sweeping attributors.

DVEmb and AdamW-influence score a training sample by how the update it
induced at its step propagates through every later step to the final model.
Both therefore (1) drive one live training trajectory and take the query
gradients at its end, (2) sweep the recorded steps latest to earliest, and
(3) carry the propagation on one of two sides: through the training side (a
parameter-space summary operator, test-independent) or through the query
side (the query gradients themselves, matrix-free).  This module holds what
that has in common:

* the trajectory pass -- into a gradient store (``train_grads``) or, with
  ``args.recompute_gradients``, into parameter snapshots
  (``train_snapshots``) from which each step's gradients are recomputed at
  attribution time by a :class:`~dattri_llm.gradient.streaming.\
ReplayGradientSource`;
* opening either kind of train source behind one interface, with the
  per-step random access (``for_steps``) the sweeps use;
* the ``propagation`` / ``loop_over_test`` options and their validation, the
  step bookkeeping (propagated vs. emitted steps, ``final_step``), the
  dense query matrix, and the score metadata.

The method-specific sweeps stay in the subclasses.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING, ClassVar

import torch
from tqdm.auto import tqdm

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.attribution.utils import (
    normalize_layer_names,
    read_lr_schedule,
    write_lr_schedule,
)
from dattri_llm.gradient import ops
from dattri_llm.gradient.snapshots import TrajectorySnapshots
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource, ReplayGradientSource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.callbacks import HookManagerCallback
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import GradientStreamer

TRAIN_GRADS_DIR = "train_grads"
TRAIN_SNAPSHOTS_DIR = "train_snapshots"
TEST_GRADS_DIR = "test_grads"

TrainSource = DiskGradientSource | ReplayGradientSource
TrainSourceSpec = (
    str
    | pathlib.Path
    | GradientStorageManager
    | TrajectorySnapshots
    | DiskGradientSource
    | ReplayGradientSource
)

PROPAGATIONS = ("train", "test")


def _dense_float(block: Gradient, dtype: torch.dtype = torch.float32) -> Gradient:
    """Materialize every layer of *block* to a ``(B, d)`` tensor of *dtype*
    (float32 by default; a narrower dtype halves what a full-width block
    holds, the products being formed in float32 per layer).  One layer is
    materialized at a time, so the peak is the narrowed block plus one
    layer's float32 gradient.
    """
    return block.map_layers(lambda _n, v, t: ops.materialize(v, t).to(dtype))


class TrajectoryAttributor(BaseInnerProductAttributor):
    """Base class of the trajectory-sweeping attributors (see the module
    docstring).  Subclasses implement the sweeps and hook into the trajectory
    pass through :meth:`trajectory_callbacks`, :meth:`on_trajectory_block`
    and :meth:`finish_trajectory`.
    """

    algorithm: ClassVar[str] = "Trajectory"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: object | None = None,
    ) -> None:
        super().__init__(args, task=task)  # type: ignore[arg-type]
        # The capture configuration of the last trajectory this attributor
        # collected; a replay must use the same one.
        self._hook_config: HookManagerConfig | None = None

    # ------------------------------------------------------------------ #
    # Collection                                                           #
    # ------------------------------------------------------------------ #

    @property
    def recompute(self) -> bool:
        """Whether the trajectory pass stores snapshots instead of gradients."""
        return bool(getattr(self.args, "recompute_gradients", False))

    def checkpoints(self) -> list[int]:
        """The trajectory regenerates from the task's first checkpoint."""
        n = self.num_checkpoints()
        if n > 1:
            warnings.warn(
                f"{self.algorithm} regenerates the trajectory from checkpoint 0; "
                f"the other {n - 1} checkpoint(s) are ignored.",
                stacklevel=2,
            )
        return [0]

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: str | None = None,
        hook_config: HookManagerConfig | None = None,
        offload_interval: int = 1,
    ) -> list[tuple[str, str]]:
        """Run the training trajectory live and cache what the sweep needs.

        The train side goes to ``<cache_dir>/train_grads`` (one gradient
        block per step) or, with ``args.recompute_gradients``, to
        ``<cache_dir>/train_snapshots`` (each step's parameters and batch,
        plus whatever the method records alongside).  The query gradients at
        the final model go to ``<cache_dir>/test_grads``.  The per-step
        learning rates actually applied are written beside the train side.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            cache_dir: Parent directory; defaults to ``args.output_dir``.
            hook_config: Capture configuration for both passes (and for the
                replay, when recomputing).
            offload_interval: Steps accumulated per gradient file when
                storing gradients.

        Returns:
            ``[(train_dir, test_dir)]``.
        """
        self.require_task("cache")
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        root = pathlib.Path(cache_dir)
        train_dir = str(
            root / (TRAIN_SNAPSHOTS_DIR if self.recompute else TRAIN_GRADS_DIR)
        )
        test_dir = str(root / TEST_GRADS_DIR)
        recorded_lr = self.collect_trajectory(
            TrajectorySnapshots(train_dir)
            if self.recompute
            else GradientStorageManager(train_dir),
            GradientStorageManager(test_dir),
            train_dataset,
            test_dataset,
            hook_config=hook_config,
            offload_interval=offload_interval,
        )
        write_lr_schedule(train_dir, recorded_lr)
        return [(train_dir, test_dir)]

    def collect_trajectory(
        self,
        train_target: GradientStorageManager | TrajectorySnapshots,
        test_store: GradientStorageManager,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        offload_interval: int = 1,
    ) -> dict[int, float]:
        """Drive the trajectory ``theta_0 -> theta_T`` into *train_target*
        (a gradient store, or a snapshot store to recompute from), then the
        frozen query pass at ``theta_T`` into *test_store*.

        Returns:
            The per-step learning rates actually applied.
        """
        self.checkpoints()
        model = self.load_checkpoint(0)
        self._hook_config = hook_config
        snapshots = (
            train_target if isinstance(train_target, TrajectorySnapshots) else None
        )
        streamer = self.generate_train_rep(
            train_dataset,
            enable_update=True,
            hook_config=hook_config,
            snapshots=snapshots,
        )
        for callback in self.trajectory_callbacks(model, streamer, snapshots):
            streamer.hook_manager.add_callback(callback)
        on_block = self.on_trajectory_block()
        if snapshots is not None:
            with streamer:
                for step, block, hashes in streamer:
                    if on_block is not None:
                        on_block(step, block, hashes)
        else:
            self.collect_gradients(
                streamer,
                train_target,  # type: ignore[arg-type]
                offload_interval=offload_interval,
                on_block=on_block,
            )
        self.collect_gradients(
            self.generate_test_rep(test_dataset, hook_config=hook_config),
            test_store,
            offload_interval=offload_interval,
        )
        self.finish_trajectory(train_target)
        return streamer.learning_rates

    def trajectory_callbacks(  # noqa: PLR6301 - overridable hook
        self,
        model: nn.Module,  # noqa: ARG002
        streamer: GradientStreamer,  # noqa: ARG002
        snapshots: TrajectorySnapshots | None,  # noqa: ARG002
    ) -> list[HookManagerCallback]:
        """Callbacks to attach to the trajectory pass (none by default)."""
        return []

    def on_trajectory_block(  # noqa: PLR6301 - overridable hook
        self,
    ) -> Callable[[int, Gradient, list[str]], None] | None:
        """A hook run on every streamed train block, after its update."""
        return None

    def finish_trajectory(
        self,
        train_target: GradientStorageManager | TrajectorySnapshots,
    ) -> None:
        """Persist anything the sweep needs beside the train side."""

    # ------------------------------------------------------------------ #
    # Sources                                                              #
    # ------------------------------------------------------------------ #

    def open_train_source(
        self,
        train_source: TrainSourceSpec,
        *,
        steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        hook_config: HookManagerConfig | None = None,
        verbose: bool = False,
    ) -> TrainSource:
        """The train side as a source with per-step random access.

        A gradient store (directory, open store, or disk source) opens as a
        :class:`DiskGradientSource`; a snapshot store (or its directory)
        opens as a :class:`ReplayGradientSource` over the task's model and
        training loss, capturing with *hook_config* -- the configuration the
        trajectory was collected with (the one of this attributor's last
        :meth:`cache` when not given).
        """
        layer_name = normalize_layer_names(layer_name)
        if isinstance(train_source, ReplayGradientSource):
            return train_source if steps is None else train_source.for_steps(steps)
        if isinstance(train_source, TrajectorySnapshots) or (
            TrajectorySnapshots.is_snapshot_dir(train_source)
        ):
            task = self.require_task("attribute_from_cache (replaying snapshots)")
            snapshots = (
                train_source
                if isinstance(train_source, TrajectorySnapshots)
                else TrajectorySnapshots(train_source)  # type: ignore[arg-type]
            )
            return ReplayGradientSource(
                task.get_model(),
                self.args,
                snapshots,
                loss_fn=self.train_loss_fn(),
                config=hook_config if hook_config is not None else self._hook_config,
                steps=steps,
                layer_name=layer_name,
                desc=f"{self.algorithm}: replay",
                verbose=verbose,
            )
        store = self.resolve_store(train_source)  # type: ignore[arg-type]
        return self.load_train_rep(
            store, steps=steps, layer_name=layer_name, verbose=verbose
        )

    @staticmethod
    def train_dir_of(train_source: TrainSource) -> str:
        """The directory a train source reads (store or snapshot root)."""
        if isinstance(train_source, ReplayGradientSource):
            return str(train_source.snapshots.root)
        return str(train_source.file_manager.save_dir)

    @staticmethod
    def available_steps(train_source: TrainSource) -> list[int]:
        """Every step the train side holds, ascending."""
        if isinstance(train_source, ReplayGradientSource):
            return sorted(train_source.snapshots.steps())
        return sorted(train_source.file_manager.available_steps())

    def recorded_lr_of(self, train_source: TrainSource) -> dict[int, float] | None:
        """The schedule recorded beside the train side, if any."""
        return read_lr_schedule(self.train_dir_of(train_source))

    @staticmethod
    def close_source(source: object) -> None:
        """Release a replay source's hooks (a no-op for other sources)."""
        if isinstance(source, ReplayGradientSource):
            source.close()

    def source_meta(
        self,
        train_source: TrainSource,
        test_store: GradientStorageManager,
    ) -> dict:
        """Score metadata describing the two sides."""
        if isinstance(train_source, ReplayGradientSource):
            return {
                "train_source": "snapshots",
                "train_dir": self.train_dir_of(train_source),
                "sample_id_key": {"train": None, "test": test_store.sample_id_key},
            }
        return {
            "train_source": "gradients",
            **self.stores_meta(train_source.file_manager, test_store),
        }

    # ------------------------------------------------------------------ #
    # Sweep bookkeeping                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_propagation(propagation: str, loop_over_test: bool) -> None:
        """Check the side the product is carried on and its blocking option."""
        if propagation not in PROPAGATIONS:
            raise ValueError(
                f"propagation must be one of {PROPAGATIONS}, got {propagation!r}.",
            )
        if propagation == "train" and loop_over_test:
            raise ValueError(
                "loop_over_test applies only to propagation='test': the "
                "train-side sweep is test-independent (its memory is the "
                "parameter-space operator, not the query matrix), so there is "
                "nothing to block over.",
            )

    @staticmethod
    def resolve_steps(
        available: list[int],
        selected_training_steps: Iterable[int] | None,
        final_step: int | None,
    ) -> tuple[list[int], set[int], int]:
        """The steps the sweep propagates through and the ones it emits.

        ``final_step`` (capital ``T``) defaults to one past the last
        available step; every available step below it is propagated, and
        *selected_training_steps* only restricts which of those become rows.

        Returns:
            ``(prop_steps, output_steps, final_step)``.
        """
        if final_step is None:
            final_step = (max(available) + 1) if available else 0
        prop_steps = [s for s in available if s < final_step]
        if not prop_steps:
            raise ValueError(
                f"No training step satisfies step < final_step ({final_step}); "
                f"available steps: {available}.",
            )
        if selected_training_steps is None:
            output_steps = set(prop_steps)
        else:
            output_steps = {int(s) for s in selected_training_steps} & set(prop_steps)
            if not output_steps:
                raise ValueError(
                    "selected_training_steps matches none of the propagated "
                    f"steps {prop_steps[:10]}...",
                )
        return prop_steps, output_steps, final_step

    def steps_bar(self, prop_steps: list[int], desc: str, verbose: bool) -> Iterable:
        """The propagated steps, latest first, behind an optional progress bar."""
        return tqdm(
            sorted(prop_steps, reverse=True),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose or not self.args.should_log,
        )

    def collect_test_matrix(
        self,
        test_source: DiskGradientSource,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Every query gradient as one dense per-layer matrix.

        Returns ``(w, test_ids)`` where ``w`` maps each layer to its
        ``(num_test, d_layer)`` final-model gradients, rows ordered by first
        appearance of each test hash (a duplicate hash keeps its last row).
        """
        device = self.args.device
        test_ids: list[str] = []
        test_index: dict[str, int] = {}
        pending: list[tuple[Gradient, list[int]]] = []
        for _step, test_g, test_hashes in test_source:
            mat = _dense_float(test_g.to(device))
            cols: list[int] = []
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
                cols.append(test_index[h])
            pending.append((mat, cols))
        num_test = len(test_ids)
        layers = list(pending[0][0].data) if pending else []
        w: dict[str, torch.Tensor] = {
            name: torch.zeros(
                num_test, pending[0][0].data[name].shape[1], device=device
            )
            for name in layers
        }
        for mat, cols in pending:
            idx = torch.as_tensor(cols, device=device)
            for name in layers:
                w[name].index_copy_(0, idx, mat.data[name])
        return w, test_ids

    @staticmethod
    def test_column_order(
        test_source: DiskGradientSource,
    ) -> tuple[list[str], dict[str, int]]:
        """The column order of the queries, from their hashes alone."""
        test_ids: list[str] = []
        test_index: dict[str, int] = {}
        for _step, _g, hashes in test_source:
            for h in hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
        return test_ids, test_index
