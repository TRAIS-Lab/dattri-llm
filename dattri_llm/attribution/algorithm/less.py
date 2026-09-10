"""LESS: optimizer-aware gradient similarity (Xia et al., 2024).

LESS scores a training sample by the cosine between the *Adam update
direction* it would induce and the raw gradient of a query, weighted by the
learning rate::

    Inf(z, z') = sum_t lr_t * cos(grad(z'; theta_t), Gamma(z; theta_t))

``Gamma`` is the sample's gradient pushed through the optimizer's pre-step
state -- :func:`~dattri_llm.gradient.ops.precondition`, which the
:class:`~dattri_llm.gradient.hooks.HookManager` applies at capture when it is
handed the optimizer.  LESS is therefore TracIn on preconditioned train
gradients with a cosine metric, in either of TracIn's two forms:

* ``enable_update=True`` -- the per-step form of the paper's definition: a
  training trajectory whose every step contributes ``lr_t`` times the cosine
  at the state the step updates from.
* ``enable_update=False`` -- the paper's practical recipe: a few frozen
  checkpoints from a warm-up run, each with its saved optimizer state, summed
  with the checkpoints' learning rates as weights.

The map is coordinate-wise, so it applies exactly to whatever entries are
captured -- every entry of the hooked layers (LESS's LoRA setting), a
``"subset_materialized"`` subset, or, projected after the map, a
``"materialized"`` random projection (the paper's 8192-dimensional features).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.attribution.utils import read_lr_schedule, write_lr_schedule
from dattri_llm.gradient import ops
from dattri_llm.gradient.streaming import GradientStreamer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.storage_manager import GradientStorageManager
    from dattri_llm.gradient.streaming import DiskGradientSource, GradientSource


def _unit_rows(block: Gradient, scale: float) -> Gradient:
    """Every sample's concatenated entries scaled to norm *scale* (dense)."""
    dense: dict[str, torch.Tensor] = {}
    for name, value in block.data.items():
        entries = (
            value
            if isinstance(value, torch.Tensor)
            else ops.materialize(value, block.layer_types[name])
        )
        dense[name] = entries.reshape(entries.shape[0], -1).float()
    norm_sq = None
    for x in dense.values():
        n = (x * x).sum(dim=1)
        norm_sq = n if norm_sq is None else norm_sq + n
    factor = scale / norm_sq.clamp_min(1e-16).sqrt()  # type: ignore[union-attr]
    return block.map_layers(lambda name, _v, _lt: dense[name] * factor[:, None])


class _WeightedSource:
    """A train source whose blocks are unit-normalized and scaled per step."""

    def __init__(
        self,
        source: GradientSource,
        device: object,
        weight: Callable[[int], float],
    ) -> None:
        self._source = source
        self._device = device
        self._weight = weight

    @property
    def reusable(self) -> bool:
        return getattr(self._source, "reusable", False)

    def __iter__(self) -> Iterator[tuple[int, Gradient, list[str]]]:
        for step, block, hashes in self._source:
            yield step, _unit_rows(block.to(self._device), self._weight(step)), hashes


class LESSAttributor(BaseInnerProductAttributor):
    """LESS attributor.

    Args:
        args: :class:`AttributionArguments`; an updating pass trains with its
            optimizer settings, as the streamer mirrors the HF ``Trainer``.
        task: The attribution task (its checkpoints are the frozen form's
            LESS checkpoints).
        optimizers: For the frozen form, one optimizer per task checkpoint,
            in order -- the optimizer state live at that checkpoint (see
            :func:`~dattri_llm.gradient.optimizer_state.optimizer_from_state_dict`
            for a saved ``optimizer.pt``).  Unused by the per-step form.
        checkpoint_weights: The frozen form's per-checkpoint weights (LESS
            uses each checkpoint's learning rate); ``None`` weights every
            checkpoint 1.
    """

    algorithm: ClassVar[str] = "LESS"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
        optimizers: Sequence[torch.optim.Optimizer] | None = None,
        checkpoint_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(args, task=task)
        self._optimizers = list(optimizers or [])
        self._weights = [float(w) for w in checkpoint_weights or []]
        if self._weights and len(self._weights) != len(self._optimizers):
            raise ValueError(
                "checkpoint_weights must match optimizers one to one, got "
                f"{len(self._weights)} weights for {len(self._optimizers)} optimizers.",
            )
        self._enable_update = False
        # Per-step row weights: a constant, a {step: weight} map, or None to
        # read the applied learning rates off the live updating streamer.
        self._step_weights: float | dict[int, float] | None = 1.0

    # ------------------------------------------------------------------ #
    # Checkpoints and collection                                           #
    # ------------------------------------------------------------------ #
    def checkpoints(self) -> list[int]:
        """The trajectory's start, or every task checkpoint with its optimizer."""
        if self._enable_update:
            return [0]
        n = self.num_checkpoints()
        if len(self._optimizers) != n:
            raise ValueError(
                f"LESS's frozen form needs one optimizer per checkpoint: the task "
                f"has {n} checkpoint(s) but {len(self._optimizers)} optimizers "
                "were given.",
            )
        return list(range(n))

    def generate_train_rep(
        self,
        train_dataset: Dataset,
        *,
        checkpoint_step: int = 0,
        enable_update: bool = False,
        hook_config: HookManagerConfig | None = None,
    ) -> GradientStreamer:
        """A **preconditioned** train streamer: the trajectory's own optimizer
        when updating, else the checkpoint's optimizer.
        """
        return GradientStreamer(
            self.require_task("attribute").get_model(),
            train_dataset,
            self.args,
            batch_size=self.args.per_device_train_batch_size,
            enable_update=enable_update,
            loss_fn=self.train_loss_fn(),
            checkpoint_step=checkpoint_step,
            config=hook_config,
            optimizer=None if enable_update else self._optimizers[checkpoint_step],
            precondition=True,
        )

    def collect_gradients(
        self,
        streamer: GradientStreamer,
        store: GradientStorageManager,
        **kwargs: object,
    ) -> GradientStorageManager:
        """Collect, recording the applied learning rates beside a trajectory
        store so :meth:`attribute_from_cache` can weight its steps.
        """
        out = super().collect_gradients(streamer, store, **kwargs)  # type: ignore[arg-type]
        if getattr(streamer, "enable_update", False):
            write_lr_schedule(str(store.save_dir), streamer.learning_rates)
        return out

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        enable_update: bool = False,
        **kwargs: object,
    ) -> list[tuple[str, str]]:
        """Cache preconditioned train and raw test gradients (see the base)."""
        self._enable_update = enable_update
        return super().cache(
            train_dataset, test_dataset, enable_update=enable_update, **kwargs
        )  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #
    def transform_test_rep(self, test_rep: Gradient) -> Gradient:  # noqa: PLR6301
        """Unit-normalized raw query gradient (dense)."""
        return _unit_rows(test_rep, 1.0)

    def score_sources(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
        *,
        loop_over_test: bool = False,
        transform_test: Callable[[Gradient], Gradient] | None = None,
    ) -> tuple[torch.Tensor, list[str], list[int], list[str]]:
        """Score with the train blocks unit-normalized and weighted per step,
        so the plain layerwise inner product is the weighted cosine.
        """
        weights = self._step_weights

        def weight(step: int) -> float:
            if weights is None:
                lrs = getattr(train_source, "learning_rates", None)
                if lrs is None or step not in lrs:
                    raise ValueError(
                        f"no learning rate recorded for step {step}; the per-step "
                        "form weights each step by the applied learning rate.",
                    )
                return float(lrs[step])
            if isinstance(weights, dict):
                if step not in weights:
                    raise ValueError(
                        f"no weight for step {step}; provided steps: "
                        f"{sorted(weights)}.",
                    )
                return weights[step]
            return weights

        return super().score_sources(
            _WeightedSource(train_source, self.args.device, weight),  # type: ignore[arg-type]
            test_source,
            loop_over_test=loop_over_test,
            transform_test=transform_test,
        )

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #
    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        verbose: bool = False,
        loop_over_test: bool = False,
        enable_update: bool = False,
        gradient_cache_residency: str | None = None,
    ) -> AttributionScore:
        """Score live, in the per-step or the frozen-checkpoint form.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Query dataset to stream.
            hook_config: Capture configuration.  Any projection must be
                ``"subset_materialized"`` or ``"materialized"`` (the map
                needs exact gradient entries; a ``"materialized"`` projection
                is applied after it).
            verbose: Accepted for API parity.
            loop_over_test: Re-stream the test blocks per train block.  In the
                per-step form this evaluates the query gradient at every
                step's parameters; ``False`` takes it once, at the start.
            enable_update: ``True`` for the per-step form (a trajectory from
                the first checkpoint, each step weighted by its applied
                learning rate); ``False`` for the frozen form over the task's
                checkpoints with their ``optimizers`` and
                ``checkpoint_weights``.
            gradient_cache_residency: As in the base class.
        """
        self._enable_update = enable_update
        self._step_weights = (
            None
            if enable_update
            else (dict(enumerate(self._weights)) if self._weights else 1.0)
        )
        return super().attribute(
            train_dataset,
            test_dataset,
            hook_config=hook_config,
            verbose=verbose,
            loop_over_test=loop_over_test,
            enable_update=enable_update,
            gradient_cache_residency=gradient_cache_residency,
        )

    def attribute_from_cache(
        self,
        train_source: str | Path | GradientStorageManager | DiskGradientSource,
        test_source: str | Path | GradientStorageManager | DiskGradientSource,
        *,
        checkpoint_weight: float | None = None,
        learning_rates: float | Mapping[int, float] | None = None,
        **kwargs: object,
    ) -> AttributionScore:
        """Score stored **preconditioned** train gradients against raw queries.

        The train store must hold gradients captured through the optimizer
        (``HookManager(optimizer=...)`` / :meth:`cache`); the test store holds
        raw query gradients.

        Args:
            train_source: Preconditioned train gradients.
            test_source: Raw query gradients.
            checkpoint_weight: One weight for every row (the frozen form's
                learning rate at that checkpoint).
            learning_rates: Per-step weights for a trajectory store, as a
                constant or a ``{step: lr}`` map.  Unset, the schedule
                :meth:`cache` recorded beside the store is used when present.
            **kwargs: Passed to the base class (``selected_training_steps``,
                ``layer_name``, ``loop_over_test``, ...).
        """
        train_store = self.resolve_store(train_source)
        meta: dict = {}
        if learning_rates is not None:
            self._step_weights = (
                {int(k): float(v) for k, v in learning_rates.items()}
                if isinstance(learning_rates, Mapping)
                else float(learning_rates)
            )
            meta["learning_rates"] = learning_rates
        elif checkpoint_weight is not None:
            self._step_weights = float(checkpoint_weight)
            meta["checkpoint_weight"] = checkpoint_weight
        else:
            recorded = read_lr_schedule(str(train_store.save_dir))
            if recorded is not None:
                self._step_weights = recorded
            elif self._step_weights is None:
                self._step_weights = 1.0
        return super().attribute_from_cache(
            train_store,
            test_source,
            algorithm_meta=meta,
            **kwargs,
        )
