"""TracIn / GradCos attribution.

Every train record is scored against every test record::

    score[i, j] = <g_train_i, g_test_j>

i.e. the full ``(num_train, num_test)`` gradient cross-gram -- there is no
train/test step alignment.  With ``normalized_grad=True`` the inner product
becomes a cosine similarity (the GradCos / CosIn variant); ``GradCos`` is
TracIn with ``normalized_grad=True`` passed to the attribute methods -- there
is no separate subclass.

TracIn is the plainest inner-product attributor: both transforms are the
identity (the test side is merely materialized when that fits the cache
budget, so every train block meets a dense test block) and the score is the
layerwise cross-gram.  Everything else -- collection, the multi-checkpoint
ensemble, the trajectory mode, the scoring loop -- is inherited from
:class:`~dattri_llm.attribution.base.BaseInnerProductAttributor`.

The result is a :class:`~dattri_llm.attribution.score.AttributionScore`.  Rows
are stamped with the step each train gradient was recorded at, so a sample
collected at several checkpoints contributes one row per checkpoint; summing a
sample's rows over steps recovers the classic dattri ``(num_train, num_test)``
matrix.  Rows and columns are identified by content hash in store order, so no
reconstruction of the original DataLoader is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.gradient import ops
from dattri_llm.utils.cache import CacheBudget

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.storage_manager import GradientStorageManager
    from dattri_llm.gradient.streaming import DiskGradientSource
    from dattri_llm.utils.cache import TensorCache


class TracInAttributor(BaseInnerProductAttributor):
    """TracIn / GradCos attributor.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory.
        task: The attribution task; required by the live methods only.

    ``normalized_grad`` (cosine / GradCos vs raw inner product / TracIn) is a
    per-attribution choice passed to :meth:`attribute` /
    :meth:`attribute_from_cache`.

    Layer selection happens at **capture** (via the ``hook_config`` of the live
    methods, or whatever was hooked when the cache was collected).  By default
    every stored layer is scored; :meth:`attribute_from_cache` additionally takes
    a ``layer_name`` read-time filter to score a subset of the stored layers.
    """

    algorithm: ClassVar[str] = "TracIn"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
    ) -> None:
        super().__init__(args, task=task)
        self._metric = "dot"

    def _set_metric(self, normalized_grad: bool) -> None:
        """Select the metric (and the algorithm name recorded in the score)."""
        self._metric = "cosine" if normalized_grad else "dot"
        self.algorithm = "GradCos" if normalized_grad else "TracIn"

    # ------------------------------------------------------------------ #
    # Hooks                                                               #
    # ------------------------------------------------------------------ #

    def transform_test_rep(self, test_rep: Gradient) -> Gradient:  # noqa: PLR6301
        """Materialize the test block when its dense form fits the cache budget.

        A dense test side makes every train block's score a bare GEMM against
        a train layer materialized once (see :meth:`inner_product`).  Over
        budget -- at full dimension the dense form is ~1 GB *per sample* --
        the block stays factorized, so both sides stay factorized and the
        kernel materializes at most one layer at a time: caching is an
        optimization and must never be the reason a run runs out of memory.
        """
        if CacheBudget(test_rep.device).fits(test_rep.materialized_nbytes):
            return test_rep.materialize()
        return test_rep

    def inner_product(
        self,
        train_rep: Gradient,
        test_rep: Gradient,
        *,
        dense_cache: TensorCache | None = None,
    ) -> torch.Tensor:
        """Layerwise cross-gram, divided by the gradient norms for GradCos."""
        dot = super().inner_product(train_rep, test_rep, dense_cache=dense_cache)
        if self._metric == "dot":
            return dot
        shared = [name for name in train_rep.data if name in test_rep.data]
        if not shared:
            return dot
        n_tr = self._norm(train_rep, shared, dense_cache)
        n_te = self._norm(test_rep, shared, None)
        return dot / (n_tr[:, None] * n_te[None, :] + 1e-8)

    @staticmethod
    def _norm(
        rep: Gradient,
        layers: list[str],
        dense_cache: TensorCache | None,
    ) -> torch.Tensor:
        """Per-sample whole-model gradient norms over *layers* ``(B,)``.

        Uses the dense copy of a layer already sitting in *dense_cache* when
        there is one, so nothing is materialized twice.
        """
        total = torch.zeros(rep.batch_size, device=rep.device)
        for name in layers:
            value = rep.data[name]
            if dense_cache is not None and name in dense_cache:
                value = dense_cache.get(name)
            norm_sq = ops.grad_norm_sq(value, rep.layer_types[name])
            total += (
                norm_sq.expand(rep.batch_size) if norm_sq.shape[0] == 1 else norm_sq
            )
        return total.clamp_min(0).sqrt()

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
        normalized_grad: bool = False,
    ) -> AttributionScore:
        """Score by collecting gradients **live** (the on-the-fly workflow).

        See :meth:`BaseInnerProductAttributor.attribute` for the shared
        arguments.  ``normalized_grad=True`` scores by cosine similarity
        (GradCos); ``False`` (default) by the raw inner product (TracIn).
        """
        self._set_metric(normalized_grad)
        return super().attribute(
            train_dataset,
            test_dataset,
            hook_config=hook_config,
            verbose=verbose,
            loop_over_test=loop_over_test,
            enable_update=enable_update,
            gradient_cache_residency=gradient_cache_residency,
            normalized_grad=normalized_grad,
        )

    def attribute_from_cache(
        self,
        train_source: str | Path | GradientStorageManager | DiskGradientSource,
        test_source: str | Path | GradientStorageManager | DiskGradientSource,
        *,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        loop_over_test: bool = False,
        algorithm_meta: dict | None = None,
        normalized_grad: bool = False,
    ) -> AttributionScore:
        """Score collected gradients; ``normalized_grad`` selects GradCos.

        See :meth:`BaseInnerProductAttributor.attribute_from_cache` for the
        shared arguments.
        """
        self._set_metric(normalized_grad)
        return super().attribute_from_cache(
            train_source,
            test_source,
            selected_training_steps=selected_training_steps,
            layer_name=layer_name,
            verbose=verbose,
            loop_over_test=loop_over_test,
            algorithm_meta=algorithm_meta,
            normalized_grad=normalized_grad,
        )
