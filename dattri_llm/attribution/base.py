"""Abstract base classes for training-data attribution algorithms.

Two levels, mirroring dattri:

* :class:`BaseAttributor` -- the contract every attributor satisfies: build it
  from :class:`AttributionArguments` (plus an optional dattri
  ``AttributionTask`` for the live workflow), ``cache`` gradients, and
  ``attribute`` either live or ``attribute_from_cache``.
* :class:`BaseInnerProductAttributor` -- the concrete workflow of every method
  whose score is an inner product between a (transformed) train representation
  and a (transformed) test representation: TracIn/GradCos, the K-FAC family,
  and DVEmb's train-side embeddings.  A new method overrides only the hooks it
  needs -- typically :meth:`transform_test_rep` and/or :meth:`inner_product` --
  and inherits collection, the checkpoint ensemble, the scoring loop, the
  dense-materialization cache, and the score assembly.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch

from dattri_llm.attribution.score import AttributionScore
from dattri_llm.attribution.utils import (
    collect_gradients,
    normalize_layer_names,
    score_sources,
    task_loss_fn,
)
from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Gradient, GradientRecord
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource, GradientStreamer
from dattri_llm.utils.cache import CACHE_RESIDENCIES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dattri.task import AttributionTask
    from torch import nn
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import GradientSource
    from dattri_llm.utils.cache import TensorCache


class BaseAttributor(ABC):
    """Base class for all attributors.

    Every attributor supports two workflows:

    1. **On-the-fly** -- :meth:`attribute` drives the task's model over the
       datasets, collecting and scoring in one run.
    2. **Store-then-attribute** -- :meth:`cache` (or any training loop wrapped
       with a :class:`~dattri_llm.gradient.hooks.HookManager`) persists the
       gradients, and :meth:`attribute_from_cache` scores them later; it needs
       no model.

    Method-specific options are keyword arguments of the two attribute
    methods (``**attribution_kwargs``), so the signatures below are the
    largest common set.
    """

    #: Name recorded in every :class:`AttributionScore` this attributor produces.
    algorithm: ClassVar[str] = "Base"

    @abstractmethod
    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the attributor.

        Args:
            args: Configuration controlling device placement, batch sizes,
                precision, DataLoader behaviour, distributed settings and the
                output directory.  See :class:`AttributionArguments`.
            task: The dattri ``AttributionTask`` supplying the model, loss,
                optional target function and checkpoints.  Required by the
                live methods (:meth:`cache`, :meth:`attribute`); unused by
                :meth:`attribute_from_cache`.
            **kwargs: Method-specific construction options.
        """

    @abstractmethod
    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: str | None = None,
        hook_config: HookManagerConfig | None = None,
        **cache_kwargs: object,
    ) -> list[tuple[str, str]]:
        """Collect (and precompute) everything :meth:`attribute_from_cache` needs.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            cache_dir: Parent directory of the gradient stores; defaults to
                ``args.output_dir``.
            hook_config: :class:`HookManagerConfig` for the internal streamers
                (which layers to hook, per-layer projection, ...).  ``None``
                uses the streamer default.
            **cache_kwargs: Method-specific collection options.

        Returns:
            ``[(train_gradients_dir, test_gradients_dir), ...]`` -- the store
            pairs to feed :meth:`attribute_from_cache`.
        """

    @abstractmethod
    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        verbose: bool = False,
        **attribution_kwargs: object,
    ) -> AttributionScore:
        """Attribute **on the fly**: collect gradients live, then score them.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            hook_config: As in :meth:`cache`.
            verbose: Show progress bars on the logging process.
            **attribution_kwargs: Method-specific options.

        Returns:
            The :class:`AttributionScore`, also persisted to ``args.output_dir``.
        """

    @abstractmethod
    def attribute_from_cache(
        self,
        train_source: str | Path | GradientStorageManager | DiskGradientSource,
        test_source: str | Path | GradientStorageManager | DiskGradientSource,
        *,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        **attribution_kwargs: object,
    ) -> AttributionScore:
        """Attribute from previously collected gradients.

        Args:
            train_source: The train gradients -- the directory written by
                :class:`GradientStorageManager` for the train pass, an open
                store of any residency (e.g. an in-RAM one filled by an
                :class:`OffloadCallback` in the same process), or a
                :class:`DiskGradientSource` over one.
            test_source: The test gradients, likewise.
            selected_training_steps: Restrict the training steps (the output
                rows) to these; ``None`` uses every step in the store.
            layer_name: Restrict scoring to this subset of the *stored* layers
                (``str`` or list); ``None`` scores every stored layer.  A
                read-time filter -- the same store can be re-queried per layer.
            verbose: Show progress bars on the logging process.
            **attribution_kwargs: Method-specific options.

        Returns:
            The :class:`AttributionScore`, also persisted to ``args.output_dir``.
        """


class BaseInnerProductAttributor(BaseAttributor):  # noqa: PLR0904 - the workflow's hook surface
    """Base class for inner-product attributors.

    The score of a train sample against a test sample is
    ``<transform_train(g_train), transform_test(g_test)>`` -- a layerwise
    inner product between two gradient blocks.  A method plugs into the
    workflow through these hooks (all have sensible defaults):

    * :meth:`generate_train_rep` / :meth:`generate_test_rep` -- live gradient
      sources (a :class:`GradientStreamer` per side).
    * :meth:`load_train_rep` / :meth:`load_test_rep` -- on-disk gradient
      sources (a :class:`DiskGradientSource` per side).
    * :meth:`prepare_scoring` -- one-time work before a scoring pass, computed
      from the sources (e.g. a preconditioner fit on the train gradients).
    * :meth:`transform_train_rep` / :meth:`transform_test_rep` -- per-block
      transforms (identity by default).
    * :meth:`inner_product` -- the ``(B_train, B_test)`` score of one pair
      (the layerwise cross-gram by default).
    * :meth:`checkpoints` -- which task checkpoints :meth:`attribute`
      ensembles over (all of them by default).

    Everything else -- collection into a store of any residency, the
    checkpoint ensemble, the scoring loop with its dense-materialization
    cache, the ``loop_over_test`` memory mode, the layer / step filters, and
    the :class:`AttributionScore` assembly -- is inherited.
    """

    algorithm: ClassVar[str] = "InnerProduct"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
    ) -> None:
        self.args = args
        self.task = task

    # ------------------------------------------------------------------ #
    # Task plumbing                                                        #
    # ------------------------------------------------------------------ #

    def require_task(self, method: str) -> AttributionTask:
        """The attribution task, or a clear error naming the live *method*."""
        if self.task is None:
            raise ValueError(
                f"{method}() (live collection) requires a ``task`` with a model; "
                "pass pre-collected gradients to attribute_from_cache() instead.",
            )
        return self.task

    def num_checkpoints(self) -> int:
        """Number of checkpoints the task provides."""
        return len(self.require_task("attribute").get_checkpoints())

    def load_checkpoint(self, index: int) -> nn.Module:
        """Load the task's *index*-th checkpoint into its model and return it."""
        task = self.require_task("attribute")
        task._load_checkpoints(index)  # noqa: SLF001 - dattri's loading entry point
        return task.get_model()

    def checkpoints(self) -> list[int]:
        """Checkpoint indices :meth:`attribute` ensembles over (all, by default).

        Single-checkpoint methods override this to ``[0]``.
        """
        return list(range(self.num_checkpoints()))

    def train_loss_fn(self) -> Callable:
        """The task's training loss in the streamer's ``(model, batch)`` form."""
        return task_loss_fn(self.require_task("attribute").original_loss_func)

    def test_loss_fn(self) -> Callable:
        """The task's target function (defaults to its loss) for the test side."""
        return task_loss_fn(self.require_task("attribute").original_target_func)

    # ------------------------------------------------------------------ #
    # Representations: live sources and on-disk sources                    #
    # ------------------------------------------------------------------ #

    def generate_train_rep(
        self,
        train_dataset: Dataset,
        *,
        checkpoint_step: int = 0,
        enable_update: bool = False,
        hook_config: HookManagerConfig | None = None,
    ) -> GradientStreamer:
        """Live train gradients: a streamer over *train_dataset*.

        Args:
            train_dataset: Dataset to stream (batches go straight to the
                task's loss as its ``data``).
            checkpoint_step: Step label stamped on the blocks of a frozen
                probe (the checkpoint index).
            enable_update: Train the model as it streams (a trajectory; each
                optimizer step is its own step label) instead of a frozen probe.
            hook_config: Capture configuration; ``None`` uses the default.
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
        )

    def generate_test_rep(
        self,
        test_dataset: Dataset,
        *,
        checkpoint_step: int = 0,
        hook_config: HookManagerConfig | None = None,
        hook_manager: object | None = None,
    ) -> GradientStreamer:
        """Live test gradients: a frozen streamer over *test_dataset*.

        Args:
            test_dataset: Dataset to stream against the task's target function.
            checkpoint_step: Step label stamped on the blocks.
            hook_config: Capture configuration; ignored when *hook_manager* is
                given.
            hook_manager: Share the train streamer's hook manager (one set of
                hooks over the model) instead of registering a second one.
        """
        return GradientStreamer(
            self.require_task("attribute").get_model(),
            test_dataset,
            self.args,
            batch_size=self.args.per_device_eval_batch_size,
            enable_update=False,
            loss_fn=self.test_loss_fn(),
            checkpoint_step=checkpoint_step,
            config=hook_config,
            hook_manager=hook_manager,
        )

    def load_train_rep(
        self,
        train_gradients: str | GradientStorageManager,
        *,
        steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        desc: str | None = None,
    ) -> DiskGradientSource:
        """Stored train gradients as a re-iterable source.

        Args:
            train_gradients: A store directory, or an open
                :class:`GradientStorageManager` of any residency.
            steps: Restrict to these stored steps (``None`` = all).
            layer_name: Restrict every block to these layers (``None`` = all).
            verbose: Show a progress bar on the logging process.
            desc: Progress-bar label; defaults to ``"<algorithm>: train"``.
        """
        store = (
            train_gradients
            if isinstance(train_gradients, GradientStorageManager)
            else GradientStorageManager(train_gradients)
        )
        return DiskGradientSource(
            store,
            self.args,
            steps=steps,
            layer_name=normalize_layer_names(layer_name),
            desc=desc if desc is not None else f"{self.algorithm}: train",
            verbose=verbose,
        )

    def load_test_rep(
        self,
        test_gradients: str | GradientStorageManager,
        *,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        desc: str | None = None,
    ) -> DiskGradientSource:
        """Stored test gradients as a re-iterable source; see :meth:`load_train_rep`."""
        store = (
            test_gradients
            if isinstance(test_gradients, GradientStorageManager)
            else GradientStorageManager(test_gradients)
        )
        return DiskGradientSource(
            store,
            self.args,
            layer_name=normalize_layer_names(layer_name),
            desc=desc if desc is not None else f"{self.algorithm}: test",
            verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Collection                                                           #
    # ------------------------------------------------------------------ #

    def collect_gradients(  # noqa: PLR6301 - overridable hook
        self,
        streamer: GradientStreamer,
        store: GradientStorageManager,
        *,
        offload_interval: int = 1,
        on_block: Callable[[int, Gradient, list[str]], None] | None = None,
    ) -> GradientStorageManager:
        """Run *streamer* to completion into *store*; see
        :func:`~dattri_llm.attribution.utils.collect_gradients`.
        """
        return collect_gradients(
            streamer,
            store,
            offload_interval=offload_interval,
            on_block=on_block,
        )

    def cache_representations(
        self,
        source: Iterable[tuple[int, Gradient, list[str]]],
        store: GradientStorageManager,
        *,
        transform: Callable[[Gradient], Gradient] | None = None,
        sample_id_key: str | int | None = None,
    ) -> GradientStorageManager:
        """Persist ``transform(block)`` for every block of *source* into *store*.

        The derived representations -- K-FAC-preconditioned test gradients,
        DVEmb data value embeddings -- are stored as **materialized** per-layer
        :class:`Gradient` records carrying the source blocks' steps and
        hashes, so scoring against the store is a plain inner product
        (e.g. ``TracInAttributor.attribute_from_cache``).  *transform*
        defaults to :meth:`transform_test_rep`; pass ``None`` explicitly via
        an identity when *source* already yields the final representation.

        Args:
            source: Blocks to transform, moved to ``args.device`` first.
            store: Destination store (any residency).
            transform: Per-block transform; default :meth:`transform_test_rep`.
            sample_id_key: Identifier scheme of the written records (a store
                inherits its source's scheme).

        Returns:
            *store*, for chaining.
        """
        transform = transform if transform is not None else self.transform_test_rep
        for step, block, hashes in source:
            rep = transform(block.to(self.args.device)).materialize().to("cpu")
            store.save_bulk(
                [
                    GradientRecord(
                        step=step,
                        input_hash=list(hashes),
                        gradient=rep,
                        sample_id_key=sample_id_key,
                    ),
                ],
            )
        return store

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: str | None = None,
        hook_config: HookManagerConfig | None = None,
        enable_update: bool = False,
        on_train_block: Callable[[int, Gradient, list[str]], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Collect the train and test gradients, live, to disk.

        Reproducing :meth:`attribute` is then *cache + attribute_from_cache*
        over the returned pairs (one per checkpoint, each internally aligned;
        summing their scores is the multi-checkpoint ensemble).

        * ``enable_update=False`` (default): one ``(train, test)`` pair per
          checkpoint in :meth:`checkpoints`, both sides frozen probes at that
          same checkpoint, under ``<cache_dir>/ckpt_<k>/``.
        * ``enable_update=True``: a single pair -- the test gradients at the
          first checkpoint, then a training **trajectory** from it (train per
          optimizer step).

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            cache_dir: Parent directory; defaults to ``args.output_dir``.
            hook_config: Capture configuration for both streamers.
            enable_update: Trajectory vs. per-checkpoint frozen probes.
            on_train_block: Hook invoked on every streamed **train** block
                (see :func:`collect_gradients`); the on-the-fly way to fit
                side quantities such as K-FAC covariances in the same pass.

        Returns:
            The ``(train_gradients_dir, test_gradients_dir)`` pairs.
        """
        self.require_task("cache")
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir

        def pair(k: int) -> tuple[str, str]:
            root = Path(cache_dir) / f"ckpt_{k}"
            return str(root / "train_grads"), str(root / "test_grads")

        pairs: list[tuple[str, str]] = []
        for k in self.checkpoints():
            self.load_checkpoint(k)
            train_dir, test_dir = pair(k)
            # Test first: with enable_update the train pass advances the model.
            self.collect_gradients(
                self.generate_test_rep(
                    test_dataset,
                    checkpoint_step=k,
                    hook_config=hook_config,
                ),
                GradientStorageManager(test_dir),
            )
            self.collect_gradients(
                self.generate_train_rep(
                    train_dataset,
                    checkpoint_step=k,
                    enable_update=enable_update,
                    hook_config=hook_config,
                ),
                GradientStorageManager(train_dir),
                on_block=on_train_block,
            )
            pairs.append((train_dir, test_dir))
            if enable_update:
                break  # a trajectory regenerates from the first checkpoint only
        return pairs

    # ------------------------------------------------------------------ #
    # Method hooks                                                         #
    # ------------------------------------------------------------------ #

    def prepare_scoring(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
    ) -> None:
        """One-time work before a scoring pass, computed from the sources.

        Called once per :meth:`attribute` checkpoint pass and once per
        :meth:`attribute_from_cache`, before any block is transformed.  The
        place to fit what :meth:`transform_test_rep` / :meth:`inner_product`
        need from the training gradients (K-FAC fits its covariances here;
        that needs ``train_source.reusable``).  No-op by default.
        """

    def transform_train_rep(self, train_rep: Gradient) -> Gradient:  # noqa: PLR6301
        """Transform one device-resident train block (identity by default).

        Inner-product attributors score ``<T_train(g_train), T_test(g_test)>``;
        this is ``T_train`` -- e.g. a dimension reduction.
        """
        return train_rep

    def transform_test_rep(self, test_rep: Gradient) -> Gradient:  # noqa: PLR6301
        """Transform one device-resident test block (identity by default).

        This is ``T_test`` -- e.g. multiplication by an inverse Fisher.  The
        result may hold layers factorized (raw or final factors) or dense;
        :meth:`inner_product` handles every combination.
        """
        return test_rep

    def inner_product(  # noqa: PLR6301 - overridable hook
        self,
        train_rep: Gradient,
        test_rep: Gradient,
        *,
        dense_cache: TensorCache | None = None,
    ) -> torch.Tensor:
        """``(B_train, B_test)`` score of one train rep against one test rep.

        The default is the layerwise cross-gram
        (:func:`~dattri_llm.gradient.ops.layerwise_cross_dot`) over the
        layers the two blocks share, with *dense_cache* -- a cache scoped to
        this train block by the scoring loop -- making a factorized train
        layer materialize once across the test blocks it meets.  A pair that
        shares no layer scores zero.
        """
        if not any(name in test_rep.data for name in train_rep.data):
            return torch.zeros(train_rep.batch_size, test_rep.batch_size)
        return ops.layerwise_cross_dot(train_rep, test_rep, dense_cache=dense_cache)

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def score_sources(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
        *,
        loop_over_test: bool = False,
        transform_test: Callable[[Gradient], Gradient] | None = None,
    ) -> tuple[torch.Tensor, list[str], list[int], list[str]]:
        """Score every train block against every test block with this
        attributor's hooks (see :func:`~dattri_llm.attribution.utils.score_sources`).

        Runs :meth:`prepare_scoring` first.  *transform_test* overrides
        :meth:`transform_test_rep` -- e.g. the identity when the test source
        already holds preconditioned representations.
        """
        self.prepare_scoring(train_source, test_source)
        return score_sources(
            train_source,
            test_source,
            self.args.device,
            inner_product=self.inner_product,
            transform_train=self.transform_train_rep,
            transform_test=(
                transform_test
                if transform_test is not None
                else self.transform_test_rep
            ),
            batch_size=self.args.per_device_train_batch_size or 1,
            loop_over_test=loop_over_test,
        )

    def build_score(
        self,
        scores: torch.Tensor,
        row_train_ids: list[str],
        row_steps: list[int],
        test_ids: list[str],
        *,
        algorithm_meta: dict | None = None,
        layer_name: list[str] | None = None,
    ) -> AttributionScore:
        """Assemble the :class:`AttributionScore` and persist it to
        ``args.output_dir``.
        """
        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta=dict(algorithm_meta or {}),
            algorithm=self.algorithm,
            layer_name=layer_name,
        )
        result.save(self.args.output_path)
        return result

    @staticmethod
    def stores_meta(
        train_store: GradientStorageManager,
        test_store: GradientStorageManager,
    ) -> dict:
        """The identifier scheme each store was collected under (``None`` =
        content hashing) -- recorded so the row/column ids are unambiguous.
        """
        return {
            "sample_id_key": {
                "train": train_store.sample_id_key,
                "test": test_store.sample_id_key,
            },
        }

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        verbose: bool = False,  # noqa: ARG002 - live streamers show no bars
        loop_over_test: bool = False,
        enable_update: bool = False,
        gradient_cache_residency: str | None = None,
        **attribution_kwargs: object,
    ) -> AttributionScore:
        """Attribute **on the fly** over the task's checkpoints.

        * ``enable_update=False`` (default) -- one frozen pass per checkpoint
          in :meth:`checkpoints`; rows are stamped with the checkpoint index
          and summed over checkpoints by the score's agnostic queries.
        * ``enable_update=True`` -- a single training **trajectory** from the
          first checkpoint; each optimizer step is its own row stamp.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            hook_config: Capture configuration for the internal streamers.
                The test streamer shares the train streamer's hooks.
            verbose: Accepted for API parity (live streams show no bars).
            loop_over_test: Re-stream the test blocks per train block (low
                memory) instead of caching them once (default).
            enable_update: Trajectory vs. frozen multi-checkpoint scoring.
            gradient_cache_residency: ``None`` (default) streams the
                gradients straight into scoring, re-running the model for any
                extra pass a method needs.  ``"memory"``/``"tiered"``/``"disk"``
                instead collects each side **once** into a store of that
                residency and scores from it -- the choice when a method
                re-reads the train gradients (K-FAC) or the captures are
                cheap to hold (projected).  Ephemeral stores are released on
                return; ``"disk"`` persists under ``args.output_dir``.
            **attribution_kwargs: Method-specific options (those of
                :meth:`attribute_from_cache`), recorded in the score's
                metadata.
        """
        self.require_task("attribute")
        if gradient_cache_residency is not None:
            if gradient_cache_residency not in CACHE_RESIDENCIES:
                raise ValueError(
                    "gradient_cache_residency must be one of "
                    f"{list(CACHE_RESIDENCIES)} or None, "
                    f"got {gradient_cache_residency!r}.",
                )
            return self._attribute_via_stores(
                train_dataset,
                test_dataset,
                residency=gradient_cache_residency,
                hook_config=hook_config,
                loop_over_test=loop_over_test,
                enable_update=enable_update,
                **attribution_kwargs,
            )

        checkpoints = self.checkpoints()
        if enable_update and len(checkpoints) > 1:
            warnings.warn(
                f"enable_update=True regenerates the trajectory from checkpoint "
                f"0; the other {len(checkpoints) - 1} provided checkpoint(s) are "
                "ignored.",
                stacklevel=2,
            )
            checkpoints = checkpoints[:1]

        row_blocks: list[torch.Tensor] = []
        row_train_ids: list[str] = []
        row_steps: list[int] = []
        test_ids: list[str] | None = None
        hooked_layers: list[str] = []
        for k in checkpoints:
            self.load_checkpoint(k)
            train = self.generate_train_rep(
                train_dataset,
                checkpoint_step=k,
                enable_update=enable_update,
                hook_config=hook_config,
            )
            test = self.generate_test_rep(
                test_dataset,
                checkpoint_step=k,
                hook_manager=train.hook_manager,
            )
            hooked_layers = list(train.hook_manager.layer_name)
            with train, test:
                sc, rids, rsteps, tids = self.score_sources(
                    train,
                    test,
                    loop_over_test=loop_over_test,
                )
            row_blocks.append(sc)
            row_train_ids.extend(rids)
            row_steps.extend(rsteps)
            test_ids = tids if test_ids is None else test_ids
        return self.build_score(
            torch.cat(row_blocks, dim=0) if row_blocks else torch.zeros(0, 0),
            row_train_ids,
            row_steps,
            test_ids or [],
            algorithm_meta={
                "n_checkpoints": len(checkpoints),
                "enable_update": enable_update,
                **attribution_kwargs,
            },
            layer_name=hooked_layers or None,
        )

    def _attribute_via_stores(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        residency: str,
        hook_config: HookManagerConfig | None,
        loop_over_test: bool,
        enable_update: bool,
        **attribution_kwargs: object,
    ) -> AttributionScore:
        """On-the-fly attribution with the gradients collected once into stores
        of *residency*, then scored from them.  Ephemeral stores are
        context-managed so a tiered spill is cleaned up on exit.
        """
        if residency == "disk":
            pairs = self.cache(
                train_dataset,
                test_dataset,
                hook_config=hook_config,
                enable_update=enable_update,
            )
            results = [
                self.attribute_from_cache(
                    train_dir,
                    test_dir,
                    loop_over_test=loop_over_test,
                    **attribution_kwargs,
                )
                for train_dir, test_dir in pairs
            ]
            return self._stack_scores(results, len(pairs))
        import tempfile

        checkpoints = self.checkpoints()[:1] if enable_update else self.checkpoints()
        results: list[AttributionScore] = []
        for k in checkpoints:
            self.load_checkpoint(k)
            with (
                GradientStorageManager(
                    tempfile.mkdtemp(prefix=f"{self.algorithm.lower()}_train_"),
                    residency=residency,
                ) as train_store,
                GradientStorageManager(
                    tempfile.mkdtemp(prefix=f"{self.algorithm.lower()}_test_"),
                    residency=residency,
                ) as test_store,
            ):
                self.collect_gradients(
                    self.generate_test_rep(
                        test_dataset,
                        checkpoint_step=k,
                        hook_config=hook_config,
                    ),
                    test_store,
                )
                self.collect_gradients(
                    self.generate_train_rep(
                        train_dataset,
                        checkpoint_step=k,
                        enable_update=enable_update,
                        hook_config=hook_config,
                    ),
                    train_store,
                )
                results.append(
                    self.attribute_from_cache(
                        train_store,
                        test_store,
                        loop_over_test=loop_over_test,
                        algorithm_meta={"gradient_cache_residency": residency},
                        **attribution_kwargs,
                    ),
                )
        return self._stack_scores(results, len(checkpoints))

    def _stack_scores(
        self,
        results: list[AttributionScore],
        n_checkpoints: int,
    ) -> AttributionScore:
        """Concatenate per-checkpoint scores into one (rows keep their stamps)."""
        if len(results) == 1:
            return results[0]
        first = results[0]
        return self.build_score(
            torch.cat([r.scores for r in results], dim=0),
            [i for r in results for i in r.row_train_ids],
            [s for r in results for s in r.row_steps],
            first.test_ids,
            algorithm_meta={**first.algorithm_meta, "n_checkpoints": n_checkpoints},
            layer_name=first.layer_name,
        )

    @staticmethod
    def resolve_store(
        source: str | Path | GradientStorageManager | DiskGradientSource,
    ) -> GradientStorageManager:
        """The open store behind a directory, a store, or a disk source."""
        if isinstance(source, GradientStorageManager):
            return source
        if isinstance(source, DiskGradientSource):
            return source.file_manager
        if isinstance(source, (str, Path)):
            return GradientStorageManager(str(source))
        raise TypeError(
            "expected a store directory, a GradientStorageManager, or a "
            f"DiskGradientSource, got {type(source).__name__}.",
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
        **attribution_kwargs: object,
    ) -> AttributionScore:
        """Score previously collected gradients (the *store-then-attribute* path).

        Every train record is scored against every test record; rows/columns
        are the train/test identifiers in store order, each row stamped with
        the step its gradient was recorded at.  The sources may be store
        directories, open stores of any residency, or disk sources -- so this
        also serves gradients held in RAM (a manual collection into a
        ``residency="memory"`` store, or :meth:`attribute`'s in-RAM path).
        Methods with extra per-call options override this method and forward
        the rest to ``super()``.

        Args:
            train_source: Train gradients -- directory, open store, or source.
            test_source: Test gradients, likewise.
            selected_training_steps: Restrict the training steps (the output
                rows) to these; ``None`` uses every step in the store
                (over-specified ranges are intersected).  The test set always
                supplies every column.
            layer_name: Restrict scoring to this subset of the *stored* layers.
            verbose: Show progress bars on the logging process.
            loop_over_test: Re-stream the test blocks per train block (low
                memory) instead of caching them once (default).
            algorithm_meta: Extra entries for the score's metadata.
            **attribution_kwargs: Recorded in the score's metadata; a method
                with its own options consumes them before calling ``super()``.
        """
        train_store = self.resolve_store(train_source)
        test_store = self.resolve_store(test_source)
        layer_name = normalize_layer_names(layer_name)
        train = self.load_train_rep(
            train_store,
            steps=selected_training_steps,
            layer_name=layer_name,
            verbose=verbose,
            desc=f"{self.algorithm}: scoring",
        )
        test = self.load_test_rep(
            test_store,
            layer_name=layer_name,
            verbose=verbose,
            desc=f"{self.algorithm}: loading test",
        )
        scores, row_train_ids, row_steps, test_ids = self.score_sources(
            train,
            test,
            loop_over_test=loop_over_test,
        )
        return self.build_score(
            scores,
            row_train_ids,
            row_steps,
            test_ids,
            algorithm_meta={
                "selected_training_steps": train.steps,
                **self.stores_meta(train_store, test_store),
                **(algorithm_meta or {}),
                **attribution_kwargs,
            },
            layer_name=layer_name,
        )
