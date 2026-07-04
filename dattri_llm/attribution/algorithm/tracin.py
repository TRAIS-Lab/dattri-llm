"""TracIn / GradCos attribution over on-disk gradients (workflow 2).

This attributor consumes :class:`~dattri_llm.gradient.gradient.Gradient`
records produced earlier by the gradient-collection pipeline and persisted to
disk via :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.  No
forward/backward pass is performed at attribution time — only inner products
between pre-stored per-sample gradients.

Every train record is scored against every test record::

    score[i, j] = <g_train_i, g_test_j>

i.e. the full ``(num_train, num_test)`` cross-gram, exactly mirroring the
:class:`~dattri_llm.attribution.algorithm.kronecker.KFACAttributor` assembly — there is no
train/test step alignment.  With ``normalized_grad=True`` the inner product
becomes a cosine similarity (the GradCos / CosIn variant);
``GradCos`` is TracIn with ``normalized_grad=True`` passed to the attribute
methods — there is no separate subclass.

The result is a :class:`~dattri_llm.attribution.score.AttributionScore`.  Rows are
stamped with the step each train gradient was recorded at (read straight off the
on-disk records), so a sample collected at several checkpoints contributes one
row per checkpoint; summing a sample's rows over steps recovers the classic
dattri ``(num_train, num_test)`` matrix.  Rows and columns are identified by
content hash in on-disk order, so no reconstruction of the original DataLoader is
required.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.attribution.base import BaseAttributor
from dattri_llm.attribution.utils import (
    collect_to_disk,
    normalize_layer_names,
    score_sources,
    task_loss_fn,
)
from dattri_llm.attribution.score import AttributionScore
from dattri_llm.gradient.streaming import (
    DiskGradientSource,
    GradientSource,
    GradientStreamer,
)
from dattri.task import AttributionTask
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManagerConfig


class TracInAttributor(BaseAttributor):
    """TracIn / GradCos attributor that consumes pre-collected on-disk gradients.

    Scores every train record against every test record (the full
    ``(num_train, num_test)`` gradient cross-gram), with no train/test step
    alignment — structurally identical to the K-FAC family, minus the Fisher
    preconditioner.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory where the score is
            persisted.  Only the ``dataloader_*`` fields, ``device``, and
            ``output_dir`` are consulted in this workflow.
        task: Accepted for parity with the workflow-1 API but unused here.

    ``normalized_grad`` (cosine / GradCos vs raw inner product / TracIn) is a
    per-attribution choice passed to :meth:`attribute` /
    :meth:`attribute_from_cache`.

    Layer selection happens at **capture** (via the ``hook_config`` of the live
    methods, or whatever was hooked when the cache was collected).  By default
    every stored layer is scored; :meth:`attribute_from_cache` additionally takes
    a ``layer_name`` read-time filter to score a subset of the stored layers.
    """

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: Optional[AttributionTask] = None,
    ) -> None:
        self.args = args
        self.task = task

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: Optional[str] = None,
        enable_update: bool = False,
        hook_config: Optional[HookManagerConfig] = None,
    ) -> List[Tuple[str, str]]:
        """Collect the gradients TracIn/GradCos needs, live, to disk.

        Runs forward/backward over the data (from the task's model, loss, and
        optional ``target_func``) and offloads the per-sample gradients so that
        :meth:`attribute_from_cache` can score them.  Reproducing :meth:`attribute`
        is then *cache + attribute_from_cache* — scoring **each returned pair** and
        **summing** the results (one pair per checkpoint, each internally aligned)::

            pairs = attr.cache(train_ds, test_ds)
            total = sum(attr.attribute_from_cache(tr, te).agnostic_matrix()[1]
                        for tr, te in pairs)

        * ``enable_update=False`` (default): one ``(train, test)`` pair **per
          checkpoint** the task provides, both sides collected **at that same
          ``θ_k``** (so each pair's score is ``⟨g_train@θ_k, g_test@θ_k⟩`` — the
          step-aligned term).  Summing the pairs is the multi-checkpoint ensemble.
        * ``enable_update=True``: a single pair — a training **trajectory** from
          checkpoint 0 (train per optimizer step) with the test gradients taken
          once at ``θ_0`` (mirroring :meth:`attribute`, which reads the test side
          before training).

        Args:
            train_dataset, test_dataset: Datasets to stream.
            cache_dir: Parent dir; each pair goes under ``<cache_dir>/ckpt_<k>/``.
                Defaults to ``args.output_dir``.
            enable_update: Trajectory (train as it streams) vs. per-checkpoint.
            hook_config: :class:`HookManagerConfig` for the internal streamers
                (which layers to hook, per-layer projection, ...).  ``None`` uses
                the streamer default (factorized hooks on every linear-family layer).

        Returns:
            A list of ``(train_gradients_dir, test_gradients_dir)`` pairs — one per
            checkpoint (frozen) or a single pair (trajectory).
        """
        if self.task is None:
            raise ValueError(
                "cache() (live collection) requires a `task` with a model; pass "
                "pre-collected gradients to attribute_from_cache() instead."
            )
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        train_loss = task_loss_fn(self.task.original_loss_func)
        test_loss = task_loss_fn(self.task.original_target_func)
        tb = self.args.per_device_train_batch_size
        vb = self.args.per_device_eval_batch_size

        def _pair(k: int) -> Tuple[str, str]:
            train_dir = os.path.join(cache_dir, f"ckpt_{k}", "train_grads")
            test_dir = os.path.join(cache_dir, f"ckpt_{k}", "test_grads")
            return train_dir, test_dir

        if enable_update:
            n_ckpt = len(self.task.get_checkpoints())
            if n_ckpt > 1:
                warnings.warn(
                    f"enable_update=True regenerates the trajectory from checkpoint "
                    f"0; the other {n_ckpt - 1} provided checkpoint(s) are ignored.",
                    stacklevel=2,
                )
            self.task._load_checkpoints(0)
            model = self.task.get_model()
            train_dir, test_dir = _pair(0)
            # Test FIRST, at θ_0 — mirroring attribute(), where score_sources reads
            # the test side before the training pass advances the model.
            collect_to_disk(
                GradientStreamer(model, test_dataset, self.args, batch_size=vb,
                                 enable_update=False, loss_fn=test_loss,
                                 config=hook_config),
                GradientFileManager(test_dir),
            )
            collect_to_disk(
                GradientStreamer(model, train_dataset, self.args, batch_size=tb,
                                 enable_update=True, loss_fn=train_loss,
                                 config=hook_config),
                GradientFileManager(train_dir),
            )
            return [(train_dir, test_dir)]

        # Frozen: a self-contained, step-aligned pair per checkpoint.
        pairs: List[Tuple[str, str]] = []
        for k in range(len(self.task.get_checkpoints())):
            self.task._load_checkpoints(k)
            model = self.task.get_model()
            train_dir, test_dir = _pair(k)
            collect_to_disk(
                GradientStreamer(model, test_dataset, self.args, batch_size=vb,
                                 enable_update=False, loss_fn=test_loss,
                                 checkpoint_step=k, config=hook_config),
                GradientFileManager(test_dir),
            )
            collect_to_disk(
                GradientStreamer(model, train_dataset, self.args, batch_size=tb,
                                 enable_update=False, loss_fn=train_loss,
                                 checkpoint_step=k, config=hook_config),
                GradientFileManager(train_dir),
            )
            pairs.append((train_dir, test_dir))
        return pairs

    @staticmethod
    def _label(normalized_grad: bool) -> Tuple[str, str]:
        """(algorithm name, similarity metric) for the normalization choice."""
        return (
            ("GradCos", "cosine") if normalized_grad else ("TracIn", "dot")
        )

    def _run(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
        *,
        loop_over_test: bool = False,
        algorithm_meta: Optional[dict] = None,
        layer_name: Optional[List[str]] = None,
        normalized_grad: bool = False,
    ) -> AttributionScore:
        """Score a train source against a test source — the shared loop.

        ``train_source`` is iterated **once**, so a single-shot trajectory stream
        (`GradientStreamer` with `enable_update=True`) works as well as a re-iterable disk
        source.  TracIn uses the raw gradients with no preconditioner, so
        ``prepare_test`` is the identity and ``score_block`` is the cross-gram
        (``"dot"``, or ``"cosine"`` for GradCos).
        """
        name, metric = self._label(normalized_grad)
        scores, row_train_ids, row_steps, test_ids = score_sources(
            train_source,
            test_source,
            self.args.device,
            prepare_test=lambda g: g,
            score_block=lambda tr, te, _n: tr.similarity(te, metric=metric, reduce="all"),
            loop_over_test=loop_over_test,
        )
        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta={"normalized_grad": normalized_grad,
                            **(algorithm_meta or {})},
            algorithm=name,
            layer_name=layer_name,
        )
        result.save(self.args.output_path)
        return result

    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        *,
        selected_training_steps: Optional[Iterable[int]] = None,
        loop_over_test: bool = False,
        verbose: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        normalized_grad: bool = False,
    ) -> AttributionScore:
        """Score pre-collected on-disk gradients (the *store-then-attribute* path).

        Every train record is scored against every test record; rows/columns are
        the train/test content hashes in on-disk order, each row stamped with the
        step its gradient was recorded at.

        Args:
            train_gradients_dir: Directory written by :class:`GradientFileManager`
                during the training pass.
            test_gradients_dir: Directory written during the test pass.
            selected_training_steps: Restrict the training checkpoints (the output
                rows) to these steps; ``None`` uses every step on disk
                (over-specified ranges are intersected).  The test set always
                supplies every column.
            loop_over_test: Re-stream the test blocks per train block (low memory)
                instead of caching them once (default).
            verbose: Show tqdm progress bars on the logging process.
            layer_name: Restrict scoring to this subset of the *stored* layers
                (``str`` or list; unknown names raise).  ``None`` (default) scores
                every stored layer.  This is a read-time filter — the cache itself
                is unchanged, so the same cache can be re-queried per layer.
            normalized_grad: ``True`` scores by cosine similarity (GradCos);
                ``False`` (default) by the raw inner product (TracIn / GradDot).

        Returns:
            An :class:`AttributionScore`, also persisted to ``args.output_dir``.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__}.attribute_from_cache requires both "
                "train_gradients_dir and test_gradients_dir."
            )
        layer_name = normalize_layer_names(layer_name)
        name, _ = self._label(normalized_grad)
        train = DiskGradientSource(
            GradientFileManager(train_gradients_dir), self.args,
            steps=selected_training_steps, layer_name=layer_name,
            desc=f"{name}: scoring", verbose=verbose,
        )
        test = DiskGradientSource(
            GradientFileManager(test_gradients_dir), self.args,
            layer_name=layer_name,
            desc=f"{name}: loading test", verbose=verbose,
        )
        return self._run(
            train, test, loop_over_test=loop_over_test,
            algorithm_meta={"selected_training_steps": train._steps},
            layer_name=layer_name,
            normalized_grad=normalized_grad,
        )

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        loop_over_test: bool = False,
        enable_update: bool = False,
        hook_config: Optional[HookManagerConfig] = None,
        normalized_grad: bool = False,
    ) -> AttributionScore:
        """Score by collecting gradients **live** (on-the-fly, workflow 1).

        Everything but the data and these two flags comes from the
        :class:`AttributionTask` (model, loss, optional ``target_func`` for the
        test side, and the checkpoint list) and from ``args`` (batch sizes,
        precision, and — when ``enable_update`` — the optimizer/scheduler, all
        settled inside :class:`GradientStreamer`).

        * ``enable_update=False`` (default) — score at **every checkpoint** the
          task provides (dattri's multi-checkpoint TracIn ensemble): one frozen
          pass per checkpoint, the rows stamped with the checkpoint index, summed
          over checkpoints at query time.
        * ``enable_update=True`` — a single training **trajectory** from the first
          checkpoint; each optimizer step is its own checkpoint row.

        Args:
            train_dataset, test_dataset: Datasets to stream (the loader batches
                are passed straight to the task's loss as its ``data``).
            loop_over_test: As in :meth:`attribute_from_cache`.
            enable_update: Train the model as it streams (trajectory) vs. frozen
                multi-checkpoint scoring.
            hook_config: :class:`HookManagerConfig` for the internal streamers
                (which layers to hook, per-layer projection, ...).  ``None`` uses
                the streamer default (factorized hooks on every linear-family
                layer).  The test streamer shares the train streamer's hooks, so
                one config governs both sides.
            normalized_grad: ``True`` scores by cosine similarity (GradCos);
                ``False`` (default) by the raw inner product (TracIn / GradDot).
        """
        if self.task is None:
            raise ValueError(
                "attribute() (live collection) requires a `task` with a model; "
                "use attribute_from_cache() for pre-collected gradients."
            )
        train_loss = task_loss_fn(self.task.original_loss_func)
        # The test side is attributed against the task's target_func (which dattri
        # defaults to the loss when none is given).
        test_loss = task_loss_fn(self.task.original_target_func)
        tb = self.args.per_device_train_batch_size
        vb = self.args.per_device_eval_batch_size
        device = self.args.device
        name, metric = self._label(normalized_grad)
        score_block = lambda tr, te, _n: tr.similarity(te, metric=metric, reduce="all")
        # Layer selection is the hook config's job; record what was hooked (the
        # scored layer set) for the AttributionScore metadata.
        hooked_layers: List[str] = []

        def _score_one(checkpoint_step: int, update: bool):
            model = self.task.get_model()
            train = GradientStreamer(
                model, train_dataset, self.args, batch_size=tb,
                enable_update=update, loss_fn=train_loss, checkpoint_step=checkpoint_step,
                config=hook_config,
            )
            test = GradientStreamer(
                model, test_dataset, self.args, batch_size=vb,
                loss_fn=test_loss, checkpoint_step=checkpoint_step,
                hook_manager=train.hook_manager,  # one set of hooks over the model
            )
            hooked_layers[:] = train.hook_manager.layer_name
            with train, test:
                return score_sources(
                    train, test, device, prepare_test=lambda g: g,
                    score_block=score_block, loop_over_test=loop_over_test,
                )

        row_blocks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        test_ids: Optional[List[str]] = None
        if enable_update:
            # Single trajectory from the first checkpoint; steps are the optimizer
            # steps (the streamer's own per-pass index).  The trajectory is
            # regenerated live, so any other provided checkpoints are unused.
            n_ckpt = len(self.task.get_checkpoints())
            if n_ckpt > 1:
                warnings.warn(
                    f"enable_update=True regenerates the trajectory from checkpoint "
                    f"0; the other {n_ckpt - 1} provided checkpoint(s) are ignored. "
                    "Use enable_update=False for multi-checkpoint (frozen) TracIn.",
                    stacklevel=2,
                )
            self.task._load_checkpoints(0)
            sc, rids, rsteps, tids = _score_one(0, update=True)
            row_blocks, row_train_ids, row_steps, test_ids = [sc], rids, rsteps, tids
        else:
            # One frozen pass per provided checkpoint; rows stamped with the
            # checkpoint index, summed over checkpoints by the agnostic query.
            for k in range(len(self.task.get_checkpoints())):
                self.task._load_checkpoints(k)
                sc, rids, _rs, tids = _score_one(k, update=False)
                row_blocks.append(sc)
                row_train_ids.extend(rids)
                row_steps.extend([k] * len(rids))
                test_ids = tids if test_ids is None else test_ids

        scores = torch.cat(row_blocks, dim=0) if row_blocks else torch.zeros(0, 0)
        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids or [],
            algorithm_meta={"n_checkpoints": len(self.task.get_checkpoints()),
                            "normalized_grad": normalized_grad},
            algorithm=name,
            layer_name=hooked_layers or None,   # what the hook manager captured
        )
        result.save(self.args.output_path)
        return result
