"""TracIn and GradCos attributors over on-disk gradients (workflow 2).

These attributors consume :class:`~dattri_llm.gradient.gradient.Gradient`
records produced earlier by the gradient-collection pipeline and persisted
to disk via :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.
No forward/backward pass is performed at attribution time — only inner
products between pre-stored per-sample gradients.

Score::

    score[i, j] = sum_k weight_k * <g_train_i^(k), g_test_j^(k)>

where ``k`` indexes the saved gradient *steps* used as TracIn ensemble
terms.  ``g_*^(k)`` is the per-sample weight gradient flattened into a
single vector across all hooked layers.  With ``normalized_grad=True`` the
inner product is replaced by a cosine similarity per ``(i, j)`` — this is
the GradCos variant.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import BaseAttributor
from dattri_llm.algorithm.task import AttributionTask
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient
from dattri_llm.gradient.utils import hash_sample


# --------------------------------------------------------------------------- #
# TracIn / GradCos                                                             #
# --------------------------------------------------------------------------- #


class TracInAttributor(BaseAttributor):
    """TracIn attributor that consumes pre-collected on-disk gradients.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory where the score tensor
            is persisted.  Only ``per_device_train_batch_size``,
            ``per_device_eval_batch_size``, the dataloader_* fields,
            ``device``, and ``output_dir`` are consulted in this workflow.
        weight_list: Per-step weights, one per included step.  Typically the
            learning rate at each saved step.  If ``None`` (and ``steps`` is
            also ``None``), every common step is included with uniform weight
            ``1.0`` — i.e. the GradDot variant.  Must have the same length as
            ``steps`` (or as the auto-discovered step list).
        normalized_grad: If ``True``, L2-normalise each per-sample flattened
            gradient before the dot product.  ``True`` corresponds to
            CosIn / GradCos; ``False`` to TracIn / GradDot.
        layer_name: Restrict the inner product to these layer names (matching
            keys of the saved :class:`Gradient`).  ``None`` uses every layer
            present in both the train and test gradients.
        steps: Saved step indices to include as TracIn ensemble terms.
            ``None`` auto-discovers the intersection of steps present in both
            ``train_gradients_dir`` and ``test_gradients_dir``.
        token_reduction: How to reduce the token dim of ``indexing="batch_token"``
            gradients before flattening.  ``"sum"`` (default) matches a
            sum-over-tokens loss reduction; ``"mean"`` matches a mean-over-tokens
            loss reduction.  This must match the loss reduction used during
            gradient collection if you want scores comparable across sequences
            of different lengths.
        task: Accepted for parity with the workflow-1 API but unused in this
            workflow.

    Raises:
        ValueError: If ``token_reduction`` is not ``"sum"`` or ``"mean"``.
    """

    def __init__(
        self,
        args: AttributionArguments,
        *,
        weight_list: Optional[Sequence[float]] = None,
        normalized_grad: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        steps: Optional[Sequence[int]] = None,
        token_reduction: str = "sum",
        task: Optional[AttributionTask] = None,
    ) -> None:
        if token_reduction not in ("sum", "mean"):
            raise ValueError(
                f"token_reduction must be 'sum' or 'mean', got {token_reduction!r}."
            )
        self.args = args
        self.task = task
        self.weight_list = list(weight_list) if weight_list is not None else None
        self.normalized_grad = normalized_grad
        if isinstance(layer_name, str):
            self.layer_name: Optional[List[str]] = [layer_name]
        elif layer_name is None:
            self.layer_name = None
        else:
            self.layer_name = list(layer_name)
        self.steps = list(steps) if steps is not None else None
        self.token_reduction = token_reduction

    def cache(self, train_dataset: Dataset) -> None:
        pass

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        train_gradients_dir: Optional[str] = None,
        test_gradients_dir: Optional[str] = None,
        loop_over_test: bool = False,
    ) -> torch.Tensor:
        """Compute the (num_train, num_test) attribution score matrix.

        Iterates the train and test datasets in their natural order with
        ``shuffle=False``; row ``i`` corresponds to ``train_dataset[i]`` and
        column ``j`` to ``test_dataset[j]``.  Each sample's gradient is
        retrieved from disk by content hash (:func:`hash_sample`), so the
        collation that produced the saved gradients must yield identical
        per-sample tensors here (same padding policy, same field ordering).

        Args:
            train_dataset: The training dataset whose influence is measured.
            test_dataset: The test (query) dataset.
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during the training pass.
            test_gradients_dir: Directory written by
                :class:`GradientFileManager` during the test forward+backward
                pass.
            loop_over_test: Controls the streaming strategy for the inner loop.

                * ``False`` (default) — test-batch gradients are loaded once
                  per step and cached on device; train batches are streamed one
                  at a time.  Peak memory: one train batch + all test batches.
                  Disk reads per step: ``N_train_batches + N_test_batches``.

                * ``True`` — both train and test batches are streamed; no
                  caching.  Peak memory: one train batch + one test batch.
                  Disk reads per step:
                  ``N_train_batches x N_test_batches + N_train_batches``.
                  Use when the test set is too large to hold on device at once.

        Returns:
            Float tensor of shape ``(len(train_dataset), len(test_dataset))``.
            Also persisted to ``args.output_dir/scores.pt`` alongside a
            ``metadata.json`` describing the run.

        Raises:
            ValueError: If either gradients dir is missing, or no common
                steps are found between the two dirs, or ``weight_list``
                length disagrees with the resolved step list.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                "TracInAttributor requires both train_gradients_dir and "
                "test_gradients_dir.  On-the-fly gradient collection (workflow 1) "
                "is not implemented yet."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)

        steps = self._resolve_steps(train_fm, test_fm)
        weights = self._resolve_weights(steps)

        train_loader = self._make_dataloader(train_dataset, train=True)
        test_loader = self._make_dataloader(test_dataset, train=False)

        n_train = len(train_dataset)
        n_test = len(test_dataset)
        device = self.args.device

        # Precompute per-batch sample hashes so we do not re-hash each step.
        train_batch_hashes = [
            [hash_sample(b, i) for i in range(_batch_size(b))] for b in train_loader
        ]
        test_batch_hashes = [
            [hash_sample(b, i) for i in range(_batch_size(b))] for b in test_loader
        ]
        if sum(len(h) for h in train_batch_hashes) != n_train:
            raise RuntimeError(
                "Train DataLoader produced a different total sample count than "
                "len(train_dataset)."
            )
        if sum(len(h) for h in test_batch_hashes) != n_test:
            raise RuntimeError(
                "Test DataLoader produced a different total sample count than "
                "len(test_dataset)."
            )

        scores = torch.zeros(n_train, n_test, dtype=torch.float)

        metric = "cosine" if self.normalized_grad else "dot"
        tok_reduction = "mean" if self.token_reduction == "mean" else None

        for step, weight in zip(steps, weights):
            if not loop_over_test:
                test_batch_grads = [
                    _load_batch_gradients(test_fm, step, hashes, self.layer_name).to(device)
                    for hashes in test_batch_hashes
                ]

            train_row = 0
            for train_hashes in train_batch_hashes:
                train_g = _load_batch_gradients(
                    train_fm, step, train_hashes, self.layer_name,
                ).to(device)
                n_tr = train_g.batch_size

                if loop_over_test:
                    test_batch_grads = [
                        _load_batch_gradients(test_fm, step, hashes, self.layer_name).to(device)
                        for hashes in test_batch_hashes
                    ]

                test_col = 0
                for test_g in test_batch_grads:
                    n_te = test_g.batch_size
                    block = train_g.similarity(
                        test_g,
                        metric=metric,
                        reduce="all",
                        token_reduction=tok_reduction,
                    )
                    scores[train_row:train_row + n_tr, test_col:test_col + n_te] += (
                        weight * block
                    ).detach().cpu()
                    test_col += n_te

                train_row += n_tr

        self._save_scores(scores, steps, weights)
        return scores

    def _resolve_steps(
        self,
        train_fm: GradientFileManager,
        test_fm: GradientFileManager,
    ) -> List[int]:
        if self.steps is not None:
            return list(self.steps)
        train_steps = {
            e["step"] for entries in train_fm.index.values() for e in entries
        }
        test_steps = {
            e["step"] for entries in test_fm.index.values() for e in entries
        }
        common = sorted(train_steps & test_steps)
        if not common:
            raise ValueError(
                "No common steps between train and test gradient directories."
            )
        return common

    def _resolve_weights(self, steps: List[int]) -> List[float]:
        if self.weight_list is None:
            return [1.0] * len(steps)
        if len(self.weight_list) != len(steps):
            raise ValueError(
                f"weight_list length ({len(self.weight_list)}) must equal the "
                f"number of steps ({len(steps)})."
            )
        return [float(w) for w in self.weight_list]

    def _make_dataloader(self, dataset: Dataset, *, train: bool) -> DataLoader:
        """Build a deterministic DataLoader (shuffle=False) from ``args``.

        Workflow 2 looks up gradients by content hash; the order in which
        the loader produces samples defines the row/column index of the
        output matrix, so shuffling is always disabled.
        """
        batch_size = (
            self.args.per_device_train_batch_size
            if train
            else self.args.per_device_eval_batch_size
        )
        kwargs: dict = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(**kwargs)

    def _save_scores(
        self,
        scores: torch.Tensor,
        steps: List[int],
        weights: List[float],
    ) -> None:
        out_dir = self.args.output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(scores, out_dir / "scores.pt")
        meta = {
            "algorithm": "GradCos" if self.normalized_grad else "TracIn",
            "normalized_grad": self.normalized_grad,
            "steps": steps,
            "weights": weights,
            "layer_name": self.layer_name,
            "token_reduction": self.token_reduction,
            "num_train": int(scores.shape[0]),
            "num_test": int(scores.shape[1]),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #


def _batch_size(batch: dict) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim > 0:
            return int(v.shape[0])
    raise ValueError("Cannot infer batch size: no batched tensor field in batch dict.")


def _load_sample_gradient(
    fm: GradientFileManager,
    step: int,
    h: str,
    layer_name: Optional[List[str]],
) -> Gradient:
    """Return a batch-size-1 :class:`Gradient` for one sample at one step."""
    rec = fm.load_by_hash(step, h)
    g = rec.gradient
    if isinstance(rec.input_hash, list):
        i = rec.input_hash.index(h)
        g = g.slice(dim="batch", index=i)
    if layer_name is not None:
        g = g.select_layers(layer_name)
    return g


def _load_batch_gradients(
    fm: GradientFileManager,
    step: int,
    hashes: List[str],
    layer_name: Optional[List[str]],
) -> Gradient:
    """Concatenate per-sample :class:`Gradient` lookups along the batch dim."""
    per_sample = [_load_sample_gradient(fm, step, h, layer_name) for h in hashes]
    out = per_sample[0]
    for g in per_sample[1:]:
        out = out.concatenate(g, dim="batch")
    return out
