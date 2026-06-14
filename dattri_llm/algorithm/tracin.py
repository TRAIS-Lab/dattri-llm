"""TracIn / GradCos attribution over on-disk gradients (workflow 2).

This attributor consumes :class:`~dattri_llm.gradient.gradient.Gradient`
records produced earlier by the gradient-collection pipeline and persisted to
disk via :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.  No
forward/backward pass is performed at attribution time — only inner products
between pre-stored per-sample gradients.

Per saved step ``k`` (a TracIn ensemble term)::

    score[(train_i, k), test_j] = weight_k * <g_train_i^(k), g_test_j^(k)>

where ``g_*^(k)`` is the per-sample weight gradient across all hooked layers.
With ``normalized_grad=True`` the inner product becomes a cosine similarity
(the GradCos / CosIn variant); ``GradCos = TracInAttributor(normalized_grad=True)``
— there is no separate subclass.

The result is a :class:`~dattri_llm.algorithm.result.AttributionScore`.  It is
*trajectory-aware* on the train side: one row per ``(train_hash, step)`` pair,
rather than a single step-summed row.  Summing a sample's rows over steps
recovers the classic dattri ``(num_train, num_test)`` matrix.  Rows and columns
are identified by content hash, read directly from the on-disk records, so no
reconstruction of the original training DataLoader is required — the ordering
is defined entirely by what is on disk.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import BaseAttributor
from dattri_llm.algorithm.result import AttributionScore
from dattri_llm.algorithm.task import AttributionTask
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient


class TracInAttributor(BaseAttributor):
    """TracIn / GradCos attributor that consumes pre-collected on-disk gradients.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory where the score is
            persisted.  Only the ``dataloader_*`` fields, ``device``, and
            ``output_dir`` are consulted in this workflow.
        weight_list: Per-step weights, one per included step (typically the
            learning rate at each saved step).  ``None`` uses uniform weight
            ``1.0`` for every step (GradDot).  Length must match the resolved
            step list.
        normalized_grad: If ``True``, use cosine similarity per ``(i, j)`` pair
            (GradCos / CosIn); if ``False``, the raw inner product (TracIn /
            GradDot).
        layer_name: Restrict the inner product to these layer names.  ``None``
            uses every layer shared between the train and test gradients.
        steps: Saved step indices to include as ensemble terms.  ``None``
            auto-discovers the intersection of steps present in both gradient
            directories.
        token_reduction: How to reduce the token dim of
            ``indexing="batch_token"`` gradients.  ``"sum"`` (default) matches a
            sum-over-tokens loss; ``"mean"`` matches a mean-over-tokens loss.
        task: Accepted for parity with the workflow-1 API but unused here.

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
        """No-op: gradients are already cached on disk in this workflow."""

    def attribute(
        self,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_gradients_dir: Optional[str] = None,
        test_gradients_dir: Optional[str] = None,
        loop_over_test: bool = False,
    ) -> AttributionScore:
        """Compute the trajectory-aware attribution score from on-disk gradients.

        Rows and columns are derived from the saved records themselves: the
        column order is the test-sample hash order at the first included step,
        and rows are appended per step in on-disk order.  ``train_dataset`` /
        ``test_dataset`` are accepted for API parity but are *not* required and
        are not used to define ordering.

        Args:
            train_dataset: Unused; kept for API parity.
            test_dataset: Unused; kept for API parity.
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during the training pass.
            test_gradients_dir: Directory written by
                :class:`GradientFileManager` during the test pass.
            loop_over_test: If ``False`` (default), the step's test blocks are
                loaded once and reused across train blocks (peak memory: all
                test blocks of one step + one train block).  If ``True``, test
                blocks are re-streamed for every train block (peak memory: one
                train + one test block), at the cost of more disk reads.

        Returns:
            An :class:`AttributionScore`.  Also persisted to
            ``args.output_dir`` as ``scores.pt`` + ``metadata.json``.

        Raises:
            ValueError: If either gradients dir is missing, no common steps are
                found, or ``weight_list`` length disagrees with the step list.
            KeyError: If a test sample present at a later step was absent at the
                first step (test set changed across the trajectory).
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
        device = self.args.device

        metric = "cosine" if self.normalized_grad else "dot"
        tok_reduction = "mean" if self.token_reduction == "mean" else None

        # Column order: the test-hash order observed at the first included step.
        test_ids = self._test_column_order(test_fm, steps[0])
        test_index = {h: c for c, h in enumerate(test_ids)}
        num_test = len(test_ids)

        # Trajectory-aware rows accumulate (one chunk per train block) on CPU;
        # the user opted into building the full (train_hash, step) matrix.
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []

        for step, weight in zip(steps, weights):
            if not loop_over_test:
                test_blocks = [
                    (g.to(device), h)
                    for g, h in self._grad_loader(test_fm, step)
                ]

            for train_g, train_hashes in self._grad_loader(train_fm, step):
                train_g = train_g.to(device)
                n_tr = train_g.batch_size
                row_buf = torch.zeros(n_tr, num_test, dtype=torch.float)

                blocks = (
                    ((g.to(device), h) for g, h in self._grad_loader(test_fm, step))
                    if loop_over_test
                    else test_blocks
                )
                for test_g, test_hashes in blocks:
                    cols = [self._require_col(test_index, h, step) for h in test_hashes]
                    block = train_g.similarity(
                        test_g,
                        metric=metric,
                        reduce="all",
                        token_reduction=tok_reduction,
                    )
                    row_buf[:, cols] = (weight * block).detach().to("cpu", torch.float)

                row_chunks.append(row_buf)
                row_train_ids.extend(train_hashes)
                row_steps.extend([step] * n_tr)

        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, num_test, dtype=torch.float)
        )

        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids,
            steps=steps,
            weights=weights,
            algorithm="GradCos" if self.normalized_grad else "TracIn",
            normalized_grad=self.normalized_grad,
            layer_name=self.layer_name,
            token_reduction=self.token_reduction,
        )
        result.save(self.args.output_path)
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_steps(
        self,
        train_fm: GradientFileManager,
        test_fm: GradientFileManager,
    ) -> List[int]:
        if self.steps is not None:
            return list(self.steps)
        common = sorted(set(train_fm.available_steps()) & set(test_fm.available_steps()))
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

    def _test_column_order(self, test_fm: GradientFileManager, step: int) -> List[str]:
        """Establish the column (test-hash) order from one step's disk layout."""
        ids: List[str] = []
        for _, hashes in self._grad_loader(test_fm, step):
            ids.extend(hashes)
        return ids

    @staticmethod
    def _require_col(test_index: dict, h: str, step: int) -> int:
        if h not in test_index:
            raise KeyError(
                f"Test sample {h[:16]}… appears at step {step} but was absent at "
                "the first step used to define columns; the test set must be "
                "consistent across the trajectory."
            )
        return test_index[h]

    def _grad_loader(self, fm: GradientFileManager, step: int) -> DataLoader:
        """A DataLoader yielding ``(Gradient_block, hashes)`` per file at *step*.

        Uses ``batch_size=1`` with an identity collate (each item is already a
        batched block) so that ``num_workers`` prefetches files in parallel.
        """
        dataset = _GradientFileDataset(fm, step, layer_name=self.layer_name)
        kwargs: dict = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "collate_fn": _identity_collate,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(**kwargs)


# --------------------------------------------------------------------------- #
# Internal: on-disk gradient loading                                          #
# --------------------------------------------------------------------------- #


def _identity_collate(batch: list) -> object:
    """Collate for a ``batch_size=1`` loader: return the single item as-is.

    :class:`_GradientFileDataset` already yields a batched :class:`Gradient`
    block per item, so the DataLoader must not stack anything; it just unwraps
    the singleton list.
    """
    (item,) = batch
    return item


class _GradientFileDataset(Dataset):
    """Yields ``(Gradient_block, hashes)`` for each on-disk file at one step.

    Reading on-disk gradients one *sample* at a time would re-read each batch
    file once per sample and rebuild batches via repeated
    :meth:`Gradient.concatenate`.  This dataset instead exposes one *file* per
    item — a single :func:`torch.load` yields a whole batch block — so a
    standard :class:`~torch.utils.data.DataLoader` with ``num_workers > 0`` can
    prefetch files in parallel and the attributor can stream blocks without
    stacking a full train/test side into one tensor.

    Args:
        file_manager: Manager opened on the gradient directory.
        step: The step whose record files this dataset iterates.
        layer_name: If given, restrict each block's gradient to these layers.
    """

    def __init__(
        self,
        file_manager: GradientFileManager,
        step: int,
        layer_name: Optional[List[str]] = None,
    ) -> None:
        self._fm = file_manager
        self._step = step
        self._layer_name = list(layer_name) if layer_name is not None else None
        self._slots = file_manager.iter_step(step)

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, i: int) -> Tuple[Gradient, List[str]]:
        file_rel, idxs = self._slots[i]
        records = self._fm.load_records(file_rel)

        grads: List[Gradient] = []
        hashes: List[str] = []
        for idx in idxs:
            rec = records[idx]
            g = rec.gradient
            if self._layer_name is not None:
                g = g.select_layers(self._layer_name)
            grads.append(g)
            h = rec.input_hash
            hashes.extend(h if isinstance(h, list) else [h])

        block = grads[0]
        for g in grads[1:]:
            block = block.concatenate(g, dim="batch")

        if block.batch_size != len(hashes):
            raise RuntimeError(
                f"Block batch size ({block.batch_size}) disagrees with hash "
                f"count ({len(hashes)}) for file {file_rel!r} at step "
                f"{self._step}."
            )
        return block, hashes
