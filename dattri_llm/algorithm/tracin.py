"""TracIn / GradCos attribution over on-disk gradients (workflow 2).

This attributor consumes :class:`~dattri_llm.gradient.gradient.Gradient`
records produced earlier by the gradient-collection pipeline and persisted to
disk via :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.  No
forward/backward pass is performed at attribution time — only inner products
between pre-stored per-sample gradients.

Per mapped step pair ``(k_train, k_test)`` (a TracIn ensemble term)::

    score[(train_i, k_train), test_j] = weight_k * <g_train_i^(k_train), g_test_j^(k_test)>

where ``g_*^(k)`` is the per-sample weight gradient across all hooked layers.
With ``normalized_grad=True`` the inner product becomes a cosine similarity
(the GradCos / CosIn variant); ``GradCos = TracInAttributor(normalized_grad=True)``
— there is no separate subclass.

The result is a :class:`~dattri_llm.algorithm.score.AttributionScore`.  It is
*trajectory-aware* on the train side: one row per ``(train_hash, step)`` pair,
rather than a single step-summed row.  Summing a sample's rows over steps
recovers the classic dattri ``(num_train, num_test)`` matrix.  Rows and columns
are identified by content hash, read directly from the on-disk records, so no
reconstruction of the original training DataLoader is required — the ordering
is defined entirely by what is on disk.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import BaseAttributor, make_gradient_multistep_dataloader
from dattri_llm.algorithm.score import AttributionScore
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
        step_map: Optional mapping from train-gradient step to test-gradient
            step.  ``None`` uses identity mapping over the common on-disk steps.
        task: Accepted for parity with the workflow-1 API but unused here.
    """

    def __init__(
        self,
        args: AttributionArguments,
        *,
        weight_list: Optional[Sequence[float]] = None,
        normalized_grad: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        step_map: Optional[Mapping[int, int]] = None,
        task: Optional[AttributionTask] = None,
    ) -> None:
        self.args = args
        self.task = task
        self.weight_list = list(weight_list) if weight_list is not None else None
        self.normalized_grad = normalized_grad
        self.step_map = dict(step_map) if step_map is not None else None
        if isinstance(layer_name, str):
            self.layer_name: Optional[List[str]] = [layer_name]
        elif layer_name is None:
            self.layer_name = None
        else:
            self.layer_name = list(layer_name)

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
        are not used to define ordering.  The test set must be consistent across
        the included steps — a sample present at one step but missing at another
        raises ``KeyError`` rather than producing a silent zero.  Test samples
        with identical content share a single column (they hash equally), so
        ``num_test`` counts distinct test samples.

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
                found, or ``weight_list`` length disagrees with the
                auto-discovered common step list.
            KeyError: If the test set changed across the trajectory — a sample
                present at a later step that was absent at the first step, or a
                column defined at the first step that is missing at a later one.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                "TracInAttributor requires both train_gradients_dir and "
                "test_gradients_dir.  On-the-fly gradient collection (workflow 1) "
                "is not implemented yet."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)

        step_pairs = self._resolve_step_pairs(train_fm, test_fm)
        train_steps = [train_step for train_step, _ in step_pairs]
        test_steps = [test_step for _, test_step in step_pairs]
        weights = self._resolve_weights(train_steps)
        device = self.args.device

        metric = "cosine" if self.normalized_grad else "dot"

        # Test files are streamed once at file granularity.  If an on-disk file
        # contains multiple requested steps, all matching records are consumed
        # from that single load.  Column order is then defined from the first
        # mapped test step after its complete hash list is known, so validation
        # does not depend on file ordering.
        test_ids: List[str] = []
        test_index: dict = {}
        test_hashes_by_step: dict[int, List[str]] = {}
        test_blocks_by_step: dict[int, List[Tuple[Gradient, List[str]]]] = {}
        for by_step in self._grad_file_loader(test_fm, sorted(set(test_steps))):
            for test_step, (test_g, test_hashes) in by_step.items():
                test_hashes_by_step.setdefault(test_step, []).extend(test_hashes)
                if not loop_over_test:
                    test_blocks_by_step.setdefault(test_step, []).append(
                        (test_g.to(device), test_hashes)
                    )

        column_step = test_steps[0]
        for h in test_hashes_by_step[column_step]:
            if h not in test_index:
                test_index[h] = len(test_ids)
                test_ids.append(h)

        for test_step in sorted(set(test_steps)):
            seen = {
                self._require_col(test_index, h, test_step)
                for h in test_hashes_by_step[test_step]
            }
            if len(seen) != len(test_ids):
                missing = [
                    f"{test_ids[c][:16]}…"
                    for c in range(len(test_ids))
                    if c not in seen
                ]
                raise KeyError(
                    f"Test sample(s) {missing} defined the columns at test step "
                    f"{column_step} but are absent at test step {test_step}; "
                    "the test set must be consistent across the trajectory."
                )

        # Trajectory-aware rows accumulate (one chunk per train block) on CPU;
        # the user opted into building the full (train_hash, step) matrix.
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        pair_by_train_step = dict(step_pairs)
        weight_by_train_step = dict(zip(train_steps, weights))

        for by_step in self._grad_file_loader(train_fm, train_steps):
            for train_step, (train_g, train_hashes) in by_step.items():
                train_g = train_g.to(device)
                test_step = pair_by_train_step[train_step]
                weight = weight_by_train_step[train_step]
                test_iter = (
                    (
                        (test_g.to(device), test_hashes)
                        for by_test_step in self._grad_file_loader(test_fm, [test_step])
                        for _, (test_g, test_hashes) in by_test_step.items()
                    )
                    if loop_over_test
                    else iter(test_blocks_by_step[test_step])
                )
                row_buf = self._row_for_train_block(
                    train_g,
                    weight,
                    test_iter,
                    train_step,
                    metric,
                    test_ids,
                    test_index,
                )
                row_chunks.append(row_buf)
                row_train_ids.extend(train_hashes)
                row_steps.extend([train_step] * train_g.batch_size)

        num_test = len(test_ids)
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
            algorithm_meta={
                "steps": train_steps,
                "train_steps": train_steps,
                "test_steps": test_steps,
                "step_map": [[tr, te] for tr, te in step_pairs],
                "weights": weights,
            },
            algorithm="GradCos" if self.normalized_grad else "TracIn",
            normalized_grad=self.normalized_grad,
            layer_name=self.layer_name,
        )
        result.save(self.args.output_path)
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_step_pairs(
        self,
        train_fm: GradientFileManager,
        test_fm: GradientFileManager,
    ) -> List[Tuple[int, int]]:
        train_steps = set(train_fm.available_steps())
        test_steps = set(test_fm.available_steps())

        if self.step_map is None:
            common = train_steps & test_steps
            if not common:
                raise ValueError(
                    "No common steps between train and test gradient directories. "
                    "Pass step_map={train_step: test_step} when the recorded "
                    "train and test step IDs differ."
                )
            return [(step, step) for step in sorted(common)]

        missing_train = sorted(s for s in self.step_map if s not in train_steps)
        missing_test = sorted({s for s in self.step_map.values() if s not in test_steps})
        if missing_train or missing_test:
            problems = []
            if missing_train:
                problems.append(
                    f"train step(s) {missing_train} not present; "
                    f"available train steps: {sorted(train_steps)}"
                )
            if missing_test:
                problems.append(
                    f"test step(s) {missing_test} not present; "
                    f"available test steps: {sorted(test_steps)}"
                )
            raise ValueError("Invalid step_map: " + "; ".join(problems))
        if not self.step_map:
            raise ValueError("step_map must not be empty.")
        return [(tr, self.step_map[tr]) for tr in sorted(self.step_map)]

    def _resolve_weights(self, steps: List[int]) -> List[float]:
        if self.weight_list is None:
            return [1.0] * len(steps)
        if len(self.weight_list) != len(steps):
            raise ValueError(
                f"weight_list length ({len(self.weight_list)}) must equal the "
                f"number of steps ({len(steps)})."
            )
        return [float(w) for w in self.weight_list]

    def _row_for_train_block(
        self,
        train_g: Gradient,
        weight: float,
        test_iter,
        step: int,
        metric: str,
        test_ids: List[str],
        test_index: dict,
    ) -> torch.Tensor:
        """Compute one train block's CPU score row ``(n_tr, num_test)``."""
        n_tr = train_g.batch_size

        def _sim(test_g: Gradient) -> torch.Tensor:
            block = train_g.similarity(
                test_g, metric=metric, reduce="all"
            )
            return (weight * block).detach().to("cpu", torch.float)

        row_buf = torch.zeros(n_tr, len(test_ids), dtype=torch.float)
        seen_cols: set = set()
        for test_g, test_hashes in test_iter:
            cols = [self._require_col(test_index, h, step) for h in test_hashes]
            row_buf[:, cols] = _sim(test_g)
            seen_cols.update(cols)
        if len(seen_cols) != len(test_ids):
            missing = [
                f"{test_ids[c][:16]}…"
                for c in range(len(test_ids))
                if c not in seen_cols
            ]
            raise KeyError(
                f"Test sample(s) {missing} defined the columns at the first "
                f"mapped test step but are absent for train step {step}; the "
                "test set must be consistent across the trajectory."
            )
        return row_buf

    @staticmethod
    def _require_col(test_index: dict, h: str, step: int) -> int:
        if h not in test_index:
            raise KeyError(
                f"Test sample {h[:16]}… appears at step {step} but was absent at "
                "the first step used to define columns; the test set must be "
                "consistent across the trajectory."
            )
        return test_index[h]

    def _grad_file_loader(self, fm: GradientFileManager, steps: List[int]):
        """DataLoader over this attributor's gradient files for *steps*."""
        return make_gradient_multistep_dataloader(fm, steps, self.args, self.layer_name)
