"""TracIn / GradCos attribution over on-disk gradients (workflow 2).

This attributor consumes :class:`~dattri_llm.gradient.gradient.Gradient`
records produced earlier by the gradient-collection pipeline and persisted to
disk via :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.  No
forward/backward pass is performed at attribution time — only inner products
between pre-stored per-sample gradients.

Every train record is scored against every test record::

    score[i, j] = <g_train_i, g_test_j>

i.e. the full ``(num_train, num_test)`` cross-gram, exactly mirroring the
:class:`~dattri_llm.algorithm.kronecker.KFACAttributor` assembly — there is no
train/test step alignment.  With ``normalized_grad=True`` the inner product
becomes a cosine similarity (the GradCos / CosIn variant);
``GradCos = TracInAttributor(normalized_grad=True)`` — there is no separate
subclass.

The result is a :class:`~dattri_llm.algorithm.score.AttributionScore`.  Rows are
stamped with the step each train gradient was recorded at (read straight off the
on-disk records), so a sample collected at several checkpoints contributes one
row per checkpoint; summing a sample's rows over steps recovers the classic
dattri ``(num_train, num_test)`` matrix.  Rows and columns are identified by
content hash in on-disk order, so no reconstruction of the original DataLoader is
required.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import (
    BaseAttributor,
    iter_gradient_blocks,
    resolve_steps,
)
from dattri_llm.algorithm.score import AttributionScore
from dattri_llm.algorithm.task import AttributionTask
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient


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
        normalized_grad: If ``True``, use cosine similarity per ``(i, j)`` pair
            (GradCos / CosIn); if ``False``, the raw inner product (TracIn /
            GradDot).
        layer_name: Restrict the inner product to these layer names.  ``None``
            uses every layer shared between the train and test gradients.
        task: Accepted for parity with the workflow-1 API but unused here.


    """

    def __init__(
        self,
        args: AttributionArguments,
        *,
        normalized_grad: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        task: Optional[AttributionTask] = None,
    ) -> None:
        self.args = args
        self.task = task
        self.normalized_grad = normalized_grad
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
        selected_training_steps: Optional[Iterable[int]] = None,
        verbose: bool = False,
    ) -> AttributionScore:
        """Compute the ``(num_train, num_test)`` attribution score from on-disk
        gradients — every train record against every test record.

        Rows and columns are derived from the saved records themselves: the
        column order is the test-sample hash order on disk, and rows are appended
        per train block in on-disk order, each stamped with the step it was
        recorded at.  ``train_dataset`` / ``test_dataset`` are accepted for API
        parity but are not used.

        Args:
            train_dataset: Unused; kept for API parity.
            test_dataset: Unused; kept for API parity.
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during the training pass.
            test_gradients_dir: Directory written by
                :class:`GradientFileManager` during the test pass.
            loop_over_test: If ``False`` (default), the test blocks are loaded
                once and reused across all train blocks (peak memory: all test
                blocks + one train block).  If ``True``, they are re-streamed for
                every train block (peak memory: one train + one test block) at
                the cost of more disk reads.
            selected_training_steps: Restrict the training checkpoints used (the
                output rows) to these steps; ``None`` (default) uses every step
                on disk.  Over-specified ranges are intersected with what is
                available.  The test set always supplies every column.
            verbose: Show tqdm progress bars on the logging process.

        Returns:
            An :class:`AttributionScore`.  Also persisted to
            ``args.output_dir`` as ``scores.pt`` + ``metadata.json``.

        Raises:
            ValueError: If either gradients dir is missing, or
                ``selected_training_steps`` matches no step in the training
                gradients.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                "TracInAttributor requires both train_gradients_dir and "
                "test_gradients_dir.  On-the-fly gradient collection (workflow 1) "
                "is not implemented yet."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)
        device = self.args.device
        metric = "cosine" if self.normalized_grad else "dot"

        # ``selected_training_steps`` picks which training checkpoints become
        # output rows; the test set always contributes every column.
        train_steps = resolve_steps(train_fm, selected_training_steps)
        test_steps = test_fm.available_steps()

        # One pass over the test files: fix the column order (disk order) and,
        # unless looping, cache each (device-resident) block for reuse.
        test_ids: List[str] = []
        test_index: dict = {}
        cached_test: List[Tuple[Gradient, List[str]]] = []
        for _step, test_g, test_hashes in iter_gradient_blocks(
            test_fm, test_steps, self.args, self.layer_name,
            desc=f"{'GradCos' if self.normalized_grad else 'TracIn'}: loading test",
            verbose=verbose,
        ):
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
            if not loop_over_test:
                cached_test.append((test_g.to(device), test_hashes))
        num_test = len(test_ids)

        # Score every selected train block against every test block.
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        for train_step, train_g, train_hashes in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc=f"{'GradCos' if self.normalized_grad else 'TracIn'}: scoring",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            row = torch.zeros(train_g.batch_size, num_test, dtype=torch.float)
            test_blocks = (
                (
                    (g.to(device), h)
                    for _s, g, h in iter_gradient_blocks(
                        test_fm, test_steps, self.args, self.layer_name
                    )
                )
                if loop_over_test
                else cached_test
            )
            for test_g, test_hashes in test_blocks:
                cols = [test_index[h] for h in test_hashes]
                block = train_g.similarity(test_g, metric=metric, reduce="all")
                row[:, cols] = block.detach().to("cpu", torch.float)
            row_chunks.append(row)
            row_train_ids.extend(train_hashes)
            row_steps.extend([train_step] * train_g.batch_size)

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
            algorithm_meta={"selected_training_steps": train_steps},
            algorithm="GradCos" if self.normalized_grad else "TracIn",
            normalized_grad=self.normalized_grad,
            layer_name=self.layer_name,
        )
        result.save(self.args.output_path)
        return result
