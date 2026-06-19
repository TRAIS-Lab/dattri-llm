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

The result is a :class:`~dattri_llm.algorithm.score.AttributionScore`.  It is
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
from torch.utils.data import Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import BaseAttributor, make_gradient_dataloader
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
                found, an explicitly requested step is absent from either
                directory, or ``weight_list`` length disagrees with the step
                list.
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

        steps = self._resolve_steps(train_fm, test_fm)
        weights = self._resolve_weights(steps)
        device = self.args.device

        metric = "cosine" if self.normalized_grad else "dot"
        tok_reduction = "mean" if self.token_reduction == "mean" else None

        # Column order (the test-hash order on disk at the first included step)
        # is discovered lazily while the first train block is scored, so the
        # test side is never loaded twice — there is no separate pre-pass that
        # reads every test file just to read its hashes.
        test_ids: List[str] = []
        test_index: dict = {}

        # Trajectory-aware rows accumulate (one chunk per train block) on CPU;
        # the user opted into building the full (train_hash, step) matrix.
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []

        for si, (step, weight) in enumerate(zip(steps, weights)):
            test_blocks = (
                None
                if loop_over_test
                else [(g.to(device), h) for g, h in self._grad_loader(test_fm, step)]
            )

            for bi, (train_g, train_hashes) in enumerate(
                self._grad_loader(train_fm, step)
            ):
                train_g = train_g.to(device)
                test_iter = (
                    ((g.to(device), h) for g, h in self._grad_loader(test_fm, step))
                    if loop_over_test
                    else iter(test_blocks)
                )
                # The first train block of the first step defines the columns;
                # every later block must match them (_require_col guards this).
                row_buf = self._row_for_train_block(
                    train_g, weight, test_iter, step, metric, tok_reduction,
                    test_ids, test_index, discover=(si == 0 and bi == 0),
                )
                row_chunks.append(row_buf)
                row_train_ids.extend(train_hashes)
                row_steps.extend([step] * train_g.batch_size)

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
            algorithm_meta={"steps": steps, "weights": weights},
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
        common = set(train_fm.available_steps()) & set(test_fm.available_steps())
        if self.steps is not None:
            # Honour the caller's order (steps[0] defines the column layout) but
            # reject any step missing from either directory — otherwise a typo'd
            # step silently contributes nothing (or yields an empty result).
            missing = [s for s in self.steps if s not in common]
            if missing:
                raise ValueError(
                    f"Requested steps {missing} are not present in both the "
                    f"train and test gradient directories. Available common "
                    f"steps: {sorted(common)}."
                )
            return list(self.steps)
        if not common:
            raise ValueError(
                "No common steps between train and test gradient directories."
            )
        return sorted(common)

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
        tok_reduction: Optional[str],
        test_ids: List[str],
        test_index: dict,
        *,
        discover: bool,
    ) -> torch.Tensor:
        """Compute one train block's CPU score row ``(n_tr, num_test)``.

        When ``discover`` is True (only the first train block of the first
        step) the column layout is built *as the test stream is consumed* —
        each unseen test hash becomes the next column — so the test side is
        scored exactly once and never pre-loaded just to read its hashes.
        Later blocks pass ``discover=False`` and resolve columns via
        :meth:`_require_col`, which rejects a test hash absent from the layout;
        the mirror case — a column present at the first step but *missing* at a
        later step — is caught here so the gap is never left as silent zeros.
        """
        n_tr = train_g.batch_size

        def _sim(test_g: Gradient) -> torch.Tensor:
            block = train_g.similarity(
                test_g, metric=metric, reduce="all", token_reduction=tok_reduction
            )
            return (weight * block).detach().to("cpu", torch.float)

        if not discover:
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
                    f"step but are absent at step {step}; the test set must be "
                    "consistent across the trajectory."
                )
            return row_buf

        # Discovery: columns are unknown until the whole test stream is seen,
        # so buffer each block's (cols, values) and assemble once num_test is
        # fixed.  Peak extra memory is one full row — the same as row_buf.
        pieces: List[Tuple[List[int], torch.Tensor]] = []
        for test_g, test_hashes in test_iter:
            cols = []
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
                cols.append(test_index[h])
            pieces.append((cols, _sim(test_g)))
        row_buf = torch.zeros(n_tr, len(test_ids), dtype=torch.float)
        for cols, vals in pieces:
            row_buf[:, cols] = vals
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

    def _grad_loader(self, fm: GradientFileManager, step: int):
        """DataLoader over this attributor's gradient files at *step*.

        Thin wrapper around :func:`~dattri_llm.algorithm.base.make_gradient_dataloader`
        that binds the attributor's ``args`` and ``layer_name``.
        """
        return make_gradient_dataloader(fm, step, self.args, self.layer_name)
