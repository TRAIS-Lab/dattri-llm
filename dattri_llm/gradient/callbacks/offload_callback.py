"""Disk-offloading callback for captured gradient records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.gradient import GradientRecord

if TYPE_CHECKING:
    from dattri_llm.gradient.file_manager import GradientFileManager


class OffloadCallback(HookManagerCallback):
    """Periodically saves :class:`GradientRecord` objects to disk.

    Accumulates records in memory and writes them to a single batch file every
    ``offload_interval`` *stored steps*.  Any remaining staged records are
    written when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect`
    context closes.

    Args:
        offload_interval: Number of stored steps to accumulate before writing
            a batch file.  Set to ``1`` for one file per step.
        file_manager: The :class:`GradientFileManager` to delegate saves to.
        recording_type: ``"per_batch"`` (default) stores one
            :class:`GradientRecord` per step.  ``"per_sample"`` slices the
            full-batch record into B individual records before staging -- useful
            when downstream code looks up gradients by a single-sample hash.
        gradient_accumulation_steps: When the wrapped training loop
            accumulates gradients over ``N`` micro-batches per optimizer
            update, pass the same ``N`` here: every window of ``N``
            consecutive capture records is merged into **one** stored record
            -- gradients batch-concatenated (differing sequence lengths are
            zero-padded), identifier lists concatenated -- and its ``step``
            is re-stamped with the callback's own optimizer-step counter, so
            on-disk steps align with the trainer's ``global_step``.  This is
            exact: parameters do not change within a window, so the merged
            record is indistinguishable from one large-batch step.  A ragged
            trailing window (fewer than ``N`` records at context end) is
            merged as-is.  The value is declared on the ``file_manager`` so
            the store's step convention is recorded and cannot be mixed.
            Default ``1``: every backward is its own stored step, with the
            capture step index kept verbatim.
    """

    def __init__(
        self,
        offload_interval: int,
        file_manager: GradientFileManager,
        recording_type: str = "per_batch",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if recording_type not in ("per_sample", "per_batch"):
            raise ValueError(
                f"recording_type must be 'per_sample' or 'per_batch', "
                f"got {recording_type!r}.",
            )
        if gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {gradient_accumulation_steps}.",
            )
        self._offload_interval = offload_interval
        self.file_manager = file_manager
        self._recording_type = recording_type
        self._accum_steps = gradient_accumulation_steps
        file_manager.declare_gradient_accumulation_steps(gradient_accumulation_steps)
        # Open accumulation window (micro-batch records not yet merged) and
        # the optimizer-step counter that re-stamps merged records.
        self._window: list[GradientRecord] = []
        self._update_step: int = 0
        self._staged: list[GradientRecord] = []
        self._staged_steps: set = set()

    def on_step_end(self, record: GradientRecord) -> None:
        """Stage the step's record(s), flushing every ``offload_interval`` steps.

        With ``gradient_accumulation_steps == N > 1``, records are first
        collected into windows of ``N`` and merged into one optimizer-step
        record before staging.
        """
        if self._accum_steps > 1:
            self._window.append(record)
            if len(self._window) < self._accum_steps:
                return
            record = self._merge_window()
        self._stage(record)

    def _merge_window(self) -> GradientRecord:
        """Merge the open window into one record stamped with the update step.

        Batch-concatenates the micro-batch gradients (unequal sequence
        lengths are zero-padded by
        :meth:`~dattri_llm.gradient.gradient.Gradient.concatenate`) and
        concatenates the identifier lists in micro-batch order.
        """
        window, self._window = self._window, []
        gradient = window[0].gradient
        for rec in window[1:]:
            gradient = gradient.concatenate(rec.gradient, dim="batch")
        hashes = [
            h
            for rec in window
            for h in (
                rec.input_hash if isinstance(rec.input_hash, list) else [rec.input_hash]
            )
        ]
        merged = GradientRecord(
            step=self._update_step,
            input_hash=hashes,
            gradient=gradient,
            sample_id_key=window[0].sample_id_key,
        )
        self._update_step += 1
        return merged

    def _stage(self, record: GradientRecord) -> None:
        records = self._expand(record)
        if (
            record.step not in self._staged_steps
            and len(self._staged_steps) >= self._offload_interval
        ):
            self._flush()
        self._staged.extend(records)
        self._staged_steps.add(record.step)

    def _expand(self, record: GradientRecord) -> list[GradientRecord]:
        """Slice into per-sample records when ``recording_type='per_sample'``."""
        if self._recording_type != "per_sample":
            return [record]
        hashes = (
            record.input_hash
            if isinstance(record.input_hash, list)
            else [record.input_hash]
        )
        batch_size = record.gradient.batch_size
        return [
            GradientRecord(
                step=record.step,
                input_hash=hashes[i],
                gradient=record.gradient.slice(dim="batch", index=i),
                sample_id_key=record.sample_id_key,
            )
            for i in range(batch_size)
        ]

    def on_context_end(self) -> None:
        """Flush any records still staged when the collect() context closes.

        A ragged trailing accumulation window (the loop ended mid-window,
        as a Trainer epoch can) is merged and staged first.
        """
        if self._window:
            self._stage(self._merge_window())
        self._flush()

    def _flush(self) -> None:
        if not self._staged:
            return
        self.file_manager.save_bulk(self._staged)
        self._staged.clear()
        self._staged_steps.clear()

    @property
    def staged(self) -> list[GradientRecord]:
        """Records staged but not yet written to disk."""
        return list(self._staged)
