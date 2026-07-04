"""Disk-offloading callback for captured gradient records."""

from __future__ import annotations

from typing import List

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import GradientRecord


class OffloadCallback(HookManagerCallback):
    """Periodically saves :class:`GradientRecord` objects to disk.

    Accumulates records in memory and writes them to a single batch file every
    ``offload_interval`` *batch steps*.  Any remaining staged records are written
    when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect` context
    closes.

    Args:
        offload_interval: Number of batch steps to accumulate before writing a
            batch file.  Set to ``1`` for one file per step.
        file_manager: The :class:`GradientFileManager` to delegate saves to.
        recording_type: ``"per_batch"`` (default) stores one
            :class:`GradientRecord` per step.  ``"per_sample"`` slices the
            full-batch record into B individual records before staging -- useful
            when downstream code looks up gradients by a single-sample hash.
    """

    def __init__(
        self,
        offload_interval: int,
        file_manager: GradientFileManager,
        recording_type: str = "per_batch",
    ) -> None:
        if recording_type not in ("per_sample", "per_batch"):
            raise ValueError(
                f"recording_type must be 'per_sample' or 'per_batch', "
                f"got {recording_type!r}."
            )
        self._offload_interval = offload_interval
        self.file_manager = file_manager
        self._recording_type = recording_type
        self._staged: List[GradientRecord] = []
        self._staged_steps: set = set()

    def on_step_end(self, record: GradientRecord) -> None:
        records = self._expand(record)
        if record.step not in self._staged_steps:
            if len(self._staged_steps) >= self._offload_interval:
                self._flush()
        self._staged.extend(records)
        self._staged_steps.add(record.step)

    def _expand(self, record: GradientRecord) -> List[GradientRecord]:
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
            )
            for i in range(batch_size)
        ]

    def on_context_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._staged:
            return
        self.file_manager.save_bulk(self._staged)
        self._staged.clear()
        self._staged_steps.clear()

    @property
    def staged(self) -> List[GradientRecord]:
        """Records staged but not yet written to disk."""
        return list(self._staged)
