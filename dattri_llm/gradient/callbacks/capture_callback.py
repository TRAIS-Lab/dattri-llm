"""In-memory single-step capture callback."""

from __future__ import annotations

from typing import Optional

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.gradient import GradientRecord


class CaptureCallback(HookManagerCallback):
    """Holds the most recent per-step :class:`GradientRecord` in memory.

    The minimal callback: it persists nothing and simply stashes the latest
    step's record on :attr:`record`, so a caller can read it back immediately
    after the step (used by the live :class:`~dattri_llm.gradient.streaming.\
GradientStreamer` to yield one block at a time).  Reset to ``None`` before each
    step by the consumer if a missed capture should be detectable.
    """

    def __init__(self) -> None:
        self.record: Optional[GradientRecord] = None

    def on_step_end(self, record: GradientRecord) -> None:
        self.record = record
