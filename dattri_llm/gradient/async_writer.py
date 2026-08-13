"""Asynchronous disk offload for gradient records.

:class:`AsyncGradientWriter` decouples gradient collection from disk IO: the
producer (a training loop under ``OffloadCallback``, or an attributor's
``collect_to_disk`` pass) hands record groups to :meth:`submit` and keeps
computing while a single background thread performs the device-to-host copy
and the store write.  Ordering and crash safety are inherited unchanged from
:class:`~dattri_llm.gradient.storage_manager.GradientStorageManager`: groups
are written strictly in submission order by one thread, through the same
``save_bulk`` (write-then-index) path.

Memory: a submitted group's payloads stay referenced (device-resident, if
captured on GPU) until written.  ``max_pending`` bounds that -- a full queue
blocks :meth:`submit`, which is the backpressure that keeps a slow disk from
accumulating unbounded GPU memory.

CUDA ordering: :meth:`submit` records an event on the producer's current
stream for each device appearing in the group; the writer waits on those
events before copying, on its own stream, so the D2H never races the
producing kernels and never stalls the compute stream.

Thread safety: while a writer is open, its ``file_manager`` must not be used
from any other thread (reads included) -- close or flush first.  The standard
workflows already satisfy this: collection writes, then the context closes
(flushing the writer), then attribution reads.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.gradient import GradientRecord

if TYPE_CHECKING:
    from typing_extensions import Self

    from dattri_llm.gradient.storage_manager import GradientStorageManager

_SENTINEL = None


class AsyncGradientWriter:
    """Write gradient record groups to a store from a background thread.

    Args:
        file_manager: Destination store; owned by the writer thread while the
            writer is open (see the module docstring's thread-safety note).
        max_pending: Groups allowed in flight before :meth:`submit` blocks
            (``>= 1``).  Each pending group holds its payload memory alive.
    """

    def __init__(
        self,
        file_manager: GradientStorageManager,
        max_pending: int = 2,
    ) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}.")
        self._fm = file_manager
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._closed = False
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="dattri-llm-gradient-writer",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------ API #

    def submit(self, records: list[GradientRecord]) -> None:
        """Enqueue one record group for writing; blocks while ``max_pending``
        groups are already in flight.  Re-raises any earlier write error.
        """
        if self._closed:
            raise RuntimeError("submit() on a closed AsyncGradientWriter.")
        self._check_error()
        if not records:
            return
        self._queue.put((list(records), self._record_events(records)))

    def flush(self) -> None:
        """Block until every submitted group is on disk; re-raise write errors."""
        self._queue.join()
        self._check_error()

    def close(self) -> None:
        """Flush, stop the writer thread, and re-raise any write error.

        Idempotent (a repeated close is a no-op); the writer cannot be reused.
        """
        if not self._closed:
            self._closed = True
            self._queue.join()
            self._queue.put(_SENTINEL)
            self._thread.join()
        self._check_error(consume=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        # Swallow nothing; on an in-flight producer exception still drain what
        # was submitted so the store stays consistent with the index.
        self.close()
        return False

    # ---------------------------------------------------------- internals #

    def _check_error(self, consume: bool = False) -> None:
        """Re-raise a writer-thread error.  Only :meth:`close` consumes it, so
        the thread shutdown always still happens there.
        """
        if self._error is not None:
            error = self._error
            if consume:
                self._error = None
            raise error

    @staticmethod
    def _record_events(
        records: list[GradientRecord],
    ) -> list[tuple[torch.device, torch.cuda.Event]]:
        """Fence the producer's in-flight kernels for every CUDA device that
        appears in *records* (no-op for CPU payloads).
        """
        events = []
        devices = {
            r.gradient.device for r in records if r.gradient.device.type == "cuda"
        }
        for device in devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            events.append((device, event))
        return events

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                return
            records, events = item
            try:
                if self._error is None:
                    self._fm.save_bulk(self._to_cpu(records, events))
            except BaseException as exc:  # noqa: BLE001 - hand to the producer
                self._error = exc
            finally:
                # task_done unconditionally: flush()/close() must not deadlock
                # on a failed group (the error is re-raised there instead).
                self._queue.task_done()

    def _to_cpu(
        self,
        records: list[GradientRecord],
        events: list[tuple[torch.device, torch.cuda.Event]],
    ) -> list[GradientRecord]:
        """Move device-resident records to CPU on the writer's own stream.

        Waits the submit-time events first, so the copy is ordered after the
        producing kernels without ever blocking the producer's stream.  CPU
        records pass through untouched (``save_bulk`` treats them as a no-op).
        """
        if not events:
            return records
        for device, event in events:
            stream = self._copy_stream(device)
            stream.wait_event(event)
        if len(events) == 1:
            ((device, _),) = events
            stream = self._copy_stream(device)
            with torch.cuda.stream(stream):
                records = [self._cpu_record(r) for r in records]
            stream.synchronize()
            return records
        # Multi-device group (rare): host-wait each fence, then plain copies.
        for _, event in events:
            event.synchronize()
        return [self._cpu_record(r) for r in records]

    @staticmethod
    def _cpu_record(record: GradientRecord) -> GradientRecord:
        if record.gradient.device.type == "cpu":
            return record
        return GradientRecord(
            step=record.step,
            input_hash=record.input_hash,
            gradient=record.gradient.to("cpu"),
            sample_id_key=record.sample_id_key,
        )

    def _copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        stream = self._copy_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            self._copy_streams[index] = stream
        return stream
