"""Unit tests for the async disk-write path: ``AsyncGradientWriter`` and its
wiring into ``OffloadCallback`` and ``collect_to_disk``.

The load-bearing property everywhere: a store produced asynchronously must be
**indistinguishable** from one produced synchronously -- same steps, same
hashes, same gradient values.  CUDA-stream staging is skipif-guarded; all
queue/thread semantics are exercised on CPU.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from dattri_llm.attribution.utils import collect_to_disk
from dattri_llm.gradient.async_writer import AsyncGradientWriter
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.gradient import Gradient, GradientRecord
from dattri_llm.gradient.storage_manager import GradientStorageManager

TIMEOUT = 10.0  # generous thread-sync bound; every wait should be instant


def make_record(step: int, b: int = 2) -> GradientRecord:
    g = torch.Generator().manual_seed(step)
    gradient = Gradient(
        representation={"l1": "materialized"},
        data={"l1": torch.randn(b, 6, generator=g)},
        layer_types={"l1": "nn.Linear"},
    )
    return GradientRecord(
        step=step,
        input_hash=[f"h{step}-{j}" for j in range(b)],
        gradient=gradient,
    )


def read_store(root) -> dict[int, dict[str, torch.Tensor]]:
    """Read a store back as ``{step: {hash: gradient row}}`` via a fresh
    manager, so the comparison goes through the real load path.
    """
    fm = GradientStorageManager(str(root))
    out: dict[int, dict[str, torch.Tensor]] = {}
    for step in fm.available_steps():
        for file_rel, by_step in fm.iter_steps(step):
            records = fm.load_records(file_rel)
            for idx in by_step[step]:
                rec = records[idx]
                hashes = (
                    rec.input_hash
                    if isinstance(rec.input_hash, list)
                    else [rec.input_hash]
                )
                for i, h in enumerate(hashes):
                    row = rec.gradient.slice(dim="batch", index=i).data["l1"]
                    out.setdefault(step, {})[h] = row
    return out


class GatedFileManager:
    """A save_bulk stub whose writes block until released."""

    def __init__(self, fail: Exception | None = None):
        self.gate = threading.Event()
        self.saved: list[list[GradientRecord]] = []
        self.fail = fail

    def save_bulk(self, records):
        if not self.gate.wait(TIMEOUT):
            raise TimeoutError("gate never released")
        if self.fail is not None:
            raise self.fail
        self.saved.append(list(records))


# --------------------------------------------------------------------------- #
# AsyncGradientWriter                                                          #
# --------------------------------------------------------------------------- #


class TestAsyncGradientWriter:
    def test_store_matches_synchronous_write(self, tmp_path):
        groups = [[make_record(0), make_record(1)], [make_record(2)]]

        sync_fm = GradientStorageManager(str(tmp_path / "sync"))
        for group in groups:
            sync_fm.save_bulk(group)

        with AsyncGradientWriter(GradientStorageManager(str(tmp_path / "async"))) as w:
            for group in groups:
                w.submit(group)

        sync_store = read_store(tmp_path / "sync")
        async_store = read_store(tmp_path / "async")
        assert sync_store.keys() == async_store.keys() == {0, 1, 2}
        for step in sync_store:
            assert sync_store[step].keys() == async_store[step].keys()
            for h in sync_store[step]:
                assert torch.equal(sync_store[step][h], async_store[step][h])

    def test_groups_written_in_submission_order(self):
        fm = GatedFileManager()
        fm.gate.set()
        with AsyncGradientWriter(fm, max_pending=1) as w:
            for i in range(6):
                w.submit([make_record(i)])
        assert [g[0].step for g in fm.saved] == list(range(6))

    def test_flush_waits_for_pending_writes(self):
        fm = GatedFileManager()
        w = AsyncGradientWriter(fm)
        w.submit([make_record(0)])
        assert fm.saved == []  # write is parked on the gate

        flushed = threading.Event()

        def flush():
            w.flush()
            flushed.set()

        t = threading.Thread(target=flush, daemon=True)
        t.start()
        assert not flushed.wait(0.1)  # flush must block while the write is
        fm.gate.set()
        assert flushed.wait(TIMEOUT)
        assert len(fm.saved) == 1
        w.close()

    def test_submit_blocks_at_max_pending(self):
        fm = GatedFileManager()
        w = AsyncGradientWriter(fm, max_pending=1)
        w.submit([make_record(0)])  # dequeued by the worker, parked on gate
        w.submit([make_record(1)])  # fills the queue slot

        third_done = threading.Event()

        def third():
            w.submit([make_record(2)])
            third_done.set()

        t = threading.Thread(target=third, daemon=True)
        t.start()
        assert not third_done.wait(0.2)  # backpressure: no slot free
        fm.gate.set()
        assert third_done.wait(TIMEOUT)
        w.close()
        assert [g[0].step for g in fm.saved] == [0, 1, 2]

    def test_write_error_reaches_producer(self):
        boom = RuntimeError("disk full")
        fm = GatedFileManager(fail=boom)
        fm.gate.set()
        w = AsyncGradientWriter(fm)
        w.submit([make_record(0)])
        with pytest.raises(RuntimeError, match="disk full"):
            w.flush()
        # The error also surfaces on submit, and close consumes it exactly once.
        with pytest.raises(RuntimeError, match="disk full"):
            w.submit([make_record(1)])
        with pytest.raises(RuntimeError, match="disk full"):
            w.close()
        w.close()  # idempotent after the error was consumed

    def test_error_skips_later_groups_but_never_deadlocks(self):
        boom = RuntimeError("boom")
        fm = GatedFileManager(fail=boom)
        fm.gate.set()
        w = AsyncGradientWriter(fm, max_pending=1)
        for i in range(4):  # would deadlock if failed groups kept the queue full
            try:
                w.submit([make_record(i)])
            except RuntimeError:
                break
        with pytest.raises(RuntimeError, match="boom"):
            w.close()
        assert fm.saved == []

    def test_submit_after_close_raises(self):
        fm = GatedFileManager()
        fm.gate.set()
        w = AsyncGradientWriter(fm)
        w.close()
        with pytest.raises(RuntimeError, match="closed"):
            w.submit([make_record(0)])

    def test_empty_submit_is_noop(self):
        fm = GatedFileManager()  # gate never set: a queued write would hang
        w = AsyncGradientWriter(fm)
        w.submit([])
        w.flush()
        w.close()
        assert fm.saved == []

    def test_invalid_max_pending(self):
        with pytest.raises(ValueError, match="max_pending"):
            AsyncGradientWriter(GatedFileManager(), max_pending=0)

    def test_submitted_list_is_snapshotted(self):
        fm = GatedFileManager()
        w = AsyncGradientWriter(fm)
        group = [make_record(0)]
        w.submit(group)
        group.clear()  # producer reuses its staging list immediately
        fm.gate.set()
        w.close()
        assert [g[0].step for g in fm.saved] == [0]


# --------------------------------------------------------------------------- #
# OffloadCallback(async_write=True)                                            #
# --------------------------------------------------------------------------- #


class TestOffloadCallbackAsync:
    def _run(self, root, *, async_write: bool, n_steps=5, interval=2):
        fm = GradientStorageManager(str(root))
        cb = OffloadCallback(
            offload_interval=interval,
            file_manager=fm,
            async_write=async_write,
        )
        for i in range(n_steps):
            cb.on_step_end(make_record(i))
        cb.on_context_end()
        return fm

    def test_async_store_matches_sync(self, tmp_path):
        self._run(tmp_path / "sync", async_write=False)
        self._run(tmp_path / "async", async_write=True)
        sync_store = read_store(tmp_path / "sync")
        async_store = read_store(tmp_path / "async")
        assert sync_store.keys() == async_store.keys()
        for step in sync_store:
            for h in sync_store[step]:
                assert torch.equal(sync_store[step][h], async_store[step][h])

    def test_context_end_drains_writer(self, tmp_path):
        self._run(tmp_path / "s", async_write=True, n_steps=3, interval=10)
        # All records were still staged at context end; they must be on disk
        # by the time on_context_end returns (writer drained synchronously).
        assert sorted(read_store(tmp_path / "s")) == [0, 1, 2]

    def test_no_writer_thread_when_sync(self, tmp_path):
        fm = GradientStorageManager(str(tmp_path / "s"))
        cb = OffloadCallback(offload_interval=1, file_manager=fm)
        cb.on_step_end(make_record(0))
        cb.on_context_end()
        assert cb._writer is None


# --------------------------------------------------------------------------- #
# collect_to_disk(async_write=...)                                             #
# --------------------------------------------------------------------------- #


class FakeStreamer:
    def __init__(self, blocks, args=None):
        self._blocks = blocks
        self._args = args
        self.hook_manager = SimpleNamespace(sample_id_key=None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._blocks)


def make_blocks(n=4):
    return [
        (i, make_record(i).gradient, [f"h{i}-0", f"h{i}-1"]) for i in range(n)
    ]


class TestCollectToDiskAsync:
    def test_async_store_matches_sync(self, tmp_path):
        collect_to_disk(
            FakeStreamer(make_blocks()),
            GradientStorageManager(str(tmp_path / "sync")),
            async_write=False,
        )
        collect_to_disk(
            FakeStreamer(make_blocks()),
            GradientStorageManager(str(tmp_path / "async")),
            async_write=True,
        )
        sync_store = read_store(tmp_path / "sync")
        async_store = read_store(tmp_path / "async")
        assert sync_store.keys() == async_store.keys()
        for step in sync_store:
            for h in sync_store[step]:
                assert torch.equal(sync_store[step][h], async_store[step][h])

    def test_default_reads_streamer_args(self, tmp_path, monkeypatch):
        created = []
        original = AsyncGradientWriter.__init__

        def spy(self, *a, **kw):
            created.append(True)
            original(self, *a, **kw)

        monkeypatch.setattr(AsyncGradientWriter, "__init__", spy)
        args = SimpleNamespace(async_disk_write=True)
        collect_to_disk(
            FakeStreamer(make_blocks(2), args=args),
            GradientStorageManager(str(tmp_path / "s")),
        )
        assert created  # args opted in -> writer used
        assert sorted(read_store(tmp_path / "s")) == [0, 1]

    def test_on_block_runs_on_producer_thread(self, tmp_path):
        main_thread = threading.current_thread()
        seen = []
        collect_to_disk(
            FakeStreamer(make_blocks(3)),
            GradientStorageManager(str(tmp_path / "s")),
            async_write=True,
            on_block=lambda *a: seen.append(threading.current_thread()),
        )
        assert len(seen) == 3
        assert all(t is main_thread for t in seen)
