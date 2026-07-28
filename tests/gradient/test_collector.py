"""Unit tests for dattri_llm gradient collection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.hooks import (
    REGISTER_ALL,
    HookManager,
    HookManagerCallback,
    HookManagerConfig,
)
from dattri_llm.gradient.ops import PARAM_GRAD_TYPES
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.utils.hashing import hash_batch, hash_sample

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


class RecordingCallback(HookManagerCallback):
    """Stores every GradientRecord emitted by the collector."""

    def __init__(self) -> None:
        self.records: list[GradientRecord] = []
        self.context_ended: int = 0

    def on_step_end(self, record: GradientRecord) -> None:
        self.records.append(record)

    def on_context_end(self) -> None:
        self.context_ended += 1


def _run_one_step(model: nn.Module, input_ids: torch.Tensor, callbacks=None):
    """Run one forward+backward step and return the collector."""
    collector = HookManager(model, callbacks=callbacks or [])
    with collector.collect():
        logits = model(input_ids)
        logits.mean().backward()
    return collector


# --------------------------------------------------------------------------- #
# hash_sample                                                                  #
# --------------------------------------------------------------------------- #


class TestHashSample:
    def test_same_sample_same_hash(self):
        sample = {"input_ids": torch.randint(0, 10, (8,))}
        assert hash_sample(sample) == hash_sample(sample)

    def test_different_samples_different_hash(self):
        batch = {"input_ids": torch.randint(0, 10, (3, 8))}
        hashes = hash_batch(batch, 3)
        # All three samples should be distinct (with overwhelming probability for random
        # data)
        assert len(set(hashes)) == 3

    def test_64_char_hex(self):
        h = hash_sample({"input_ids": torch.zeros(4, dtype=torch.long)})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash_across_batches(self):
        row = torch.randint(0, 100, (8,))
        batch_a = {"input_ids": torch.stack([row, torch.zeros(8, dtype=torch.long)])}
        batch_b = {"input_ids": torch.stack([torch.zeros(8, dtype=torch.long), row])}
        # Same content row -> same hash regardless of batch position, and the
        # single-sample hash of the raw row matches both.
        assert hash_batch(batch_a, 2)[0] == hash_batch(batch_b, 2)[1]
        assert hash_batch(batch_a, 2)[0] == hash_sample({"input_ids": row})

    def test_multiple_fields_used(self):
        ids = torch.zeros(4, dtype=torch.long)
        mask_a = {"input_ids": ids, "attention_mask": torch.ones(4)}
        mask_b = {"input_ids": ids, "attention_mask": torch.zeros(4)}
        # Same input_ids but different attention_mask -> different hash
        assert hash_sample(mask_a) != hash_sample(mask_b)

    def test_non_tensor_values_skipped(self):
        sample = {
            "input_ids": torch.zeros(4, dtype=torch.long),
            "some_flag": True,  # non-tensor, should be ignored
        }
        # Should not raise
        h = hash_sample(sample)
        assert len(h) == 64

    def test_hash_batch_order_matches_capture(self):
        batch = {"x": torch.randn(3, 5), "y": torch.randn(3, 2)}
        expected = [
            hash_sample({"x": batch["x"][i], "y": batch["y"][i]}) for i in range(3)
        ]
        assert hash_batch(batch, 3) == expected

    def test_hash_batch_skips_non_batch_first(self):
        # A field whose leading dim disagrees with the batch (e.g. a
        # sequence-first or broadcast tensor) carries no per-sample identity
        # and is skipped rather than raising (it used to crash the capture
        # hooks mid-training).
        batch = {"x": torch.randn(3, 5), "pos": torch.randn(7, 3)}
        assert hash_batch(batch, batch_size=3) == hash_batch(
            {"x": batch["x"]},
            batch_size=3,
        )


class TestRecordBatchSizeFromGradient:
    """The record's identity hashes use the *gradient's* batch size.

    Regression: the manager guessed the batch size from the first captured
    input tensor's leading dim.  A broadcast kwarg (position_ids of shape
    (1, T)) arriving before input_ids made it infer batch size 1, so a
    B-sample step was labelled with a single hash of the shared broadcast
    row -- caught only later, as a batch-size mismatch at save time.
    """

    class _PosFirstNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(100, 8)
            self.pos = nn.Embedding(16, 8)
            self.fc = nn.Linear(8, 4)

        # kwargs order matters: position_ids is declared before input_ids.
        def forward(self, position_ids=None, input_ids=None):
            return self.fc(self.embed(input_ids) + self.pos(position_ids))

    def test_broadcast_first_kwarg_labels_every_sample(self):
        torch.manual_seed(0)
        B, T = 4, 6
        model = self._PosFirstNet()
        recorder = RecordingCallback()
        collector = HookManager(
            model,
            config=HookManagerConfig(linear_io=[r"fc"]),
            callbacks=[recorder],
        )
        input_ids = torch.randint(0, 100, (B, T))
        position_ids = torch.arange(T).unsqueeze(0)  # (1, T) broadcast
        try:
            with collector.collect():
                out = model(position_ids=position_ids, input_ids=input_ids)
                out.pow(2).sum().backward()
        finally:
            collector.remove()

        record = recorder.records[0]
        assert record.gradient.batch_size == B
        # One hash per sample, each identifying its input_ids row; the
        # broadcast position row does not enter the identity.
        assert record.input_hash == [
            hash_sample({"input_ids": input_ids[i]}) for i in range(B)
        ]


# --------------------------------------------------------------------------- #
# GradientId / GradientRecord                                                  #
# --------------------------------------------------------------------------- #


class TestGradientRecord:
    def test_repr(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        rec = cb.records[0]
        r = repr(rec)
        assert "step=0" in r
        assert "input_hash=" in r

    def test_equality(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        # One record per step; input_hash is a list of B hashes
        rec = cb.records[0]
        assert isinstance(rec.input_hash, list)
        assert len(set(rec.input_hash)) == len(rec.input_hash)  # all hashes distinct


# --------------------------------------------------------------------------- #
# HookManager -- init and introspection                                   #
# --------------------------------------------------------------------------- #


class TestHookManagerInit:
    def test_layer_names_populated(self, tiny_model):
        collector = HookManager(tiny_model)
        # Default config hooks every linear-family layer regardless of name:
        # embedding, attn_proj, mlp.0, mlp.2, lm_head (mlp.1 is ReLU).
        assert set(collector.layer_names) == {
            "embedding",
            "attn_proj",
            "mlp.0",
            "mlp.2",
            "lm_head",
        }
        collector.remove()

    def test_custom_patterns(self, tiny_model):
        collector = HookManager(
            tiny_model,
            config=HookManagerConfig(linear_io=[r"mlp\.0"]),
        )
        assert collector.layer_names == ["mlp.0"]
        collector.remove()

    def test_steps_collected_initial(self, tiny_model):
        collector = HookManager(tiny_model)
        assert collector.steps_collected == 0
        collector.remove()


# --------------------------------------------------------------------------- #
# HookManager -- collect() context manager                                #
# --------------------------------------------------------------------------- #


class TestCollectContextManager:
    def test_no_record_without_backward(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"])  # forward only, no backward
        assert len(cb.records) == 0
        assert cb.context_ended == 1
        collector.remove()

    def test_one_record_per_step(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            logits = tiny_model(tiny_batch["input_ids"])
            logits.mean().backward()
        assert len(cb.records) == 1
        collector.remove()

    def test_context_end_fires_once(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            logits = tiny_model(tiny_batch["input_ids"])
            logits.mean().backward()
        assert cb.context_ended == 1
        collector.remove()

    def test_steps_collected_increments(self, tiny_model, tiny_batch):
        collector = HookManager(tiny_model)
        with collector.collect():
            logits = tiny_model(tiny_batch["input_ids"])
            logits.mean().backward()
        assert collector.steps_collected == 1
        collector.remove()

    def test_multiple_steps_within_one_collect(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        n_steps = 3
        with collector.collect():
            for _ in range(n_steps):
                logits = tiny_model(tiny_batch["input_ids"])
                logits.mean().backward()
        assert len(cb.records) == n_steps
        assert collector.steps_collected == n_steps
        collector.remove()

    def test_not_collecting_outside_context(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        # Forward+backward outside collect() should not emit records
        logits = tiny_model(tiny_batch["input_ids"])
        logits.mean().backward()
        assert len(cb.records) == 0
        assert collector.steps_collected == 0
        collector.remove()


# --------------------------------------------------------------------------- #
# HookManager -- record content                                           #
# --------------------------------------------------------------------------- #


class TestRecordContent:
    def test_record_type(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        _run_one_step(tiny_model, tiny_batch["input_ids"], callbacks=[cb])
        for rec in cb.records:
            assert isinstance(rec, GradientRecord)
            assert isinstance(rec.step, int)
            assert isinstance(rec.input_hash, list)  # list of B hashes
            assert isinstance(rec.gradient, Gradient)

    def test_record_batch_size(self, tiny_model, tiny_batch):
        B = tiny_batch["input_ids"].shape[0]
        cb = RecordingCallback()
        _run_one_step(tiny_model, tiny_batch["input_ids"], callbacks=[cb])
        for rec in cb.records:
            assert rec.gradient.batch_size == B

    def test_record_step_ids(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        n_steps = 2
        with collector.collect():
            for _ in range(n_steps):
                logits = tiny_model(tiny_batch["input_ids"])
                logits.mean().backward()
        steps = [rec.step for rec in cb.records]
        # One record per step
        assert steps == [0, 1]
        collector.remove()

    def test_record_has_layer_data(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        _run_one_step(tiny_model, tiny_batch["input_ids"], callbacks=[cb])
        for rec in cb.records:
            # Default config hooks every linear-family layer:
            # embedding, attn_proj, mlp.0, mlp.2, lm_head.
            assert len(rec.gradient.layer_names) == 5
            assert "embedding" in rec.gradient.layer_names

    def test_gradient_finite(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        _run_one_step(tiny_model, tiny_batch["input_ids"], callbacks=[cb])
        for rec in cb.records:
            mat = rec.gradient.materialize()
            for t in mat.data.values():
                assert torch.isfinite(t).all()


# --------------------------------------------------------------------------- #
# HookManager -- sample hashing                                           #
# --------------------------------------------------------------------------- #


class TestSampleHashing:
    def test_different_samples_different_hashes(self, tiny_model):
        torch.manual_seed(0)
        ids = torch.randint(0, tiny_model.vocab_size, (3, tiny_model.seq_len))
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(ids).mean().backward()
        # One record per step; input_hash is a list of 3 hashes
        rec = cb.records[0]
        assert isinstance(rec.input_hash, list)
        assert len(set(rec.input_hash)) == 3  # all three samples distinct
        collector.remove()

    def test_same_sample_across_two_steps_same_hash(self, tiny_model):
        """The same input repeated in a second step should produce the same hash."""
        torch.manual_seed(7)
        ids = torch.randint(0, tiny_model.vocab_size, (2, tiny_model.seq_len))
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            for _ in range(2):
                tiny_model(ids).mean().backward()
        # Two records (one per step), each with 2 hashes
        assert len(cb.records) == 2
        assert (
            cb.records[0].input_hash == cb.records[1].input_hash
        )  # same inputs -> same hashes
        assert cb.records[0].step == 0
        assert cb.records[1].step == 1
        collector.remove()


# --------------------------------------------------------------------------- #
# OffloadCallback                                                              #
# --------------------------------------------------------------------------- #


class TestGradientStorageManager:
    # A valid 64-char SHA-256 hex string for use in deterministic tests.
    _HASH_A = "abcdef01" * 8  # 64 chars
    _HASH_B = "12345678" * 8  # 64 chars

    def _make_record(
        self,
        step: int,
        input_hash: str,
        tiny_model,
        tiny_batch,
    ) -> GradientRecord:
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        gradient = cb.records[0].gradient
        return GradientRecord(step=step, input_hash=input_hash, gradient=gradient)

    def test_save_creates_file(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            location = manager.save(rec)
            assert (Path(tmpdir) / location).exists()

    def test_save_updates_index(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            rec = self._make_record(7, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            assert self._HASH_A in manager.index
            entries = manager.index[self._HASH_A]
            assert len(entries) == 1
            assert entries[0]["step"] == 7
            assert entries[0]["idx"] == 0

    def test_index_log_written_after_save(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            assert (Path(tmpdir) / "index.jsonl").exists()
            assert (Path(tmpdir) / "index_meta.json").exists()

    def test_index_log_appends_one_line_per_save(self, tiny_model, tiny_batch):
        """The log grows by exactly one line per save, never rewriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            log = Path(tmpdir) / "index.jsonl"
            for step in range(3):
                rec = self._make_record(step, self._HASH_A, tiny_model, tiny_batch)
                manager.save_bulk([rec])
                assert len(log.read_text().splitlines()) == step + 1

    def test_index_meta_written_once_not_per_save(self, tiny_model, tiny_batch):
        """The settings sidecar never changes, so only the first save writes it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            meta = Path(tmpdir) / "index_meta.json"

            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save_bulk([rec])
            assert meta.exists()
            first_mtime = meta.stat().st_mtime_ns

            for step in range(1, 5):
                rec = self._make_record(step, f"{step:064x}", tiny_model, tiny_batch)
                manager.save_bulk([rec])
            assert meta.stat().st_mtime_ns == first_mtime
            # No temp file left behind by the one write that did happen.
            assert list(Path(tmpdir).glob("*.tmp")) == []

    def test_index_meta_precedes_first_log_line(self, tiny_model, tiny_batch):
        """A log never exists without the settings needed to interpret it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk(
                [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)],
            )
            meta = json.loads((Path(tmpdir) / "index_meta.json").read_text())
            assert meta["sample_id_key"] is None
            assert meta["gradient_accumulation_steps"] == 1
            assert meta["format"] == GradientStorageManager._INDEX_FORMAT

    def test_index_meta_rewritten_when_settings_change(
        self,
        tiny_model,
        tiny_batch,
    ):
        """Skipping the rewrite must not skip a genuine settings change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk(
                [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)],
            )
            meta = Path(tmpdir) / "index_meta.json"
            assert json.loads(meta.read_text())["gradient_accumulation_steps"] == 1

            manager.declare_gradient_accumulation_steps(4)
            manager.save_bulk(
                [self._make_record(1, self._HASH_B, tiny_model, tiny_batch)],
            )
            assert json.loads(meta.read_text())["gradient_accumulation_steps"] == 4
            # And a reopened store adopts the updated convention.
            assert GradientStorageManager(tmpdir).gradient_accumulation_steps == 4

    def test_index_write_cost_does_not_grow_with_store_size(
        self,
        tiny_model,
        tiny_batch,
    ):
        """Each save writes only its own delta, so the increment stays flat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            log = Path(tmpdir) / "index.jsonl"
            increments = []
            previous = 0
            for step in range(12):
                rec = self._make_record(step, f"{step:064x}", tiny_model, tiny_batch)
                manager.save_bulk([rec])
                size = log.stat().st_size
                increments.append(size - previous)
                previous = size
            assert max(increments) <= min(increments) * 1.5

    def test_reopened_store_recovers_every_entry(self, tiny_model, tiny_batch):
        """Expanding the log rebuilds the exact in-memory index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk(
                [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)],
            )
            manager.save_bulk(
                [self._make_record(3, self._HASH_B, tiny_model, tiny_batch)],
            )

            recovered = GradientStorageManager(tmpdir)
            assert recovered.index == manager.index
            assert recovered.lookup_by_hash(self._HASH_A) == [(0, 0)]
            assert recovered.lookup_by_hash(self._HASH_B) == [(3, 0)]
            assert recovered.load_all_by_hash(self._HASH_B)[0].step == 3

    def test_truncated_final_log_line_is_skipped(self, tiny_model, tiny_batch):
        """A crash mid-append loses only the save it was writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk(
                [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)],
            )
            manager.save_bulk(
                [self._make_record(1, self._HASH_B, tiny_model, tiny_batch)],
            )

            # Chop the final line in half, as a hard kill mid-append would.
            log = Path(tmpdir) / "index.jsonl"
            lines = log.read_text().splitlines()
            log.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2])

            with pytest.warns(UserWarning, match="truncated"):
                recovered = GradientStorageManager(tmpdir)
            # The completed save survives; only the truncated one is lost.
            assert recovered.lookup_by_hash(self._HASH_A) == [(0, 0)]
            assert self._HASH_B not in recovered.index

    def test_save_timings_recorded(self, tiny_model, tiny_batch):
        """Every save phase is timed, so slow offloading can be attributed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            assert all(p["calls"] == 0 for p in manager.timing.values())

            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save_bulk([rec])

            timing = manager.timing
            assert set(timing) == {
                "to_cpu",
                "write_group",
                "index_update",
                "index_write",
                "spill",
            }
            assert all(p["calls"] == 1 for p in timing.values())
            assert all(p["seconds"] >= 0.0 for p in timing.values())
            assert "write_group" in manager.timing_report()

            manager.reset_timing()
            assert all(p["calls"] == 0 for p in manager.timing.values())

    def test_timings_recorded_for_every_residency(self, tiny_model, tiny_batch):
        """Ephemeral residencies are timed too -- index_write is just ~0."""
        for residency in ("memory", "tiered"):
            with tempfile.TemporaryDirectory() as tmpdir:
                with GradientStorageManager(tmpdir, residency=residency) as manager:
                    rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
                    manager.save_bulk([rec])
                    assert all(p["calls"] == 1 for p in manager.timing.values())
                # No index persisted for the ephemeral residencies.
                assert list(Path(tmpdir).rglob("index.jsonl")) == []

    def test_tiered_spill_is_timed(self, tiny_model, tiny_batch):
        """A tiered spill is a full torch.save per group, so it must be timed.

        It runs after the four write phases; leaving it outside the report let
        a spilling store look nearly free while doing real disk work.
        """
        # budget_bytes=0 -> every save immediately evicts to the spill dir.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            GradientStorageManager(
                tmpdir,
                residency="tiered",
                budget_bytes=0,
            ) as manager,
        ):
            for step in range(3):
                manager.save_bulk(
                    [
                        self._make_record(
                            step,
                            f"{step:064x}",
                            tiny_model,
                            tiny_batch,
                        ),
                    ],
                )
            assert list(Path(tmpdir).rglob("tiered_spill_*/*.pt"))
            assert manager.timing["spill"]["seconds"] > 0.0
            assert manager.timing["spill"]["calls"] == 3

    def test_spill_time_is_zero_without_spilling(self, tiny_model, tiny_batch):
        """A disk store never spills, so the phase reads 0 rather than absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk(
                [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)],
            )
            assert manager.timing["spill"]["calls"] == 1
            assert manager.timing["spill"]["seconds"] < 1e-3

    def test_index_write_is_atomic_under_crash(
        self,
        tiny_model,
        tiny_batch,
        monkeypatch,
    ):
        """A crash mid-meta-write must leave the previous index readable.

        The sidecar is only rewritten when its settings actually change, so
        the crash is staged on such a change -- the one case where an
        already-populated store rewrites it and a torn write could destroy
        the settings a reader needs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)

            real_dump = json.dump

            def crashing_dump(obj, fp, *args, **kwargs):
                # Emit a truncated payload, then die -- as a hard kill
                # mid-write would.
                fp.write('{"format": 2, "sample_id_key": null, "trunc')
                raise RuntimeError("simulated crash mid-write")

            monkeypatch.setattr(
                "dattri_llm.gradient.storage_manager.json.dump",
                crashing_dump,
            )
            manager.declare_gradient_accumulation_steps(4)  # forces a rewrite
            rec2 = self._make_record(1, self._HASH_B, tiny_model, tiny_batch)
            with pytest.raises(RuntimeError, match="simulated crash"):
                manager.save(rec2)
            monkeypatch.setattr(
                "dattri_llm.gradient.storage_manager.json.dump",
                real_dump,
            )

            # The on-disk sidecar is the pre-crash version, not truncated JSON:
            # a fresh manager still loads every previously indexed record.
            recovered = GradientStorageManager(tmpdir)
            assert self._HASH_A in recovered.index
            assert self._HASH_B not in recovered.index
            records = recovered.load_all_by_hash(self._HASH_A)
            assert [r.step for r in records] == [0]

            # The failed write was not recorded as done, so the next save retries.
            manager.save(self._make_record(2, self._HASH_B, tiny_model, tiny_batch))
            meta = json.loads((Path(tmpdir) / "index_meta.json").read_text())
            assert meta["gradient_accumulation_steps"] == 4

    def test_load_all_by_hash(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            for step in [0, 2, 5]:
                rec = self._make_record(step, self._HASH_A, tiny_model, tiny_batch)
                manager.save(rec)
            records = manager.load_all_by_hash(self._HASH_A)
            assert [r.step for r in records] == [0, 2, 5]

    def test_load_all_by_hash_unknown_raises(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            with pytest.raises(KeyError):
                manager.load_all_by_hash(self._HASH_B)

    def test_load_all_inputs_equivalent_to_by_hash(self, tiny_model, tiny_batch):
        """The inputs/by-hash pair differ only by how the sample is identified."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            offload = OffloadCallback(
                offload_interval=100,
                file_manager=manager,
                recording_type="per_sample",
            )
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            sample1 = {k: v[1] for k, v in inputs.items()}
            h = hash_sample(sample1)
            by_inputs = manager.load_all(sample1)
            by_hash = manager.load_all_by_hash(h)
            assert [r.input_hash for r in by_inputs] == [r.input_hash for r in by_hash]
            assert by_inputs[0].input_hash == h
            collector.remove()

    def test_index_loaded_from_disk_on_construction(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = GradientStorageManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager1.save(rec)

            manager2 = GradientStorageManager(tmpdir)
            assert self._HASH_A in manager2.index
            assert manager2.index[self._HASH_A][0]["step"] == 0

    def test_duplicate_step_not_duplicated_in_index(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            manager.save(rec)  # same step saved again
            steps = [e["step"] for e in manager.index[self._HASH_A]]
            assert steps.count(0) == 1


class TestBatchSaving:
    """Tests for GradientStorageManager.save_bulk and OffloadCallback flushing."""

    _HASH_A = "abcdef01" * 8
    _HASH_B = "12345678" * 8

    def _make_record(self, step, input_hash, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        return GradientRecord(
            step=step,
            input_hash=input_hash,
            gradient=cb.records[0].gradient,
        )

    def test_save_bulk_creates_one_file(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            location = manager.save_bulk(recs)
            assert (Path(tmpdir) / location).exists()
            assert location.startswith("batch_")
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1

    def test_save_bulk_indexes_all_hashes(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            manager.save_bulk(recs)
            assert self._HASH_A in manager.index
            assert self._HASH_B in manager.index
            # Both point to the same batch file but different idx
            assert manager.index[self._HASH_A][0]["idx"] == 0
            assert manager.index[self._HASH_B][0]["idx"] == 1

    def test_save_bulk_load_round_trip(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            manager.save_bulk(recs)
            loaded_a = manager.load_all_by_hash(self._HASH_A)[0]
            loaded_b = manager.load_all_by_hash(self._HASH_B)[0]
            assert isinstance(loaded_a, GradientRecord)
            assert isinstance(loaded_b, GradientRecord)
            assert loaded_a.input_hash == self._HASH_A
            assert loaded_b.input_hash == self._HASH_B

    def test_next_batch_id_continues_after_reload(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = GradientStorageManager(tmpdir)
            recs = [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)]
            manager1.save_bulk(recs)  # writes batch_000000.pt

            manager2 = GradientStorageManager(tmpdir)
            recs2 = [self._make_record(1, self._HASH_B, tiny_model, tiny_batch)]
            manager2.save_bulk(recs2)  # should write batch_000001.pt

            batch_files = sorted(Path(tmpdir).glob("batch_*.pt"))
            assert len(batch_files) == 2
            assert batch_files[1].name == "batch_000001.pt"

    def test_per_batch_input_hash_indexed(self, tiny_model, tiny_batch):
        """Per-batch records carry input_hash as a list; the file manager
        indexes all of them.
        """
        inputs = {"input_ids": tiny_batch["input_ids"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            offload = OffloadCallback(offload_interval=100, file_manager=manager)
            cfg = HookManagerConfig(linear_io=REGISTER_ALL)
            collector = HookManager(tiny_model, config=cfg, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            # Each sample hash should be indexed even though only one file was saved.
            B = tiny_batch["input_ids"].shape[0]
            indexed_hashes = set(manager.index.keys())
            assert len(indexed_hashes) == B
            collector.remove()

    def test_offload_groups_one_step_into_one_file(self, tiny_model, tiny_batch):
        """One batch step -> one batch file (offload_interval=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            offload = OffloadCallback(offload_interval=1, file_manager=manager)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            # B per-sample records in one batch file
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1
            assert len(list(Path(tmpdir).glob("step_*.pt"))) == 0
            collector.remove()

    def test_offload_load_all_per_sample(self, tiny_model, tiny_batch):
        """Every sample written by OffloadCallback(per_sample) is retrievable
        from its raw inputs, one record per step.
        """
        inputs = {"input_ids": tiny_batch["input_ids"]}
        B = tiny_batch["input_ids"].shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            offload = OffloadCallback(
                offload_interval=1,
                file_manager=manager,
                recording_type="per_sample",
            )
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            for i in range(B):
                records = manager.load_all({k: v[i] for k, v in inputs.items()})
                assert len(records) == 1
                assert isinstance(records[0], GradientRecord)
            collector.remove()


class TestLookupAndLoadSample:
    """input_hash -> (step, sample_idx) -> file: position-indexed per-sample lookup."""

    H: ClassVar[list[str]] = ["aa" * 32, "bb" * 32, "cc" * 32]  # three sample hashes

    def _batch_record(self, step, hashes):
        """A per-batch record whose rows are distinguishable: row i at step s
        holds the constant ``100*s + i``.
        """
        B = len(hashes)
        data = torch.arange(B, dtype=torch.float).unsqueeze(1).repeat(1, 4) + 100 * step
        g = Gradient(
            representation={"l": "materialized"},
            data={"l": data},
            layer_types={"l": "nn.Linear"},
        )
        return GradientRecord(step=step, input_hash=hashes, gradient=g)

    def _shuffled_store(self, tmpdir):
        """Two steps with the SAME samples at different batch positions --
        the shuffling scenario the (step, sample) index exists for.
        """
        manager = GradientStorageManager(tmpdir)
        manager.save_bulk([self._batch_record(0, [self.H[0], self.H[1], self.H[2]])])
        manager.save_bulk([self._batch_record(1, [self.H[2], self.H[0], self.H[1]])])
        return manager

    def test_lookup_returns_step_sample_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._shuffled_store(tmpdir)
            # H[2] sat at position 2 in step 0 and position 0 in step 1.
            assert manager.lookup_by_hash(self.H[2]) == [(0, 2), (1, 0)]
            assert manager.lookup_by_hash(self.H[0]) == [(0, 0), (1, 1)]

    def test_load_sample_slices_the_indexed_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._shuffled_store(tmpdir)
            for _h_i, h in enumerate(self.H):
                for step, sample_idx in manager.lookup_by_hash(h):
                    g = manager.load_sample_by_hash(h, step, sample_idx)
                    # row value encodes (step, position): 100*step + sample_idx.
                    expected = float(100 * step + sample_idx)
                    assert g.data["l"].shape[0] == 1
                    assert torch.allclose(g.data["l"], torch.full((1, 4), expected))

    def test_load_sample_missing_pair_raises_with_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._shuffled_store(tmpdir)
            with pytest.raises(KeyError, match=r"Available \(step, sample_idx\) pairs"):
                manager.load_sample_by_hash(self.H[0], step=7, sample_idx=0)

    def test_lookup_unknown_hash_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._shuffled_store(tmpdir)
            with pytest.raises(KeyError, match="not in index"):
                manager.lookup_by_hash("ff" * 32)

    def test_inputs_and_by_hash_pairs_agree(self):
        """Each load-family pair differs only by how the sample is identified."""
        x = torch.arange(12, dtype=torch.float).reshape(3, 4)
        hashes = hash_batch({"x": x}, 3)
        sample1 = {"x": x[1]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientStorageManager(tmpdir)
            manager.save_bulk([self._batch_record(0, hashes)])
            assert manager.lookup(sample1) == manager.lookup_by_hash(hashes[1])
            step, sample_idx = manager.lookup(sample1)[0]
            g_a = manager.load_sample(sample1, step, sample_idx)
            g_b = manager.load_sample_by_hash(hashes[1], step, sample_idx)
            assert torch.equal(g_a.data["l"], g_b.data["l"])


class TestOffloadCallback:
    def _make_offload(self, tmpdir, offload_interval=100):
        manager = GradientStorageManager(tmpdir)
        offload = OffloadCallback(
            offload_interval=offload_interval,
            file_manager=manager,
        )
        return manager, offload

    def test_files_written_after_context(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            _manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            # OffloadCallback always writes batch files
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1
            collector.remove()

    def test_index_log_written(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            assert (Path(tmpdir) / "index.jsonl").exists()
            assert len(manager.index) == tiny_batch["input_ids"].shape[0]
            collector.remove()

    def test_index_maps_hash_to_steps(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                for _ in range(2):
                    tiny_model(tiny_batch["input_ids"]).mean().backward()
            for entries in manager.index.values():
                steps = [e["step"] for e in entries]
                assert steps == sorted(steps)
                assert len(steps) == 2
            collector.remove()

    def test_load_round_trip(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            for h in manager.index:
                for step, sample_idx in manager.lookup_by_hash(h):
                    g = manager.load_sample_by_hash(h, step, sample_idx)
                    assert g.batch_size == 1
            collector.remove()

    def test_load_all_via_manager(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                for _ in range(3):
                    tiny_model(tiny_batch["input_ids"]).mean().backward()
            some_hash = next(iter(manager.index))
            records = manager.load_all_by_hash(some_hash)
            assert len(records) == 3
            collector.remove()

    def test_periodic_flush(self, tiny_model, tiny_batch):
        # offload_interval=2: flush every 2 batch steps -> 4 steps / 2 = 2 batch files
        with tempfile.TemporaryDirectory() as tmpdir:
            _manager, offload = self._make_offload(tmpdir, offload_interval=2)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                for _ in range(4):
                    tiny_model(tiny_batch["input_ids"]).mean().backward()
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 2
            collector.remove()

    def test_staged_property(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            _manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
                assert len(offload.staged) == 1  # one per-batch record per step
            assert len(offload.staged) == 0
            collector.remove()

    def test_manager_accessible_for_queries(self, tiny_model, tiny_batch):
        """file_manager is the query handle -- offload only owns saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            # All loading goes through the manager, not the offload callback.
            assert offload.file_manager is manager
            some_hash = next(iter(manager.index))
            records = manager.load_all_by_hash(some_hash)
            assert len(records) == 1
            collector.remove()


# --------------------------------------------------------------------------- #
# HookManager -- remove                                                   #
# --------------------------------------------------------------------------- #


class TestHookManagerRemove:
    def test_remove_clears_layer_names(self, tiny_model):
        collector = HookManager(tiny_model)
        assert len(collector.layer_names) > 0
        collector.remove()
        assert collector.layer_names == []

    def test_hooks_inactive_after_remove(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        collector.remove()
        # After remove, hooks are gone -- no records should be emitted
        # (collect() itself would raise an AttributeError since buffers cleared,
        # but running without collect() also should not emit records)
        logits = tiny_model(tiny_batch["input_ids"])
        logits.mean().backward()
        assert len(cb.records) == 0


# --------------------------------------------------------------------------- #
# HookManager.get_gradient -- last-step gradient cache                          #
# --------------------------------------------------------------------------- #


class TestGetGradient:
    def test_after_completed_step(self, tiny_model, tiny_batch):
        # A completed step clears the buffers; get_gradient must still return
        # the assembled gradient from the single-slot cache.
        hm = HookManager(tiny_model)
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
            g = hm.get_gradient()
        assert set(g.layer_names) == {
            "embedding",
            "attn_proj",
            "mlp.0",
            "mlp.2",
            "lm_head",
        }

    def test_cache_is_same_object_as_record(self, tiny_model, tiny_batch):
        # The cached gradient is exactly the object handed to on_step_end --
        # no duplicate copy is kept in memory.
        cb = RecordingCallback()
        hm = HookManager(tiny_model, callbacks=[cb])
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert hm.get_gradient() is cb.records[-1].gradient

    def test_cache_tracks_latest_step(self, tiny_model, tiny_batch):
        # Across multiple steps only the most recent is cached.
        hm = HookManager(tiny_model)
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
            g1 = hm.get_gradient()
            tiny_model(tiny_batch["input_ids"]).mean().backward()
            g2 = hm.get_gradient()
        assert g1 is not g2

    def test_cache_persists_after_context_exit(self, tiny_model, tiny_batch):
        hm = HookManager(tiny_model)
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        # Still retrievable after the collect() context closes.
        assert len(hm.get_gradient().layer_names) == 5

    def test_new_collect_clears_stale_cache(self, tiny_model, tiny_batch):
        hm = HookManager(tiny_model)
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        # Entering a fresh context drops the prior cache; before any backward
        # there is nothing to return.
        with hm.collect():
            with pytest.raises(RuntimeError):
                hm.get_gradient()
            tiny_model(tiny_batch["input_ids"]).mean().backward()

    def test_raises_when_nothing_captured(self, tiny_model):
        hm = HookManager(tiny_model)
        with pytest.raises(RuntimeError):
            hm.get_gradient()

    def test_remove_clears_cache(self, tiny_model, tiny_batch):
        hm = HookManager(tiny_model)
        with hm.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        hm.remove()
        assert hm._last_gradient is None


# --------------------------------------------------------------------------- #
# HookManagerConfig                                                            #
# --------------------------------------------------------------------------- #


class TestHookManagerConfig:
    # -- default -------------------------------------------------------------

    def test_default_selectors(self):
        cfg = HookManagerConfig()
        assert cfg.hook_types == {}
        assert cfg.linear_io is None
        assert cfg.param_grad is None
        assert cfg.is_default

    # -- hook_types assignment (the basic control) -----------------------------

    def test_hook_types_assignment(self):
        cfg = HookManagerConfig(
            hook_types={"mlp.0": "linear_io", "lm_head": "param_grad"},
        )
        assert cfg.hook_types == {"mlp.0": "linear_io", "lm_head": "param_grad"}
        assert not cfg.is_default

    def test_hook_types_invalid_value_raises(self):
        with pytest.raises(ValueError, match="not a valid hook type"):
            HookManagerConfig(hook_types={"mlp.0": "bogus"})

    def test_hook_types_wrong_type_raises(self):
        with pytest.raises(TypeError, match="hook_types must be a dict"):
            HookManagerConfig(hook_types=["mlp.0"])  # type: ignore[arg-type]

    # -- REGISTER_ALL ----------------------------------------------------------

    def test_register_all_is_singleton(self):
        assert HookManagerConfig(linear_io=REGISTER_ALL).linear_io is REGISTER_ALL

    def test_linear_io_register_all(self):
        cfg = HookManagerConfig(linear_io=REGISTER_ALL)
        assert cfg.linear_io is REGISTER_ALL
        assert cfg.param_grad is None
        assert not cfg.is_default

    def test_param_grad_register_all(self):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        assert cfg.param_grad is REGISTER_ALL
        assert cfg.linear_io is None
        assert not cfg.is_default

    # -- pattern lists ---------------------------------------------------------

    def test_linear_io_pattern(self):
        cfg = HookManagerConfig(linear_io=[r"mlp\.0"])
        assert cfg.linear_io == [r"mlp\.0"]

    def test_param_grad_pattern(self):
        cfg = HookManagerConfig(param_grad=[r"mlp\.[02]"])
        assert cfg.param_grad == [r"mlp\.[02]"]

    def test_both_selectors(self):
        cfg = HookManagerConfig(linear_io=[r"mlp\."], param_grad=[r"lm_head"])
        assert cfg.linear_io == [r"mlp\."]
        assert cfg.param_grad == [r"lm_head"]

    # -- validation ------------------------------------------------------------

    def test_invalid_selector_type_raises(self):
        with pytest.raises(TypeError, match="must be None, REGISTER_ALL"):
            HookManagerConfig(linear_io="mlp")  # type: ignore[arg-type]

    def test_non_string_pattern_raises(self):
        with pytest.raises(TypeError, match="regex strings"):
            HookManagerConfig(param_grad=[123])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# HookManager -- param_grad hook type                                          #
# --------------------------------------------------------------------------- #


class TestHookManagerParamGrad:
    def test_param_layer_names_empty_for_linear_io_only(self, tiny_model):
        collector = HookManager(tiny_model)
        assert collector.param_layer_names == []
        collector.remove()

    def test_param_layer_names_populated_for_param_grad(self, tiny_model):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        collector = HookManager(tiny_model, config=cfg)
        assert len(collector.param_layer_names) > 0
        collector.remove()

    def test_param_grad_only_emits_one_record_per_step(self, tiny_model, tiny_batch):
        """param_grad-only per-batch mode emits one record per step (not per sample)."""
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        cb = RecordingCallback()
        B = tiny_batch["input_ids"].shape[0]
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert len(cb.records) == 1  # one batch record
        assert isinstance(cb.records[0].input_hash, list)
        assert len(cb.records[0].input_hash) == B
        collector.remove()

    def test_param_grad_entries_in_gradient(self, tiny_model, tiny_batch):
        """Records from param_grad-only per-batch mode contain materialized entries."""
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        cb = RecordingCallback()
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        g = cb.records[0].gradient
        param_grad_layers = [
            n for n, t in (g.layer_types or {}).items() if t == PARAM_GRAD_TYPES
        ]
        assert len(param_grad_layers) > 0
        collector.remove()

    def test_param_grad_tensors_are_finite(self, tiny_model, tiny_batch):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        cb = RecordingCallback()
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        g = cb.records[0].gradient
        for name, val in g.data.items():
            if (g.layer_types or {}).get(name) == PARAM_GRAD_TYPES:
                assert isinstance(val, torch.Tensor)
                assert torch.isfinite(val).all(), f"{name} has NaN/Inf"
        collector.remove()

    def test_combined_per_batch_emits_one_record_per_step(self, tiny_model, tiny_batch):
        """Combined linear_io + param_grad per-batch: one record per step.

        Uses disjoint selectors (mlp.* via linear_io, lm_head via param_grad)
        so each family claims a distinct set of layers.
        """
        cfg = HookManagerConfig(linear_io=[r"mlp\."], param_grad=[r"lm_head"])
        cb = RecordingCallback()
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert len(cb.records) == 1
        g = cb.records[0].gradient
        types = set((g.layer_types or {}).values())
        # linear_io layers store their canonical class names (e.g. "nn.Linear")
        # rather than the generic marker.
        non_param_types = types - {PARAM_GRAD_TYPES}
        assert len(non_param_types) > 0, (
            "Expected at least one canonical layer type (linear_io layers) in types"
        )
        assert PARAM_GRAD_TYPES in types
        collector.remove()

    def test_per_batch_linear_io_only_emits_one_record_per_step(
        self,
        tiny_model,
        tiny_batch,
    ):
        """per_batch with linear_io only emits one record per step."""
        cfg = HookManagerConfig(linear_io=REGISTER_ALL)
        cb = RecordingCallback()
        B = tiny_batch["input_ids"].shape[0]
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert len(cb.records) == 1
        assert isinstance(cb.records[0].input_hash, list)
        assert len(cb.records[0].input_hash) == B
        collector.remove()

    def test_param_grad_step_increments(self, tiny_model, tiny_batch):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        collector = HookManager(tiny_model, config=cfg)
        with collector.collect():
            for _ in range(3):
                tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert collector.steps_collected == 3
        collector.remove()

    def test_param_grad_step_ids_in_records(self, tiny_model, tiny_batch):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        cb = RecordingCallback()
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            for _ in range(3):
                tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert [r.step for r in cb.records] == [0, 1, 2]
        collector.remove()

    def test_param_grad_custom_patterns(self, tiny_model, tiny_batch):
        cfg = HookManagerConfig(param_grad=[r"mlp\.0"])
        cb = RecordingCallback()
        collector = HookManager(tiny_model, config=cfg, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        g = cb.records[0].gradient
        param_grad_keys = [
            n for n, t in (g.layer_types or {}).items() if t == PARAM_GRAD_TYPES
        ]
        assert all(k.startswith("mlp.0.") for k in param_grad_keys)
        collector.remove()

    def test_param_grad_selector_set(self, tiny_model):
        """param_grad selector is recorded on HookManagerConfig."""
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        assert cfg.param_grad is REGISTER_ALL

    def test_remove_clears_param_layer_names(self, tiny_model):
        cfg = HookManagerConfig(param_grad=REGISTER_ALL)
        collector = HookManager(tiny_model, config=cfg)
        assert len(collector.param_layer_names) > 0
        collector.remove()
        assert collector.param_layer_names == []


# --------------------------------------------------------------------------- #
# DDP-like multi-rank storage tests                                            #
# (simulated: two GradientStorageManagers each given a fake rank via monkeypatch) #
# --------------------------------------------------------------------------- #


class TestGradientStorageManagerDDP:
    """Simulate two DDP ranks writing to the same root save_dir."""

    _HASH_A = "aaaaaaaa" * 8  # 64 chars
    _HASH_B = "bbbbbbbb" * 8

    def _make_record(self, step, input_hash, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        return GradientRecord(
            step=step,
            input_hash=input_hash,
            gradient=cb.records[0].gradient,
        )

    def _manager_for_rank(self, tmpdir, rank, monkeypatch):
        """Return a GradientStorageManager that behaves as if running on *rank*."""
        import dattri_llm.gradient.storage_manager as fm_module

        monkeypatch.setattr(fm_module, "dist_rank", lambda: rank)
        return GradientStorageManager(tmpdir)

    def test_each_rank_writes_to_own_subdir(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        import dattri_llm.gradient.storage_manager as fm_module

        monkeypatch.setattr(fm_module, "dist_rank", lambda: 0)
        m0 = GradientStorageManager(str(tmp_path))
        rec0 = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m0.save_bulk([rec0])

        monkeypatch.setattr(fm_module, "dist_rank", lambda: 1)
        m1 = GradientStorageManager(str(tmp_path))
        rec1 = self._make_record(0, self._HASH_B, tiny_model, tiny_batch)
        m1.save_bulk([rec1])

        assert (tmp_path / "rank_0").is_dir()
        assert (tmp_path / "rank_1").is_dir()
        assert len(list((tmp_path / "rank_0").glob("batch_*.pt"))) == 1
        assert len(list((tmp_path / "rank_1").glob("batch_*.pt"))) == 1

    def test_no_batch_id_collision(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """Both ranks start _next_batch_id at 0 but write to different subdirs."""
        import dattri_llm.gradient.storage_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "dist_rank", lambda r=rank: r)
            m = GradientStorageManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_bulk([rec])

        # Both write batch_000000.pt but in separate dirs -- no data loss.
        assert (tmp_path / "rank_0" / "batch_000000.pt").exists()
        assert (tmp_path / "rank_1" / "batch_000000.pt").exists()

    def test_rank_resolved_lazily_at_first_save(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        """A manager constructed before ``init_process_group`` (rank probe
        returns None) must still route to rank_N/ once the process group
        exists at save time -- construction order must not matter.
        """
        import dattri_llm.gradient.storage_manager as fm_module

        # Construction happens pre-init on every "rank": probe sees nothing.
        monkeypatch.setattr(fm_module, "dist_rank", lambda: None)
        managers = {rank: GradientStorageManager(str(tmp_path)) for rank in (0, 1)}

        # By save time the process group is up and the true rank is visible.
        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "dist_rank", lambda r=rank: r)
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            managers[rank].save_bulk([rec])

        # Nothing landed in the root; each rank got its own subdirectory.
        assert not list(tmp_path.glob("*.pt"))
        assert not (tmp_path / "index.jsonl").exists()
        assert (tmp_path / "rank_0" / "batch_000000.pt").exists()
        assert (tmp_path / "rank_1" / "batch_000000.pt").exists()

        # The routing freezes at the first save: a later probe change (e.g.
        # destroy_process_group) must not switch directories mid-run.
        monkeypatch.setattr(fm_module, "dist_rank", lambda: None)
        rec = self._make_record(1, self._HASH_A, tiny_model, tiny_batch)
        managers[0].save_bulk([rec])
        assert (tmp_path / "rank_0" / "batch_000001.pt").exists()
        assert not list(tmp_path.glob("*.pt"))

    def test_each_rank_has_own_index_log(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        import dattri_llm.gradient.storage_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "dist_rank", lambda r=rank: r)
            m = GradientStorageManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_bulk([rec])

        # The identifier scheme lives in the per-rank settings sidecar.
        meta0 = json.loads((tmp_path / "rank_0" / "index_meta.json").read_text())
        assert meta0["sample_id_key"] is None

        def logged_hashes(rank: int) -> set[str]:
            log = (tmp_path / f"rank_{rank}" / "index.jsonl").read_text()
            return {
                h
                for line in log.splitlines()
                for record in json.loads(line)["records"]
                for h in record["hashes"]
            }

        # Each rank's log only contains its own hash.
        hashes0, hashes1 = logged_hashes(0), logged_hashes(1)
        assert self._HASH_A in hashes0
        assert self._HASH_B not in hashes0
        assert self._HASH_B in hashes1
        assert self._HASH_A not in hashes1

    def test_index_entries_use_rank_relative_paths(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        import dattri_llm.gradient.storage_manager as fm_module

        monkeypatch.setattr(fm_module, "dist_rank", lambda: 0)
        m = GradientStorageManager(str(tmp_path))
        rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m.save_bulk([rec])

        entry = m.index[self._HASH_A][0]
        assert entry["file"].startswith("rank_0/")

    def test_fresh_manager_merges_all_ranks(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        """A reader opened after training sees gradients from every rank."""
        import dattri_llm.gradient.storage_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "dist_rank", lambda r=rank: r)
            m = GradientStorageManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_bulk([rec])

        # Reader: no distributed context.
        monkeypatch.setattr(fm_module, "dist_rank", lambda: None)
        reader = GradientStorageManager(str(tmp_path))
        assert self._HASH_A in reader.index
        assert self._HASH_B in reader.index

    def test_load_across_ranks(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """A reader opened after training can load records from any rank."""
        import dattri_llm.gradient.storage_manager as fm_module

        # Simulate training: rank 0 and rank 1 each write their own records.
        monkeypatch.setattr(fm_module, "dist_rank", lambda: 0)
        m0 = GradientStorageManager(str(tmp_path))
        m0.save_bulk([self._make_record(0, self._HASH_A, tiny_model, tiny_batch)])

        monkeypatch.setattr(fm_module, "dist_rank", lambda: 1)
        m1 = GradientStorageManager(str(tmp_path))
        m1.save_bulk([self._make_record(0, self._HASH_B, tiny_model, tiny_batch)])

        # Simulate post-training analysis: fresh reader with no distributed context.
        monkeypatch.setattr(fm_module, "dist_rank", lambda: None)
        reader = GradientStorageManager(str(tmp_path))
        loaded_a = reader.load_all_by_hash(self._HASH_A)[0]
        loaded_b = reader.load_all_by_hash(self._HASH_B)[0]
        assert loaded_a.input_hash == self._HASH_A
        assert loaded_b.input_hash == self._HASH_B

    def test_non_distributed_still_writes_to_root(
        self,
        tmp_path,
        monkeypatch,
        tiny_model,
        tiny_batch,
    ):
        """Single-GPU path: no rank_N/ subdirectory created."""
        import dattri_llm.gradient.storage_manager as fm_module

        monkeypatch.setattr(fm_module, "dist_rank", lambda: None)
        m = GradientStorageManager(str(tmp_path))
        rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m.save_bulk([rec])

        assert not any(tmp_path.glob("rank_*"))
        assert (tmp_path / "batch_000000.pt").exists()
        assert (tmp_path / "index.jsonl").exists()

    def test_linear_io_pattern_filters_layers(self, tiny_model, tiny_batch):
        """A linear_io regex selector restricts collection to matching layers."""
        cb = RecordingCallback()
        collector = HookManager(
            tiny_model,
            config=HookManagerConfig(linear_io=[r"mlp\.0"]),
            callbacks=[cb],
        )
        assert collector.layer_names == ["mlp.0"]
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert len(cb.records) == 1  # one record per step
        collector.remove()


class TestIndexMergeRoundTrip:
    """Reading the append-only index back: expansion, merge, retrieval.

    The DDP tests above write one single-hash record per rank, so ``idx`` and
    ``sample_idx`` are always 0 and each log holds a single line.  These cover
    what only appears at scale -- several lines per log, several records per
    line, several samples per record -- and assert on the retrieved gradient
    *values*, not merely that a hash is present in the index.
    """

    @staticmethod
    def _batch_record(step: int, hashes: list[str]) -> GradientRecord:
        """A record whose rows are distinguishable: row i holds ``100*step + i``."""
        data = torch.arange(len(hashes), dtype=torch.float)
        data = data.unsqueeze(1).repeat(1, 4) + 100 * step
        return GradientRecord(
            step=step,
            input_hash=hashes,
            gradient=Gradient(
                representation={"l": "materialized"},
                data={"l": data},
                layer_types={"l": "nn.Linear"},
            ),
        )

    @staticmethod
    def _hash(tag: str) -> str:
        return (tag + "0" * 64)[:64]

    def _write_two_ranks(self, tmp_path, monkeypatch):
        """Two ranks x two saves x two records x three samples.

        Every axis the log line encodes varies: the file (rank), the record
        position within a save (``idx``), and the sample position within a
        record (``sample_idx``).

        Returns:
            ``{hash: (step, sample_idx)}`` for every sample written.
        """
        import dattri_llm.gradient.storage_manager as sm_module

        expected: dict[str, tuple[int, int]] = {}
        step = 0
        for rank in (0, 1):
            monkeypatch.setattr(sm_module, "dist_rank", lambda r=rank: r)
            manager = GradientStorageManager(str(tmp_path))
            for save in range(2):
                group = []
                for rec_idx in range(2):
                    hashes = [
                        self._hash(f"r{rank}s{save}q{rec_idx}i{i}") for i in range(3)
                    ]
                    for sample_idx, h in enumerate(hashes):
                        expected[h] = (step, sample_idx)
                    group.append(self._batch_record(step, hashes))
                    step += 1
                manager.save_bulk(group)
        return expected

    def test_merged_index_retrieves_every_sample_from_every_rank(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A fresh reader resolves every sample to the right gradient row."""
        import dattri_llm.gradient.storage_manager as sm_module

        expected = self._write_two_ranks(tmp_path, monkeypatch)

        # Post-training reader: no distributed context, merges both ranks.
        monkeypatch.setattr(sm_module, "dist_rank", lambda: None)
        reader = GradientStorageManager(str(tmp_path))

        assert len(reader.index) == len(expected) == 24
        for h, (step, sample_idx) in expected.items():
            assert reader.lookup_by_hash(h) == [(step, sample_idx)]
            g = reader.load_sample_by_hash(h, step, sample_idx)
            # Row value encodes (step, position), so a wrong file, wrong
            # record within the file, or wrong row all show up here.
            assert g.data["l"].shape[0] == 1
            assert torch.allclose(
                g.data["l"],
                torch.full((1, 4), float(100 * step + sample_idx)),
            )

    def test_merged_index_keeps_every_occurrence_of_a_repeated_sample(
        self,
        tmp_path,
        monkeypatch,
    ):
        """One sample seen at several steps keeps one entry per occurrence."""
        import dattri_llm.gradient.storage_manager as sm_module

        shared = self._hash("shared")
        # rank 0 sees it at steps 0 and 2, rank 1 at step 5 -- and at a
        # different batch position each time, as shuffling would produce.
        layout = {0: [(0, 0), (2, 1)], 1: [(5, 2)]}
        for rank, occurrences in layout.items():
            monkeypatch.setattr(sm_module, "dist_rank", lambda r=rank: r)
            manager = GradientStorageManager(str(tmp_path))
            for step, position in occurrences:
                hashes = [self._hash(f"pad{step}_{i}") for i in range(3)]
                hashes[position] = shared
                manager.save_bulk([self._batch_record(step, hashes)])

        monkeypatch.setattr(sm_module, "dist_rank", lambda: None)
        reader = GradientStorageManager(str(tmp_path))

        assert reader.lookup_by_hash(shared) == [(0, 0), (2, 1), (5, 2)]
        for step, sample_idx in reader.lookup_by_hash(shared):
            g = reader.load_sample_by_hash(shared, step, sample_idx)
            assert torch.allclose(
                g.data["l"],
                torch.full((1, 4), float(100 * step + sample_idx)),
            )
        # load_all_by_hash returns the whole record per occurrence, in step order.
        assert [r.step for r in reader.load_all_by_hash(shared)] == [0, 2, 5]

    def test_every_line_of_a_multi_save_log_is_replayed(self, tmp_path):
        """All lines are read back, not just the last one appended."""
        manager = GradientStorageManager(str(tmp_path))
        expected = {}
        for step in range(8):
            hashes = [self._hash(f"s{step}i{i}") for i in range(3)]
            for sample_idx, h in enumerate(hashes):
                expected[h] = (step, sample_idx)
            manager.save_bulk([self._batch_record(step, hashes)])

        log_lines = (tmp_path / "index.jsonl").read_text().splitlines()
        assert len(log_lines) == 8

        reader = GradientStorageManager(str(tmp_path))
        assert reader.index == manager.index
        for h, (step, sample_idx) in expected.items():
            assert reader.lookup_by_hash(h) == [(step, sample_idx)]

    def test_expand_log_line_recovers_sample_positions(self):
        """One log line unpacks into one entry per sample, positions intact."""
        from dattri_llm.gradient.storage_manager import _expand_log_line

        out = _expand_log_line(
            {
                "file": "rank_1/batch_000002.pt",
                "records": [
                    {"idx": 0, "step": 4, "hashes": ["ha", "hb"]},
                    {"idx": 1, "step": 5, "hashes": ["hc"]},
                ],
            },
        )
        assert out == {
            "ha": [
                {
                    "file": "rank_1/batch_000002.pt",
                    "idx": 0,
                    "step": 4,
                    "sample_idx": 0,
                },
            ],
            "hb": [
                {
                    "file": "rank_1/batch_000002.pt",
                    "idx": 0,
                    "step": 4,
                    "sample_idx": 1,
                },
            ],
            "hc": [
                {
                    "file": "rank_1/batch_000002.pt",
                    "idx": 1,
                    "step": 5,
                    "sample_idx": 0,
                },
            ],
        }

    def test_merge_index_appends_distinct_and_skips_duplicates(self):
        """Merging keeps every distinct occurrence and never doubles one."""
        from dattri_llm.gradient.storage_manager import _merge_index

        first = {"file": "a.pt", "idx": 0, "step": 0, "sample_idx": 0}
        second = {"file": "b.pt", "idx": 0, "step": 1, "sample_idx": 3}

        merged: dict[str, list[dict]] = {}
        _merge_index(merged, {"h": [first]})
        _merge_index(merged, {"h": [second]})
        assert merged == {"h": [first, second]}

        # An identical entry (compared by value, not identity) is not re-added.
        _merge_index(merged, {"h": [dict(first)]})
        assert merged == {"h": [first, second]}

    def test_reader_without_distributed_context_ignores_rank_layout(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Merged entries stay loadable through paths relative to the root."""
        import dattri_llm.gradient.storage_manager as sm_module

        expected = self._write_two_ranks(tmp_path, monkeypatch)
        monkeypatch.setattr(sm_module, "dist_rank", lambda: None)
        reader = GradientStorageManager(str(tmp_path))

        files = {e["file"] for entries in reader.index.values() for e in entries}
        assert {f.split("/")[0] for f in files} == {"rank_0", "rank_1"}
        # Every referenced path resolves under the root, from a reader that
        # never knew which rank produced it.
        for rel in files:
            assert (tmp_path / rel).exists()
        assert len(reader.load_records(next(iter(files)))) == 2
        assert len(expected) == 24


class TestResidency:
    """disk / memory / tiered residency backends of GradientStorageManager."""

    @staticmethod
    def _batch_record(step: int, seed: int) -> GradientRecord:
        torch.manual_seed(seed)
        data = {f"L{layer}": torch.randn(4, 32) for layer in range(3)}
        gradient = Gradient(
            representation=dict.fromkeys(data, "materialized"),
            data=data,
            layer_types=dict.fromkeys(data, "nn.Linear"),
        )
        return GradientRecord(
            step=step,
            input_hash=[f"h{step}_{i}" for i in range(4)],
            gradient=gradient,
        )

    _GROUP_BYTES = 3 * 4 * 32 * 4  # 3 layers x (4x32) float32

    def _collect(self, residency, tmpdir, **kw):
        fm = GradientStorageManager(tmpdir, residency=residency, **kw)
        for step in range(5):
            fm.save_bulk([self._batch_record(step, seed=step)])
        blocks = {}
        for loc, _by_step in fm.iter_steps(list(range(5))):
            for rec in fm.load_records(loc):
                blocks[rec.step] = rec.gradient.data["L0"].clone()
        return fm, blocks

    def test_rejects_unknown_residency(self):
        with (
            tempfile.TemporaryDirectory() as d,
            pytest.raises(ValueError, match="residency must be one of"),
        ):
            GradientStorageManager(d, residency="cloud")

    def test_memory_writes_no_files_and_matches_disk(self):
        with tempfile.TemporaryDirectory() as dd, tempfile.TemporaryDirectory() as dm:
            _, disk_blocks = self._collect("disk", dd)
            fm_mem, mem_blocks = self._collect("memory", dm)
            # Nothing serialized to disk.
            assert list(Path(dm).rglob("*.pt")) == []
            assert list(Path(dm).rglob("index.jsonl")) == []
            assert list(Path(dm).rglob("index_meta.json")) == []
            assert fm_mem.residency == "memory"
            # Identical records read back (same seeded data).
            for step in range(5):
                assert torch.equal(disk_blocks[step], mem_blocks[step])

    def test_disk_creates_one_file_per_group(self):
        with tempfile.TemporaryDirectory() as d:
            self._collect("disk", d)
            assert len(list(Path(d).glob("batch_*.pt"))) == 5

    def test_tiered_spills_oldest_and_reads_transparently(self):
        with tempfile.TemporaryDirectory() as dd, tempfile.TemporaryDirectory() as dt:
            _, disk_blocks = self._collect("disk", dd)
            # Budget for ~2 groups resident -> the 3 oldest spill to disk.
            budget = int(self._GROUP_BYTES * 2.5)
            fm, tiered_blocks = self._collect("tiered", dt, budget_bytes=budget)
            spilled = list(Path(dt).rglob("*.pt"))
            assert len(spilled) == 3  # 5 groups - 2 kept resident
            assert fm._mem_bytes <= budget
            # Reads are transparent across the memory/disk split.
            for step in range(5):
                assert torch.equal(disk_blocks[step], tiered_blocks[step])

    def test_tiered_auto_budget_when_unspecified(self):
        with tempfile.TemporaryDirectory() as d:
            fm = GradientStorageManager(d, residency="tiered")
            assert fm._budget_bytes is not None
            assert fm._budget_bytes > 0

    def test_tiered_close_removes_spill_files(self):
        with tempfile.TemporaryDirectory() as d:
            budget = int(self._GROUP_BYTES * 2.5)
            fm = GradientStorageManager(d, residency="tiered", budget_bytes=budget)
            for step in range(5):
                fm.save_bulk([self._batch_record(step, seed=step)])
            assert len(list(Path(d).rglob("*.pt"))) == 3  # spilled
            fm.close()
            # Spill dir gone; save_dir left pristine.
            assert list(Path(d).rglob("*.pt")) == []
            assert list(Path(d).iterdir()) == []

    def test_tiered_context_manager_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            budget = int(self._GROUP_BYTES * 2.5)
            with GradientStorageManager(
                d,
                residency="tiered",
                budget_bytes=budget,
            ) as fm:
                for step in range(5):
                    fm.save_bulk([self._batch_record(step, seed=step)])
                assert len(list(Path(d).rglob("*.pt"))) == 3
            assert list(Path(d).iterdir()) == []  # cleaned on __exit__

    def test_close_is_idempotent_and_noop_for_disk(self):
        with tempfile.TemporaryDirectory() as d:
            fm = GradientStorageManager(d, residency="disk")
            fm.save_bulk([self._batch_record(0, seed=0)])
            fm.close()
            fm.close()  # idempotent
            # Disk files are the durable store -- untouched by close().
            assert len(list(Path(d).glob("batch_*.pt"))) == 1

    def test_memory_ignores_existing_disk_index(self):
        with tempfile.TemporaryDirectory() as d:
            disk = GradientStorageManager(d, residency="disk")
            for step in range(3):
                disk.save_bulk([self._batch_record(step, seed=step)])
            # A fresh memory store on the same dir must not adopt the disk index.
            mem = GradientStorageManager(d, residency="memory")
            assert mem.index == {}


class TestDiskFormat:
    """disk_format='memmap' -- flat .bin + .meta for materialized groups, with a
    transparent pickle fallback for anything holding factorized factors.
    """

    @staticmethod
    def _materialized_record(step: int, seed: int) -> GradientRecord:
        torch.manual_seed(seed)
        data = {
            "L0": torch.randn(4, 32),
            "L1": torch.randn(4, 16),
            "ids": torch.arange(4 * 8).reshape(4, 8),  # int64 layer
        }
        gradient = Gradient(
            representation=dict.fromkeys(data, "materialized"),
            data=data,
            layer_types=dict.fromkeys(data, "nn.Linear"),
        )
        return GradientRecord(
            step=step,
            input_hash=[f"h{step}_{i}" for i in range(4)],
            gradient=gradient,
        )

    @staticmethod
    def _factorized_record(step: int, seed: int) -> GradientRecord:
        torch.manual_seed(seed)
        gradient = Gradient(
            representation={"L0": "factorized"},
            data={
                "L0": Factorized(
                    activation=torch.randn(4, 2, 8),
                    pre_activation_grad=torch.randn(4, 2, 8),
                ),
            },
            layer_types={"L0": "nn.Linear"},
            validate_on_init=False,
        )
        return GradientRecord(
            step=step,
            input_hash=[f"h{step}_{i}" for i in range(4)],
            gradient=gradient,
        )

    def _collect(self, disk_format, tmpdir, record_fn):
        fm = GradientStorageManager(tmpdir, disk_format=disk_format)
        for step in range(5):
            fm.save_bulk([record_fn(step, seed=step)])
        return fm

    def test_rejects_unknown_disk_format(self):
        with (
            tempfile.TemporaryDirectory() as d,
            pytest.raises(ValueError, match="disk_format must be one of"),
        ):
            GradientStorageManager(d, disk_format="parquet")

    def test_memmap_writes_bin_and_meta_no_pt(self):
        with tempfile.TemporaryDirectory() as d:
            self._collect("memmap", d, self._materialized_record)
            assert len(list(Path(d).glob("*.mmap.bin"))) == 5
            assert len(list(Path(d).glob("*.mmap.meta"))) == 5
            assert list(Path(d).glob("*.pt")) == []

    def test_memmap_matches_pickle(self):
        with tempfile.TemporaryDirectory() as dp, tempfile.TemporaryDirectory() as dm:
            self._collect("pickle", dp, self._materialized_record)
            self._collect("memmap", dm, self._materialized_record)
            # A plain reader (default pickle format) must read .mmap handles by
            # suffix -- the on-disk store is self-describing.
            pk, mm = GradientStorageManager(dp), GradientStorageManager(dm)
            for (fp, _sp), (fmm, _sm) in zip(
                pk.iter_steps(list(range(5))),
                mm.iter_steps(list(range(5))),
                strict=True,
            ):
                rp, rm = pk.load_records(fp), mm.load_records(fmm)
                for a, b in zip(rp, rm, strict=True):
                    assert a.step == b.step
                    assert a.input_hash == b.input_hash
                    for layer in a.gradient.data:
                        ta, tb = a.gradient.data[layer], b.gradient.data[layer]
                        assert ta.dtype == tb.dtype  # dtype preserved (incl int64)
                        assert torch.equal(ta, tb)

    def test_memmap_per_sample_slicing(self):
        """The by-hash index resolves a single sample through the mmap loader."""
        with tempfile.TemporaryDirectory() as dm, tempfile.TemporaryDirectory() as dp:
            mm = self._collect("memmap", dm, self._materialized_record)
            pk = self._collect("pickle", dp, self._materialized_record)
            # Sample "h3_2" is row 2 of the step-3 block.
            entry_mm = mm.index["h3_2"][0]
            entry_pk = pk.index["h3_2"][0]
            rec_mm = mm._load_entry(entry_mm)
            rec_pk = pk._load_entry(entry_pk)
            assert torch.equal(
                rec_mm.gradient.data["L0"],
                rec_pk.gradient.data["L0"],
            )

    def test_memmap_reconstructed_tensors_are_writable(self):
        with tempfile.TemporaryDirectory() as d:
            mm = self._collect("memmap", d, self._materialized_record)
            loc = next(iter(mm.iter_steps([0])))[0]
            rec = mm.load_records(loc)[0]
            # Copy-on-write mapping -> writable without touching the file.
            rec.gradient.data["L0"].add_(1.0)  # must not raise

    def test_memmap_falls_back_to_pickle_for_factorized(self):
        with tempfile.TemporaryDirectory() as dm, tempfile.TemporaryDirectory() as dp:
            fm = self._collect("memmap", dm, self._factorized_record)
            # Factorized groups cannot be memmapped -> .pt, no .mmap.
            assert len(list(Path(dm).glob("*.pt"))) == 5
            assert list(Path(dm).glob("*.mmap.bin")) == []
            self._collect("pickle", dp, self._factorized_record)
            mm, pk = GradientStorageManager(dm), GradientStorageManager(dp)
            for (fmm, _), (fp, _) in zip(
                mm.iter_steps(list(range(5))),
                pk.iter_steps(list(range(5))),
                strict=True,
            ):
                for a, b in zip(mm.load_records(fmm), pk.load_records(fp), strict=True):
                    fa = a.gradient.data["L0"]
                    fb = b.gradient.data["L0"]
                    assert torch.equal(fa.activation, fb.activation)
                    assert torch.equal(fa.pre_activation_grad, fb.pre_activation_grad)
            assert fm.disk_format == "memmap"

    def test_memmap_mixed_group_falls_back(self):
        """A group mixing a materialized and a factorized layer -> pickle."""
        with tempfile.TemporaryDirectory() as d:
            fm = GradientStorageManager(d, disk_format="memmap")
            grad = Gradient(
                representation={"mat": "materialized", "fac": "factorized"},
                data={
                    "mat": torch.randn(4, 8),
                    "fac": Factorized(
                        activation=torch.randn(4, 2, 8),
                        pre_activation_grad=torch.randn(4, 2, 8),
                    ),
                },
                layer_types={"mat": "nn.Linear", "fac": "nn.Linear"},
                validate_on_init=False,
            )
            record = GradientRecord(
                step=0,
                input_hash=["a", "b", "c", "d"],
                gradient=grad,
            )
            fm.save_bulk([record])
            assert len(list(Path(d).glob("*.pt"))) == 1
            assert list(Path(d).glob("*.mmap.bin")) == []

    def test_memmap_reopen_does_not_overwrite_existing_groups(self):
        """Reopening a memmap store resumes the counter instead of restarting.

        The auto-name counter has to see ``batch_<id>.mmap.bin``, not just
        ``batch_<id>.pt``; otherwise a second collection into the same dir
        rewrites ``batch_000000.mmap`` while the index still points at it, and
        the first run's samples silently read back as the second run's data.
        """
        with tempfile.TemporaryDirectory() as d:
            first = GradientStorageManager(d, disk_format="memmap")
            first.save_bulk([self._materialized_record(0, seed=0)])
            expected = self._materialized_record(0, seed=0).gradient.data["L0"]

            second = GradientStorageManager(d, disk_format="memmap")
            second.save_bulk([self._materialized_record(1, seed=99)])

            assert sorted(p.name for p in Path(d).glob("*.mmap.bin")) == [
                "batch_000000.mmap.bin",
                "batch_000001.mmap.bin",
            ]
            # Run 1's sample still resolves to run 1's gradient.
            reader = GradientStorageManager(d, disk_format="memmap")
            got = reader.load_sample_by_hash("h0_0", 0, 0)
            assert torch.equal(got.data["L0"], expected[0:1])

    def test_memmap_reopen_keeps_every_step_loadable(self):
        """Two collections into one memmap dir keep both runs' steps intact."""
        with tempfile.TemporaryDirectory() as d:
            for run in range(2):
                fm = GradientStorageManager(d, disk_format="memmap")
                for step in range(3):
                    fm.save_bulk(
                        [self._materialized_record(run * 3 + step, seed=step)],
                    )
            reader = GradientStorageManager(d, disk_format="memmap")
            assert reader.available_steps() == [0, 1, 2, 3, 4, 5]
            assert len(list(Path(d).glob("*.mmap.bin"))) == 6


class TestKroneckerCovarianceCallback:
    """The inline covariance callback must reproduce the block-level
    KroneckerAccumulator's ``(A, G)`` -- i.e. fitting during capture from the
    raw per-layer factors matches fitting from the assembled factorized blocks.
    """

    @staticmethod
    def _reference(records):
        from dattri_llm.gradient.ops import KroneckerAccumulator

        acc = KroneckerAccumulator()
        for r in records:
            layers = [
                n for n, v in r.gradient.data.items() if isinstance(v, Factorized)
            ]
            acc.update(r.gradient, layers)
        return acc.result()

    def _collect(self, model, xs):
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback

        rec = RecordingCallback()
        cov = KroneckerCovarianceCallback()
        hm = HookManager(model, callbacks=[rec, cov])
        with hm.collect():
            for x in xs:
                model(x).pow(2).sum().backward()
                model.zero_grad(set_to_none=True)
        return rec, cov

    def test_matches_block_accumulator_single_step(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
        rec, cov = self._collect(model, [torch.randn(5, 6)])
        ref, got = self._reference(rec.records), cov.result()
        assert set(got) == set(ref)
        for layer in ref:
            a_ref, g_ref = ref[layer]
            a_got, g_got = got[layer]
            assert torch.allclose(a_got, a_ref, atol=1e-5), f"A {layer}"
            assert torch.allclose(g_got, g_ref, atol=1e-5), f"G {layer}"

    def test_matches_block_accumulator_multi_step(self):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 5, bias=False), nn.ReLU(), nn.Linear(5, 3))
        xs = [torch.randn(3, 4) for _ in range(4)]
        rec, cov = self._collect(model, xs)
        ref, got = self._reference(rec.records), cov.result()
        assert set(got) == set(ref)
        for layer in ref:
            a_ref, g_ref = ref[layer]
            a_got, g_got = got[layer]
            assert torch.allclose(a_got, a_ref, atol=1e-5), f"A {layer}"
            assert torch.allclose(g_got, g_ref, atol=1e-5), f"G {layer}"

    def test_reset_clears_state(self):
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback

        torch.manual_seed(2)
        model = nn.Sequential(nn.Linear(4, 4))
        cov = KroneckerCovarianceCallback()
        hm = HookManager(model, callbacks=[cov])
        with hm.collect():
            model(torch.randn(3, 4)).pow(2).sum().backward()
        assert cov.result()
        cov.reset()
        assert cov.result() == {}


class TestProjectedCovarianceCallback:
    """Under a LoGRA projection, the covariance callback receives the *projected*
    factors (a-side projected in the forward hook), so it fits compact
    ``(proj_dim, proj_dim)`` covariances -- the logix-style factors that match a
    ``logra_materialized`` compact store.
    """

    PROJ = 4

    def _config(self, style):
        from dattri_llm.gradient.hooks import REGISTER_ALL, HookManagerConfig

        return HookManagerConfig(
            linear_io=REGISTER_ALL,
            projection={
                "__default__": {
                    "style": style,
                    "proj_dim": self.PROJ,
                    "proj_max_batch_size": 32,
                    "proj_type": "rademacher",
                    "proj_seed": 0,
                },
            },
        )

    def _collect(self, style):
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 8))
        rec = RecordingCallback()
        cov = KroneckerCovarianceCallback()
        hm = HookManager(model, config=self._config(style), callbacks=[rec, cov])
        with hm.collect():
            model(torch.randn(5, 16)).pow(2).sum().backward()
        return rec, cov

    def test_covariances_are_compact_and_match_blocks(self):
        from dattri_llm.gradient.ops import KroneckerAccumulator

        # logra_factorized keeps the projected factors in the block, so the block
        # KroneckerAccumulator is the reference for the compact covariance.
        rec, cov = self._collect("logra_factorized")
        got = cov.result()
        assert got, "no covariances collected"
        for layer, (a_cov, g_cov) in got.items():
            assert a_cov.shape == (self.PROJ, self.PROJ), f"{layer} A {a_cov.shape}"
            assert g_cov.shape == (self.PROJ, self.PROJ), f"{layer} G {g_cov.shape}"

        acc = KroneckerAccumulator()
        for r in rec.records:
            layers = [
                n for n, v in r.gradient.data.items() if isinstance(v, Factorized)
            ]
            acc.update(r.gradient, layers)
        ref = acc.result()
        assert set(got) == set(ref)
        for layer in ref:
            a_ref, g_ref = ref[layer]
            a_got, g_got = got[layer]
            assert torch.allclose(a_got, a_ref, atol=1e-5), f"A {layer}"
            assert torch.allclose(g_got, g_ref, atol=1e-5), f"G {layer}"

    def test_compact_store_still_gets_projected_covariance(self):
        # logra_materialized stores a compact (B, k*k) block (no factors), yet the
        # callback still fits the compact (k, k) covariances from the projected
        # factors emitted at capture -- the whole point of the logix-style path.
        _rec, cov = self._collect("logra_materialized")
        got = cov.result()
        assert got
        for layer, (a_cov, g_cov) in got.items():
            assert a_cov.shape == (self.PROJ, self.PROJ), f"{layer} A {a_cov.shape}"
            assert g_cov.shape == (self.PROJ, self.PROJ), f"{layer} G {g_cov.shape}"
