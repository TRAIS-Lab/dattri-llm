"""Unit tests for dattri_llm gradient collection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dattri_llm.gradient.hooks import (
    REGISTER_ALL,
    HookManager,
    HookManagerCallback,
    HookManagerConfig,
    OffloadCallback,
)
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient, GradientRecord
from dattri_llm.gradient.utils import hash_sample


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
        inputs = {"input_ids": torch.randint(0, 10, (3, 8))}
        h1 = hash_sample(inputs, 0)
        h2 = hash_sample(inputs, 0)
        assert h1 == h2

    def test_different_samples_different_hash(self):
        inputs = {"input_ids": torch.randint(0, 10, (3, 8))}
        hashes = [hash_sample(inputs, i) for i in range(3)]
        # All three samples should be distinct (with overwhelming probability for random data)
        assert len(set(hashes)) == 3

    def test_64_char_hex(self):
        inputs = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
        h = hash_sample(inputs, 0)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash_across_batches(self):
        row = torch.randint(0, 100, (8,))
        batch_a = {"input_ids": torch.stack([row, torch.zeros(8, dtype=torch.long)])}
        batch_b = {"input_ids": torch.stack([torch.zeros(8, dtype=torch.long), row])}
        # Same content row → same hash regardless of batch position
        assert hash_sample(batch_a, 0) == hash_sample(batch_b, 1)

    def test_multiple_fields_used(self):
        ids = torch.zeros(2, 4, dtype=torch.long)
        mask_a = {"input_ids": ids, "attention_mask": torch.ones(2, 4)}
        mask_b = {"input_ids": ids, "attention_mask": torch.zeros(2, 4)}
        # Same input_ids but different attention_mask → different hash
        assert hash_sample(mask_a, 0) != hash_sample(mask_b, 0)

    def test_non_tensor_values_skipped(self):
        inputs = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "some_flag": True,  # non-tensor, should be ignored
        }
        # Should not raise
        h = hash_sample(inputs, 0)
        assert len(h) == 64


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
# HookManager — init and introspection                                   #
# --------------------------------------------------------------------------- #


class TestHookManagerInit:
    def test_layer_names_populated(self, tiny_model):
        collector = HookManager(tiny_model)
        # Default config hooks every linear-family layer regardless of name:
        # embedding, attn_proj, mlp.0, mlp.2, lm_head (mlp.1 is ReLU).
        assert set(collector.layer_names) == {
            "embedding", "attn_proj", "mlp.0", "mlp.2", "lm_head",
        }
        collector.remove()

    def test_custom_patterns(self, tiny_model):
        collector = HookManager(
            tiny_model, config=HookManagerConfig(linear_io=[r"mlp\.0"])
        )
        assert collector.layer_names == ["mlp.0"]
        collector.remove()

    def test_steps_collected_initial(self, tiny_model):
        collector = HookManager(tiny_model)
        assert collector.steps_collected == 0
        collector.remove()


# --------------------------------------------------------------------------- #
# HookManager — collect() context manager                                #
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
# HookManager — record content                                           #
# --------------------------------------------------------------------------- #


class TestRecordContent:
    def test_record_type(self, tiny_model, tiny_batch):
        cb = RecordingCallback()
        _run_one_step(tiny_model, tiny_batch["input_ids"], callbacks=[cb])
        for rec in cb.records:
            assert isinstance(rec, GradientRecord)
            assert isinstance(rec.step, int)
            assert isinstance(rec.input_hash, list)   # list of B hashes
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
# HookManager — sample hashing                                           #
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
        assert cb.records[0].input_hash == cb.records[1].input_hash  # same inputs → same hashes
        assert cb.records[0].step == 0
        assert cb.records[1].step == 1
        collector.remove()


# --------------------------------------------------------------------------- #
# OffloadCallback                                                              #
# --------------------------------------------------------------------------- #


class TestGradientFileManager:
    # A valid 64-char SHA-256 hex string for use in deterministic tests.
    _HASH_A = "abcdef01" * 8   # 64 chars
    _HASH_B = "12345678" * 8   # 64 chars

    def _make_record(self, step: int, input_hash: str, tiny_model, tiny_batch) -> GradientRecord:
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        gradient = cb.records[0].gradient
        return GradientRecord(step=step, input_hash=input_hash, gradient=gradient)

    def test_save_creates_file(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            path = manager.save(rec)
            assert path.exists()

    def test_save_updates_index(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(7, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            assert self._HASH_A in manager.index
            entries = manager.index[self._HASH_A]
            assert len(entries) == 1
            assert entries[0]["step"] == 7
            assert entries[0]["idx"] == 0

    def test_index_json_written_after_save(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            assert (Path(tmpdir) / "index.json").exists()

    def test_load_by_hash_round_trip(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(3, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            loaded = manager.load_by_hash(3, self._HASH_A)
            assert isinstance(loaded, GradientRecord)
            assert loaded.step == 3
            assert loaded.input_hash == self._HASH_A

    def test_load_by_hash_wrong_step_raises(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            with pytest.raises(KeyError):
                manager.load_by_hash(99, self._HASH_A)

    def test_load_by_inputs(self, tiny_model, tiny_batch):
        """load() accepts model inputs directly and hashes them internally."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=100, file_manager=manager,
                                      recording_type="per_sample")
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            loaded = manager.load(0, inputs, sample_idx=0)
            assert isinstance(loaded, GradientRecord)
            assert loaded.step == 0
            assert loaded.gradient.batch_size == 1
            collector.remove()

    def test_load_inputs_equivalent_to_by_hash(self, tiny_model, tiny_batch):
        """load() and load_by_hash() return the same record."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=100, file_manager=manager,
                                      recording_type="per_sample")
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            h = hash_sample(inputs, sample_idx=0)
            by_hash = manager.load_by_hash(0, h)
            by_inputs = manager.load(0, inputs, sample_idx=0)
            assert by_hash.input_hash == by_inputs.input_hash
            collector.remove()

    def test_load_all_by_hash(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            for step in [0, 2, 5]:
                rec = self._make_record(step, self._HASH_A, tiny_model, tiny_batch)
                manager.save(rec)
            records = manager.load_all_by_hash(self._HASH_A)
            assert [r.step for r in records] == [0, 2, 5]

    def test_load_all_by_hash_unknown_raises(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            with pytest.raises(KeyError):
                manager.load_all_by_hash(self._HASH_B)

    def test_load_all_by_inputs(self, tiny_model, tiny_batch):
        """load_all() loads all records for all samples in the batch.

        Uses per_sample recording so each sample gets its own record.
        """
        inputs = {"input_ids": tiny_batch["input_ids"]}
        B = tiny_batch["input_ids"].shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=100, file_manager=manager,
                                      recording_type="per_sample")
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            # load_all returns all unique records for all samples in the batch
            records = manager.load_all(inputs)
            assert len(records) == B
            collector.remove()

    def test_load_all_covers_all_samples(self, tiny_model, tiny_batch):
        """load_all() returns one record per unique sample in the batch (per_sample mode)."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        B = tiny_batch["input_ids"].shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=100, file_manager=manager,
                                      recording_type="per_sample")
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            records = manager.load_all(inputs)
            input_hashes = {hash_sample(inputs, i) for i in range(B)}
            record_hashes = {r.input_hash for r in records}  # str per per_sample record
            assert record_hashes == input_hashes
            collector.remove()

    def test_index_loaded_from_disk_on_construction(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = GradientFileManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager1.save(rec)

            manager2 = GradientFileManager(tmpdir)
            assert self._HASH_A in manager2.index
            assert manager2.index[self._HASH_A][0]["step"] == 0

    def test_duplicate_step_not_duplicated_in_index(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
            manager.save(rec)
            manager.save(rec)  # same step saved again
            steps = [e["step"] for e in manager.index[self._HASH_A]]
            assert steps.count(0) == 1


class TestBatchSaving:
    """Tests for GradientFileManager.save_batch and OffloadCallback(batch_save=True)."""

    _HASH_A = "abcdef01" * 8
    _HASH_B = "12345678" * 8

    def _make_record(self, step, input_hash, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        return GradientRecord(step=step, input_hash=input_hash, gradient=cb.records[0].gradient)

    def test_save_batch_creates_one_file(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            path = manager.save_batch(recs)
            assert path.exists()
            assert path.name.startswith("batch_")
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1

    def test_save_batch_indexes_all_hashes(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            manager.save_batch(recs)
            assert self._HASH_A in manager.index
            assert self._HASH_B in manager.index
            # Both point to the same batch file but different idx
            assert manager.index[self._HASH_A][0]["idx"] == 0
            assert manager.index[self._HASH_B][0]["idx"] == 1

    def test_save_batch_load_round_trip(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            recs = [
                self._make_record(0, self._HASH_A, tiny_model, tiny_batch),
                self._make_record(0, self._HASH_B, tiny_model, tiny_batch),
            ]
            manager.save_batch(recs)
            loaded_a = manager.load_by_hash(0, self._HASH_A)
            loaded_b = manager.load_by_hash(0, self._HASH_B)
            assert isinstance(loaded_a, GradientRecord)
            assert isinstance(loaded_b, GradientRecord)
            assert loaded_a.input_hash == self._HASH_A
            assert loaded_b.input_hash == self._HASH_B

    def test_next_batch_id_continues_after_reload(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = GradientFileManager(tmpdir)
            recs = [self._make_record(0, self._HASH_A, tiny_model, tiny_batch)]
            manager1.save_batch(recs)  # writes batch_000000.pt

            manager2 = GradientFileManager(tmpdir)
            recs2 = [self._make_record(1, self._HASH_B, tiny_model, tiny_batch)]
            manager2.save_batch(recs2)  # should write batch_000001.pt

            batch_files = sorted(Path(tmpdir).glob("batch_*.pt"))
            assert len(batch_files) == 2
            assert batch_files[1].name == "batch_000001.pt"

    def test_per_batch_input_hash_indexed(self, tiny_model, tiny_batch):
        """Per-batch records carry input_hash as a list; the file manager indexes all of them."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
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
        """One batch step → one batch file (offload_interval=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=1, file_manager=manager)
            collector = HookManager(tiny_model, callbacks=[offload])
            B = tiny_batch["input_ids"].shape[0]
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            # B per-sample records in one batch file
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1
            assert len(list(Path(tmpdir).glob("step_*.pt"))) == 0
            collector.remove()

    def test_offload_load_by_hash_after_batch_write(self, tiny_model, tiny_batch):
        """Records written by OffloadCallback(per_sample) can be loaded by sample hash."""
        inputs = {"input_ids": tiny_batch["input_ids"]}
        B = tiny_batch["input_ids"].shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GradientFileManager(tmpdir)
            offload = OffloadCallback(offload_interval=1, file_manager=manager,
                                      recording_type="per_sample")
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(**inputs).mean().backward()
            records = manager.load_all(inputs)
            assert len(records) == B
            assert all(isinstance(r, GradientRecord) for r in records)
            collector.remove()


class TestOffloadCallback:
    def _make_offload(self, tmpdir, offload_interval=100):
        manager = GradientFileManager(tmpdir)
        offload = OffloadCallback(offload_interval=offload_interval, file_manager=manager)
        return manager, offload

    def test_files_written_after_context(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            # OffloadCallback always writes batch files
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 1
            collector.remove()

    def test_index_json_written(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
            assert (Path(tmpdir) / "index.json").exists()
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
            for h, entries in manager.index.items():
                for entry in entries:
                    s = entry["step"]
                    rec = manager.load_by_hash(s, h)
                    assert isinstance(rec, GradientRecord)
                    assert rec.step == s
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
        # offload_interval=2: flush every 2 batch steps → 4 steps / 2 = 2 batch files
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir, offload_interval=2)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                for _ in range(4):
                    tiny_model(tiny_batch["input_ids"]).mean().backward()
            assert len(list(Path(tmpdir).glob("batch_*.pt"))) == 2
            collector.remove()

    def test_staged_property(self, tiny_model, tiny_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, offload = self._make_offload(tmpdir)
            collector = HookManager(tiny_model, callbacks=[offload])
            with collector.collect():
                tiny_model(tiny_batch["input_ids"]).mean().backward()
                assert len(offload.staged) == 1  # one per-batch record per step
            assert len(offload.staged) == 0
            collector.remove()

    def test_manager_accessible_for_queries(self, tiny_model, tiny_batch):
        """file_manager is the query handle — offload only owns saving."""
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
# HookManager — remove                                                   #
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
        # After remove, hooks are gone — no records should be emitted
        # (collect() itself would raise an AttributeError since buffers cleared,
        # but running without collect() also should not emit records)
        logits = tiny_model(tiny_batch["input_ids"])
        logits.mean().backward()
        assert len(cb.records) == 0


# --------------------------------------------------------------------------- #
# HookManagerConfig                                                            #
# --------------------------------------------------------------------------- #


class TestHookManagerConfig:
    # ── default ─────────────────────────────────────────────────────────────

    def test_default_selectors(self):
        cfg = HookManagerConfig()
        assert cfg.hook_types == {}
        assert cfg.linear_io is None
        assert cfg.param_grad is None
        assert cfg.is_default

    # ── hook_types assignment (the basic control) ─────────────────────────────

    def test_hook_types_assignment(self):
        cfg = HookManagerConfig(
            hook_types={"mlp.0": "linear_io", "lm_head": "param_grad"}
        )
        assert cfg.hook_types == {"mlp.0": "linear_io", "lm_head": "param_grad"}
        assert not cfg.is_default

    def test_hook_types_invalid_value_raises(self):
        with pytest.raises(ValueError, match="not a valid hook type"):
            HookManagerConfig(hook_types={"mlp.0": "bogus"})

    def test_hook_types_wrong_type_raises(self):
        with pytest.raises(TypeError, match="hook_types must be a dict"):
            HookManagerConfig(hook_types=["mlp.0"])  # type: ignore[arg-type]

    # ── REGISTER_ALL ──────────────────────────────────────────────────────────

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

    # ── pattern lists ─────────────────────────────────────────────────────────

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

    # ── validation ────────────────────────────────────────────────────────────

    def test_invalid_selector_type_raises(self):
        with pytest.raises(TypeError, match="must be None, REGISTER_ALL"):
            HookManagerConfig(linear_io="mlp")  # type: ignore[arg-type]

    def test_non_string_pattern_raises(self):
        with pytest.raises(TypeError, match="regex strings"):
            HookManagerConfig(param_grad=[123])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# HookManager — param_grad hook type                                          #
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
            n for n, t in (g.layer_types or {}).items() if t == "param_grad"
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
            if (g.layer_types or {}).get(name) == "param_grad":
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
        non_param_types = types - {"param_grad"}
        assert len(non_param_types) > 0, (
            "Expected at least one canonical layer type (linear_io layers) in types"
        )
        assert "param_grad" in types
        collector.remove()

    def test_per_batch_linear_io_only_emits_one_record_per_step(self, tiny_model, tiny_batch):
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
            n for n, t in (g.layer_types or {}).items() if t == "param_grad"
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
# (simulated: two GradientFileManagers each given a fake rank via monkeypatch) #
# --------------------------------------------------------------------------- #


class TestGradientFileManagerDDP:
    """Simulate two DDP ranks writing to the same root save_dir."""

    _HASH_A = "aaaaaaaa" * 8  # 64 chars
    _HASH_B = "bbbbbbbb" * 8

    def _make_record(self, step, input_hash, tiny_model, tiny_batch):
        cb = RecordingCallback()
        collector = HookManager(tiny_model, callbacks=[cb])
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        collector.remove()
        return GradientRecord(step=step, input_hash=input_hash, gradient=cb.records[0].gradient)

    def _manager_for_rank(self, tmpdir, rank, monkeypatch):
        """Return a GradientFileManager that behaves as if running on *rank*."""
        import dattri_llm.gradient.file_manager as fm_module
        monkeypatch.setattr(fm_module, "_dist_rank", lambda: rank)
        return GradientFileManager(tmpdir)

    def test_each_rank_writes_to_own_subdir(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        import dattri_llm.gradient.file_manager as fm_module

        monkeypatch.setattr(fm_module, "_dist_rank", lambda: 0)
        m0 = GradientFileManager(str(tmp_path))
        rec0 = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m0.save_batch([rec0])

        monkeypatch.setattr(fm_module, "_dist_rank", lambda: 1)
        m1 = GradientFileManager(str(tmp_path))
        rec1 = self._make_record(0, self._HASH_B, tiny_model, tiny_batch)
        m1.save_batch([rec1])

        assert (tmp_path / "rank_0").is_dir()
        assert (tmp_path / "rank_1").is_dir()
        assert len(list((tmp_path / "rank_0").glob("batch_*.pt"))) == 1
        assert len(list((tmp_path / "rank_1").glob("batch_*.pt"))) == 1

    def test_no_batch_id_collision(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """Both ranks start _next_batch_id at 0 but write to different subdirs."""
        import dattri_llm.gradient.file_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "_dist_rank", lambda r=rank: r)
            m = GradientFileManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_batch([rec])

        # Both write batch_000000.pt but in separate dirs — no data loss.
        assert (tmp_path / "rank_0" / "batch_000000.pt").exists()
        assert (tmp_path / "rank_1" / "batch_000000.pt").exists()

    def test_each_rank_has_own_index_json(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        import dattri_llm.gradient.file_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "_dist_rank", lambda r=rank: r)
            m = GradientFileManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_batch([rec])

        idx0 = json.loads((tmp_path / "rank_0" / "index.json").read_text())
        idx1 = json.loads((tmp_path / "rank_1" / "index.json").read_text())
        # Each rank's index only contains its own hash.
        assert self._HASH_A in idx0 and self._HASH_B not in idx0
        assert self._HASH_B in idx1 and self._HASH_A not in idx1

    def test_index_entries_use_rank_relative_paths(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        import dattri_llm.gradient.file_manager as fm_module

        monkeypatch.setattr(fm_module, "_dist_rank", lambda: 0)
        m = GradientFileManager(str(tmp_path))
        rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m.save_batch([rec])

        entry = m.index[self._HASH_A][0]
        assert entry["file"].startswith("rank_0/")

    def test_fresh_manager_merges_all_ranks(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """A reader opened after training sees gradients from every rank."""
        import dattri_llm.gradient.file_manager as fm_module

        for rank, h in [(0, self._HASH_A), (1, self._HASH_B)]:
            monkeypatch.setattr(fm_module, "_dist_rank", lambda r=rank: r)
            m = GradientFileManager(str(tmp_path))
            rec = self._make_record(0, h, tiny_model, tiny_batch)
            m.save_batch([rec])

        # Reader: no distributed context.
        monkeypatch.setattr(fm_module, "_dist_rank", lambda: None)
        reader = GradientFileManager(str(tmp_path))
        assert self._HASH_A in reader.index
        assert self._HASH_B in reader.index

    def test_load_across_ranks(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """A reader opened after training can load records from any rank."""
        import dattri_llm.gradient.file_manager as fm_module

        # Simulate training: rank 0 and rank 1 each write their own records.
        monkeypatch.setattr(fm_module, "_dist_rank", lambda: 0)
        m0 = GradientFileManager(str(tmp_path))
        m0.save_batch([self._make_record(0, self._HASH_A, tiny_model, tiny_batch)])

        monkeypatch.setattr(fm_module, "_dist_rank", lambda: 1)
        m1 = GradientFileManager(str(tmp_path))
        m1.save_batch([self._make_record(0, self._HASH_B, tiny_model, tiny_batch)])

        # Simulate post-training analysis: fresh reader with no distributed context.
        monkeypatch.setattr(fm_module, "_dist_rank", lambda: None)
        reader = GradientFileManager(str(tmp_path))
        loaded_a = reader.load_by_hash(0, self._HASH_A)
        loaded_b = reader.load_by_hash(0, self._HASH_B)
        assert loaded_a.input_hash == self._HASH_A
        assert loaded_b.input_hash == self._HASH_B

    def test_non_distributed_still_writes_to_root(self, tmp_path, monkeypatch, tiny_model, tiny_batch):
        """Single-GPU path: no rank_N/ subdirectory created."""
        import dattri_llm.gradient.file_manager as fm_module

        monkeypatch.setattr(fm_module, "_dist_rank", lambda: None)
        m = GradientFileManager(str(tmp_path))
        rec = self._make_record(0, self._HASH_A, tiny_model, tiny_batch)
        m.save_batch([rec])

        assert not any(tmp_path.glob("rank_*"))
        assert (tmp_path / "batch_000000.pt").exists()
        assert (tmp_path / "index.json").exists()

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
