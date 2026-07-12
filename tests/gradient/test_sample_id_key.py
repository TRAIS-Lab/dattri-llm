"""Tests for ``sample_id_key``: user-designated sample identifiers.

By default a sample's identity is the SHA-256 content hash of its model
inputs.  ``HookManager(sample_id_key=...)`` routes to the alternative scheme:
identifiers are read directly from a designated input field (a kwarg name, or
a positional index for tuple inputs) and stringified -- an ``idx`` column of
``[32, 42]`` yields ``input_hash == ["32", "42"]``.  The chosen key is stamped
on every record and on the on-disk store, so readers can disambiguate the
scheme at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks import CaptureCallback, OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.utils.hashing import hash_batch

B, IN_DIM, OUT_DIM = 4, 6, 3


class KwargModel(nn.Module):
    """Accepts (and ignores) an ``idx`` kwarg carrying the sample ids."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None):
        return self.fc(x)


class PositionalModel(nn.Module):
    """Takes the ids as a second positional argument (tuple-style inputs)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor, idx: torch.Tensor):
        return self.fc(x)


def _step(model, cb=None, sample_id_key=None, **fwd_kwargs):
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[cb] if cb else [],
        sample_id_key=sample_id_key,
    )
    try:
        with hm.collect():
            model.zero_grad()
            model(**fwd_kwargs).pow(2).sum().backward()
    finally:
        hm.remove()


class TestManagerRouting:
    def test_kwarg_field_becomes_identifier(self):
        cb = CaptureCallback()
        idx = torch.tensor([32, 42, 51, 1])
        _step(KwargModel(), cb, "idx", x=torch.randn(B, IN_DIM), idx=idx)
        rec = cb.record
        assert rec.input_hash == ["32", "42", "51", "1"]
        assert rec.sample_id_key == "idx"

    def test_positional_index_becomes_identifier(self):
        cb = CaptureCallback()
        model = PositionalModel()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
            sample_id_key=1,
        )
        try:
            with hm.collect():
                ids = torch.tensor([7, 8, 9, 10])
                model(torch.randn(B, IN_DIM), ids).sum().backward()
        finally:
            hm.remove()
        assert cb.record.input_hash == ["7", "8", "9", "10"]
        assert cb.record.sample_id_key == 1

    def test_default_is_content_hash(self):
        cb = CaptureCallback()
        x = torch.randn(B, IN_DIM)
        _step(KwargModel(), cb, None, x=x)
        assert cb.record.input_hash == hash_batch({"x": x}, B)
        assert cb.record.sample_id_key is None

    def test_missing_field_raises(self):
        with pytest.raises(KeyError, match="sample_id_key"):
            _step(KwargModel(), None, "idx", x=torch.randn(B, IN_DIM))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="batch size"):
            _step(
                KwargModel(),
                None,
                "idx",
                x=torch.randn(B, IN_DIM),
                idx=torch.tensor([1, 2]),  # 2 ids for a batch of 4
            )


class TestFileManagerRoundTrip:
    def _collect(self, tmp_path) -> tuple[GradientFileManager, torch.Tensor]:
        fm = GradientFileManager(str(tmp_path))
        cb = OffloadCallback(1, fm, recording_type="per_batch")
        x = torch.randn(B, IN_DIM)
        _step(KwargModel(), cb, "idx", x=x, idx=torch.tensor([32, 42, 51, 1]))
        return fm, x

    def test_lookup_by_id_value(self, tmp_path):
        fm, _ = self._collect(tmp_path)
        # int and str forms are interchangeable (str-normalised keys).
        assert fm.lookup_by_hash(42) == [(0, 1)]
        assert fm.lookup_by_hash("42") == [(0, 1)]
        grad = fm.load_sample_by_hash(42, 0, 1)
        assert grad.batch_size == 1

    def test_lookup_by_inputs_rereads_the_column(self, tmp_path):
        fm, x = self._collect(tmp_path)
        assert fm.lookup({"x": x[2], "idx": torch.tensor(51)}) == [(0, 2)]
        assert fm.lookup({"idx": 51}) == [(0, 2)]  # only the id field matters

    def test_lookup_by_inputs_without_column_raises(self, tmp_path):
        fm, x = self._collect(tmp_path)
        with pytest.raises(KeyError, match="sample_id_key"):
            fm.lookup({"x": x[2]})

    def test_scheme_persisted_and_reloaded(self, tmp_path):
        self._collect(tmp_path)
        payload = json.loads((Path(tmp_path) / "index.json").read_text())
        assert payload["sample_id_key"] == "idx"
        fresh = GradientFileManager(str(tmp_path))
        assert fresh.sample_id_key == "idx"
        assert fresh.lookup_by_hash(32) == [(0, 0)]

    def test_mixed_scheme_store_raises(self, tmp_path):
        fm, _x = self._collect(tmp_path)
        cb = OffloadCallback(1, fm, recording_type="per_batch")
        with pytest.raises(ValueError, match="mix identifier schemes"):
            _step(KwargModel(), cb, None, x=torch.randn(B, IN_DIM))

    def test_hash_scheme_store_unchanged(self, tmp_path):
        fm = GradientFileManager(str(tmp_path))
        cb = OffloadCallback(1, fm, recording_type="per_batch")
        x = torch.randn(B, IN_DIM)
        _step(KwargModel(), cb, None, x=x)
        assert fm.sample_id_key is None
        h = hash_batch({"x": x}, B)[0]
        assert fm.lookup_by_hash(h) == [(0, 0)]
        assert fm.lookup({"x": x[0]}) == [(0, 0)]  # content-hash path intact
