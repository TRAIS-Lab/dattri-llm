"""Tests for content hashing of model inputs (``hash_sample`` / ``hash_batch``).

The load-bearing property is capture/lookup agreement: the capture side hashes
tensor rows sliced from the collated batch, while the lookup side hashes
whatever ``dataset[i]`` yields -- often plain Python lists (the Hugging Face
default).  The two must agree, broadcast fields must not poison the batch
hash, and a dict with nothing identifying the sample must fail loudly instead
of silently giving every sample the same identity.
"""

from __future__ import annotations

import pytest
import torch

from dattri_llm.utils.hashing import hash_batch, hash_sample

B, T = 3, 5


def _batch() -> dict:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50, (B, T), generator=g)
    return {"input_ids": ids, "labels": ids.clone()}


class TestHashSample:
    def test_list_matches_tensor_row(self):
        """dataset[i] yielding plain lists must hash like the tensor row."""
        batch = _batch()
        row = {k: v[0] for k, v in batch.items()}
        as_lists = {k: v[0].tolist() for k, v in batch.items()}
        assert hash_sample(as_lists) == hash_sample(row)

    def test_scalar_matches_zero_dim_tensor(self):
        assert hash_sample({"y": 7}) == hash_sample({"y": torch.tensor(7)})

    def test_non_digestible_fields_are_skipped(self):
        """A text column alongside the token ids does not change the hash."""
        row = {k: v[0] for k, v in _batch().items()}
        with_text = dict(row, text="some raw string")
        assert hash_sample(with_text) == hash_sample(row)

    def test_nothing_digestible_raises(self):
        """Regression: an all-skipped dict hashed to the empty digest, giving
        every sample the same identity.
        """
        with pytest.raises(ValueError, match="no digestible field"):
            hash_sample({"text": "hello", "meta": None})
        with pytest.raises(ValueError, match="no digestible field"):
            hash_sample({})

    def test_shape_invariance(self):
        """(T,), (1, T), and batch[i] forms hash identically (raw bytes)."""
        row = torch.arange(T)
        assert hash_sample({"x": row}) == hash_sample({"x": row.unsqueeze(0)[0]})


class TestHashBatch:
    def test_matches_per_sample_slices(self):
        batch = _batch()
        hashes = hash_batch(batch)
        assert len(hashes) == B
        for i in range(B):
            assert hashes[i] == hash_sample({k: v[i] for k, v in batch.items()})

    def test_broadcast_fields_are_skipped(self):
        """Regression: a broadcast kwarg (e.g. position_ids of shape (1, T))
        raised NotImplementedError from inside the capture hooks, crashing
        training.  It carries no per-sample identity and is now skipped.
        """
        batch = _batch()
        with_pos = dict(batch, position_ids=torch.arange(T).unsqueeze(0))
        assert hash_batch(with_pos) == hash_batch(batch)

    def test_batch_size_inferred_as_largest_leading_dim(self):
        """Inference is robust to a broadcast field appearing first."""
        batch = {"position_ids": torch.arange(T).unsqueeze(0), **_batch()}
        assert len(hash_batch(batch)) == B

    def test_list_field_sliced_per_sample(self):
        batch = _batch()
        with_list = dict(batch, weight=[1.5, 2.5, 3.5])
        hashes = hash_batch(with_list)
        expected = [
            hash_sample(
                {
                    "input_ids": batch["input_ids"][i],
                    "labels": batch["labels"][i],
                    "weight": w,
                },
            )
            for i, w in enumerate([1.5, 2.5, 3.5])
        ]
        assert hashes == expected

    def test_only_broadcast_fields_raises(self):
        with pytest.raises(ValueError, match="no batch-first field"):
            hash_batch({"position_ids": torch.arange(T).unsqueeze(0)}, batch_size=B)

    def test_no_tensor_fields_raises(self):
        with pytest.raises(ValueError, match="could not infer a batch size"):
            hash_batch({"text": "hello"})
