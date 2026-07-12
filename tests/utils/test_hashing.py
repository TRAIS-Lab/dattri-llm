"""Tests for content hashing of model inputs (``hash_sample`` / ``hash_batch``).

The load-bearing property is capture/lookup agreement: the capture side hashes
tensor rows sliced from the collated batch, while the lookup side hashes
whatever ``dataset[i]`` yields -- often plain Python lists (the Hugging Face
default).  The two must agree, broadcast fields must not poison the batch
hash, and a dict with nothing identifying the sample must fail loudly instead
of silently giving every sample the same identity.
"""

from __future__ import annotations

import warnings

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
        hashes = hash_batch(batch, B)
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
        assert hash_batch(with_pos, B) == hash_batch(batch, B)

    def test_unambiguous_inference_matches_explicit(self):
        """Fields agreeing on one leading dim infer silently and identically
        to the explicit call (backward-compatible no-arg form).
        """
        batch = _batch()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert hash_batch(batch) == hash_batch(batch, B)

    def test_inference_ignores_broadcast_fields(self):
        """A size-1 leading dim is a broadcast, not a batch-size candidate --
        inference stays silent and correct with position_ids first.
        """
        batch = {"position_ids": torch.arange(T).unsqueeze(0), **_batch()}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(hash_batch(batch)) == B

    def test_ambiguous_leading_dims_warn_and_prefer_most_common(self):
        """Regression: max-leading-dim inference let an auxiliary field like
        cu_seqlens of shape (B+1,) win, excluding input_ids from the hash
        entirely and keying every sample by an offset.  Disagreeing fields
        now warn, and the most common dim (the real batch) is used.
        """
        batch = _batch()
        with_cu = dict(batch, cu_seqlens=torch.arange(B + 1))
        with pytest.warns(UserWarning, match="conflicting leading dimensions"):
            hashes = hash_batch(with_cu)
        assert hashes == hash_batch(batch, B)

    def test_oversized_auxiliary_field_is_skipped(self):
        """A packed-attention cu_seqlens of shape (B+1,) is not per-sample:
        with the true batch size supplied it is skipped silently, and the
        real fields keep identifying the samples.
        """
        batch = _batch()
        with_cu = dict(batch, cu_seqlens=torch.arange(B + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert hash_batch(with_cu, B) == hash_batch(batch, B)

    def test_non_positive_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            hash_batch(_batch(), 0)

    def test_list_field_sliced_per_sample(self):
        batch = _batch()
        with_list = dict(batch, weight=[1.5, 2.5, 3.5])
        hashes = hash_batch(with_list, B)
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
        # Explicit size: nothing left after skipping non-tensor fields.
        with pytest.raises(ValueError, match="no batch-first field"):
            hash_batch({"text": "hello"}, batch_size=B)
        # Inferred: nothing to infer a batch size from in the first place.
        with pytest.raises(ValueError, match="could not infer a batch size"):
            hash_batch({"text": "hello"})
