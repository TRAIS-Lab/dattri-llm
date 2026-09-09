"""Unit tests for the shared attribution machinery in ``attribution/utils.py``:
argument normalization, the collection engine, host-side re-batching, and the
``score_sources`` scoring loop.

Everything runs on CPU with tiny in-memory sources; the disk- and
model-backed end-to-end paths are covered by the attributor test modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dattri_llm.attribution.utils import (
    collect_gradients,
    normalize_layer_names,
    score_sources,
    task_loss_fn,
)
from dattri_llm.gradient.gradient import Factorized, Gradient
from dattri_llm.gradient.streaming import rebatch_blocks
from dattri_llm.utils.cache import TensorCache

B, T, D_IN, D_OUT = 2, 4, 3, 5


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def make_materialized_block(seed: int = 0, b: int = B) -> Gradient:
    g = torch.Generator().manual_seed(seed)
    return Gradient(
        representation={"l1": "materialized"},
        data={"l1": torch.randn(b, D_IN * D_OUT, generator=g)},
        layer_types={"l1": "nn.Linear"},
    )


def make_factorized_block(seed: int = 0, b: int = B) -> Gradient:
    g = torch.Generator().manual_seed(seed)
    return Gradient(
        representation={"l1": "factorized"},
        data={
            "l1": Factorized(
                activation=torch.randn(b, T, D_IN, generator=g),
                pre_activation_grad=torch.randn(b, T, D_OUT, generator=g),
            ),
        },
        layer_types={"l1": "nn.Linear"},
        indexing={"l1": "batch_token"},
    )


def make_stream(n: int = 4, factory=make_materialized_block) -> list:
    return [(i, factory(seed=i), [f"h{i}-{j}" for j in range(B)]) for i in range(n)]


class FakeSource:
    """A minimal in-memory GradientSource."""

    def __init__(self, blocks, args=None, reusable=True):
        self._blocks = blocks
        self.args = args
        self._reusable = reusable
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    @property
    def reusable(self):
        return self._reusable


class FakeStreamer:
    """Mimics a GradientStreamer: context manager + block iterator."""

    def __init__(self, blocks, sample_id_key=None):
        self._blocks = blocks
        self.hook_manager = SimpleNamespace(sample_id_key=sample_id_key)
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False

    def __iter__(self):
        if not self.entered:
            raise RuntimeError("iterated outside the collection context")
        return iter(self._blocks)


class FakeFileManager:
    def __init__(self):
        self.calls: list[list] = []

    def save_bulk(self, records):
        self.calls.append(list(records))


def _flat_rows(g: Gradient) -> torch.Tensor:
    value = g.data["l1"]
    if isinstance(value, Factorized):
        bf = value.as_batch_first()
        # Token-summed outer product, flattened per sample.
        return torch.einsum(
            "bti,bto->boi", bf.activation, bf.pre_activation_grad
        ).flatten(1)
    return value


def _dot_inner_product(train_rep, test_rep, *, dense_cache=None):
    return _flat_rows(train_rep) @ _flat_rows(test_rep).T


# --------------------------------------------------------------------------- #
# Small adapters                                                               #
# --------------------------------------------------------------------------- #


class TestNormalizeLayerNames:
    def test_none_passes_through(self):
        assert normalize_layer_names(None) is None

    def test_str_becomes_list(self):
        assert normalize_layer_names("mlp.fc1") == ["mlp.fc1"]

    def test_list_is_copied(self):
        names = ["a", "b"]
        out = normalize_layer_names(names)
        assert out == names
        assert out is not names


class TestTaskLossFn:
    def test_adapts_functorch_loss_to_streamer_convention(self):
        model = nn.Linear(3, 1, bias=False)
        batch = torch.ones(2, 3)

        def dattri_loss(params, data):
            return torch.func.functional_call(model, params, (data,)).sum()

        loss_fn = task_loss_fn(dattri_loss)
        expected = model(batch).sum()
        assert torch.allclose(loss_fn(model, batch), expected)


# --------------------------------------------------------------------------- #
# collect_gradients                                                            #
# --------------------------------------------------------------------------- #


class TestCollectGradients:
    def test_records_carry_block_identity(self):
        blocks = make_stream(3)
        fm = FakeFileManager()
        streamer = FakeStreamer(blocks, sample_id_key="idx")
        assert collect_gradients(streamer, fm) is fm

        records = [r for call in fm.calls for r in call]
        assert [r.step for r in records] == [0, 1, 2]
        assert [r.input_hash for r in records] == [h for _, _, h in blocks]
        assert all(r.sample_id_key == "idx" for r in records)
        assert records[1].gradient is blocks[1][1]
        assert streamer.entered == 1
        assert streamer.exited == 1

    @pytest.mark.parametrize(
        ("interval", "expected_call_sizes"),
        [(1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (10, [5])],
    )
    def test_offload_interval_batches_saves(self, interval, expected_call_sizes):
        fm = FakeFileManager()
        collect_gradients(
            FakeStreamer(make_stream(5)),
            fm,
            offload_interval=interval,
        )
        assert [len(call) for call in fm.calls] == expected_call_sizes

    def test_on_block_sees_every_block_once(self):
        blocks = make_stream(3)
        seen = []
        collect_gradients(
            FakeStreamer(blocks),
            FakeFileManager(),
            on_block=lambda step, grad, hashes: seen.append((step, hashes)),
        )
        assert seen == [(s, h) for s, _, h in blocks]

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError, match="offload_interval"):
            collect_gradients(FakeStreamer([]), FakeFileManager(), offload_interval=0)


# --------------------------------------------------------------------------- #
# rebatch_blocks                                                               #
# --------------------------------------------------------------------------- #


class TestRebatchBlocks:
    def test_batches_at_boundary_and_remainder(self):
        source = make_stream(5)  # 10 samples in blocks of 2
        out = list(rebatch_blocks(iter(source), batch_size=4))
        assert [g.batch_size for _, g, _ in out] == [4, 4, 2]
        # Row order and metadata follow the source's sample order.
        all_ids = [i for _, _, ids in out for i in ids]
        assert all_ids == [h for _, _, hashes in source for h in hashes]
        all_steps = [s for steps, _, _ in out for s in steps]
        assert all_steps == [s for s, g, _ in source for _ in range(g.batch_size)]

    def test_concatenation_preserves_values(self):
        source = make_stream(3)
        ((_, big, _),) = list(rebatch_blocks(iter(source), batch_size=100))
        expected = torch.cat([g.data["l1"] for _, g, _ in source])
        assert torch.equal(big.data["l1"], expected)

    def test_factorized_blocks_pass_through_unbatched(self):
        source = make_stream(3, factory=make_factorized_block)
        out = list(rebatch_blocks(iter(source), batch_size=100))
        assert len(out) == 3
        assert out[0][1] is source[0][1]  # untouched, not copied

    def test_factorized_forces_flush_and_keeps_order(self):
        source = [
            (0, make_materialized_block(seed=0), ["m0-a", "m0-b"]),
            (1, make_factorized_block(seed=1), ["f1-a", "f1-b"]),
            (2, make_materialized_block(seed=2), ["m2-a", "m2-b"]),
        ]
        out = list(rebatch_blocks(iter(source), batch_size=100))
        all_ids = [i for _, _, ids in out for i in ids]
        assert all_ids == ["m0-a", "m0-b", "f1-a", "f1-b", "m2-a", "m2-b"]

    def test_empty_source(self):
        assert list(rebatch_blocks(iter([]), batch_size=4)) == []

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            list(rebatch_blocks(iter([]), batch_size=0))


# --------------------------------------------------------------------------- #
# score_sources                                                                #
# --------------------------------------------------------------------------- #


class TestScoreSources:
    def _sources(self, n_train=3, n_test=2, reusable=True):
        train = FakeSource(make_stream(n_train))
        test = FakeSource(
            [
                (0, make_materialized_block(seed=100 + i), [f"t{i}-0", f"t{i}-1"])
                for i in range(n_test)
            ],
            reusable=reusable,
        )
        return train, test

    def _oracle(self, train, test):
        tr = torch.cat([_flat_rows(g) for _, g, _ in train._blocks])
        te = torch.cat([_flat_rows(g) for _, g, _ in test._blocks])
        return tr @ te.T

    def test_scores_match_oracle(self):
        train, test = self._sources()
        scores, row_ids, row_steps, test_ids = score_sources(
            train,
            test,
            "cpu",
            inner_product=_dot_inner_product,
        )
        assert torch.allclose(scores, self._oracle(train, test), atol=1e-6)
        assert row_ids == [h for _, _, hashes in train._blocks for h in hashes]
        assert test_ids == [h for _, _, hashes in test._blocks for h in hashes]
        assert row_steps == [s for s, g, _ in train._blocks for _ in range(B)]

    def test_columns_land_by_first_seen_hash(self):
        train = FakeSource([(0, make_materialized_block(seed=0), ["r0", "r1"])])
        t_a, t_b = make_materialized_block(seed=1), make_materialized_block(seed=2)
        # Column order is the order hashes are first seen across test blocks.
        test = FakeSource([(0, t_a, ["a0", "a1"]), (0, t_b, ["b0", "b1"])])
        scores, _, _, test_ids = score_sources(
            train,
            test,
            "cpu",
            inner_product=_dot_inner_product,
        )
        assert test_ids == ["a0", "a1", "b0", "b1"]
        train_g = train._blocks[0][1]
        assert torch.allclose(scores[:, :2], _flat_rows(train_g) @ _flat_rows(t_a).T)
        assert torch.allclose(scores[:, 2:], _flat_rows(train_g) @ _flat_rows(t_b).T)

    @pytest.mark.parametrize("batch_size", [1, 3, 100])
    def test_scores_and_rows_invariant_to_batch_size(self, batch_size):
        train, test = self._sources(n_train=4, n_test=1)
        scores, ids, steps, _ = score_sources(
            train,
            test,
            "cpu",
            inner_product=_dot_inner_product,
            batch_size=batch_size,
        )
        assert ids == [h for _, _, hashes in train._blocks for h in hashes]
        assert steps == [s for s, g, _ in train._blocks for _ in range(g.batch_size)]
        assert torch.allclose(scores, self._oracle(train, test), atol=1e-6)

    def test_mixed_stream_stays_ordered(self):
        train = FakeSource(
            [
                (0, make_materialized_block(seed=0), ["m0-a", "m0-b"]),
                (1, make_factorized_block(seed=1), ["f1-a", "f1-b"]),
                (2, make_materialized_block(seed=2), ["m2-a", "m2-b"]),
            ],
        )
        test = FakeSource(
            [(0, make_materialized_block(seed=99, b=3), ["t0", "t1", "t2"])]
        )
        scores, ids, steps, _ = score_sources(
            train,
            test,
            "cpu",
            inner_product=_dot_inner_product,
            batch_size=100,
        )
        assert ids == ["m0-a", "m0-b", "f1-a", "f1-b", "m2-a", "m2-b"]
        assert steps == [0, 0, 1, 1, 2, 2]
        assert scores.shape == (6, 3)

    def test_empty_train_source(self):
        train, test = self._sources(n_train=0, n_test=1)
        scores, ids, steps, test_ids = score_sources(
            train,
            test,
            "cpu",
            inner_product=_dot_inner_product,
        )
        assert scores.shape == (0, 2)
        assert ids == []
        assert steps == []
        assert len(test_ids) == 2

    def test_loop_over_test_matches_cached(self):
        train_a, test_a = self._sources()
        cached = score_sources(
            train_a,
            test_a,
            "cpu",
            inner_product=_dot_inner_product,
        )
        train_b, test_b = self._sources()
        looped = score_sources(
            train_b,
            test_b,
            "cpu",
            inner_product=_dot_inner_product,
            loop_over_test=True,
        )
        for got, want in zip(looped, cached, strict=True):
            if isinstance(got, torch.Tensor):
                assert torch.allclose(got, want, atol=1e-6)
            else:
                assert got == want
        # Cached mode iterates the test source once; looping re-streams it
        # once per train block (plus the column-discovery pass).
        assert test_a.passes == 1
        assert test_b.passes == 1 + len(train_b._blocks)

    def test_loop_over_test_requires_reusable_source(self):
        train, test = self._sources(reusable=False)
        with pytest.raises(ValueError, match="reusable"):
            score_sources(
                train,
                test,
                "cpu",
                inner_product=_dot_inner_product,
                loop_over_test=True,
            )

    def test_transforms_feed_inner_product(self):
        train, test = self._sources(n_train=1, n_test=1)
        seen = []
        train_marker = make_materialized_block(seed=7)
        test_marker = make_materialized_block(seed=8)

        def inner_product(train_rep, test_rep, *, dense_cache=None):
            seen.append((train_rep, test_rep))
            return torch.zeros(train_rep.batch_size, test_rep.batch_size)

        score_sources(
            train,
            test,
            "cpu",
            inner_product=inner_product,
            transform_train=lambda g: train_marker,
            transform_test=lambda g: test_marker,
        )
        assert seen == [(train_marker, test_marker)]

    def test_dense_test_blocks_are_merged_into_one(self):
        """Dense test reps are stacked into a single block (one GEMM per layer
        per train block), with the column order preserved.
        """
        seen = []

        def inner_product(train_rep, test_rep, *, dense_cache=None):
            seen.append((test_rep.batch_size, dense_cache))
            return _dot_inner_product(train_rep, test_rep)

        train, test = self._sources(n_train=2, n_test=3)
        scores, _, _, test_ids = score_sources(
            train,
            test,
            "cpu",
            inner_product=inner_product,
        )
        assert seen == [(6, None), (6, None)]  # one merged 6-column block
        assert test_ids == [h for _, _, hashes in test._blocks for h in hashes]
        assert torch.allclose(scores, self._oracle(train, test), atol=1e-6)

    def test_dense_cache_is_scoped_to_the_train_block(self):
        """With several (factorized, hence unmergeable) test blocks the loop
        hands ``inner_product`` a cache that lives for one train block only;
        with a single test block there is nothing to reuse and no cache.
        """
        caches: list = []

        def inner_product(train_rep, test_rep, *, dense_cache=None):
            caches.append(dense_cache)
            if dense_cache is not None:
                dense_cache.get_or_compute("l1", lambda: _flat_rows(train_rep))
            return _dot_inner_product(train_rep, test_rep)

        train = FakeSource(make_stream(2))
        test = FakeSource(
            [
                (0, make_factorized_block(seed=100 + i), [f"t{i}-0", f"t{i}-1"])
                for i in range(3)
            ],
        )
        score_sources(train, test, "cpu", inner_product=inner_product)
        assert all(isinstance(c, TensorCache) for c in caches)
        # One cache per train block, shared by that block's three test blocks.
        assert len({id(c) for c in caches}) == 2
        assert caches[0] is caches[1] is caches[2]
        assert caches[3] is not caches[0]

        caches.clear()
        train, test = self._sources(n_train=2, n_test=1)
        score_sources(train, test, "cpu", inner_product=inner_product)
        assert caches == [None, None]
