"""The trajectory attributors' shared options: query-side propagation,
recompute-from-snapshots, and their agreement with the stored, training-side
sweeps.  DVEmb and AdamW-influence are exercised through the same fixtures.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import tests.attribution.test_adamw_influence as TA
from dattri_llm.attribution.algorithm import adamw_influence
from dattri_llm.attribution.algorithm.adamw_influence import AdamWInfluenceAttributor
from dattri_llm.attribution.algorithm.dvemb import DVEmbAttributor
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.gradient.snapshots import TrajectorySnapshots

HOOKS = HookManagerConfig(linear_io=[f"{n}$" for n in TA.LAYERS])


def _same_scores(a, b, atol=1e-5):
    assert a.test_ids == b.test_ids
    order_a = {
        k: i for i, k in enumerate(zip(a.row_train_ids, a.row_steps, strict=True))
    }
    order_b = {
        k: i for i, k in enumerate(zip(b.row_train_ids, b.row_steps, strict=True))
    }
    assert set(order_a) == set(order_b)
    for key, i in order_a.items():
        assert torch.allclose(a.scores[i], b.scores[order_b[key]], atol=atol), key


def _run_adamw(tmp_path, *, recompute, **kw):
    _, task, train, test = TA._setup()
    args = dataclasses.replace(TA._args(tmp_path), recompute_gradients=recompute)
    return AdamWInfluenceAttributor(args, task=task).attribute(
        train, test, hook_config=HOOKS, **kw
    )


def _run_dvemb(tmp_path, *, recompute, **kw):
    _, task, train, test = TA._setup()
    args = dataclasses.replace(TA._args(tmp_path), recompute_gradients=recompute)
    return DVEmbAttributor(args, task=task).attribute(
        train, test, hook_config=HOOKS, learning_rate=args.learning_rate, **kw
    )


class TestAdamWInfluenceSides:
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_query_side_matches_training_side(self, tmp_path, loop_over_test):
        ref = _run_adamw(tmp_path / "w", recompute=False)
        got = _run_adamw(
            tmp_path / "u",
            recompute=False,
            propagation="test",
            loop_over_test=loop_over_test,
        )
        _same_scores(ref, got)

    def test_query_side_chunks_samples(self, tmp_path, monkeypatch):
        ref = _run_adamw(tmp_path / "w", recompute=False)
        # One sample per push: the chunked accumulation must not change anything.
        monkeypatch.setattr(adamw_influence, "_SWEEP_CHUNK_BYTES", 1)
        got = _run_adamw(tmp_path / "u", recompute=False, propagation="test")
        _same_scores(ref, got)

    def test_narrow_block_dtype_is_close(self, tmp_path):
        ref = _run_adamw(tmp_path / "w", recompute=False)
        got = _run_adamw(
            tmp_path / "u",
            recompute=False,
            propagation="test",
            block_dtype=torch.bfloat16,
        )
        _same_scores(ref, got, atol=2e-2 * float(ref.scores.abs().max()))

    def test_training_side_rejects_loop_over_test(self, tmp_path):
        with pytest.raises(ValueError, match="loop_over_test"):
            _run_adamw(tmp_path, recompute=False, loop_over_test=True)


class TestRecompute:
    def test_cache_layout(self, tmp_path):
        _, task, train, test = TA._setup()
        args = dataclasses.replace(TA._args(tmp_path), recompute_gradients=True)
        attr = AdamWInfluenceAttributor(args, task=task)
        ((train_dir, test_dir),) = attr.cache(train, test, hook_config=HOOKS)
        assert train_dir.endswith("train_snapshots")
        assert TrajectorySnapshots.is_snapshot_dir(train_dir)
        snaps = TrajectorySnapshots(train_dir)
        assert snaps.steps() == list(range(TA.N_TRAIN // TA.BATCH))
        assert snaps.dynamics_steps() == snaps.steps()
        assert not (tmp_path / "train_grads").exists()
        # The query side is still a gradient store.
        assert (tmp_path / "test_grads").is_dir()
        assert not TrajectorySnapshots.is_snapshot_dir(test_dir)

    @pytest.mark.parametrize("propagation", ["train", "test"])
    def test_adamw_recompute_matches_stored(self, tmp_path, propagation):
        ref = _run_adamw(tmp_path / "stored", recompute=False, propagation=propagation)
        got = _run_adamw(tmp_path / "replay", recompute=True, propagation=propagation)
        _same_scores(ref, got)

    @pytest.mark.parametrize("propagation", ["train", "test"])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_dvemb_recompute_matches_stored(
        self, tmp_path, propagation, loop_over_test
    ):
        if propagation == "train" and loop_over_test:
            pytest.skip("loop_over_test is a test-side option")
        kw = {"propagation": propagation, "loop_over_test": loop_over_test}
        ref = _run_dvemb(tmp_path / "stored", recompute=False, **kw)
        got = _run_dvemb(tmp_path / "replay", recompute=True, **kw)
        _same_scores(ref, got)

    def test_replay_needs_the_task(self, tmp_path):
        _, task, train, test = TA._setup()
        args = dataclasses.replace(TA._args(tmp_path), recompute_gradients=True)
        ((train_dir, test_dir),) = AdamWInfluenceAttributor(args, task=task).cache(
            train, test, hook_config=HOOKS
        )
        with pytest.raises(ValueError, match="requires a ``task``"):
            AdamWInfluenceAttributor(args).attribute_from_cache(
                train_dir, test_dir, hook_config=HOOKS
            )

    def test_stored_cache_scored_from_a_fresh_attributor(self, tmp_path):
        """Store-then-attribute over snapshots: a new attributor with the task
        and the capture config replays the cached trajectory.
        """
        _, task, train, test = TA._setup()
        args = dataclasses.replace(TA._args(tmp_path), recompute_gradients=True)
        ((train_dir, test_dir),) = AdamWInfluenceAttributor(args, task=task).cache(
            train, test, hook_config=HOOKS
        )
        got = AdamWInfluenceAttributor(args, task=task).attribute_from_cache(
            train_dir, test_dir, hook_config=HOOKS, propagation="test"
        )
        ref = _run_adamw(tmp_path / "stored", recompute=False)
        _same_scores(ref, got)
