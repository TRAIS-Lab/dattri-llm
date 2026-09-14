"""``attribution_granularity="token"``: one score row per training token
position, summing back to the instance-level score exactly -- for every
inner-product attributor, since each score is bilinear in the train gradient.
"""

from __future__ import annotations

import pytest
import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.attribution.score import AttributionScore
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import CaptureCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

IN, HID, OUT = 6, 10, 4
T = 5  # tokens per sequence
N_TRAIN, N_TEST = 6, 3
SEED = 0


class SeqMLP(nn.Module):
    """Position-wise MLP over ``(B, T, IN)`` -- every layer keeps a token axis."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class SeqDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def _task_and_data():
    torch.manual_seed(SEED)
    model = SeqMLP().eval()
    g = torch.Generator().manual_seed(SEED)
    train = SeqDataset(
        torch.randn(N_TRAIN, T, IN, generator=g),
        torch.randn(N_TRAIN, T, OUT, generator=g),
    )
    test = SeqDataset(
        torch.randn(N_TEST, T, IN, generator=g),
        torch.randn(N_TEST, T, OUT, generator=g),
    )

    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return (
        AttributionTask(loss_func=loss_func, model=model, checkpoints=[ckpt]),
        train,
        test,
    )


def _args(out_dir, batch=2) -> AttributionArguments:
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
    )


def _capture(model, x, capture_style="factorized"):
    cb = CaptureCallback()
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL, capture_style=capture_style),
        callbacks=[cb],
    )
    with hm.collect():
        model(x).pow(2).sum().backward()
    hm.remove()
    return cb.record.gradient


class TestKernel:
    def test_dense_side_matches_factorized_side(self):
        torch.manual_seed(1)
        model = SeqMLP()
        tr = _capture(model, torch.randn(3, T, IN))
        te = _capture(model, torch.randn(2, T, IN))
        te_dense = te.materialize()
        want = ops.layerwise_cross_dot_per_token(tr, te)
        got = ops.layerwise_cross_dot_per_token(tr, te_dense)
        assert got.shape == (3, T, 2)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
        # ... and both sum over positions to the instance-level cross-gram.
        torch.testing.assert_close(
            got.sum(1), ops.layerwise_cross_dot(tr, te_dense), atol=1e-5, rtol=1e-4
        )

    def test_similarity_per_token_accepts_a_dense_other(self):
        torch.manual_seed(2)
        model = SeqMLP()
        tr = _capture(model, torch.randn(2, T, IN))
        te = _capture(model, torch.randn(2, T, IN))
        torch.testing.assert_close(
            tr.similarity_per_token(te.materialize()),
            tr.similarity_per_token(te),
            atol=1e-5,
            rtol=1e-4,
        )

    def test_materialized_train_side_raises(self):
        torch.manual_seed(3)
        model = SeqMLP()
        tr = _capture(model, torch.randn(2, T, IN), capture_style="materialized")
        te = _capture(model, torch.randn(2, T, IN))
        with pytest.raises(ValueError, match="no token axis"):
            ops.layerwise_cross_dot_per_token(tr, te)


def _check_token_score(score: AttributionScore, instance: AttributionScore) -> None:
    assert score.granularity == "token"
    assert instance.granularity == "instance"
    assert score.num_rows == N_TRAIN * T
    assert score.row_token_ids == list(range(T)) * N_TRAIN
    assert score.test_ids == instance.test_ids
    # The instance-level accessors of a token score give the instance score.
    ids_t, m_t = score.agnostic_matrix()
    ids_i, m_i = instance.agnostic_matrix()
    assert ids_t == ids_i
    torch.testing.assert_close(m_t, m_i, atol=1e-4, rtol=1e-4)
    for h in ids_t:
        positions, block = score.token_scores(h)
        assert positions == list(range(T))
        torch.testing.assert_close(
            block.sum(0), instance.trajectory_agnostic(h), atol=1e-4, rtol=1e-4
        )
        steps, rows = score.trajectory_aware(h)
        assert steps == [0]
        torch.testing.assert_close(rows[0], block.sum(0), atol=1e-4, rtol=1e-4)
    t_ids, t_m = score.step_matrix(0)
    torch.testing.assert_close(t_m, m_i, atol=1e-4, rtol=1e-4)
    assert t_ids == ids_i
    torch.testing.assert_close(
        score.score_at(ids_t[0], 0, score.test_ids[1]),
        instance.score_at(ids_t[0], 0, score.test_ids[1]),
        atol=1e-4,
        rtol=1e-4,
    )
    assert score.algorithm_meta["attribution_granularity"] == "token"


class TestAttributors:
    @pytest.mark.parametrize("normalized_grad", [False, True])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_tracin_live(self, tmp_path, normalized_grad, loop_over_test):
        task, tr, te = _task_and_data()
        attr = TracInAttributor(_args(tmp_path / "a"), task=task)
        instance = attr.attribute(
            tr, te, normalized_grad=normalized_grad, loop_over_test=loop_over_test
        )
        token = attr.attribute(
            tr,
            te,
            normalized_grad=normalized_grad,
            loop_over_test=loop_over_test,
            attribution_granularity="token",
        )
        _check_token_score(token, instance)

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_kronecker_live(self, tmp_path, cls):
        task, tr, te = _task_and_data()
        attr = cls(_args(tmp_path / "a"), task=task)
        instance = attr.attribute(tr, te, damping=1e-2)
        token = attr.attribute(tr, te, damping=1e-2, attribution_granularity="token")
        _check_token_score(token, instance)

    def test_kfac_from_cache_with_preconditioned_store(self, tmp_path):
        task, tr, te = _task_and_data()
        attr = KFACAttributor(_args(tmp_path / "a"), task=task)
        ((train_dir, test_dir),) = attr.cache(tr, te)
        instance = attr.attribute_from_cache(train_dir, test_dir, damping=1e-2)
        token = attr.attribute_from_cache(
            train_dir,
            test_dir,
            damping=1e-2,
            loop_over_test=True,
            preconditioned_test_cache_residency="memory",
            attribution_granularity="token",
        )
        _check_token_score(token, instance)

    def test_round_trips_through_disk(self, tmp_path):
        task, tr, te = _task_and_data()
        attr = TracInAttributor(_args(tmp_path / "a"), task=task)
        token = attr.attribute(tr, te, attribution_granularity="token")
        loaded = AttributionScore.load(attr.args.output_path)
        assert loaded.row_token_ids == token.row_token_ids
        torch.testing.assert_close(loaded.scores, token.scores)
        assert loaded.granularity == "token"

    def test_unknown_granularity_rejected(self, tmp_path):
        task, tr, te = _task_and_data()
        attr = TracInAttributor(_args(tmp_path / "a"), task=task)
        with pytest.raises(ValueError, match="attribution_granularity"):
            attr.attribute(tr, te, attribution_granularity="sentence")

    def test_token_scores_needs_token_granularity(self, tmp_path):
        task, tr, te = _task_and_data()
        attr = TracInAttributor(_args(tmp_path / "a"), task=task)
        instance = attr.attribute(tr, te)
        with pytest.raises(ValueError, match="token-level"):
            instance.token_scores(instance.train_ids[0])
