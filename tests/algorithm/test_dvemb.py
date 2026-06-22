"""Correctness tests for :class:`DVEmbAttributor` (on-disk workflow).

DVEmb corrects the TracIn inner product for how a training update at step
``t_s`` propagates through every later step up to the final model ``θ_T``:

    I(z*, t_s) = η_{t_s} · g_val(θ_T)ᵀ [∏_{k=t_s+1}^{T-1}(I − η_k H_k)] g(θ_{t_s}, z*)

with ``H_k = Σ_{z∈B_k} g(θ_k, z) g(θ_k, z)ᵀ`` (empirical Fisher of the batch).

The oracle here builds the per-step Fisher and the explicit matrix product over
full concatenated parameter gradients — independent of the attributor's
per-layer, test-side back-propagation — and the two must agree.  Setting the
learning rate so the Fisher correction vanishes (``H=0`` reached by using a
single training step) recovers TracIn, which is checked too.

The collection / model fixtures are shared with ``test_tracin``.
"""

from __future__ import annotations

import pytest
import torch

import tests.algorithm.test_tracin as TT
from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.dvemb import DVEmbAttributor
from dattri_llm.gradient.utils import hash_sample

LAYERS = TT.LAYER_NAMES  # ["mlp.fc1", "mlp.fc2"]


def _args(out_dir):
    return AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def _full_grads(model, sd, x, y):
    """Per-sample full (concatenated) weight gradients at checkpoint *sd*."""
    return TT._grads_at(model, sd, x, y, normalized=False)


def _dvemb_oracle(model, train_sds, x_tr, y_tr, test_sd, x_te, y_te, lr, final_step):
    """Explicit DVEmb influence: (num_train_rows grouped by step, num_test).

    Returns ``{step: (B, num_test) tensor}`` of η · g_valᵀ M g(z*).
    """
    steps = list(range(len(train_sds)))
    prop_steps = [s for s in steps if s < final_step]
    # Full gradients per step and the final-model test gradients.
    g_tr = {s: _full_grads(model, train_sds[s], x_tr, y_tr) for s in prop_steps}
    g_te = _full_grads(model, test_sd, x_te, y_te)              # (num_test, P)
    dim = g_te.shape[1]
    H = {s: g_tr[s].T @ g_tr[s] for s in prop_steps}            # Σ_z g gᵀ
    eye = torch.eye(dim)

    out = {}
    for ts in prop_steps:
        M = eye.clone()
        for k in prop_steps:
            if ts < k < final_step:
                M = M @ (eye - lr * H[k])
        # η · g_valᵀ M g(z*)  → (num_train, num_test)
        out[ts] = lr * (g_tr[ts] @ M @ g_te.T)
    return out


@pytest.fixture()
def collected(tmp_path):
    """Two-checkpoint train dir + single-checkpoint (final-model) test dir."""
    torch.manual_seed(TT.SEED)
    model = TT.MLP().eval()
    sd0, sd1 = TT._make_checkpoints(model)
    # A distinct "final model" checkpoint for the test gradients (θ_T).
    g = torch.Generator().manual_seed(TT.SEED + 7)
    sdT = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in sd1.items()}
    x_tr, y_tr, x_te, y_te = TT._make_data()

    train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
    TT._collect_to_disk(model, [sd0, sd1], x_tr, y_tr, train_dir)
    TT._collect_to_disk(model, [sdT], x_te, y_te, test_dir)

    train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(TT.N_TRAIN)]
    test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(TT.N_TEST)]
    return dict(
        model=model, train_sds=[sd0, sd1], test_sd=sdT,
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        train_dir=train_dir, test_dir=test_dir,
        train_hashes=train_hashes, test_hashes=test_hashes,
    )


def _make_attr(out_dir, lr):
    return DVEmbAttributor(_args(out_dir), learning_rate=lr, layer_name=LAYERS)


class TestDVEmbOnDisk:
    @pytest.mark.parametrize("lr", [0.1, 0.5])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_matches_explicit_oracle(self, collected, tmp_path, lr, loop_over_test):
        """Full propagated influence matches the explicit-matrix oracle."""
        res = _make_attr(tmp_path / f"o_{lr}_{loop_over_test}", lr).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=loop_over_test,
            final_step=2,
        )
        oracle = _dvemb_oracle(
            collected["model"], collected["train_sds"],
            collected["x_tr"], collected["y_tr"],
            collected["test_sd"], collected["x_te"], collected["y_te"],
            lr=lr, final_step=2,
        )
        # Per-step rows must match the oracle for that step.
        for step, (train_ids, matrix) in res.step_matrices().items():
            want = oracle[step][[collected["train_hashes"].index(h) for h in train_ids]]
            cols = [collected["test_hashes"].index(h) for h in res.test_ids]
            assert torch.allclose(matrix, want[:, cols], atol=1e-5, rtol=1e-4), (
                f"step {step}: max diff "
                f"{(matrix - want[:, cols]).abs().max().item():.2e}"
            )

    def test_loop_modes_agree(self, collected, tmp_path):
        a = _make_attr(tmp_path / "a", 0.3).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=False, final_step=2,
        )
        b = _make_attr(tmp_path / "b", 0.3).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=True, final_step=2,
        )
        assert a.test_ids == b.test_ids
        assert a.row_train_ids == b.row_train_ids
        assert a.row_steps == b.row_steps
        assert torch.equal(a.scores, b.scores)

    def test_reduces_to_tracin_at_final_step(self, collected, tmp_path):
        """The last step (t_s = T−1) has an empty propagation product, so its
        DVEmb rows are exactly η · ⟨g_train, g_test⟩ — plain TracIn."""
        lr = 0.4
        res = _make_attr(tmp_path / "o", lr).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            final_step=2,
        )
        # Oracle: η · g(θ_1) gᵀ_val (no Fisher factor for the final train step).
        g_tr1 = _full_grads(
            collected["model"], collected["train_sds"][1],
            collected["x_tr"], collected["y_tr"],
        )
        g_te = _full_grads(
            collected["model"], collected["test_sd"],
            collected["x_te"], collected["y_te"],
        )
        tracin = lr * (g_tr1 @ g_te.T)
        train_ids, matrix = res.step_matrix(1)
        rows = [collected["train_hashes"].index(h) for h in train_ids]
        cols = [collected["test_hashes"].index(h) for h in res.test_ids]
        assert torch.allclose(matrix, tracin[rows][:, cols], atol=1e-5, rtol=1e-4)

    def test_selected_steps_filter_rows_but_not_propagation(self, collected, tmp_path):
        """Restricting output to step 0 still propagates through step 1: the
        step-0 rows keep their full (I − η H_1) correction."""
        lr = 0.5
        res = _make_attr(tmp_path / "o", lr).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            selected_training_steps=[0], final_step=2,
        )
        assert sorted(set(res.row_steps)) == [0]
        oracle = _dvemb_oracle(
            collected["model"], collected["train_sds"],
            collected["x_tr"], collected["y_tr"],
            collected["test_sd"], collected["x_te"], collected["y_te"],
            lr=lr, final_step=2,
        )
        train_ids, matrix = res.step_matrix(0)
        rows = [collected["train_hashes"].index(h) for h in train_ids]
        cols = [collected["test_hashes"].index(h) for h in res.test_ids]
        assert torch.allclose(
            matrix, oracle[0][rows][:, cols], atol=1e-5, rtol=1e-4
        )

    def test_per_step_learning_rate_mapping(self, collected, tmp_path):
        """A {step: η} mapping uses each step's own rate in score and Fisher."""
        lrs = {0: 0.3, 1: 0.6}
        res = DVEmbAttributor(
            _args(tmp_path / "o"), learning_rate=lrs, layer_name=LAYERS
        ).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            final_step=2,
        )
        # Oracle with per-step rates: η_{t_s} scale and η_k factors.
        model, train_sds = collected["model"], collected["train_sds"]
        g_tr = {s: _full_grads(model, train_sds[s], collected["x_tr"], collected["y_tr"])
                for s in (0, 1)}
        g_te = _full_grads(model, collected["test_sd"], collected["x_te"], collected["y_te"])
        dim = g_te.shape[1]
        eye = torch.eye(dim)
        H1 = g_tr[1].T @ g_tr[1]
        oracle = {
            0: lrs[0] * (g_tr[0] @ (eye - lrs[1] * H1) @ g_te.T),
            1: lrs[1] * (g_tr[1] @ g_te.T),
        }
        for step, (train_ids, matrix) in res.step_matrices().items():
            rows = [collected["train_hashes"].index(h) for h in train_ids]
            cols = [collected["test_hashes"].index(h) for h in res.test_ids]
            assert torch.allclose(
                matrix, oracle[step][rows][:, cols], atol=1e-5, rtol=1e-4
            )

    def test_missing_gradients_dir_raises(self, collected, tmp_path):
        attr = _make_attr(tmp_path / "o", 0.1)
        with pytest.raises(ValueError, match=r"train_gradients_dir"):
            attr.attribute(test_gradients_dir=str(collected["test_dir"]), final_step=2)

    def test_no_step_below_final_raises(self, collected, tmp_path):
        attr = _make_attr(tmp_path / "o", 0.1)
        with pytest.raises(ValueError, match=r"step < final_step"):
            attr.attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
                final_step=0,
            )
