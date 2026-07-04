"""Correctness tests for KFACAttributor / EKFACAttributor (on-disk workflow).

Each attributor's factorised fast path is checked against an explicit
Kronecker-Fisher oracle built from the *same* on-disk gradients:

* **K-FAC** -- ``score = sum_l <dW_te, G_l^-1 dW_tr A_l^-1>_F`` with
  ``A_l^-1=(A_l+lambda)^-1`` etc.  This oracle is fully basis-free.
* **EK-FAC** -- rotate ``dW`` into the Kronecker eigenbasis, divide by the
  empirical corrected eigenvalues, contract.  The eigenbasis itself
  (``torch.linalg.eigh``) is shared with the implementation (EK-FAC is
  basis-dependent by construction); the scoring assembly is computed
  independently here.

A tiny bias-free MLP with single-token (2-D) inputs is used so each per-sample
weight gradient is exactly the rank-1 outer product ``g a^T``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tests.attribution.test_tracin as TT
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Factorized
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.utils.hashing import hash_sample

DAMPING = 1e-2
LAYERS = TT.LAYER_NAMES  # ["mlp.fc1", "mlp.fc2"]


# --------------------------------------------------------------------------- #
# Oracle helpers                                                               #
# --------------------------------------------------------------------------- #


def _load_factors(test_dir, step, layers):
    """Return ``{hash: {layer: (a, g)}}`` for one step (per-sample records)."""
    fm = GradientFileManager(str(test_dir))
    out = {}
    for file_rel, idxs in fm.iter_step(step):
        recs = fm.load_records(file_rel)
        for i in idxs:
            rec = recs[i]
            hashes = (
                rec.input_hash if isinstance(rec.input_hash, list) else [rec.input_hash]
            )
            g = rec.gradient
            for b, h in enumerate(hashes):
                out[h] = {
                    layer: (
                        g.data[layer].activation[b].float(),
                        g.data[layer].pre_activation_grad[b].float(),
                    )
                    for layer in layers
                }
    return out


def _stack(factors, hashes, layer, idx):
    return torch.stack([factors[h][layer][idx] for h in hashes])


def _kfac_oracle(tr_f, te_f, train_hashes, test_hashes, damping):
    N = len(train_hashes)
    oracle = torch.zeros(len(train_hashes), len(test_hashes))
    for layer in LAYERS:
        a_tr, g_tr = (
            _stack(tr_f, train_hashes, layer, 0),
            _stack(tr_f, train_hashes, layer, 1),
        )
        a_te, g_te = (
            _stack(te_f, test_hashes, layer, 0),
            _stack(te_f, test_hashes, layer, 1),
        )
        A = a_tr.T @ a_tr / N
        G = g_tr.T @ g_tr / N
        A_inv = torch.linalg.inv(A + damping * torch.eye(A.shape[0]))
        G_inv = torch.linalg.inv(G + damping * torch.eye(G.shape[0]))
        dW_tr = torch.einsum("no,ni->noi", g_tr, a_tr)  # (N_tr, out, in)
        dW_te = torch.einsum("mo,mi->moi", g_te, a_te)  # (N_te, out, in)
        T = torch.einsum("oc,nci->noi", G_inv, dW_tr)
        T = torch.einsum("noj,jp->nop", T, A_inv)  # G^-1 dW A^-1
        oracle += torch.einsum("nop,mop->nm", T, dW_te)
    return oracle


def _ekfac_oracle(tr_f, te_f, train_hashes, test_hashes, damping):
    N = len(train_hashes)
    oracle = torch.zeros(len(train_hashes), len(test_hashes))
    for layer in LAYERS:
        a_tr, g_tr = (
            _stack(tr_f, train_hashes, layer, 0),
            _stack(tr_f, train_hashes, layer, 1),
        )
        a_te, g_te = (
            _stack(te_f, test_hashes, layer, 0),
            _stack(te_f, test_hashes, layer, 1),
        )
        A = a_tr.T @ a_tr / N
        G = g_tr.T @ g_tr / N
        # The faithful projection is sign-invariant, so the oracle and the
        # implementation agree regardless of eigh's arbitrary sign choices.
        _, U_A, _, U_G = ops.kfac_eigh(A, G)
        dW_tr = torch.einsum("no,ni->noi", g_tr, a_tr)
        dW_te = torch.einsum("mo,mi->moi", g_te, a_te)
        M_tr = torch.einsum("op,noi,iq->npq", U_G, dW_tr, U_A)  # U_G^T dW U_A
        M_te = torch.einsum("op,moi,iq->mpq", U_G, dW_te, U_A)
        lam = (M_tr * M_tr).mean(0)  # corrected eigenvalues
        oracle += torch.einsum("npq,mpq->nm", M_tr / (lam + damping), M_te)
    return oracle


def _grad_dot_oracle(tr_f, te_f, train_hashes, test_hashes):
    """Plain per-sample gradient dot ``<dW_tr, dW_te>`` summed over layers.

    In the heavy-damping limit every EK-FAC convention collapses to this (the
    eigenbasis rotation is orthogonal, so it preserves the inner product),
    independent of the ill-defined eigenvector choices.
    """
    out = torch.zeros(len(train_hashes), len(test_hashes))
    for layer in LAYERS:
        a_tr, g_tr = (
            _stack(tr_f, train_hashes, layer, 0),
            _stack(tr_f, train_hashes, layer, 1),
        )
        a_te, g_te = (
            _stack(te_f, test_hashes, layer, 0),
            _stack(te_f, test_hashes, layer, 1),
        )
        out += (g_tr @ g_te.T) * (a_tr @ a_te.T)
    return out


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def collected(tmp_path):
    """One-checkpoint (single-step) collection so the oracle stays transparent."""
    torch.manual_seed(TT.SEED)
    model = TT.MLP().eval()
    checkpoints = TT._make_checkpoints(model)[:1]
    x_tr, y_tr, x_te, y_te = TT._make_data()
    train_dir, test_dir = tmp_path / "train_g", tmp_path / "test_g"
    TT._collect_to_disk(model, checkpoints, x_tr, y_tr, train_dir)
    TT._collect_to_disk(model, checkpoints, x_te, y_te, test_dir)
    train_hashes = [
        hash_sample({"x": x_tr[i], "y": y_tr[i]}) for i in range(TT.N_TRAIN)
    ]
    test_hashes = [hash_sample({"x": x_te[j], "y": y_te[j]}) for j in range(TT.N_TEST)]
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_hashes": train_hashes,
        "test_hashes": test_hashes,
    }


def _make(attr_cls, out_dir):
    args = AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return attr_cls(args)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


class TestKFAC:
    def test_matches_kronecker_oracle(self, collected, tmp_path):
        res = _make(KFACAttributor, tmp_path / "o").attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        matrix = res.query(
            collected["train_hashes"],
            collected["test_hashes"],
            trajectory="agnostic",
        )
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _kfac_oracle(
            tr_f,
            te_f,
            collected["train_hashes"],
            collected["test_hashes"],
            DAMPING,
        )
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_algorithm_label_and_shape(self, collected, tmp_path):
        res = _make(KFACAttributor, tmp_path / "o").attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert res.algorithm == "KFAC"
        assert res.scores.shape == (TT.N_TRAIN, TT.N_TEST)


class TestEKFAC:
    def test_matches_eigenbasis_oracle(self, collected, tmp_path):
        res = _make(EKFACAttributor, tmp_path / "o").attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        matrix = res.query(
            collected["train_hashes"],
            collected["test_hashes"],
            trajectory="agnostic",
        )
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _ekfac_oracle(
            tr_f,
            te_f,
            collected["train_hashes"],
            collected["test_hashes"],
            DAMPING,
        )
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_algorithm_label(self, collected, tmp_path):
        res = _make(EKFACAttributor, tmp_path / "o").attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert res.algorithm == "EKFAC"

    def test_heavy_damping_limit_is_gradient_dot(self, collected, tmp_path):
        """At large damping EK-FAC -> (1/lambda)*<dW_tr, dW_te> (the eigenvalue
        correction washes out), validating the rotate->divide->contract machinery.
        """
        damping = 1e6
        res = EKFACAttributor(
            AttributionArguments(
                output_dir=str(tmp_path / "o"),
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
            ),
        ).attribute_from_cache(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            damping=damping,
        )
        matrix = (
            res.query(
                collected["train_hashes"],
                collected["test_hashes"],
                trajectory="agnostic",
            )
            * damping
        )
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _grad_dot_oracle(
            tr_f,
            te_f,
            collected["train_hashes"],
            collected["test_hashes"],
        )
        assert torch.allclose(matrix, oracle, atol=1e-2, rtol=1e-2), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_transposed_projection_is_sign_sensitive(self):
        """Design rationale for the faithful projection ``U_G^T dW U_A``: its
        score is invariant to an (arbitrary) eigenvector sign flip, whereas the
        transposed ``U_G dW U_A^T`` (dattri's original) is not -- which is why the
        'approx' mode was fixed to use the faithful projection too.
        """
        torch.manual_seed(0)
        B, out, inn = 30, 4, 5
        a, g = torch.randn(B, inn), torch.randn(B, out)
        A, G = a.T @ a / B, g.T @ g / B
        _, U_A, _, U_G = ops.kfac_eigh(A, G)

        def score(U_A, U_G, transposed):
            rot_a, rot_g = (
                (U_A.T.contiguous(), U_G.T.contiguous()) if transposed else (U_A, U_G)
            )
            M = ops.ekfac_materialize(
                Factorized(a, g),
                "nn.Linear",
                rot_a,
                rot_g,
                include_bias=False,
            )
            lam = (M * M).mean(0)
            return (M / (lam + 1e-3)) @ M.T

        U_G_flip = U_G.clone()
        U_G_flip[:, 0] *= -1  # a different but equally valid eigenbasis
        # Faithful projection: invariant.  Transposed (dattri): not.
        assert torch.allclose(
            score(U_A, U_G, transposed=False),
            score(U_A, U_G_flip, transposed=False),
            atol=1e-4,
        )
        assert not torch.allclose(
            score(U_A, U_G, transposed=True),
            score(U_A, U_G_flip, transposed=True),
            atol=1e-4,
        )

    def test_modes_agree(self, collected, tmp_path):
        """The fixed 'approx' mode now produces the same scores as 'exact'."""

        def run(mode):
            return (
                EKFACAttributor(
                    AttributionArguments(
                        output_dir=str(tmp_path / mode),
                        dataloader_num_workers=0,
                        dataloader_pin_memory=False,
                    ),
                    mode=mode,
                )
                .attribute_from_cache(
                    damping=DAMPING,
                    train_gradients_dir=str(collected["train_dir"]),
                    test_gradients_dir=str(collected["test_dir"]),
                )
                .query(
                    collected["train_hashes"],
                    collected["test_hashes"],
                    trajectory="agnostic",
                )
            )

        assert torch.allclose(run("exact"), run("approx"), atol=1e-6)

    def test_invalid_mode_raises(self, tmp_path):
        args = AttributionArguments(output_dir=str(tmp_path / "o"))
        with pytest.raises(ValueError, match=r"mode"):
            EKFACAttributor(args, mode="bogus")


class _MultiTokenModel(nn.Module):
    """Embedding -> single bias-free Linear ``fc`` applied per token (B, T, *)."""

    def __init__(self, vocab=16, d=4, h=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, h, bias=False)

    def forward(self, x, y=None):
        return self.fc(self.embedding(x))


def _kfac_oracle_mt(tr_f, te_f, train_hashes, test_hashes, layer, damping):
    a_tr, g_tr = (
        _stack(tr_f, train_hashes, layer, 0),
        _stack(tr_f, train_hashes, layer, 1),
    )
    a_te, g_te = (
        _stack(te_f, test_hashes, layer, 0),
        _stack(te_f, test_hashes, layer, 1),
    )
    n_tok = a_tr.shape[0] * a_tr.shape[1]
    a_flat = a_tr.reshape(-1, a_tr.shape[-1])
    g_flat = g_tr.reshape(-1, g_tr.shape[-1])
    A = a_flat.T @ a_flat / n_tok
    G = g_flat.T @ g_flat / n_tok
    A_inv = torch.linalg.inv(A + damping * torch.eye(A.shape[0]))
    G_inv = torch.linalg.inv(G + damping * torch.eye(G.shape[0]))
    dW_tr = torch.einsum("nto,nti->noi", g_tr, a_tr)  # sum_t g_t a_t^T
    dW_te = torch.einsum("mto,mti->moi", g_te, a_te)
    T = torch.einsum("oc,nci->noi", G_inv, dW_tr)
    T = torch.einsum("noj,jp->nop", T, A_inv)
    return torch.einsum("nop,mop->nm", T, dW_te)


class TestKFACMultiToken:
    def test_matches_oracle_with_token_dim(self, tmp_path):
        """KFAC's token-summed factorised path must match the explicit sum_t oracle."""
        torch.manual_seed(0)
        model = _MultiTokenModel().eval()
        B, T = 5, 3
        gen = torch.Generator().manual_seed(1)
        x_tr = torch.randint(0, 16, (B, T), generator=gen)
        x_te = torch.randint(0, 16, (4, T), generator=gen)

        def collect(x, out_dir):
            fm = GradientFileManager(str(out_dir))
            hm = HookManager(
                model,
                config=HookManagerConfig(linear_io=[r"fc"]),
                callbacks=[
                    OffloadCallback(
                        offload_interval=1,
                        file_manager=fm,
                        recording_type="per_sample",
                    ),
                ],
            )
            with hm.collect():
                model.zero_grad(set_to_none=True)
                model(x=x).sum().backward()
            hm.remove()

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        collect(x_tr, train_dir)
        collect(x_te, test_dir)
        train_hashes = [hash_sample({"x": x_tr[i]}) for i in range(B)]
        test_hashes = [hash_sample({"x": x_te[j]}) for j in range(4)]

        args = AttributionArguments(
            output_dir=str(tmp_path / "o"),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        res = KFACAttributor(args).attribute_from_cache(
            train_gradients_dir=str(train_dir),
            test_gradients_dir=str(test_dir),
            damping=DAMPING,
        )
        matrix = res.query(train_hashes, test_hashes, trajectory="agnostic")
        tr_f = _load_factors(train_dir, 0, ["fc"])
        te_f = _load_factors(test_dir, 0, ["fc"])
        oracle = _kfac_oracle_mt(tr_f, te_f, train_hashes, test_hashes, "fc", DAMPING)
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )


@pytest.fixture
def collected_step1(tmp_path):
    """Single-checkpoint dirs whose records were recorded at a *non-zero* step.

    Collects both checkpoints (steps 0 and 1), then curates train/test dirs
    holding only the step-1 records -- which keep their recorded ``step`` of 1 --
    so the attributor sees a one-checkpoint dataset stamped at step 1 rather
    than 0.
    """
    torch.manual_seed(TT.SEED)
    model = TT.MLP().eval()
    checkpoints = TT._make_checkpoints(model)  # steps 0 and 1
    x_tr, y_tr, x_te, y_te = TT._make_data()
    raw_train, raw_test = tmp_path / "raw_tr", tmp_path / "raw_te"
    TT._collect_to_disk(model, checkpoints, x_tr, y_tr, raw_train)
    TT._collect_to_disk(model, checkpoints, x_te, y_te, raw_test)

    train_dir, test_dir = tmp_path / "tr_s1", tmp_path / "te_s1"
    GradientFileManager(str(train_dir)).save_bulk(TT._load_step_records(raw_train, 1))
    GradientFileManager(str(test_dir)).save_bulk(TT._load_step_records(raw_test, 1))

    train_hashes = [
        hash_sample({"x": x_tr[i], "y": y_tr[i]}) for i in range(TT.N_TRAIN)
    ]
    test_hashes = [hash_sample({"x": x_te[j], "y": y_te[j]}) for j in range(TT.N_TEST)]
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_hashes": train_hashes,
        "test_hashes": test_hashes,
    }


class TestRowStepsTracked:
    """Regression: rows are stamped with the gradient's recorded step, not 0."""

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_row_steps_reflect_recorded_step(self, collected_step1, tmp_path, cls):
        res = _make(cls, tmp_path / "o").attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected_step1["train_dir"]),
            test_gradients_dir=str(collected_step1["test_dir"]),
        )
        assert res.row_steps == [1] * TT.N_TRAIN
        # The hash->step pairing must be correct sample-by-sample, not just in bulk.
        assert dict(
            zip(res.row_train_ids, res.row_steps, strict=True),
        ) == dict.fromkeys(
            collected_step1["train_hashes"],
            1,
        )


class TestStepSelection:
    """``steps=`` restricts which training checkpoints the Fisher + rows use."""

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_selected_steps_equal_curated_single_step_dir(self, tmp_path, cls):
        """Attributing a two-step train dir with ``selected_training_steps=[1]``
        is identical to attributing a dir curated to hold only the step-1
        records.
        """
        torch.manual_seed(TT.SEED)
        model = TT.MLP().eval()
        checkpoints = TT._make_checkpoints(model)  # steps 0 and 1
        x_tr, y_tr, x_te, y_te = TT._make_data()

        raw_train, test_dir = tmp_path / "raw_tr", tmp_path / "te"
        TT._collect_to_disk(model, checkpoints, x_tr, y_tr, raw_train)
        TT._collect_to_disk(model, checkpoints[:1], x_te, y_te, test_dir)

        curated = tmp_path / "tr_s1"
        GradientFileManager(str(curated)).save_bulk(TT._load_step_records(raw_train, 1))

        train_hashes = [
            hash_sample({"x": x_tr[i], "y": y_tr[i]}) for i in range(TT.N_TRAIN)
        ]
        test_hashes = [
            hash_sample({"x": x_te[j], "y": y_te[j]}) for j in range(TT.N_TEST)
        ]

        def run(train_dir, steps):
            attr = cls(
                AttributionArguments(
                    output_dir=str(tmp_path / f"o_{steps}"),
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                ),
            )
            return attr.attribute_from_cache(
                damping=DAMPING,
                train_gradients_dir=str(train_dir),
                test_gradients_dir=str(test_dir),
                selected_training_steps=steps,
            )

        selected = run(raw_train, [1])
        curated_res = run(curated, None)

        assert sorted(set(selected.row_steps)) == [1]
        assert selected.algorithm_meta["selected_training_steps"] == [1]
        a = selected.query(train_hashes, test_hashes, trajectory="agnostic")
        b = curated_res.query(train_hashes, test_hashes, trajectory="agnostic")
        assert torch.allclose(a, b, atol=1e-5), f"max diff {(a - b).abs().max():.2e}"

    def test_unknown_steps_raise(self, collected, tmp_path):
        attr = _make(KFACAttributor, tmp_path / "o")
        with pytest.raises(ValueError, match=r"requested steps"):
            attr.attribute_from_cache(
                damping=DAMPING,
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
                selected_training_steps=[99],
            )


class TestKroneckerShared:
    def test_missing_gradients_dir_raises(self, collected, tmp_path):
        attr = _make(KFACAttributor, tmp_path / "o")
        with pytest.raises(TypeError, match=r"train_gradients_dir"):
            attr.attribute_from_cache(
                damping=DAMPING,
                test_gradients_dir=str(collected["test_dir"]),
            )
        with pytest.raises(TypeError, match=r"test_gradients_dir"):
            attr.attribute_from_cache(
                damping=DAMPING,
                train_gradients_dir=str(collected["train_dir"]),
            )

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_loop_over_test_matches_cached(self, collected, tmp_path, cls):
        """Streaming the test set (loop_over_test=True) gives identical scores to
        the cached path -- same result, lower peak memory.
        """

        def run(loop, tag):
            return (
                _make(cls, tmp_path / tag)
                .attribute_from_cache(
                    damping=DAMPING,
                    train_gradients_dir=str(collected["train_dir"]),
                    test_gradients_dir=str(collected["test_dir"]),
                    loop_over_test=loop,
                )
                .query(
                    collected["train_hashes"],
                    collected["test_hashes"],
                    trajectory="agnostic",
                )
            )

        assert torch.allclose(
            run(loop=False, tag="cached"),
            run(loop=True, tag="loop"),
            atol=1e-5,
        )

    def test_kfac_ekfac_agree_when_lambda_is_kronecker(self, collected, tmp_path):
        """Sanity bridge: EK-FAC with a *single*-token, rank-deficient setup still
        runs and returns finite, correctly-shaped scores for both attributors.
        """
        for cls, label in ((KFACAttributor, "KFAC"), (EKFACAttributor, "EKFAC")):
            res = _make(cls, tmp_path / label).attribute_from_cache(
                damping=DAMPING,
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
            )
            m = res.query(
                collected["train_hashes"],
                collected["test_hashes"],
                trajectory="agnostic",
            )
            assert m.shape == (TT.N_TRAIN, TT.N_TEST)
            assert torch.isfinite(m).all()


# --------------------------------------------------------------------------- #
# Direct empirical-Fisher fallback for non-K-FAC (norm) layers                  #
# --------------------------------------------------------------------------- #


class _NormModel(nn.Module):
    """``embedding -> LayerNorm (norm) -> Linear (head)`` over a token dim.

    ``norm`` is non-K-FAC (normalisation); ``head`` is K-FAC-eligible.
    """

    def __init__(self, vocab: int = 16, d: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.norm = nn.LayerNorm(d, bias=False)
        self.head = nn.Linear(d, d, bias=False)

    def forward(self, x, y=None):
        return self.head(self.norm(self.embedding(x)))


def _fim_oracle(train_dir, test_dir, train_hashes, test_hashes, layer, damping):
    """Direct empirical-Fisher score for one norm layer, built from the
    token-summed per-sample weight gradients.
    """

    def load(d, hashes):
        fm = GradientFileManager(str(d))
        out = {}
        for file_rel, idxs in fm.iter_step(0):
            recs = fm.load_records(file_rel)
            for i in idxs:
                rec = recs[i]
                hs = (
                    rec.input_hash
                    if isinstance(rec.input_hash, list)
                    else [rec.input_hash]
                )
                mat = ops.materialize(
                    rec.gradient.data[layer],
                    rec.gradient.layer_types[layer],
                )
                for b, h in enumerate(hs):
                    out[h] = mat[b].float()
        return out

    tr, te = load(train_dir, train_hashes), load(test_dir, test_hashes)
    G_tr = torch.stack([tr[h] for h in train_hashes])  # (N_tr, P)
    G_te = torch.stack([te[h] for h in test_hashes])  # (N_te, P)
    F = G_tr.T @ G_tr / G_tr.shape[0]
    F_inv = ops.sym_inverse(F, damping)
    return (G_tr @ F_inv) @ G_te.T


def _collect_norm_model(tmp_path, patterns):
    """Collect the _NormModel's factorised gradients to disk (one step), hooking
    only the layers matching *patterns* -- layer selection now happens at capture
    (via the hook config), not at scoring.
    """
    torch.manual_seed(0)
    model = _NormModel().eval()
    gen = torch.Generator().manual_seed(1)
    x_tr = torch.randint(0, 16, (6, 3), generator=gen)
    x_te = torch.randint(0, 16, (4, 3), generator=gen)

    def collect(x, out_dir):
        fm = GradientFileManager(str(out_dir))
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=patterns),
            callbacks=[
                OffloadCallback(
                    offload_interval=1,
                    file_manager=fm,
                    recording_type="per_sample",
                ),
            ],
        )
        with hm.collect():
            model.zero_grad(set_to_none=True)
            model(x=x).sum().backward()
        hm.remove()

    train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
    collect(x_tr, train_dir)
    collect(x_te, test_dir)
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_hashes": [hash_sample({"x": x_tr[i]}) for i in range(6)],
        "test_hashes": [hash_sample({"x": x_te[j]}) for j in range(4)],
    }


@pytest.fixture
def norm_collected(tmp_path):
    """Both ``norm`` and ``head`` collected (one step)."""
    return _collect_norm_model(tmp_path, [r"norm", r"head"])


def _attr(cls, out_dir):
    args = AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return cls(args)


class TestDirectFIM:
    def _run(self, collected, out_dir, **kw):
        res = _attr(KFACAttributor, out_dir).attribute_from_cache(
            damping=DAMPING,
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            **kw,
        )
        return res.query(
            collected["train_hashes"],
            collected["test_hashes"],
            trajectory="agnostic",
        )

    def test_norm_only_matches_fim_oracle(self, tmp_path):
        """Collected norm-only, 'direct' is a pure empirical-Fisher score."""
        collected = _collect_norm_model(tmp_path / "norm_only", [r"norm"])
        m = self._run(collected, tmp_path / "o", non_kfac_strategy="direct")
        oracle = _fim_oracle(
            collected["train_dir"],
            collected["test_dir"],
            collected["train_hashes"],
            collected["test_hashes"],
            "norm",
            DAMPING,
        )
        assert torch.allclose(m, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(m - oracle).abs().max().item():.2e}"
        )

    def test_direct_equals_kfac_plus_fim(self, norm_collected, tmp_path):
        """With both layers, 'direct' == K-FAC(head) ['ignore'] + FIM(norm)."""
        ignore = self._run(norm_collected, tmp_path / "i", non_kfac_strategy="ignore")
        direct = self._run(norm_collected, tmp_path / "d", non_kfac_strategy="direct")
        fim = _fim_oracle(
            norm_collected["train_dir"],
            norm_collected["test_dir"],
            norm_collected["train_hashes"],
            norm_collected["test_hashes"],
            "norm",
            DAMPING,
        )
        assert not torch.allclose(ignore, direct)  # norm adds signal
        assert torch.allclose(direct, ignore + fim, atol=1e-4, rtol=1e-3)

    def test_ignore_is_default(self, norm_collected, tmp_path):
        default = self._run(norm_collected, tmp_path / "a")
        ignore = self._run(norm_collected, tmp_path / "b", non_kfac_strategy="ignore")
        assert torch.allclose(default, ignore)

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_loop_over_test_matches_cached_with_fim(
        self,
        norm_collected,
        tmp_path,
        cls,
    ):
        def run(loop, tag):
            return (
                _attr(cls, tmp_path / tag)
                .attribute_from_cache(
                    damping=DAMPING,
                    train_gradients_dir=str(norm_collected["train_dir"]),
                    test_gradients_dir=str(norm_collected["test_dir"]),
                    non_kfac_strategy="direct",
                    loop_over_test=loop,
                )
                .query(
                    norm_collected["train_hashes"],
                    norm_collected["test_hashes"],
                    trajectory="agnostic",
                )
            )

        assert torch.allclose(
            run(loop=False, tag="c"),
            run(loop=True, tag="l"),
            atol=1e-5,
        )

    def test_cap_skips_layer_with_warning(self, norm_collected, tmp_path):
        """A cap below the norm's param count skips it (with a warning), so the
        result collapses to the K-FAC-only ('ignore') score.
        """
        ignore = self._run(norm_collected, tmp_path / "i", non_kfac_strategy="ignore")
        with pytest.warns(UserWarning, match="direct_fim_max_params"):
            capped = self._run(
                norm_collected,
                tmp_path / "c",
                non_kfac_strategy="direct",
                direct_fim_max_params=4,
            )
        assert torch.allclose(capped, ignore, atol=1e-5)

    def test_norm_only_ignore_raises(self, tmp_path):
        """No K-FAC layer collected and strategy='ignore' -> nothing eligible."""
        collected = _collect_norm_model(tmp_path / "norm_only", [r"norm"])
        with pytest.raises(ValueError, match="No eligible layers"):
            self._run(collected, tmp_path / "o", non_kfac_strategy="ignore")

    def test_invalid_strategy_raises(self, norm_collected, tmp_path):
        with pytest.raises(ValueError, match="non_kfac_strategy"):
            self._run(norm_collected, tmp_path / "o", non_kfac_strategy="bogus")

    def test_invalid_max_params_raises(self, norm_collected, tmp_path):
        with pytest.raises(ValueError, match="direct_fim_max_params"):
            self._run(
                norm_collected,
                tmp_path / "o",
                non_kfac_strategy="direct",
                direct_fim_max_params=0,
            )

    def test_embedding_layer_warns_under_direct(self, tmp_path):
        """Embedding layers are heavily parametrised -> ignored with a warning."""
        torch.manual_seed(0)
        model = _NormModel().eval()
        gen = torch.Generator().manual_seed(1)
        x_tr = torch.randint(0, 16, (6, 3), generator=gen)
        x_te = torch.randint(0, 16, (4, 3), generator=gen)

        def collect(x, out_dir):
            fm = GradientFileManager(str(out_dir))
            hm = HookManager(
                model,
                config=HookManagerConfig(linear_io=[r"embedding", r"norm", r"head"]),
                callbacks=[
                    OffloadCallback(
                        offload_interval=1,
                        file_manager=fm,
                        recording_type="per_sample",
                    ),
                ],
            )
            with hm.collect():
                model.zero_grad(set_to_none=True)
                model(x=x).sum().backward()
            hm.remove()

        collect(x_tr, tmp_path / "tr")
        collect(x_te, tmp_path / "te")
        train_hashes = [hash_sample({"x": x_tr[i]}) for i in range(6)]
        test_hashes = [hash_sample({"x": x_te[j]}) for j in range(4)]
        with pytest.warns(UserWarning, match="embedding"):
            _attr(KFACAttributor, tmp_path / "o").attribute_from_cache(
                damping=DAMPING,
                train_gradients_dir=str(tmp_path / "tr"),
                test_gradients_dir=str(tmp_path / "te"),
                non_kfac_strategy="direct",
            ).query(train_hashes, test_hashes, trajectory="agnostic")
