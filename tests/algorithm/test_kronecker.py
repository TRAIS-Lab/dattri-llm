"""Correctness tests for KFACAttributor / EKFACAttributor (on-disk workflow).

Each attributor's factorised fast path is checked against an explicit
Kronecker-Fisher oracle built from the *same* on-disk gradients:

* **K-FAC** — ``score = Σ_l ⟨∇W_te, G_l⁻¹ ∇W_tr A_l⁻¹⟩_F`` with
  ``A_l⁻¹=(A_l+λ)⁻¹`` etc.  This oracle is fully basis-free.
* **EK-FAC** — rotate ``∇W`` into the Kronecker eigenbasis, divide by the
  empirical corrected eigenvalues, contract.  The eigenbasis itself
  (``torch.linalg.eigh``) is shared with the implementation (EK-FAC is
  basis-dependent by construction); the scoring assembly is computed
  independently here.

A tiny bias-free MLP with single-token (2-D) inputs is used so each per-sample
weight gradient is exactly the rank-1 outer product ``g aᵀ``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tests.algorithm.test_tracin as TT
from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.utils import hash_sample

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
            hashes = rec.input_hash if isinstance(rec.input_hash, list) else [rec.input_hash]
            g = rec.gradient
            for b, h in enumerate(hashes):
                out[h] = {
                    l: (
                        g.data[l].activation[b].float(),
                        g.data[l].pre_activation_grad[b].float(),
                    )
                    for l in layers
                }
    return out


def _stack(factors, hashes, layer, idx):
    return torch.stack([factors[h][layer][idx] for h in hashes])


def _kfac_oracle(tr_f, te_f, train_hashes, test_hashes, damping):
    N = len(train_hashes)
    oracle = torch.zeros(len(train_hashes), len(test_hashes))
    for l in LAYERS:
        a_tr, g_tr = _stack(tr_f, train_hashes, l, 0), _stack(tr_f, train_hashes, l, 1)
        a_te, g_te = _stack(te_f, test_hashes, l, 0), _stack(te_f, test_hashes, l, 1)
        A = a_tr.T @ a_tr / N
        G = g_tr.T @ g_tr / N
        A_inv = torch.linalg.inv(A + damping * torch.eye(A.shape[0]))
        G_inv = torch.linalg.inv(G + damping * torch.eye(G.shape[0]))
        dW_tr = torch.einsum("no,ni->noi", g_tr, a_tr)  # (N_tr, out, in)
        dW_te = torch.einsum("mo,mi->moi", g_te, a_te)  # (N_te, out, in)
        T = torch.einsum("oc,nci->noi", G_inv, dW_tr)
        T = torch.einsum("noj,jp->nop", T, A_inv)       # G⁻¹ ∇W A⁻¹
        oracle += torch.einsum("nop,mop->nm", T, dW_te)
    return oracle


def _ekfac_oracle(tr_f, te_f, train_hashes, test_hashes, damping):
    N = len(train_hashes)
    oracle = torch.zeros(len(train_hashes), len(test_hashes))
    for l in LAYERS:
        a_tr, g_tr = _stack(tr_f, train_hashes, l, 0), _stack(tr_f, train_hashes, l, 1)
        a_te, g_te = _stack(te_f, test_hashes, l, 0), _stack(te_f, test_hashes, l, 1)
        A = a_tr.T @ a_tr / N
        G = g_tr.T @ g_tr / N
        # The faithful projection is sign-invariant, so the oracle and the
        # implementation agree regardless of eigh's arbitrary sign choices.
        _, U_A, _, U_G = ops.kfac_eigh(A, G)
        dW_tr = torch.einsum("no,ni->noi", g_tr, a_tr)
        dW_te = torch.einsum("mo,mi->moi", g_te, a_te)
        M_tr = torch.einsum("op,noi,iq->npq", U_G, dW_tr, U_A)  # U_Gᵀ ∇W U_A
        M_te = torch.einsum("op,moi,iq->mpq", U_G, dW_te, U_A)
        lam = (M_tr * M_tr).mean(0)                              # corrected eigenvalues
        oracle += torch.einsum("npq,mpq->nm", M_tr / (lam + damping), M_te)
    return oracle


def _grad_dot_oracle(tr_f, te_f, train_hashes, test_hashes):
    """Plain per-sample gradient dot ``⟨∇W_tr, ∇W_te⟩`` summed over layers.

    In the heavy-damping limit every EK-FAC convention collapses to this (the
    eigenbasis rotation is orthogonal, so it preserves the inner product),
    independent of the ill-defined eigenvector choices.
    """
    out = torch.zeros(len(train_hashes), len(test_hashes))
    for l in LAYERS:
        a_tr, g_tr = _stack(tr_f, train_hashes, l, 0), _stack(tr_f, train_hashes, l, 1)
        a_te, g_te = _stack(te_f, test_hashes, l, 0), _stack(te_f, test_hashes, l, 1)
        out += (g_tr @ g_te.T) * (a_tr @ a_te.T)
    return out


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def collected(tmp_path):
    """One-checkpoint (single-step) collection so the oracle stays transparent."""
    torch.manual_seed(TT.SEED)
    model = TT.MLP().eval()
    checkpoints = TT._make_checkpoints(model)[:1]
    x_tr, y_tr, x_te, y_te = TT._make_data()
    train_dir, test_dir = tmp_path / "train_g", tmp_path / "test_g"
    TT._collect_to_disk(model, checkpoints, x_tr, y_tr, train_dir)
    TT._collect_to_disk(model, checkpoints, x_te, y_te, test_dir)
    train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(TT.N_TRAIN)]
    test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(TT.N_TEST)]
    return dict(train_dir=train_dir, test_dir=test_dir,
                train_hashes=train_hashes, test_hashes=test_hashes)


def _make(attr_cls, out_dir):
    args = AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return attr_cls(args, damping=DAMPING, layer_name=LAYERS)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


class TestKFAC:
    def test_matches_kronecker_oracle(self, collected, tmp_path):
        res = _make(KFACAttributor, tmp_path / "o").attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        matrix = res.query(collected["train_hashes"], collected["test_hashes"],
                           trajectory="agnostic")
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _kfac_oracle(tr_f, te_f, collected["train_hashes"],
                              collected["test_hashes"], DAMPING)
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_algorithm_label_and_shape(self, collected, tmp_path):
        res = _make(KFACAttributor, tmp_path / "o").attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert res.algorithm == "KFAC"
        assert res.scores.shape == (TT.N_TRAIN, TT.N_TEST)


class TestEKFAC:
    def test_matches_eigenbasis_oracle(self, collected, tmp_path):
        res = _make(EKFACAttributor, tmp_path / "o").attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        matrix = res.query(collected["train_hashes"], collected["test_hashes"],
                           trajectory="agnostic")
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _ekfac_oracle(tr_f, te_f, collected["train_hashes"],
                               collected["test_hashes"], DAMPING)
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_algorithm_label(self, collected, tmp_path):
        res = _make(EKFACAttributor, tmp_path / "o").attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert res.algorithm == "EKFAC"

    def test_heavy_damping_limit_is_gradient_dot(self, collected, tmp_path):
        """At large damping EK-FAC → (1/λ)·⟨∇W_tr, ∇W_te⟩ (the eigenvalue
        correction washes out), validating the rotate→divide→contract machinery."""
        damping = 1e6
        res = EKFACAttributor(
            AttributionArguments(output_dir=str(tmp_path / "o"),
                                 dataloader_num_workers=0, dataloader_pin_memory=False),
            damping=damping, layer_name=LAYERS,
        ).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        matrix = res.query(collected["train_hashes"], collected["test_hashes"],
                           trajectory="agnostic") * damping
        tr_f = _load_factors(collected["train_dir"], 0, LAYERS)
        te_f = _load_factors(collected["test_dir"], 0, LAYERS)
        oracle = _grad_dot_oracle(tr_f, te_f, collected["train_hashes"],
                                  collected["test_hashes"])
        assert torch.allclose(matrix, oracle, atol=1e-2, rtol=1e-2), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_transposed_projection_is_sign_sensitive(self):
        """Design rationale for the faithful projection ``U_Gᵀ ∇W U_A``: its
        score is invariant to an (arbitrary) eigenvector sign flip, whereas the
        transposed ``U_G ∇W U_Aᵀ`` (dattri's original) is not — which is why the
        'approx' mode was fixed to use the faithful projection too."""
        torch.manual_seed(0)
        B, out, inn = 30, 4, 5
        a, g = torch.randn(B, inn), torch.randn(B, out)
        A, G = a.T @ a / B, g.T @ g / B
        _, U_A, _, U_G = ops.kfac_eigh(A, G)

        def score(U_A, U_G, transposed):
            rot_a, rot_g = (
                (U_A.T.contiguous(), U_G.T.contiguous()) if transposed else (U_A, U_G)
            )
            M = ops.ekfac_materialize(a, g, "nn.Linear", rot_a, rot_g, include_bias=False)
            lam = (M * M).mean(0)
            return (M / (lam + 1e-3)) @ M.T

        U_G_flip = U_G.clone()
        U_G_flip[:, 0] *= -1  # a different but equally valid eigenbasis
        # Faithful projection: invariant.  Transposed (dattri): not.
        assert torch.allclose(score(U_A, U_G, False), score(U_A, U_G_flip, False), atol=1e-4)
        assert not torch.allclose(score(U_A, U_G, True), score(U_A, U_G_flip, True), atol=1e-4)

    def test_modes_agree(self, collected, tmp_path):
        """The fixed 'approx' mode now produces the same scores as 'exact'."""
        def run(mode):
            return EKFACAttributor(
                AttributionArguments(output_dir=str(tmp_path / mode),
                                     dataloader_num_workers=0, dataloader_pin_memory=False),
                damping=1e-2, layer_name=LAYERS, mode=mode,
            ).attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
            ).query(collected["train_hashes"], collected["test_hashes"], trajectory="agnostic")
        assert torch.allclose(run("exact"), run("approx"), atol=1e-6)

    def test_invalid_mode_raises(self, tmp_path):
        args = AttributionArguments(output_dir=str(tmp_path / "o"))
        with pytest.raises(ValueError, match=r"mode"):
            EKFACAttributor(args, mode="bogus")


class _MultiTokenModel(nn.Module):
    """Embedding → single bias-free Linear ``fc`` applied per token (B, T, ·)."""

    def __init__(self, vocab=16, d=4, h=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, h, bias=False)

    def forward(self, x, y=None):
        return self.fc(self.embedding(x))


def _kfac_oracle_mt(tr_f, te_f, train_hashes, test_hashes, layer, damping):
    a_tr, g_tr = _stack(tr_f, train_hashes, layer, 0), _stack(tr_f, train_hashes, layer, 1)
    a_te, g_te = _stack(te_f, test_hashes, layer, 0), _stack(te_f, test_hashes, layer, 1)
    n_tok = a_tr.shape[0] * a_tr.shape[1]
    a_flat = a_tr.reshape(-1, a_tr.shape[-1])
    g_flat = g_tr.reshape(-1, g_tr.shape[-1])
    A = a_flat.T @ a_flat / n_tok
    G = g_flat.T @ g_flat / n_tok
    A_inv = torch.linalg.inv(A + damping * torch.eye(A.shape[0]))
    G_inv = torch.linalg.inv(G + damping * torch.eye(G.shape[0]))
    dW_tr = torch.einsum("nto,nti->noi", g_tr, a_tr)  # Σ_t gₜ aₜᵀ
    dW_te = torch.einsum("mto,mti->moi", g_te, a_te)
    T = torch.einsum("oc,nci->noi", G_inv, dW_tr)
    T = torch.einsum("noj,jp->nop", T, A_inv)
    return torch.einsum("nop,mop->nm", T, dW_te)


class TestKFACMultiToken:
    def test_matches_oracle_with_token_dim(self, tmp_path):
        """KFAC's token-summed factorised path must match the explicit Σ_t oracle."""
        torch.manual_seed(0)
        model = _MultiTokenModel().eval()
        B, T = 5, 3
        gen = torch.Generator().manual_seed(1)
        x_tr = torch.randint(0, 16, (B, T), generator=gen)
        x_te = torch.randint(0, 16, (4, T), generator=gen)

        def collect(x, out_dir):
            fm = GradientFileManager(str(out_dir))
            hm = HookManager(model, config=HookManagerConfig(mlp_name_patterns=[r"fc"]),
                             callbacks=[OffloadCallback(offload_interval=1, file_manager=fm,
                                                        recording_type="per_sample")])
            with hm.collect():
                model.zero_grad(set_to_none=True)
                model(x=x).sum().backward()
            hm.remove()

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        collect(x_tr, train_dir)
        collect(x_te, test_dir)
        train_hashes = [hash_sample({"x": x_tr}, i) for i in range(B)]
        test_hashes = [hash_sample({"x": x_te}, j) for j in range(4)]

        args = AttributionArguments(output_dir=str(tmp_path / "o"),
                                    dataloader_num_workers=0, dataloader_pin_memory=False)
        res = KFACAttributor(args, damping=DAMPING, layer_name=["fc"]).attribute(
            train_gradients_dir=str(train_dir), test_gradients_dir=str(test_dir),
        )
        matrix = res.query(train_hashes, test_hashes, trajectory="agnostic")
        tr_f = _load_factors(train_dir, 0, ["fc"])
        te_f = _load_factors(test_dir, 0, ["fc"])
        oracle = _kfac_oracle_mt(tr_f, te_f, train_hashes, test_hashes, "fc", DAMPING)
        assert torch.allclose(matrix, oracle, atol=1e-4, rtol=1e-3), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )


class TestKroneckerShared:
    def test_missing_gradients_dir_raises(self, collected, tmp_path):
        attr = _make(KFACAttributor, tmp_path / "o")
        with pytest.raises(ValueError, match=r"train_gradients_dir"):
            attr.attribute(test_gradients_dir=str(collected["test_dir"]))

    @pytest.mark.parametrize("cls", [KFACAttributor, EKFACAttributor])
    def test_loop_over_test_matches_cached(self, collected, tmp_path, cls):
        """Streaming the test set (loop_over_test=True) gives identical scores to
        the cached path — same result, lower peak memory."""
        def run(loop, tag):
            return _make(cls, tmp_path / tag).attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
                loop_over_test=loop,
            ).query(collected["train_hashes"], collected["test_hashes"],
                    trajectory="agnostic")
        assert torch.allclose(run(False, "cached"), run(True, "loop"), atol=1e-5)

    def test_kfac_ekfac_agree_when_lambda_is_kronecker(self, collected, tmp_path):
        """Sanity bridge: EK-FAC with a *single*-token, rank-deficient setup still
        runs and returns finite, correctly-shaped scores for both attributors."""
        for cls, label in ((KFACAttributor, "KFAC"), (EKFACAttributor, "EKFAC")):
            res = _make(cls, tmp_path / label).attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
            )
            m = res.query(collected["train_hashes"], collected["test_hashes"],
                          trajectory="agnostic")
            assert m.shape == (TT.N_TRAIN, TT.N_TEST)
            assert torch.isfinite(m).all()
