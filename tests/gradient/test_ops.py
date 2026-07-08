"""Ops-level unit tests for dattri_llm.gradient.ops.

These tests operate directly on synthetic factor tensors ``(a, g)`` -- they do
**not** run real modules.  End-to-end checks against ``param.grad`` (running a
forward+backward pass through an ``nn.Module`` and comparing to the captured
weight gradient) live in ``test_ground_truth.py``.

Two suites:

1.  **Algebraic identity tests** (all supported layer types): verify that
    ``pairwise_dot``, ``grad_norm_sq``, and ``dot`` are self-consistent with
    ``materialize`` -- i.e. the "ghost-scoring" identities hold:

        _pairwise_dot(a, g, lt)[i, j]  ==  mat[i] * mat[j]
        _grad_norm_sq(a, g, lt)[i]     ==  ||mat[i]||^2
        _dot(a1,g1, a2,g2, lt)[i]      ==  mat1[i] * mat2[i]

    where ``mat = _materialize(a, g, lt)`` is the true per-sample weight
    gradient (positions summed -- for norm layers this includes the
    cross-position terms).  Factors are supplied already in the preprocessed
    form (no ``module_kwargs``), so these tests exercise the core einsum
    kernels independently of layer-specific preprocessing.

2.  **K-FAC / FIM tests**: covariance-factor shapes, the NotImplementedError
    contract for norm/embedding layers, and streaming-equals-batch consistency
    for the accumulators.
"""

from __future__ import annotations

import types

import pytest
import torch
from dattri.func.projection import random_project

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized
from dattri_llm.gradient.ops import (
    FisherAccumulator,
    KroneckerAccumulator,
    LayerFisherAccumulator,
    LayerKroneckerAccumulator,
    _dot,
    _fim,
    _grad_norm_sq,
    _kfac,
    _materialize,
    _pairwise_dot,
)

# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------

B = 4  # batch size
T = 6  # token / spatial positions
D_IN = 8  # in-features / C_in
D_OUT = 12  # out-features / C_out
VOCAB = 32
E = 10  # embedding dim


# ---------------------------------------------------------------------------
# Factories -- build preprocessed (a, g) tensors for each layer type
# ---------------------------------------------------------------------------


def _linear_2d():
    return torch.randn(B, D_IN), torch.randn(B, D_OUT)


def _linear_3d():
    return torch.randn(B, T, D_IN), torch.randn(B, T, D_OUT)


def _conv():
    # (B, L, in_patch), (B, L, C_out)  -- L is the spatial dimension
    return torch.randn(B, T, D_IN), torch.randn(B, T, D_OUT)


def _conv_transpose():
    # (B, L, C_in), (B, L, out_patch)
    return torch.randn(B, T, D_IN), torch.randn(B, T, D_OUT)


def _norm_2d():
    return torch.randn(B, D_IN), torch.randn(B, D_IN)


def _norm_3d():
    return torch.randn(B, T, D_IN), torch.randn(B, T, D_IN)


def _embedding():
    ids = torch.randint(0, VOCAB, (B, T))
    return ids, torch.randn(B, T, E)


# Embedding materialization requires the layer's true vocab size (the width
# cannot be inferred from the factors); both embedding-family test cases hold
# per-token Embedding-form data, so they share one kwargs dict.
_EMB_MK = {"has_bias": False, "num_embeddings": VOCAB, "padding_idx": None}


def _mat(a, g, lt, **kw):
    """``_materialize`` with the kwargs the strict embedding API requires."""
    if lt in ("nn.Embedding", "nn.EmbeddingBag"):
        return _materialize(a, g, "nn.Embedding", module_kwargs=_EMB_MK, **kw)
    return _materialize(a, g, lt, **kw)


# (test-id tag, factory).  The "-3d" suffix only disambiguates test ids that
# reuse the same layer_type string; _lt() strips it before calling ops.
LAYER_CASES = [
    ("nn.Linear", _linear_2d),
    ("nn.Linear-3d", _linear_3d),
    ("nn.Bilinear", _linear_2d),
    ("nn.Conv1d", _conv),
    ("nn.Conv2d", _conv),
    ("nn.Conv3d", _conv),
    ("nn.ConvTranspose1d", _conv_transpose),
    ("nn.ConvTranspose2d", _conv_transpose),
    ("nn.ConvTranspose3d", _conv_transpose),
    ("nn.LayerNorm", _norm_2d),
    ("nn.LayerNorm-3d", _norm_3d),
    ("nn.RMSNorm", _norm_2d),
    ("nn.GroupNorm", _norm_2d),
    ("nn.InstanceNorm1d", _norm_2d),
    ("nn.Embedding", _embedding),
    ("nn.EmbeddingBag", _embedding),
]


def _lt(tag: str) -> str:
    """Strip the test-id-only '-3d' suffix to recover the ops layer_type string."""
    return tag.replace("-3d", "")


_PARAMS = [pytest.param(_lt(tag), *fn(), id=tag) for tag, fn in LAYER_CASES]
_PARAMS_CROSS = [
    pytest.param(_lt(tag), *fn(), *fn(), id=tag) for tag, fn in LAYER_CASES
]


# ---------------------------------------------------------------------------
# Algebraic identity tests
# ---------------------------------------------------------------------------


class TestPairwiseDotIdentity:
    """_pairwise_dot(a, g, lt)[i, j]  ==  mat[i] * mat[j]."""

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_matches_materialized_gram(self, lt, a, g):
        mat = _mat(a, g, lt).float()  # true per-sample weight grads
        expected = mat @ mat.T  # (B, B)
        actual = _pairwise_dot(a, g, lt).float()
        assert actual.shape == (B, B)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), (
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"
        )

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_symmetric(self, lt, a, g):
        K = _pairwise_dot(a, g, lt).float()
        assert torch.allclose(K, K.T, atol=1e-5)

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_positive_semidefinite_diagonal(self, lt, a, g):
        K = _pairwise_dot(a, g, lt).float()
        assert (K.diagonal() >= -1e-6).all()


class TestGradNormSqIdentity:
    """_grad_norm_sq(a, g, lt)[i]  ==  ||mat[i]||^2."""

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_matches_materialized_norm(self, lt, a, g):
        mat = _mat(a, g, lt).float()  # true per-sample weight grads
        expected = mat.pow(2).sum(-1)  # (B,)
        actual = _grad_norm_sq(a, g, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), (
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"
        )

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_equals_pairwise_diagonal(self, lt, a, g):
        diag = _pairwise_dot(a, g, lt).float().diagonal()
        norms = _grad_norm_sq(a, g, lt).float()
        assert torch.allclose(diag, norms, atol=1e-4, rtol=1e-4)


class TestDotIdentity:
    """_dot(a1,g1, a2,g2, lt)[i]  ==  mat1[i] * mat2[i]."""

    @pytest.mark.parametrize(("lt", "a1", "g1", "a2", "g2"), _PARAMS_CROSS)
    def test_matches_materialized_dot(self, lt, a1, g1, a2, g2):
        mat1 = _mat(a1, g1, lt).float()  # true per-sample weight grads
        mat2 = _mat(a2, g2, lt).float()
        expected = (mat1 * mat2).sum(-1)  # (B,)
        actual = _dot(a1, g1, a2, g2, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), (
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"
        )

    @pytest.mark.parametrize(("lt", "a", "g"), _PARAMS)
    def test_self_dot_equals_grad_norm_sq(self, lt, a, g):
        """_dot(a,g, a,g, lt) == _grad_norm_sq(a, g, lt)."""
        self_dot = _dot(a, g, a, g, lt).float()
        norms = _grad_norm_sq(a, g, lt).float()
        assert torch.allclose(self_dot, norms, atol=1e-4, rtol=1e-4)

    def test_embedding_padding_idx_matches_autograd(self):
        """With padding_idx, _materialize and _dot must match autograd, whose
        embedding backward zeroes the pad row of weight.grad (regression: pad
        positions' contributions were included).
        """
        from torch import nn

        pad = 0
        emb = nn.Embedding(VOCAB, E, padding_idx=pad)
        ids = torch.randint(1, VOCAB, (B, T))
        ids[:, -2:] = pad  # padded tail
        g = torch.randn(B, T, E)
        mk = {"has_bias": False, "num_embeddings": VOCAB, "padding_idx": pad}

        # Autograd ground truth: per-sample weight gradients.
        true_grads = []
        for i in range(B):
            emb.weight.grad = None
            (emb(ids[i : i + 1]) * g[i : i + 1]).sum().backward()
            true_grads.append(emb.weight.grad.detach().flatten())
        true_mat = torch.stack(true_grads).float()  # (B, VOCAB * E)

        mat = _materialize(ids, g, "nn.Embedding", module_kwargs=mk).float()
        assert torch.allclose(mat, true_mat, atol=1e-5), (
            f"max diff {(mat - true_mat).abs().max():.2e}"
        )

        expected = (true_mat * true_mat).sum(-1)
        actual = _dot(
            ids,
            g,
            ids,
            g,
            "nn.Embedding",
            module_kwargs1=mk,
            module_kwargs2=mk,
        ).float()
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), (
            f"max diff {(expected - actual).abs().max():.2e}"
        )

    @pytest.mark.parametrize("mode", ["sum", "mean"])
    def test_embedding_bag_padding_idx_matches_autograd(self, mode):
        """With padding_idx, the bag expansion must exclude pad tokens and,
        for mode='mean', divide by each bag's non-pad count (regression: pad
        positions received gradient and the divisor was the full T).
        """
        from torch import nn

        pad = 0
        bag = nn.EmbeddingBag(VOCAB, E, mode=mode, padding_idx=pad)
        ids = torch.randint(1, VOCAB, (B, T))
        ids[:, -2:] = pad  # padded tail, varying real content per bag
        ids[1, -3] = pad  # one bag with a different non-pad count
        g = torch.randn(B, E)  # per-bag output gradient
        mk = {
            "has_bias": False,
            "num_embeddings": VOCAB,
            "mode": mode,
            "padding_idx": pad,
        }

        # Autograd ground truth: per-bag weight gradients.
        true_grads = []
        for i in range(B):
            bag.weight.grad = None
            (bag(ids[i : i + 1]) * g[i : i + 1]).sum().backward()
            true_grads.append(bag.weight.grad.detach().flatten())
        true_mat = torch.stack(true_grads).float()  # (B, VOCAB * E)

        mat = _materialize(ids, g, "nn.EmbeddingBag", module_kwargs=mk).float()
        assert torch.allclose(mat, true_mat, atol=1e-5), (
            f"[{mode}] max diff {(mat - true_mat).abs().max():.2e}"
        )

    def test_embedding_width_is_batch_independent(self):
        """Materialized embedding gradients are always (B, num_embeddings * E),
        whatever token range a batch touches, so cross-batch dots are
        well-defined (regression: width was max(token) + 1 per batch and
        mixed-width embedding dots crashed on a shape mismatch).
        """
        ids_lo = torch.randint(0, 5, (B, T))  # small token ids only
        ids_hi = torch.randint(VOCAB - 5, VOCAB, (B, T))  # large ids only
        g_lo, g_hi = torch.randn(B, T, E), torch.randn(B, T, E)
        m_lo = _materialize(ids_lo, g_lo, "nn.Embedding", module_kwargs=_EMB_MK)
        m_hi = _materialize(ids_hi, g_hi, "nn.Embedding", module_kwargs=_EMB_MK)
        assert m_lo.shape == m_hi.shape == (B, VOCAB * E)
        # Disjoint token ranges -> exactly orthogonal weight gradients.
        assert torch.equal((m_lo * m_hi).sum(-1), torch.zeros(B))
        # Declared width must bound the captured ids -- corrupt kwargs raise.
        with pytest.raises(ValueError, match="out of range"):
            _materialize(
                ids_hi,
                g_hi,
                "nn.Embedding",
                module_kwargs={
                    "has_bias": False,
                    "num_embeddings": 5,
                    "padding_idx": None,
                },
            )

    def test_embedding_different_token_counts(self):
        """Embedding _dot with T1 != T2 (regression: side 2's scatter used
        side 1's token count and crashed on mismatched sequence lengths).
        """
        t1, t2 = T, T + 3
        ids1 = torch.randint(0, VOCAB, (B, t1))
        ids2 = torch.randint(0, VOCAB, (B, t2))
        g1 = torch.randn(B, t1, E)
        g2 = torch.randn(B, t2, E)
        mat1 = _materialize(ids1, g1, "nn.Embedding", module_kwargs=_EMB_MK).float()
        mat2 = _materialize(ids2, g2, "nn.Embedding", module_kwargs=_EMB_MK).float()
        expected = (mat1 * mat2).sum(-1)  # (B,)
        actual = _dot(ids1, g1, ids2, g2, "nn.Embedding").float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), (
            f"max diff {(expected - actual).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# K-FAC / FIM tests
# ---------------------------------------------------------------------------


class TestKFACShapes:
    @pytest.mark.parametrize(
        ("lt", "a", "g"),
        [
            pytest.param("nn.Linear", *_linear_2d(), id="linear-2d"),
            pytest.param("nn.Linear", *_linear_3d(), id="linear-3d"),
            pytest.param("nn.Conv1d", *_conv(), id="conv1d"),
            pytest.param("nn.ConvTranspose1d", *_conv_transpose(), id="convT1d"),
        ],
    )
    def test_kfac_shapes(self, lt, a, g):
        A, G = _kfac(a, g, lt)
        assert A.shape == (D_IN, D_IN)
        assert G.shape == (D_OUT, D_OUT)

    @pytest.mark.parametrize("lt", ["nn.LayerNorm", "nn.Embedding"])
    def test_kfac_not_implemented(self, lt):
        a, g = _norm_2d() if "Norm" in lt else _embedding()
        with pytest.raises(NotImplementedError):
            _kfac(a, g, lt)

    @pytest.mark.parametrize(
        ("lt", "a", "g"),
        [
            pytest.param("nn.Linear", *_linear_2d(), id="linear-2d"),
            pytest.param("nn.Linear", *_linear_3d(), id="linear-3d"),
            pytest.param("nn.Conv1d", *_conv(), id="conv1d"),
        ],
    )
    def test_fim_shape(self, lt, a, g):
        F = _fim(a, g, lt)
        assert F.shape[0] == F.shape[1]  # square


class TestKfacGradientFreeRows:
    """Padded / fully masked token rows (g exactly 0) are excluded from the
    K-FAC covariance factors and from the normalizing row count.
    """

    def _padded(self):
        """(B, T, d) factors where the last 2 of 5 positions are 'padded':

        nonzero activations (as real pads have) but exactly-zero gradients.
        """
        torch.manual_seed(0)
        a = torch.randn(B, 5, D_IN)
        g = torch.randn(B, 5, D_OUT)
        g[:, 3:] = 0.0
        return a, g

    def test_padded_rows_excluded(self):
        a, g = self._padded()
        A_pad, G_pad = _kfac(a, g, "nn.Linear")
        # Reference: the same data with the padded positions physically removed.
        A_ref, G_ref = _kfac(a[:, :3], g[:, :3], "nn.Linear")
        assert torch.allclose(A_pad, A_ref, atol=1e-6)
        assert torch.allclose(G_pad, G_ref, atol=1e-6)

    def test_accumulator_matches_one_shot(self):
        a, g = self._padded()
        acc = LayerKroneckerAccumulator()
        acc.update(a, g, "nn.Linear")
        A_s, G_s = acc.result()
        A_b, G_b = _kfac(a, g, "nn.Linear")
        assert torch.allclose(A_s, A_b, atol=1e-6)
        assert torch.allclose(G_s, G_b, atol=1e-6)

    def test_padding_invariance(self):
        # Padding the same sequences to a longer length must not change (A, G).
        a, g = self._padded()
        a_long = torch.cat([a, torch.randn(B, 4, D_IN)], dim=1)  # pads: a != 0
        g_long = torch.cat([g, torch.zeros(B, 4, D_OUT)], dim=1)  # g == 0
        A1, G1 = _kfac(a, g, "nn.Linear")
        A2, G2 = _kfac(a_long, g_long, "nn.Linear")
        assert torch.allclose(A1, A2, atol=1e-6)
        assert torch.allclose(G1, G2, atol=1e-6)

    def test_all_rows_gradient_free_raises(self):
        a = torch.randn(B, 3, D_IN)
        g = torch.zeros(B, 3, D_OUT)
        with pytest.raises(ValueError, match="zero"):
            _kfac(a, g, "nn.Linear")
        acc = LayerKroneckerAccumulator()
        acc.update(a, g, "nn.Linear")
        with pytest.raises(RuntimeError, match="gradient-carrying"):
            acc.result()


class TestStreamingAccumulators:
    def test_kfac_streaming_equals_batch(self):
        a1, g1 = _linear_2d()
        a2, g2 = _linear_2d()
        lt = "nn.Linear"

        A_b, G_b = _kfac(torch.cat([a1, a2]), torch.cat([g1, g2]), lt)

        acc = LayerKroneckerAccumulator()
        acc.update(a1, g1, lt)
        acc.update(a2, g2, lt)
        A_s, G_s = acc.result()

        assert torch.allclose(A_b, A_s, atol=1e-5)
        assert torch.allclose(G_b, G_s, atol=1e-5)

    def test_fim_streaming_equals_batch(self):
        a1, g1 = _linear_2d()
        a2, g2 = _linear_2d()
        lt = "nn.Linear"

        F_b = _fim(torch.cat([a1, a2]), torch.cat([g1, g2]), lt)

        acc = LayerFisherAccumulator()
        acc.update(a1, g1, lt)
        acc.update(a2, g2, lt)
        F_s = acc.result()

        assert torch.allclose(F_b, F_s, atol=1e-5)

    def test_fim_update_from_grad_equals_update(self):
        """update_from_grad on the materialized gradient matches update()."""
        a, g = _linear_2d()
        lt = "nn.Linear"

        ref = LayerFisherAccumulator()
        ref.update(a, g, lt)

        direct = LayerFisherAccumulator()
        direct.update_from_grad(_materialize(a, g, lt))

        assert torch.allclose(ref.result(), direct.result(), atol=1e-6)

    def test_reset_clears_state(self):
        acc = LayerKroneckerAccumulator()
        a, g = _linear_2d()
        acc.update(a, g, "nn.Linear")
        acc.reset()
        with pytest.raises(RuntimeError):
            acc.result()


# ---------------------------------------------------------------------------
# materialize -- token-collapse (default) vs per_token=True for norm layers
# ---------------------------------------------------------------------------


class TestMaterializeCollapse:
    def test_norm_sums_tokens(self):
        a, g = _norm_3d()  # (B, T, I)
        per_token = _materialize(a, g, "nn.LayerNorm", per_token=True)  # (B, T*I)
        wg = _materialize(a, g, "nn.LayerNorm")  # (B, I), default collapse
        assert wg.shape == (B, D_IN)
        assert torch.allclose(wg, per_token.reshape(B, T, D_IN).sum(1), atol=1e-5)

    def test_norm_dim_independent_of_seq_len(self):
        m1 = _materialize(*[torch.randn(B, 3, D_IN) for _ in range(2)], "nn.LayerNorm")
        m2 = _materialize(*[torch.randn(B, 7, D_IN) for _ in range(2)], "nn.LayerNorm")
        assert m1.shape == m2.shape == (B, D_IN)

    @pytest.mark.parametrize(
        ("lt", "factory"),
        [
            ("nn.Linear", _linear_3d),
            ("nn.Embedding", _embedding),
        ],
    )
    def test_per_token_noop_for_token_contracting_types(self, lt, factory):
        a, g = factory()
        assert torch.allclose(
            _mat(a, g, lt, per_token=True),
            _mat(a, g, lt),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Random projection -- TRAK (materialized) and LoGRA (factorized) units
# ---------------------------------------------------------------------------

_PROJ = {"proj_max_batch_size": 8, "proj_type": "rademacher", "proj_seed": 0}


class TestProjection:
    def test_materialized_collapses_to_proj_dim(self):
        a, g = _linear_3d()
        f = Factorized(a, g, {"has_bias": False})
        out = ops.project_materialized(
            f,
            "nn.Linear",
            random_project,
            proj_dim=64,
            **_PROJ,
        )
        assert out.shape == (B, 64)

    def test_factorized_keeps_structure(self):
        a, g = _linear_3d()
        f = Factorized(a, g, {"has_bias": False})
        a_p, g_p = ops.project_factorized(
            f,
            "nn.Linear",
            random_project,
            proj_dim=32,
            **_PROJ,
        )
        assert a_p.shape == (B, T, 32)
        assert g_p.shape == (B, T, 32)

    def test_factorized_factors_use_independent_seeds(self):
        # Same proj_dim & base seed, but a uses seed+1, so the two projectors differ.
        a, g = torch.randn(B, T, D_IN), torch.randn(B, T, D_IN)  # square: d_in == d_out
        a_p, g_p = ops.project_factorized(
            Factorized(a, g, {"has_bias": False}),
            "nn.Linear",
            random_project,
            proj_dim=32,
            **_PROJ,
        )
        # Project an identical input through both factor slots; outputs must differ.
        assert not torch.allclose(a_p, g_p)

    def test_factorized_rejects_non_outer_product(self):
        # Norm gradients are diagonal (a * g), not an outer product.
        a, g = _norm_3d()
        with pytest.raises(ValueError, match="factorized projection is undefined"):
            ops.project_factorized(
                Factorized(a, g, {"has_bias": False}),
                "nn.LayerNorm",
                random_project,
                proj_dim=16,
                **_PROJ,
            )

    def test_factorized_embedding_matches_projected_true_gradient(self):
        """Embedding LoGRA projection (one-hot inputs): materializing the
        projected factors as a linear layer must equal ``G^T dW^T A`` computed
        from the true per-sample embedding gradient and the projectors' own
        matrices (recovered by projecting identity bases).
        """
        proj_dim = 16
        ids, g = _embedding()
        a_p, g_p = ops.project_factorized(
            Factorized(ids, g, _EMB_MK),
            "nn.Embedding",
            random_project,
            proj_dim=proj_dim,
            **_PROJ,
        )
        assert a_p.shape == (B, T, proj_dim)
        assert g_p.shape == (B, T, proj_dim)

        # Recover the two projection matrices by projecting identity bases
        # with the same seeds (_project_factorized uses seed for g, seed+1
        # for a).
        seed = _PROJ["proj_seed"]
        kw = {k: v for k, v in _PROJ.items() if k != "proj_seed"}
        A = ops._apply_projector(
            random_project,
            torch.eye(VOCAB),
            proj_dim=proj_dim,
            proj_seed=seed + 1,
            **kw,
        )  # (VOCAB, p)
        G = ops._apply_projector(
            random_project,
            torch.eye(E),
            proj_dim=proj_dim,
            proj_seed=seed,
            **kw,
        )  # (E, p)

        # True per-sample embedding gradient dW_i is (VOCAB, E); the projected
        # factorized gradient must be G^T dW_i^T A, materialized linear-style.
        got = _materialize(a_p, g_p, "nn.Linear").float()  # (B, p*p)
        dw = _mat(ids, g, "nn.Embedding").reshape(B, VOCAB, E).float()
        expected = torch.einsum("ep,bve,vq->bpq", G, dw, A).reshape(B, -1)
        assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (
            f"max diff {(got - expected).abs().max():.2e}"
        )

    def test_materialized_accepts_dense_tensor(self):
        dense = torch.randn(B, 100)
        out = ops.project_materialized(
            dense,
            "nn.Linear",
            random_project,
            proj_dim=16,
            **_PROJ,
        )
        assert out.shape == (B, 16)

    def test_projection_approximately_preserves_gram(self):
        # Johnson-Lindenstrauss: random projection preserves pairwise dot products.
        torch.manual_seed(0)
        a, g = torch.randn(16, T, D_IN), torch.randn(16, T, D_OUT)
        f = Factorized(a, g, {"has_bias": False})
        full = _materialize(a, g, "nn.Linear").float()
        gram = full @ full.T
        proj = ops.project_materialized(
            f,
            "nn.Linear",
            random_project,
            proj_dim=2048,
            **_PROJ,
        )
        corr = torch.corrcoef(torch.stack([(proj @ proj.T).flatten(), gram.flatten()]))[
            0,
            1,
        ]
        assert corr > 0.9


# ---------------------------------------------------------------------------
# Multi-layer accumulators -- fan a Gradient block out to per-layer estimators
# ---------------------------------------------------------------------------


def _fake_gradient(layers):
    """``layers`` maps name -> (layer_type, a, g)."""
    return types.SimpleNamespace(
        data={n: Factorized(a, g) for n, (_lt, a, g) in layers.items()},
        layer_types={n: lt for n, (lt, _a, _g) in layers.items()},
    )


class TestMultiLayerAccumulators:
    def test_kronecker_matches_per_layer(self):
        a1, g1 = _linear_2d()
        a2, g2 = _linear_2d()
        block = _fake_gradient(
            {"fc": ("nn.Linear", a1, g1), "out": ("nn.Linear", a2, g2)},
        )

        multi = KroneckerAccumulator()
        multi.update(block, ["fc", "out"])
        res = multi.result()

        for name, (a, g) in {"fc": (a1, g1), "out": (a2, g2)}.items():
            ref = LayerKroneckerAccumulator()
            ref.update(a, g, "nn.Linear")
            A_ref, G_ref = ref.result()
            assert torch.allclose(res[name][0], A_ref, atol=1e-6)
            assert torch.allclose(res[name][1], G_ref, atol=1e-6)

    def test_fisher_matches_per_layer(self):
        a, g = _norm_3d()
        block = _fake_gradient({"ln": ("nn.LayerNorm", a, g)})

        multi = FisherAccumulator()
        multi.update(block, ["ln"])

        ref = LayerFisherAccumulator()
        ref.update(a, g, "nn.LayerNorm")
        assert torch.allclose(multi.result()["ln"], ref.result(), atol=1e-6)

    def test_fisher_skips_layers_over_cap(self):
        a, g = _norm_3d()  # norm param dim = I
        block = _fake_gradient({"ln": ("nn.LayerNorm", a, g)})

        multi = FisherAccumulator(max_params=D_IN - 1)
        multi.update(block, ["ln"])
        assert "ln" not in multi.result()
        assert multi.skipped == {"ln": D_IN}

    def test_fisher_within_cap_kept(self):
        a, g = _norm_3d()
        block = _fake_gradient({"ln": ("nn.LayerNorm", a, g)})
        multi = FisherAccumulator(max_params=D_IN)
        multi.update(block, ["ln"])
        assert "ln" in multi.result()
        assert multi.skipped == {}


# ---------------------------------------------------------------------------
# Factorized-input wrappers (``_f``) -- delegate to the raw ops, handle layout
# ---------------------------------------------------------------------------


def _seq_first(f: Factorized) -> Factorized:
    """A sequence-first twin of a batch-first ``(B, T, ...)`` factor pair."""
    return Factorized(
        f.activation.transpose(0, 1).contiguous(),
        f.pre_activation_grad.transpose(0, 1).contiguous(),
        module_kwargs=f.module_kwargs,
        batch_first=False,
    )


class TestFactorizedWrappers:
    """Each ``_f`` wrapper equals its raw counterpart on batch-first input, and
    produces the same result for a sequence-first capture (handled internally
    via ``as_batch_first``).
    """

    def test_materialize_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        assert torch.allclose(
            ops.materialize(f, "nn.Linear"),
            _materialize(a, g, "nn.Linear"),
        )

    def test_materialize_norm_f_matches_raw(self):
        a, g = _norm_3d()
        f = Factorized(a, g)
        assert torch.allclose(
            ops.materialize(f, "nn.LayerNorm"),
            _materialize(a, g, "nn.LayerNorm"),
        )

    def test_grad_norm_sq_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        assert torch.allclose(
            ops.grad_norm_sq(f, "nn.Linear"),
            _grad_norm_sq(a, g, "nn.Linear"),
        )

    def test_kfac_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        A1, G1 = ops.kfac(f, "nn.Linear")
        A2, G2 = _kfac(a, g, "nn.Linear")
        assert torch.allclose(A1, A2)
        assert torch.allclose(G1, G2)

    def test_cross_dot_f_matches_raw(self):
        a1, g1 = _linear_3d()
        a2, g2 = _linear_3d()
        f1, f2 = Factorized(a1, g1), Factorized(a2, g2)
        assert torch.allclose(
            ops.cross_dot(f1, f2, "nn.Linear"),
            ops._cross_dot(a1, g1, a2, g2, "nn.Linear"),
        )

    @pytest.mark.parametrize(
        ("lt", "factory"),
        [
            ("nn.Linear", _linear_3d),
            ("nn.LayerNorm", _norm_3d),
        ],
    )
    def test_materialize_f_seq_first_equals_batch_first(self, lt, factory):
        a, g = factory()
        bf = Factorized(a, g)
        sf = _seq_first(bf)
        assert torch.allclose(
            ops.materialize(bf, lt),
            ops.materialize(sf, lt),
            atol=1e-5,
        )

    def test_materialize_norm_seq_first_equals_batch_first(self):
        bf = Factorized(*_norm_3d())
        sf = _seq_first(bf)
        assert torch.allclose(
            ops.materialize(bf, "nn.LayerNorm"),
            ops.materialize(sf, "nn.LayerNorm"),
            atol=1e-5,
        )

    def test_kfac_cross_f_seq_first_equals_batch_first(self):
        bf1, bf2 = Factorized(*_linear_3d()), Factorized(*_linear_3d())
        A, _G = _kfac(bf1.activation, bf1.pre_activation_grad, "nn.Linear")
        A_inv = ops.sym_inverse(A, 1e-3)
        G_inv = ops.sym_inverse(_G, 1e-3)
        sf1, sf2 = _seq_first(bf1), _seq_first(bf2)
        assert torch.allclose(
            ops.kfac_cross(bf1, bf2, "nn.Linear", A_inv, G_inv),
            ops.kfac_cross(sf1, sf2, "nn.Linear", A_inv, G_inv),
            atol=1e-4,
        )

    def test_ekfac_materialize_f_seq_first_equals_batch_first(self):
        bf = Factorized(*_linear_3d())
        A, G = _kfac(bf.activation, bf.pre_activation_grad, "nn.Linear")
        _sA, U_A, _sG, U_G = ops.kfac_eigh(A, G)
        sf = _seq_first(bf)
        assert torch.allclose(
            ops.ekfac_materialize(bf, "nn.Linear", U_A, U_G),
            ops.ekfac_materialize(sf, "nn.Linear", U_A, U_G),
            atol=1e-4,
        )
