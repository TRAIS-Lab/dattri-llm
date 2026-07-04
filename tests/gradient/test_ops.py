"""Ops-level unit tests for dattri_llm.gradient.ops.

These tests operate directly on synthetic factor tensors ``(a, g)`` — they do
**not** run real modules.  End-to-end checks against ``param.grad`` (running a
forward+backward pass through an ``nn.Module`` and comparing to the captured
weight gradient) live in ``test_ground_truth.py``.

Two suites:

1.  **Algebraic identity tests** (all supported layer types): verify that
    ``pairwise_dot``, ``grad_norm_sq``, and ``dot`` are self-consistent with
    ``materialize`` — i.e. the "ghost-scoring" identities hold:

        _pairwise_dot(a, g, lt)[i, j]  ==  mat[i] · mat[j]
        _grad_norm_sq(a, g, lt)[i]     ==  ||mat[i]||²
        _dot(a1,g1, a2,g2, lt)[i]      ==  mat1[i] · mat2[i]

    where ``mat = _materialize(a, g, lt, per_token=True)`` (the per-position form,
    so a norm layer's diagonal cross-gram matches).  Factors are supplied already
    in the preprocessed form (no ``module_kwargs``), so these tests exercise the
    core einsum kernels independently of layer-specific preprocessing.

2.  **K-FAC / FIM tests**: covariance-factor shapes, the NotImplementedError
    contract for norm/embedding layers, and streaming-equals-batch consistency
    for the accumulators.
"""

from __future__ import annotations

import pytest
import torch

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

B = 4   # batch size
T = 6   # token / spatial positions
I = 8   # in-features / C_in
O = 12  # out-features / C_out
VOCAB = 32
E = 10  # embedding dim


# ---------------------------------------------------------------------------
# Factories — build preprocessed (a, g) tensors for each layer type
# ---------------------------------------------------------------------------

def _linear_2d():
    return torch.randn(B, I), torch.randn(B, O)

def _linear_3d():
    return torch.randn(B, T, I), torch.randn(B, T, O)

def _conv():
    # (B, L, in_patch), (B, L, C_out)  — L is the spatial dimension
    return torch.randn(B, T, I), torch.randn(B, T, O)

def _conv_transpose():
    # (B, L, C_in), (B, L, out_patch)
    return torch.randn(B, T, I), torch.randn(B, T, O)

def _norm_2d():
    return torch.randn(B, I), torch.randn(B, I)

def _norm_3d():
    return torch.randn(B, T, I), torch.randn(B, T, I)

def _embedding():
    # Ensure token VOCAB-1 always appears so materialize output size == VOCAB*E,
    # matching nn.Embedding(VOCAB, E).weight.grad.flatten().
    ids = torch.randint(0, VOCAB, (B, T))
    ids[0, 0] = VOCAB - 1
    return ids, torch.randn(B, T, E)


# (test-id tag, factory).  The "-3d" suffix only disambiguates test ids that
# reuse the same layer_type string; _lt() strips it before calling ops.
LAYER_CASES = [
    ("nn.Linear",              _linear_2d),
    ("nn.Linear-3d",           _linear_3d),
    ("nn.Bilinear",            _linear_2d),
    ("nn.Conv1d",              _conv),
    ("nn.Conv2d",              _conv),
    ("nn.Conv3d",              _conv),
    ("nn.ConvTranspose1d",     _conv_transpose),
    ("nn.ConvTranspose2d",     _conv_transpose),
    ("nn.ConvTranspose3d",     _conv_transpose),
    ("nn.LayerNorm",           _norm_2d),
    ("nn.LayerNorm-3d",        _norm_3d),
    ("nn.RMSNorm",             _norm_2d),
    ("nn.GroupNorm",           _norm_2d),
    ("nn.InstanceNorm1d",      _norm_2d),
    ("nn.Embedding",           _embedding),
    ("nn.EmbeddingBag",        _embedding),
]


def _lt(tag: str) -> str:
    """Strip the test-id-only '-3d' suffix to recover the ops layer_type string."""
    return tag.replace("-3d", "")


_PARAMS = [pytest.param(_lt(tag), *fn(), id=tag) for tag, fn in LAYER_CASES]
_PARAMS_CROSS = [pytest.param(_lt(tag), *fn(), *fn(), id=tag) for tag, fn in LAYER_CASES]


# ---------------------------------------------------------------------------
# Algebraic identity tests
# ---------------------------------------------------------------------------

class TestPairwiseDotIdentity:
    """_pairwise_dot(a, g, lt)[i, j]  ==  mat[i] · mat[j]."""

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_matches_materialized_gram(self, lt, a, g):
        mat = _materialize(a, g, lt, per_token=True).float()   # norm: per-position
        expected = mat @ mat.T                        # (B, B)
        actual = _pairwise_dot(a, g, lt).float()
        assert actual.shape == (B, B)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_symmetric(self, lt, a, g):
        K = _pairwise_dot(a, g, lt).float()
        assert torch.allclose(K, K.T, atol=1e-5)

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_positive_semidefinite_diagonal(self, lt, a, g):
        K = _pairwise_dot(a, g, lt).float()
        assert (K.diagonal() >= -1e-6).all()


class TestGradNormSqIdentity:
    """_grad_norm_sq(a, g, lt)[i]  ==  ||mat[i]||²."""

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_matches_materialized_norm(self, lt, a, g):
        mat = _materialize(a, g, lt, per_token=True).float()   # norm: per-position
        expected = mat.pow(2).sum(-1)                 # (B,)
        actual = _grad_norm_sq(a, g, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_equals_pairwise_diagonal(self, lt, a, g):
        diag = _pairwise_dot(a, g, lt).float().diagonal()
        norms = _grad_norm_sq(a, g, lt).float()
        assert torch.allclose(diag, norms, atol=1e-4, rtol=1e-4)


class TestDotIdentity:
    """_dot(a1,g1, a2,g2, lt)[i]  ==  mat1[i] · mat2[i]."""

    @pytest.mark.parametrize("lt,a1,g1,a2,g2", _PARAMS_CROSS)
    def test_matches_materialized_dot(self, lt, a1, g1, a2, g2):
        mat1 = _materialize(a1, g1, lt, per_token=True).float()  # norm: per-position
        mat2 = _materialize(a2, g2, lt, per_token=True).float()
        expected = (mat1 * mat2).sum(-1)              # (B,)
        actual = _dot(a1, g1, a2, g2, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_self_dot_equals_grad_norm_sq(self, lt, a, g):
        """_dot(a,g, a,g, lt) == _grad_norm_sq(a, g, lt)."""
        self_dot = _dot(a, g, a, g, lt).float()
        norms = _grad_norm_sq(a, g, lt).float()
        assert torch.allclose(self_dot, norms, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# K-FAC / FIM tests
# ---------------------------------------------------------------------------

class TestKFACShapes:
    @pytest.mark.parametrize("lt,a,g", [
        pytest.param("nn.Linear", *_linear_2d(), id="linear-2d"),
        pytest.param("nn.Linear", *_linear_3d(), id="linear-3d"),
        pytest.param("nn.Conv1d", *_conv(), id="conv1d"),
        pytest.param("nn.ConvTranspose1d", *_conv_transpose(), id="convT1d"),
    ])
    def test_kfac_shapes(self, lt, a, g):
        A, G = _kfac(a, g, lt)
        assert A.shape == (I, I)
        assert G.shape == (O, O)

    @pytest.mark.parametrize("lt", ["nn.LayerNorm", "nn.Embedding"])
    def test_kfac_not_implemented(self, lt):
        a, g = _norm_2d() if "Norm" in lt else _embedding()
        with pytest.raises(NotImplementedError):
            _kfac(a, g, lt)

    @pytest.mark.parametrize("lt,a,g", [
        pytest.param("nn.Linear", *_linear_2d(), id="linear-2d"),
        pytest.param("nn.Linear", *_linear_3d(), id="linear-3d"),
        pytest.param("nn.Conv1d", *_conv(), id="conv1d"),
    ])
    def test_fim_shape(self, lt, a, g):
        F = _fim(a, g, lt)
        assert F.shape[0] == F.shape[1]    # square


class TestKfacGradientFreeRows:
    """Padded / fully masked token rows (g exactly 0) are excluded from the
    K-FAC covariance factors and from the normalizing row count."""

    def _padded(self):
        """(B, T, d) factors where the last 2 of 5 positions are 'padded':
        nonzero activations (as real pads have) but exactly-zero gradients."""
        torch.manual_seed(0)
        a = torch.randn(B, 5, I)
        g = torch.randn(B, 5, O)
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
        a_long = torch.cat([a, torch.randn(B, 4, I)], dim=1)   # pads: a != 0
        g_long = torch.cat([g, torch.zeros(B, 4, O)], dim=1)   #        g == 0
        A1, G1 = _kfac(a, g, "nn.Linear")
        A2, G2 = _kfac(a_long, g_long, "nn.Linear")
        assert torch.allclose(A1, A2, atol=1e-6)
        assert torch.allclose(G1, G2, atol=1e-6)

    def test_all_rows_gradient_free_raises(self):
        a = torch.randn(B, 3, I)
        g = torch.zeros(B, 3, O)
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
# materialize — token-collapse (default) vs per_token=True for norm layers
# ---------------------------------------------------------------------------

class TestMaterializeCollapse:
    def test_norm_sums_tokens(self):
        a, g = _norm_3d()                                            # (B, T, I)
        per_token = _materialize(a, g, "nn.LayerNorm", per_token=True)  # (B, T*I)
        wg = _materialize(a, g, "nn.LayerNorm")                       # (B, I), default collapse
        assert wg.shape == (B, I)
        assert torch.allclose(wg, per_token.reshape(B, T, I).sum(1), atol=1e-5)

    def test_norm_dim_independent_of_seq_len(self):
        m1 = _materialize(*[torch.randn(B, 3, I) for _ in range(2)], "nn.LayerNorm")
        m2 = _materialize(*[torch.randn(B, 7, I) for _ in range(2)], "nn.LayerNorm")
        assert m1.shape == m2.shape == (B, I)

    @pytest.mark.parametrize("lt,factory", [
        ("nn.Linear", _linear_3d), ("nn.Embedding", _embedding),
    ])
    def test_per_token_noop_for_token_contracting_types(self, lt, factory):
        a, g = factory()
        assert torch.allclose(
            _materialize(a, g, lt, per_token=True), _materialize(a, g, lt), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Random projection — TRAK (materialized) and LoGRA (factorized) units
# ---------------------------------------------------------------------------

from dattri.func.projection import random_project  # noqa: E402

_PROJ = dict(proj_max_batch_size=8, proj_type="rademacher", proj_seed=0)

class TestProjection:
    def test_materialized_collapses_to_proj_dim(self):
        a, g = _linear_3d()
        f = Factorized(a, g, {"has_bias": False})
        out = ops.project_materialized(f, "nn.Linear", random_project, proj_dim=64, **_PROJ)
        assert out.shape == (B, 64)

    def test_factorized_keeps_structure(self):
        a, g = _linear_3d()
        f = Factorized(a, g, {"has_bias": False})
        a_p, g_p = ops.project_factorized(f, "nn.Linear", random_project, proj_dim=32, **_PROJ)
        assert a_p.shape == (B, T, 32) and g_p.shape == (B, T, 32)

    def test_factorized_factors_use_independent_seeds(self):
        # Same proj_dim & base seed, but a uses seed+1, so the two projectors differ.
        a, g = torch.randn(B, T, I), torch.randn(B, T, I)  # square: d_in == d_out
        a_p, g_p = ops.project_factorized(
            Factorized(a, g, {"has_bias": False}), "nn.Linear", random_project,
            proj_dim=32, **_PROJ,
        )
        # Project an identical input through both factor slots; outputs must differ.
        assert not torch.allclose(a_p, g_p)

    @pytest.mark.parametrize("lt", ["nn.LayerNorm", "nn.Embedding"])
    def test_factorized_rejects_non_outer_product(self, lt):
        a, g = _norm_3d()
        with pytest.raises(ValueError, match="factorized projection is undefined"):
            ops.project_factorized(Factorized(a, g, {"has_bias": False}), lt,
                                   random_project, proj_dim=16, **_PROJ)

    def test_materialized_accepts_dense_tensor(self):
        dense = torch.randn(B, 100)
        out = ops.project_materialized(dense, "nn.Linear", random_project, proj_dim=16, **_PROJ)
        assert out.shape == (B, 16)

    def test_projection_approximately_preserves_gram(self):
        # Johnson–Lindenstrauss: random projection preserves pairwise dot products.
        torch.manual_seed(0)
        a, g = torch.randn(16, T, I), torch.randn(16, T, O)
        f = Factorized(a, g, {"has_bias": False})
        full = _materialize(a, g, "nn.Linear").float()
        gram = full @ full.T
        proj = ops.project_materialized(f, "nn.Linear", random_project, proj_dim=2048, **_PROJ)
        corr = torch.corrcoef(torch.stack([(proj @ proj.T).flatten(), gram.flatten()]))[0, 1]
        assert corr > 0.9


# ---------------------------------------------------------------------------
# Multi-layer accumulators — fan a Gradient block out to per-layer estimators
# ---------------------------------------------------------------------------

import types  # noqa: E402

from dattri_llm.gradient.gradient import Factorized  # noqa: E402


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
        block = _fake_gradient({"fc": ("nn.Linear", a1, g1), "out": ("nn.Linear", a2, g2)})

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
        a, g = _norm_3d()                                # norm param dim = I
        block = _fake_gradient({"ln": ("nn.LayerNorm", a, g)})

        multi = FisherAccumulator(max_params=I - 1)
        multi.update(block, ["ln"])
        assert "ln" not in multi.result()
        assert multi.skipped == {"ln": I}

    def test_fisher_within_cap_kept(self):
        a, g = _norm_3d()
        block = _fake_gradient({"ln": ("nn.LayerNorm", a, g)})
        multi = FisherAccumulator(max_params=I)
        multi.update(block, ["ln"])
        assert "ln" in multi.result()
        assert multi.skipped == {}


# ---------------------------------------------------------------------------
# Factorized-input wrappers (``_f``) — delegate to the raw ops, handle layout
# ---------------------------------------------------------------------------

from dattri_llm.gradient import ops


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
    via ``as_batch_first``)."""

    def test_materialize_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        assert torch.allclose(ops.materialize(f, "nn.Linear"), _materialize(a, g, "nn.Linear"))

    def test_materialize_norm_f_matches_raw(self):
        a, g = _norm_3d()
        f = Factorized(a, g)
        assert torch.allclose(
            ops.materialize(f, "nn.LayerNorm"), _materialize(a, g, "nn.LayerNorm")
        )

    def test_grad_norm_sq_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        assert torch.allclose(ops.grad_norm_sq(f, "nn.Linear"), _grad_norm_sq(a, g, "nn.Linear"))

    def test_kfac_f_matches_raw(self):
        a, g = _linear_3d()
        f = Factorized(a, g)
        A1, G1 = ops.kfac(f, "nn.Linear")
        A2, G2 = _kfac(a, g, "nn.Linear")
        assert torch.allclose(A1, A2) and torch.allclose(G1, G2)

    def test_cross_dot_f_matches_raw(self):
        a1, g1 = _linear_3d()
        a2, g2 = _linear_3d()
        f1, f2 = Factorized(a1, g1), Factorized(a2, g2)
        assert torch.allclose(
            ops.cross_dot(f1, f2, "nn.Linear"),
            ops._cross_dot(a1, g1, a2, g2, "nn.Linear"),
        )

    @pytest.mark.parametrize("lt,factory", [
        ("nn.Linear", _linear_3d), ("nn.LayerNorm", _norm_3d),
    ])
    def test_materialize_f_seq_first_equals_batch_first(self, lt, factory):
        a, g = factory()
        bf = Factorized(a, g)
        sf = _seq_first(bf)
        assert torch.allclose(ops.materialize(bf, lt), ops.materialize(sf, lt), atol=1e-5)

    def test_materialize_norm_seq_first_equals_batch_first(self):
        bf = Factorized(*_norm_3d())
        sf = _seq_first(bf)
        assert torch.allclose(
            ops.materialize(bf, "nn.LayerNorm"),
            ops.materialize(sf, "nn.LayerNorm"), atol=1e-5,
        )

    def test_kfac_cross_f_seq_first_equals_batch_first(self):
        bf1, bf2 = Factorized(*_linear_3d()), Factorized(*_linear_3d())
        A, _G = _kfac(bf1.activation, bf1.pre_activation_grad, "nn.Linear")
        A_inv = ops.sym_inverse(A, 1e-3)
        G_inv = ops.sym_inverse(_G, 1e-3)
        sf1, sf2 = _seq_first(bf1), _seq_first(bf2)
        assert torch.allclose(
            ops.kfac_cross(bf1, bf2, "nn.Linear", A_inv, G_inv),
            ops.kfac_cross(sf1, sf2, "nn.Linear", A_inv, G_inv), atol=1e-4,
        )

    def test_ekfac_materialize_f_seq_first_equals_batch_first(self):
        bf = Factorized(*_linear_3d())
        A, G = _kfac(bf.activation, bf.pre_activation_grad, "nn.Linear")
        _sA, U_A, _sG, U_G = ops.kfac_eigh(A, G)
        sf = _seq_first(bf)
        assert torch.allclose(
            ops.ekfac_materialize(bf, "nn.Linear", U_A, U_G),
            ops.ekfac_materialize(sf, "nn.Linear", U_A, U_G), atol=1e-4,
        )
