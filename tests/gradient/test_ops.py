"""Ops-level unit tests for dattri_llm.gradient.ops.

These tests operate directly on synthetic factor tensors ``(a, g)`` — they do
**not** run real modules.  End-to-end checks against ``param.grad`` (running a
forward+backward pass through an ``nn.Module`` and comparing to the captured
weight gradient) live in ``test_ground_truth.py``.

Two suites:

1.  **Algebraic identity tests** (all supported layer types): verify that
    ``pairwise_dot``, ``grad_norm_sq``, and ``dot`` are self-consistent with
    ``materialize`` — i.e. the "ghost-scoring" identities hold:

        pairwise_dot(a, g, lt)[i, j]  ==  mat[i] · mat[j]
        grad_norm_sq(a, g, lt)[i]     ==  ||mat[i]||²
        dot(a1,g1, a2,g2, lt)[i]      ==  mat1[i] · mat2[i]

    where ``mat = materialize(a, g, lt)``.  Factors are supplied already in the
    preprocessed form (no ``module_kwargs``), so these tests exercise the core
    einsum kernels independently of layer-specific preprocessing.

2.  **K-FAC / FIM tests**: covariance-factor shapes, the NotImplementedError
    contract for norm/embedding layers, and streaming-equals-batch consistency
    for the accumulators.
"""

from __future__ import annotations

import pytest
import torch

from dattri_llm.gradient.ops import (
    FIMAccumulator,
    KFACAccumulator,
    dot,
    fim,
    grad_norm_sq,
    kfac,
    materialize,
    pairwise_dot,
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
    """pairwise_dot(a, g, lt)[i, j]  ==  mat[i] · mat[j]."""

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_matches_materialized_gram(self, lt, a, g):
        mat = materialize(a, g, lt).float()          # (B, d)
        expected = mat @ mat.T                        # (B, B)
        actual = pairwise_dot(a, g, lt).float()
        assert actual.shape == (B, B)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_symmetric(self, lt, a, g):
        K = pairwise_dot(a, g, lt).float()
        assert torch.allclose(K, K.T, atol=1e-5)

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_positive_semidefinite_diagonal(self, lt, a, g):
        K = pairwise_dot(a, g, lt).float()
        assert (K.diagonal() >= -1e-6).all()


class TestGradNormSqIdentity:
    """grad_norm_sq(a, g, lt)[i]  ==  ||mat[i]||²."""

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_matches_materialized_norm(self, lt, a, g):
        mat = materialize(a, g, lt).float()          # (B, d)
        expected = mat.pow(2).sum(-1)                 # (B,)
        actual = grad_norm_sq(a, g, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_equals_pairwise_diagonal(self, lt, a, g):
        diag = pairwise_dot(a, g, lt).float().diagonal()
        norms = grad_norm_sq(a, g, lt).float()
        assert torch.allclose(diag, norms, atol=1e-4, rtol=1e-4)


class TestDotIdentity:
    """dot(a1,g1, a2,g2, lt)[i]  ==  mat1[i] · mat2[i]."""

    @pytest.mark.parametrize("lt,a1,g1,a2,g2", _PARAMS_CROSS)
    def test_matches_materialized_dot(self, lt, a1, g1, a2, g2):
        mat1 = materialize(a1, g1, lt).float()
        mat2 = materialize(a2, g2, lt).float()
        expected = (mat1 * mat2).sum(-1)              # (B,)
        actual = dot(a1, g1, a2, g2, lt).float()
        assert actual.shape == (B,)
        assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), \
            f"[{lt}] max diff {(expected - actual).abs().max():.2e}"

    @pytest.mark.parametrize("lt,a,g", _PARAMS)
    def test_self_dot_equals_grad_norm_sq(self, lt, a, g):
        """dot(a,g, a,g, lt) == grad_norm_sq(a, g, lt)."""
        self_dot = dot(a, g, a, g, lt).float()
        norms = grad_norm_sq(a, g, lt).float()
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
        A, G = kfac(a, g, lt)
        assert A.shape == (I, I)
        assert G.shape == (O, O)

    @pytest.mark.parametrize("lt", ["nn.LayerNorm", "nn.Embedding"])
    def test_kfac_not_implemented(self, lt):
        a, g = _norm_2d() if "Norm" in lt else _embedding()
        with pytest.raises(NotImplementedError):
            kfac(a, g, lt)

    @pytest.mark.parametrize("lt,a,g", [
        pytest.param("nn.Linear", *_linear_2d(), id="linear-2d"),
        pytest.param("nn.Linear", *_linear_3d(), id="linear-3d"),
        pytest.param("nn.Conv1d", *_conv(), id="conv1d"),
    ])
    def test_fim_shape(self, lt, a, g):
        F = fim(a, g, lt)
        assert F.shape[0] == F.shape[1]    # square


class TestStreamingAccumulators:
    def test_kfac_streaming_equals_batch(self):
        a1, g1 = _linear_2d()
        a2, g2 = _linear_2d()
        lt = "nn.Linear"

        A_b, G_b = kfac(torch.cat([a1, a2]), torch.cat([g1, g2]), lt)

        acc = KFACAccumulator()
        acc.update(a1, g1, lt)
        acc.update(a2, g2, lt)
        A_s, G_s = acc.result()

        assert torch.allclose(A_b, A_s, atol=1e-5)
        assert torch.allclose(G_b, G_s, atol=1e-5)

    def test_fim_streaming_equals_batch(self):
        a1, g1 = _linear_2d()
        a2, g2 = _linear_2d()
        lt = "nn.Linear"

        F_b = fim(torch.cat([a1, a2]), torch.cat([g1, g2]), lt)

        acc = FIMAccumulator()
        acc.update(a1, g1, lt)
        acc.update(a2, g2, lt)
        F_s = acc.result()

        assert torch.allclose(F_b, F_s, atol=1e-5)

    def test_reset_clears_state(self):
        acc = KFACAccumulator()
        a, g = _linear_2d()
        acc.update(a, g, "nn.Linear")
        acc.reset()
        with pytest.raises(RuntimeError):
            acc.result()
