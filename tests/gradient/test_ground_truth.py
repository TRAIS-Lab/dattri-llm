"""Ground-truth tests: HookManager factorized gradients vs param.grad.

For each supported layer type the test:

1. Generates B fixed single-sample inputs.
2. Runs each input individually through a small wrapper model and records
   ``weight.grad`` (and ``bias.grad`` when relevant) as the ground truth.
3. Concatenates the B inputs into one batch and runs a single
   forward+backward pass under ``HookManager``, collecting ``Factorized``
   gradients.
4. Calls ``materialize``, ``grad_norm_sq``, and ``pairwise_dot`` on the
   Factorized data and asserts they match the per-sample reference values.

Layer types covered
-------------------
* ``nn.Linear``                           -- 2-D (no-bias, with-bias), 3-D token
* ``nn.NonDynamicallyQuantizableLinear``  -- same math as nn.Linear (no-bias, with-bias)
* ``transformers.pytorch_utils.Conv1D``   -- weight stored transposed (I, O);
  always has bias
* ``nn.Embedding``
* ``nn.EmbeddingBag``                     -- mode='sum', mode='mean'
* ``nn.LayerNorm``                        -- no-bias, with-bias
* ``nn.RMSNorm``                          -- affine weight (no bias in PyTorch)
* ``nn.GroupNorm``                        -- no-bias, with-bias (spatial > 1)
* ``nn.InstanceNorm2d``                   -- no-bias, with-bias (spatial > 1)
* ``nn.Conv1d``                           -- no-bias, with-bias
* ``nn.Conv2d``                           -- no-bias, with-bias
* ``nn.Conv3d``                           -- no-bias, with-bias
* ``nn.ConvTranspose1d/2d/3d``            -- no-bias, with-bias

Notes on normalization-layer ground truth
------------------------------------------
Per the factorized-gradient formulation, norm layers use the *per-position*
(diagonal) convention: ``materialize`` returns the un-summed per-position
products, and summing them over positions recovers the true ``weight.grad``.
For LayerNorm/RMSNorm tested with a single position (2-D input) this equals
``param.grad`` directly.  GroupNorm/InstanceNorm intrinsically have spatial
positions, so their tests check ``materialize`` summed-over-positions against
``param.grad``, and check ``grad_norm_sq`` / ``pairwise_dot`` against an
independent per-position reference (x_hat from the affine-free functional norm,
g controlled by an output multiplier).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import (
    HookManager,
    HookManagerCallback,
    HookManagerConfig,
)

try:
    from transformers.pytorch_utils import Conv1D as _HF_Conv1D

    _HAS_HF = True
except ImportError:
    _HF_Conv1D = None
    _HAS_HF = False

_HAS_RMSNORM = hasattr(nn, "RMSNorm")


# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------

B = 4  # batch size for the batch pass
T = 5  # token / sequence length
D_IN = 8  # in-features / C_in
D_OUT = 6  # out-features / C_out
VOCAB = 32
E = 10  # embedding dimension


# ---------------------------------------------------------------------------
# Tiny wrapper models (layer inside MLP-named submodule or always-hooked type)
# ---------------------------------------------------------------------------


class _LinearModel(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _NonDynQuantLinearModel(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = False) -> None:
        super().__init__()
        self.mlp = NonDynamicallyQuantizableLinear(in_f, out_f, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _HFConv1DModel(nn.Module):
    """HF Conv1D wrapper -- weight is (nx, nf) = (I, O), bias is always (nf,) = (O,)."""

    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        # Conv1D(nf, nx): weight shape (nx, nf) = (in_f, out_f)
        self.mlp = _HF_Conv1D(out_f, in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _EmbeddingModel(nn.Module):
    def __init__(self, vocab: int, emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class _LayerNormModel(nn.Module):
    def __init__(self, size: int, bias: bool = False) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(size, elementwise_affine=True, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class _Conv1dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _Conv2dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _Conv3dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Conv3d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _RMSNormModel(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.ln = nn.RMSNorm(size, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class _QKNormModel(nn.Module):
    """RMSNorm over the last dim of a 4-D ``(B, T, heads, head_dim)`` input --
    the Qwen3 QK-norm pattern.  The ``(head_dim,)`` weight is shared across
    heads, so the head axis is a broadcast axis that the weight gradient sums
    over, exactly like the token axis.
    """

    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.ln = nn.RMSNorm(head_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        return self.ln(x.view(b, t, self.heads, self.head_dim)).flatten(2)


class _ChannelNormModel(nn.Module):
    """Wraps a per-channel norm (GroupNorm/InstanceNorm) and scales the output
    by a fixed buffer ``R`` so the captured output gradient is non-trivial and
    identical across the per-sample reference and batch passes (g == R).
    """

    def __init__(self, norm: nn.Module, r_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.norm = norm
        torch.manual_seed(7)
        self.register_buffer("R", torch.randn(1, *r_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * self.R


class _ConvTranspose1dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.ConvTranspose1d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _ConvTranspose2dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _ConvTranspose3dModel(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.mlp = nn.ConvTranspose3d(c_in, c_out, kernel_size=k, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _EmbeddingBagModel(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, mode: str) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, emb_dim, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CaptureCB(HookManagerCallback):
    def __init__(self) -> None:
        self.records: list = []

    def on_step_end(self, record) -> None:
        self.records.append(record)


def _run_batch(
    model: nn.Module,
    x: torch.Tensor,
    name_patterns: list[str] | None = None,
):
    """One HookManager-collected batch pass; returns the assembled Gradient.

    ``name_patterns`` is forwarded as the ``linear_io`` selector to restrict
    collection to specific layers.  Pass a pattern like ``["mlp"]`` to target a
    single layer instead of hooking every linear-family layer (the default).
    """
    cb = _CaptureCB()
    config = (
        HookManagerConfig(linear_io=name_patterns)
        if name_patterns is not None
        else None
    )
    mgr = HookManager(model, config=config, callbacks=[cb])
    with mgr.collect():
        model.zero_grad()
        model(x).sum().backward()
    mgr.remove()
    assert cb.records, "HookManager did not fire on_step_end"
    return cb.records[0].gradient


def _per_sample_weight_grads(
    model: nn.Module,
    layer_attr: str,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Run each input individually; return [weight.grad.flatten(), ...] in order."""
    layer = getattr(model, layer_attr)
    grads: list[torch.Tensor] = []
    for x in inputs:
        model.zero_grad()
        model(x).sum().backward()
        grads.append(layer.weight.grad.detach().clone().flatten())
    return grads


def _per_sample_bias_grads(
    model: nn.Module,
    layer_attr: str,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Run each input individually; return [bias.grad.flatten(), ...] in order."""
    layer = getattr(model, layer_attr)
    grads: list[torch.Tensor] = []
    for x in inputs:
        model.zero_grad()
        model(x).sum().backward()
        grads.append(layer.bias.grad.detach().clone().flatten())
    return grads


def _per_sample_hf_weight_grads(
    model: nn.Module,
    layer_attr: str,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """For HF Conv1D: weight is (I, O); return weight.grad.T.flatten() = (O*I)
    to match materialize.
    """
    layer = getattr(model, layer_attr)
    grads: list[torch.Tensor] = []
    for x in inputs:
        model.zero_grad()
        model(x).sum().backward()
        grads.append(layer.weight.grad.detach().clone().T.flatten())
    return grads


def _channel_norm_diag_grad(
    xhat: torch.Tensor,
    R: torch.Tensor,
    has_bias: bool,
) -> torch.Tensor:
    """Independent per-position (diagonal) gradient for a per-channel norm.

    ``xhat`` and ``R`` are ``(1, C, *spatial)`` (R == the captured output grad
    g).  Returns the flattened per-position gradient vector ``[x_hat*g | g]`` laid
    out position-major to match the library's ``materialize`` ordering.
    """
    c = xhat.shape[1]
    gamma = (xhat * R).reshape(c, -1).permute(1, 0)  # (S, C)
    parts = [gamma]
    if has_bias:
        parts.append(R.reshape(c, -1).permute(1, 0))  # (S, C)  beta contrib = 1*g
    return torch.cat(parts, dim=-1).flatten()  # (S * feat,)


# Thin wrappers that call ops on a Gradient layer, passing module_kwargs through.


def _materialize(gradient, name: str, include_bias: bool = True) -> torch.Tensor:
    lt = gradient.layer_types[name]
    # per_token=True keeps norm layers' un-summed per-position products, which the
    # norm ground-truth tests sum over positions to recover param.grad.
    return ops.materialize(
        gradient.data[name],
        lt,
        include_bias,
        per_token=True,
    ).float()


def _grad_norm_sq(gradient, name: str, include_bias: bool = True) -> torch.Tensor:
    lt = gradient.layer_types[name]
    return ops.grad_norm_sq(gradient.data[name], lt, include_bias).float()


def _pairwise_dot(gradient, name: str, include_bias: bool = True) -> torch.Tensor:
    lt = gradient.layer_types[name]
    return ops.pairwise_dot(gradient.data[name], lt, include_bias).float()


# ---------------------------------------------------------------------------
# nn.Linear
# ---------------------------------------------------------------------------


class TestLinearGroundTruth:
    """nn.Linear -- 2-D input (no-bias), 2-D with-bias, 3-D token input."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _LinearModel(D_IN, D_OUT, bias=False)

    # -- 2-D input: (B, I) --------------------------------------------------

    def _inputs_2d(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN) for _ in range(B)]

    def test_materialize_2d(self) -> None:
        inputs = self._inputs_2d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq_2d(self) -> None:
        inputs = self._inputs_2d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot_2d(self) -> None:
        inputs = self._inputs_2d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3), (
                    f"K[{i},{j}] expected {expected:.4f} got {K[i, j]:.4f}"
                )

    # -- 3-D input: (B, T, I) -- token sequence; materialize sums over T -----

    def _inputs_3d(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, T, D_IN) for _ in range(B)]

    def test_materialize_3d(self) -> None:
        inputs = self._inputs_3d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq_3d(self) -> None:
        inputs = self._inputs_3d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot_3d(self) -> None:
        inputs = self._inputs_3d()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    # -- 2-D with bias -------------------------------------------------------

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _LinearModel(D_IN, D_OUT, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        """Bias augmentation: mat[i].reshape(O, I+1)[:, :I] == weight.grad
        and mat[i].reshape(O, I+1)[:, I] == bias.grad.
        """
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            m = mat[i].reshape(D_OUT, D_IN + 1)
            assert torch.allclose(m[:, :D_IN].flatten(), w_ref[i].float(), atol=1e-4)
            assert torch.allclose(m[:, D_IN], b_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.NonDynamicallyQuantizableLinear
# ---------------------------------------------------------------------------


class TestNonDynQuantLinearGroundTruth:
    """nn.NonDynamicallyQuantizableLinear -- behaves identically to nn.Linear."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _NonDynQuantLinearModel(D_IN, D_OUT, bias=False)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4)

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _NonDynQuantLinearModel(D_IN, D_OUT, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            m = mat[i].reshape(D_OUT, D_IN + 1)
            assert torch.allclose(m[:, :D_IN].flatten(), w_ref[i].float(), atol=1e-4)
            assert torch.allclose(m[:, D_IN], b_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# transformers.pytorch_utils.Conv1D
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_HF, reason="transformers not installed")
class TestHFConv1DGroundTruth:
    """transformers.pytorch_utils.Conv1D -- weight stored as (nx, nf)=(I, O), transposed
    relative to nn.Linear.  Bias is always present (fixed in the constructor).

    materialize produces (B, O*(I+1)) where
    mat[i].reshape(O, I+1)[:, :I] == weight.grad.T
    and mat[i].reshape(O, I+1)[:, I] == bias.grad.
    """

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _HFConv1DModel(D_IN, D_OUT)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        w_ref = _per_sample_hf_weight_grads(self.model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            m = mat[i].reshape(D_OUT, D_IN + 1)
            assert torch.allclose(m[:, :D_IN].flatten(), w_ref[i].float(), atol=1e-4), (
                f"weight part mismatch at sample {i}"
            )
            assert torch.allclose(m[:, D_IN], b_ref[i].float(), atol=1e-4), (
                f"bias part mismatch at sample {i}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        w_ref = _per_sample_hf_weight_grads(self.model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        w_ref = _per_sample_hf_weight_grads(self.model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.Embedding
# ---------------------------------------------------------------------------


class TestEmbeddingGroundTruth:
    """nn.Embedding -- token IDs as activation, grad w.r.t. output as grad."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _EmbeddingModel(VOCAB, E)

    def _inputs(self) -> list[torch.Tensor]:
        """Each sample contains VOCAB-1 so materialize output == VOCAB*E."""
        torch.manual_seed(1)
        out = []
        for _ in range(B):
            ids = torch.randint(0, VOCAB, (1, T))
            ids[0, 0] = VOCAB - 1
            out.append(ids)
        return out

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-5), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        norms = _grad_norm_sq(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        K = _pairwise_dot(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# nn.LayerNorm
# ---------------------------------------------------------------------------


class TestLayerNormGroundTruth:
    """nn.LayerNorm -- 2-D input (no-bias and with-bias).

    The forward hook delivers the *raw* (pre-normalization) input; module_kwargs
    carries ``normalized_shape`` and ``eps`` so that ``preprocess_factorized``
    computes x_hat = (x - mu)/sqrt(sigma^2+eps) internally before ``materialize``.

    For 2-D inputs the materialize output equals ``weight.grad`` directly
    (per-sample, no token summing needed).
    """

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _LayerNormModel(D_IN, bias=False)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        norms = _grad_norm_sq(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        K = _pairwise_dot(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-4)

    # -- with bias ----------------------------------------------------------

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _LayerNormModel(D_IN, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        """With bias, preprocess_factorized returns a_aug=(B,2I), g_aug=(B,2I).

        mat[i][:I] == weight.grad  and  mat[i][I:] == bias.grad.
        """
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "ln", inputs)
        b_ref = _per_sample_bias_grads(model, "ln", inputs)
        mat = _materialize(_run_batch(model, torch.cat(inputs)), "ln")  # (B, 2*I)
        for i in range(B):
            assert torch.allclose(mat[i][:D_IN], w_ref[i].float(), atol=1e-4), (
                f"weight grad mismatch sample {i}"
            )
            assert torch.allclose(mat[i][D_IN:], b_ref[i].float(), atol=1e-4), (
                f"bias grad mismatch sample {i}"
            )

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "ln", inputs)
        b_ref = _per_sample_bias_grads(model, "ln", inputs)
        norms = _grad_norm_sq(_run_batch(model, torch.cat(inputs)), "ln")
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "ln", inputs)
        b_ref = _per_sample_bias_grads(model, "ln", inputs)
        K = _pairwise_dot(_run_batch(model, torch.cat(inputs)), "ln")
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# nn.Conv1d
# ---------------------------------------------------------------------------


class TestConv1dGroundTruth:
    """nn.Conv1d -- raw (N, C_in, L) hook data; im2col unfolds before materialize."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _Conv1dModel(D_IN, D_OUT, k=3, bias=False)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 10) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    # -- with bias ----------------------------------------------------------

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _Conv1dModel(D_IN, D_OUT, k=3, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN, 10) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        """mat[i].reshape(C_out, patch+1)[:, :patch] == weight.grad,
        mat[i].reshape(C_out, patch+1)[:, patch] == bias.grad.
        """
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        patch = D_IN * 3  # C_in * k
        for i in range(B):
            m = mat[i].reshape(D_OUT, patch + 1)
            assert torch.allclose(m[:, :patch].flatten(), w_ref[i].float(), atol=1e-4)
            assert torch.allclose(m[:, patch], b_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.Conv2d
# ---------------------------------------------------------------------------


class TestConv2dGroundTruth:
    """nn.Conv2d -- raw (N, C_in, H, W) hook data; im2col unfolds before materialize."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _Conv2dModel(D_IN, D_OUT, k=2, bias=False)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 6, 6) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    # -- with bias ----------------------------------------------------------

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _Conv2dModel(D_IN, D_OUT, k=2, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN, 6, 6) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        """mat[i].reshape(C_out, patch+1)[:, :patch] == weight.grad,
        mat[i].reshape(C_out, patch+1)[:, patch] == bias.grad.
        """
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        patch = D_IN * 2 * 2  # C_in * kH * kW
        for i in range(B):
            m = mat[i].reshape(D_OUT, patch + 1)
            assert torch.allclose(m[:, :patch].flatten(), w_ref[i].float(), atol=1e-4)
            assert torch.allclose(m[:, patch], b_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.Conv3d
# ---------------------------------------------------------------------------


class TestConv3dGroundTruth:
    """nn.Conv3d -- raw (N, C_in, D, H, W) hook data; 3-D im2col before materialize."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _Conv3dModel(D_IN, D_OUT, k=2, bias=False)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 4, 4, 4) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    # -- with bias ----------------------------------------------------------

    def _bias_model_and_inputs(self):
        torch.manual_seed(0)
        model = _Conv3dModel(D_IN, D_OUT, k=2, bias=True)
        torch.manual_seed(1)
        inputs = [torch.randn(1, D_IN, 4, 4, 4) for _ in range(B)]
        return model, inputs

    def test_materialize_with_bias(self) -> None:
        """mat[i].reshape(C_out, patch+1)[:, :patch] == weight.grad,
        mat[i].reshape(C_out, patch+1)[:, patch] == bias.grad.
        """
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        patch = D_IN * 2 * 2 * 2  # C_in * kD * kH * kW
        for i in range(B):
            m = mat[i].reshape(D_OUT, patch + 1)
            assert torch.allclose(m[:, :patch].flatten(), w_ref[i].float(), atol=1e-4)
            assert torch.allclose(m[:, patch], b_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model, inputs = self._bias_model_and_inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.RMSNorm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RMSNORM, reason="nn.RMSNorm requires PyTorch >= 2.4")
class TestRMSNormGroundTruth:
    """nn.RMSNorm -- 2-D input, affine weight (PyTorch RMSNorm has no bias).

    The forward hook delivers the raw input; module_kwargs carries
    ``normalized_shape``/``eps`` so preprocess computes x_hat = x/sqrt(mean(x^2)+eps).
    With a single position the materialize output equals ``weight.grad``.
    """

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _RMSNormModel(D_IN)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN) for _ in range(B)]

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        norms = _grad_norm_sq(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        K = _pairwise_dot(_run_batch(self.model, torch.cat(inputs)), "ln")
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# nn.RMSNorm over a 4-D per-head input (Qwen3-style QK-norm)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RMSNORM, reason="nn.RMSNorm requires PyTorch >= 2.4")
class TestQKNormGroundTruth:
    """nn.RMSNorm fed ``(B, T, heads, head_dim)`` -- the extra head axis is a
    broadcast axis and must be folded into the positions, not flattened into
    the features (see ``_fold_broadcast_axes``).  Regression test for the
    Qwen3 QK-norm case, where the per-sample gradient previously came out
    ``(heads, head_dim)``-shaped and pairwise dots silently dropped the
    cross-head terms.
    """

    HEADS, HEAD_DIM = 3, 4

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _QKNormModel(self.HEADS, self.HEAD_DIM)
        with torch.no_grad():  # non-unit weight so x_hat scaling matters
            self.model.ln.weight.copy_(torch.randn(self.HEAD_DIM).abs() + 0.5)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, T, self.HEADS * self.HEAD_DIM) for _ in range(B)]

    def test_materialize_per_token(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "ln")
        # per_token=True flattens the folded (T*heads) positions with the
        # features; positions sum back to weight.grad.
        assert mat.shape == (B, T * self.HEADS * self.HEAD_DIM)
        for i in range(B):
            got = mat[i].reshape(T * self.HEADS, self.HEAD_DIM).sum(0)
            assert torch.allclose(got, ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(got - ref[i]).abs().max():.2e}"
            )

    def test_materialize_summed(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "ln", inputs)
        gradient = _run_batch(self.model, torch.cat(inputs))
        mat = ops.materialize(
            gradient.data["ln"],
            gradient.layer_types["ln"],
        ).float()
        # per_token=False contracts every broadcast axis (tokens AND heads)
        # straight to the weight gradient.
        assert mat.shape == (B, self.HEAD_DIM)
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )


# ---------------------------------------------------------------------------
# Shared driver for per-channel norms (GroupNorm / InstanceNorm)
# ---------------------------------------------------------------------------


class _ChannelNormChecks:
    """Mixin providing materialize / grad_norm_sq / pairwise_dot checks for a
    per-channel norm wrapped in :class:`_ChannelNormModel`.

    Subclasses set ``self.model`` (a ``_ChannelNormModel``), ``self.C`` (number
    of channels), ``self.xhat_fn`` (callable ``x -> x_hat`` via the affine-free
    functional norm), and provide ``_inputs()``.
    """

    def _xhat(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _diag(self, x: torch.Tensor, has_bias: bool) -> torch.Tensor:
        R = self.model.R
        return _channel_norm_diag_grad(self._xhat(x), R, has_bias)

    # -- with bias (affine=True, bias folded) --------------------------------

    def test_materialize_with_bias(self) -> None:
        inputs = self._inputs()
        w_ref = _per_sample_weight_grads(self.model, "norm", inputs)
        b_ref = _per_sample_bias_grads(self.model, "norm", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "norm")
        C = self.C
        for i in range(B):
            summed = mat[i].reshape(-1, 2 * C).sum(0)  # sum over positions
            assert torch.allclose(summed[:C], w_ref[i].float(), atol=1e-4), (
                f"gamma mismatch sample {i}"
            )
            assert torch.allclose(summed[C:], b_ref[i].float(), atol=1e-4), (
                f"beta mismatch sample {i}"
            )

    def test_grad_norm_sq_with_bias(self) -> None:
        inputs = self._inputs()
        norms = _grad_norm_sq(_run_batch(self.model, torch.cat(inputs)), "norm")
        for i in range(B):
            ref = self._diag(inputs[i], has_bias=True).pow(2).sum()
            assert torch.allclose(norms[i], ref, atol=1e-4), (
                f"sample {i}: {norms[i]:.4f} vs {ref:.4f}"
            )

    def test_pairwise_dot_with_bias(self) -> None:
        inputs = self._inputs()
        K = _pairwise_dot(_run_batch(self.model, torch.cat(inputs)), "norm")
        diags = [self._diag(x, has_bias=True) for x in inputs]
        for i in range(B):
            for j in range(B):
                expected = (diags[i] * diags[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)

    # -- no bias (include_bias=False: only the gamma gradient is kept) --------

    def test_materialize_no_bias(self) -> None:
        inputs = self._inputs()
        w_ref = _per_sample_weight_grads(self.model, "norm", inputs)
        mat = _materialize(
            _run_batch(self.model, torch.cat(inputs)),
            "norm",
            include_bias=False,
        )
        C = self.C
        for i in range(B):
            summed = mat[i].reshape(-1, C).sum(0)
            assert torch.allclose(summed, w_ref[i].float(), atol=1e-4)

    def test_grad_norm_sq_no_bias(self) -> None:
        inputs = self._inputs()
        norms = _grad_norm_sq(
            _run_batch(self.model, torch.cat(inputs)),
            "norm",
            include_bias=False,
        )
        for i in range(B):
            ref = self._diag(inputs[i], has_bias=False).pow(2).sum()
            assert torch.allclose(norms[i], ref, atol=1e-4)

    def test_pairwise_dot_no_bias(self) -> None:
        inputs = self._inputs()
        K = _pairwise_dot(
            _run_batch(self.model, torch.cat(inputs)),
            "norm",
            include_bias=False,
        )
        diags = [self._diag(x, has_bias=False) for x in inputs]
        for i in range(B):
            for j in range(B):
                expected = (diags[i] * diags[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# nn.GroupNorm
# ---------------------------------------------------------------------------


class TestGroupNormGroundTruth(_ChannelNormChecks):
    """nn.GroupNorm -- input (N, C, L) with spatial L > 1; 2 groups over C=8."""

    NUM_GROUPS = 2
    L = 5

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.C = D_IN
        norm = nn.GroupNorm(self.NUM_GROUPS, self.C, affine=True)
        self.model = _ChannelNormModel(norm, (self.C, self.L))

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, self.C, self.L) for _ in range(B)]

    def _xhat(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x,
            self.NUM_GROUPS,
            weight=None,
            bias=None,
            eps=self.model.norm.eps,
        )


# ---------------------------------------------------------------------------
# nn.InstanceNorm2d
# ---------------------------------------------------------------------------


class TestInstanceNorm2dGroundTruth(_ChannelNormChecks):
    """nn.InstanceNorm2d -- input (N, C, H, W); per-channel normalization."""

    H = 3
    W = 3

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.C = 4
        norm = nn.InstanceNorm2d(self.C, affine=True)
        self.model = _ChannelNormModel(norm, (self.C, self.H, self.W))

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, self.C, self.H, self.W) for _ in range(B)]

    def _xhat(self, x: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(x, eps=self.model.norm.eps)


# ---------------------------------------------------------------------------
# nn.ConvTranspose1d / 2d / 3d
# ---------------------------------------------------------------------------


class _ConvTransposeChecks:
    """Mixin: weight (and optional bias) ground-truth checks for a transposed
    convolution at ``self.model.mlp``.  ``self.P`` is C_out*prodkernel and
    ``self.C_in`` the number of input channels.
    """

    def _inputs(self) -> list[torch.Tensor]:
        raise NotImplementedError

    def test_materialize(self) -> None:
        model = self.make_model(bias=False)
        inputs = self._inputs()
        ref = _per_sample_weight_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-4), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        model = self.make_model(bias=False)
        inputs = self._inputs()
        ref = _per_sample_weight_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        model = self.make_model(bias=False)
        inputs = self._inputs()
        ref = _per_sample_weight_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3, rtol=1e-3)

    # -- with bias: augmented factors (L+1, C_in+1) x (L+1, P+C_out) ----------

    def test_materialize_with_bias(self) -> None:
        model = self.make_model(bias=True)
        inputs = self._inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        mat = _materialize(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        C_in, P, C_out = self.C_in, self.P, self.C_out
        for i in range(B):
            m = mat[i].reshape(C_in + 1, P + C_out)
            assert torch.allclose(
                m[:C_in, :P].flatten(),
                w_ref[i].float(),
                atol=1e-4,
            ), f"weight mismatch sample {i}"
            assert torch.allclose(m[C_in, P:], b_ref[i].float(), atol=1e-4), (
                f"bias mismatch sample {i}"
            )

    def test_grad_norm_sq_with_bias(self) -> None:
        model = self.make_model(bias=True)
        inputs = self._inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        norms = _grad_norm_sq(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        for i in range(B):
            ref_norm = w_ref[i].float().pow(2).sum() + b_ref[i].float().pow(2).sum()
            assert torch.allclose(norms[i], ref_norm, atol=1e-4)

    def test_pairwise_dot_with_bias(self) -> None:
        model = self.make_model(bias=True)
        inputs = self._inputs()
        w_ref = _per_sample_weight_grads(model, "mlp", inputs)
        b_ref = _per_sample_bias_grads(model, "mlp", inputs)
        K = _pairwise_dot(
            _run_batch(model, torch.cat(inputs), name_patterns=["mlp"]),
            "mlp",
        )
        full_ref = [
            torch.cat([w.float(), b.float()]) for w, b in zip(w_ref, b_ref, strict=True)
        ]
        for i in range(B):
            for j in range(B):
                expected = (full_ref[i] * full_ref[j]).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-3, rtol=1e-3)


class TestConvTranspose1dGroundTruth(_ConvTransposeChecks):
    K = 3

    def setup_method(self) -> None:
        self.C_in, self.C_out = D_IN, D_OUT
        self.P = D_OUT * self.K

    def make_model(self, bias: bool) -> nn.Module:
        torch.manual_seed(0)
        return _ConvTranspose1dModel(D_IN, D_OUT, k=self.K, bias=bias)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 6) for _ in range(B)]


class TestConvTranspose2dGroundTruth(_ConvTransposeChecks):
    K = 2

    def setup_method(self) -> None:
        self.C_in, self.C_out = D_IN, D_OUT
        self.P = D_OUT * self.K * self.K

    def make_model(self, bias: bool) -> nn.Module:
        torch.manual_seed(0)
        return _ConvTranspose2dModel(D_IN, D_OUT, k=self.K, bias=bias)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 4, 4) for _ in range(B)]


class TestConvTranspose3dGroundTruth(_ConvTransposeChecks):
    K = 2

    def setup_method(self) -> None:
        self.C_in, self.C_out = D_IN, D_OUT
        self.P = D_OUT * self.K * self.K * self.K

    def make_model(self, bias: bool) -> nn.Module:
        torch.manual_seed(0)
        return _ConvTranspose3dModel(D_IN, D_OUT, k=self.K, bias=bias)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        return [torch.randn(1, D_IN, 3, 3, 3) for _ in range(B)]


# ---------------------------------------------------------------------------
# nn.EmbeddingBag
# ---------------------------------------------------------------------------


class _EmbeddingBagChecks:
    """Mixin: EmbeddingBag ground-truth checks for a given reduction mode.

    Each sample contains VOCAB-1 so the materialized output spans VOCAB*E.
    """

    MODE = "sum"

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _EmbeddingBagModel(VOCAB, E, mode=self.MODE)

    def _inputs(self) -> list[torch.Tensor]:
        torch.manual_seed(1)
        out = []
        for _ in range(B):
            ids = torch.randint(0, VOCAB, (1, T))
            ids[0, 0] = VOCAB - 1
            out.append(ids)
        return out

    def test_materialize(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        mat = _materialize(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            assert torch.allclose(mat[i], ref[i].float(), atol=1e-5), (
                f"sample {i} max diff {(mat[i] - ref[i]).abs().max():.2e}"
            )

    def test_grad_norm_sq(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        norms = _grad_norm_sq(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            assert torch.allclose(norms[i], ref[i].float().pow(2).sum(), atol=1e-4)

    def test_pairwise_dot(self) -> None:
        inputs = self._inputs()
        ref = _per_sample_weight_grads(self.model, "emb", inputs)
        K = _pairwise_dot(_run_batch(self.model, torch.cat(inputs)), "emb")
        for i in range(B):
            for j in range(B):
                expected = (ref[i].float() * ref[j].float()).sum()
                assert torch.allclose(K[i, j], expected, atol=1e-4)


class TestEmbeddingBagSumGroundTruth(_EmbeddingBagChecks):
    MODE = "sum"


class TestEmbeddingBagMeanGroundTruth(_EmbeddingBagChecks):
    MODE = "mean"
