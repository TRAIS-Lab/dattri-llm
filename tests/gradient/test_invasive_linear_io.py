"""Consistency and correctness tests for the ``invasive_linear_io`` hook type.

``invasive_linear_io`` behaves exactly like ``linear_io`` (same captured
factorized gradient, same materialized weight gradient, same projection
behaviour, same step bookkeeping) EXCEPT that it overrides each ``nn.Linear``'s
forward with a custom autograd ``Function`` that skips the weight/bias gradient.
These tests pin that equivalence in every configuration, plus the invasive-only
behaviours (no ``weight.grad``, forward restored on remove, gated by collection).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.hooks import (
    INVASIVE_LINEAR_IO,
    LINEAR_IO,
    HookManager,
    HookManagerConfig,
)
from dattri_llm.gradient.hooks.config import resolve_hook_assignments


class _CaptureCB(HookManagerCallback):
    def __init__(self) -> None:
        self.records: list = []

    def on_step_end(self, record) -> None:
        self.records.append(record)


class _MLP(nn.Module):
    """Two linears (one with bias, one without) around a nonlinearity."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 4, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _seeded_mlp() -> _MLP:
    torch.manual_seed(1234)
    m = _MLP()
    for p in m.parameters():
        nn.init.normal_(p)
    return m


def _collect(model: nn.Module, x: torch.Tensor, **hook_kw):
    """Run one collected fwd/bwd; return (assembled Gradient, model)."""
    cb = _CaptureCB()
    mgr = HookManager(model, config=HookManagerConfig(**hook_kw), callbacks=[cb])
    with mgr.collect():
        model.zero_grad(set_to_none=True)
        model(x).pow(2).sum().backward()
    mgr.remove()
    assert cb.records, "HookManager did not fire on_step_end"
    return cb.records[-1].gradient, model


def _assert_gradients_identical(g_ref, g_inv) -> None:
    ref, inv = g_ref.materialize(), g_inv.materialize()
    assert sorted(ref.data) == sorted(inv.data)
    for layer in ref.data:
        r, v = ref.data[layer], inv.data[layer]
        assert r.shape == v.shape, layer
        assert torch.allclose(r, v, atol=1e-6, rtol=1e-5), (
            layer,
            (r - v).abs().max().item(),
        )


# --------------------------------------------------------------------------- #
# 1. Captured gradient is identical to linear_io                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(5, 8), (3, 4, 8)], ids=["2d", "3d"])
def test_invasive_matches_linear_io_unprojected(shape):
    x = torch.randn(*shape)
    g_lin, _ = _collect(_seeded_mlp(), x, linear_io=[r"fc"])
    g_inv, _ = _collect(_seeded_mlp(), x, invasive_linear_io=[r"fc"])
    _assert_gradients_identical(g_lin, g_inv)


@pytest.mark.parametrize("style", ["logra_factorized", "logra_materialized"])
@pytest.mark.parametrize("shape", [(5, 8), (3, 4, 8)], ids=["2d", "3d"])
def test_invasive_matches_linear_io_projected(style, shape):
    x = torch.randn(*shape)
    proj = {
        "__default__": {
            "style": style,
            "proj_dim": 4,
            "proj_max_batch_size": 32,
            "proj_type": "rademacher",
            "proj_seed": 0,
        },
    }
    g_lin, _ = _collect(_seeded_mlp(), x, linear_io=[r"fc"], projection=proj)
    g_inv, _ = _collect(_seeded_mlp(), x, invasive_linear_io=[r"fc"], projection=proj)
    _assert_gradients_identical(g_lin, g_inv)


def test_invasive_similarity_matches_linear_io():
    """Downstream ops (per-layer cosine similarity) agree."""
    x = torch.randn(5, 8)
    g_lin, _ = _collect(_seeded_mlp(), x, linear_io=[r"fc"])
    g_inv, _ = _collect(_seeded_mlp(), x, invasive_linear_io=[r"fc"])
    sim_lin = g_lin.similarity(g_lin, metric="dot")
    sim_inv = g_inv.similarity(g_inv, metric="dot")
    for layer in sim_lin:
        assert torch.allclose(sim_lin[layer], sim_inv[layer], atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. Ground truth: materialized capture == true per-sample weight grad        #
# --------------------------------------------------------------------------- #


def test_invasive_materialize_equals_reference_weight_grad():
    """Single Linear, 2D input, no bias: materialized == per-sample weight.grad."""
    torch.manual_seed(0)
    layer = nn.Linear(8, 5, bias=False)
    x = torch.randn(4, 8)

    g_inv, _ = _collect(copy.deepcopy(layer), x, invasive_linear_io=[""])
    mat = g_inv.materialize().data[""]  # (B, out*in)
    mat = mat.reshape(4, 5, 8)

    # Reference: run each sample individually through a fresh copy.
    for i in range(4):
        ref = copy.deepcopy(layer)
        ref.zero_grad(set_to_none=True)
        ref(x[i : i + 1]).pow(2).sum().backward()
        assert torch.allclose(mat[i], ref.weight.grad, atol=1e-5), i


# --------------------------------------------------------------------------- #
# 3. Invasive-only behaviour: weight.grad / bias.grad are skipped             #
# --------------------------------------------------------------------------- #


def test_invasive_skips_weight_and_bias_grad():
    x = torch.randn(5, 8)
    _, m_lin = _collect(_seeded_mlp(), x, linear_io=[r"fc"])
    _, m_inv = _collect(_seeded_mlp(), x, invasive_linear_io=[r"fc"])
    # linear_io leaves the ordinary weight/bias grads populated ...
    assert all(p.grad is not None for p in m_lin.parameters())
    # ... invasive leaves every fc weight/bias grad as None.
    assert m_inv.fc1.weight.grad is None
    assert m_inv.fc1.bias.grad is None
    assert m_inv.fc2.weight.grad is None


def test_mixed_linear_io_and_invasive():
    """A mix of families completes the step; each captures, grads differ."""
    x = torch.randn(5, 8)
    m = _seeded_mlp()
    cb = _CaptureCB()
    cfg = HookManagerConfig(
        hook_types={"fc1": LINEAR_IO, "fc2": INVASIVE_LINEAR_IO},
    )
    mgr = HookManager(m, config=cfg, callbacks=[cb])
    with mgr.collect():
        m.zero_grad(set_to_none=True)
        m(x).pow(2).sum().backward()
    mgr.remove()
    assert cb.records, "mixed config never completed a step"
    g = cb.records[-1].gradient
    assert sorted(g.materialize().data) == ["fc1", "fc2"]
    # fc1 (linear_io) keeps weight.grad; fc2 (invasive) skips it.
    assert m.fc1.weight.grad is not None
    assert m.fc2.weight.grad is None


# --------------------------------------------------------------------------- #
# 4. Config resolution                                                        #
# --------------------------------------------------------------------------- #


def test_explicit_assignment_resolves():
    m = _seeded_mlp()
    cfg = HookManagerConfig(hook_types={"fc1": INVASIVE_LINEAR_IO})
    assert resolve_hook_assignments(m, cfg) == {"fc1": INVASIVE_LINEAR_IO}


def test_selector_resolves():
    m = _seeded_mlp()
    cfg = HookManagerConfig(invasive_linear_io=[r"fc"])
    assert resolve_hook_assignments(m, cfg) == {
        "fc1": INVASIVE_LINEAR_IO,
        "fc2": INVASIVE_LINEAR_IO,
    }


def test_conflict_between_families_raises():
    m = _seeded_mlp()
    cfg = HookManagerConfig(
        hook_types={"fc1": LINEAR_IO},
        invasive_linear_io=[r"fc1"],
    )
    with pytest.raises(ValueError, match="conflicting hook types"):
        resolve_hook_assignments(m, cfg)


def test_non_linear_layer_assigned_invasive_raises():
    class WithNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8)

        def forward(self, x):
            return self.norm(x)

    m = WithNorm()
    cfg = HookManagerConfig(hook_types={"norm": INVASIVE_LINEAR_IO})
    with pytest.raises(ValueError, match=r"not nn\.Linear"):
        resolve_hook_assignments(m, cfg)


def test_invasive_selector_is_not_default():
    assert not HookManagerConfig(invasive_linear_io=[r"fc"]).is_default
    assert HookManagerConfig().is_default


# --------------------------------------------------------------------------- #
# 5. Lifecycle: forward override installed/reverted correctly                 #
# --------------------------------------------------------------------------- #


def test_forward_restored_after_remove():
    m = _seeded_mlp()
    x = torch.randn(5, 8)
    before = m(x).detach().clone()

    mgr = HookManager(m, config=HookManagerConfig(invasive_linear_io=[r"fc"]))
    with mgr.collect():
        m(x).pow(2).sum().backward()
    mgr.remove()

    # Output identical to pre-registration, and the class forward is back.
    assert torch.allclose(m(x), before, atol=1e-6)
    assert "forward" not in vars(m.fc1)  # instance override dropped
    # weight.grad now computed again (no override active).
    m.zero_grad(set_to_none=True)
    m(x).pow(2).sum().backward()
    assert m.fc1.weight.grad is not None


def test_remove_is_idempotent_with_invasive():
    m = _seeded_mlp()
    mgr = HookManager(m, config=HookManagerConfig(invasive_linear_io=[r"fc"]))
    mgr.remove()
    mgr.remove()  # must not raise
    assert "forward" not in vars(m.fc1)


def test_rearm_after_remove():
    m = _seeded_mlp()
    x = torch.randn(5, 8)
    mgr = HookManager(m, config=HookManagerConfig(invasive_linear_io=[r"fc"]))
    mgr.remove()
    mgr.register()  # re-arm
    cb = _CaptureCB()
    mgr.add_callback(cb)
    with mgr.collect():
        m.zero_grad(set_to_none=True)
        m(x).pow(2).sum().backward()
    assert cb.records, "re-armed manager did not capture"
    assert m.fc1.weight.grad is None  # override active again
    mgr.remove()
    assert "forward" not in vars(m.fc1)


def test_collect_deregister_on_exit_restores_forward():
    m = _seeded_mlp()
    x = torch.randn(5, 8)
    cb = _CaptureCB()
    mgr = HookManager(
        m, config=HookManagerConfig(invasive_linear_io=[r"fc"]), callbacks=[cb]
    )
    with mgr.collect(deregister_on_exit=True):
        m.zero_grad(set_to_none=True)
        m(x).pow(2).sum().backward()
    assert cb.records
    assert "forward" not in vars(m.fc1)  # forward restored on exit
    m.zero_grad(set_to_none=True)
    m(x).pow(2).sum().backward()
    assert m.fc1.weight.grad is not None


# --------------------------------------------------------------------------- #
# 6. Gating: the override only intervenes while collecting                    #
# --------------------------------------------------------------------------- #


def test_not_collecting_computes_weight_grad():
    """Registered but outside a collect() context, weight.grad is computed."""
    m = _seeded_mlp()
    x = torch.randn(5, 8)
    mgr = HookManager(m, config=HookManagerConfig(invasive_linear_io=[r"fc"]))
    mgr.register()  # armed but not collecting
    m.zero_grad(set_to_none=True)
    m(x).pow(2).sum().backward()
    # Not collecting -> override falls back to the original forward.
    assert m.fc1.weight.grad is not None
    mgr.remove()


# --------------------------------------------------------------------------- #
# 7. Multi-invocation (a reused Linear) matches linear_io                     #
# --------------------------------------------------------------------------- #


def test_multi_invocation_matches_linear_io():
    class Reuse(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(8, 8)

        def forward(self, x):
            return self.shared(self.shared(x))

    x = torch.randn(5, 8)
    torch.manual_seed(7)
    a = Reuse()
    b = copy.deepcopy(a)
    g_lin, _ = _collect(a, x, linear_io=[r"shared"])
    g_inv, _ = _collect(b, x, invasive_linear_io=[r"shared"])
    _assert_gradients_identical(g_lin, g_inv)
