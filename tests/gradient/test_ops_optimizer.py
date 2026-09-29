"""``ops.precondition`` against real ``torch.optim`` steps, and the
AdamW-influence kernels against their explicit dense forms.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient import ops

K, B, P = 6, 3, 6
HP = {"lr": 0.05, "step": 4, "beta1": 0.9, "beta2": 0.99}
LR = 0.1

# (constructor, number of steps before the checked one).  RAdam's rectifier
# only switches on after a few steps; both branches are covered.
OPTIMIZERS = [
    ("sgd", lambda p: torch.optim.SGD(p, lr=LR), 3),
    ("sgd_momentum_first", lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9), 0),
    (
        "sgd_momentum",
        lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9, dampening=0.1),
        3,
    ),
    (
        "sgd_nesterov",
        lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9, nesterov=True),
        3,
    ),
    ("adam_first", lambda p: torch.optim.Adam(p, lr=LR), 0),
    ("adam", lambda p: torch.optim.Adam(p, lr=LR, betas=(0.8, 0.95)), 3),
    ("adam_amsgrad", lambda p: torch.optim.Adam(p, lr=LR, amsgrad=True), 3),
    ("adamw", lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=0.0), 3),
    ("adamax", lambda p: torch.optim.Adamax(p, lr=LR), 3),
    ("nadam", lambda p: torch.optim.NAdam(p, lr=LR), 3),
    ("radam_early", lambda p: torch.optim.RAdam(p, lr=LR), 2),
    ("radam", lambda p: torch.optim.RAdam(p, lr=LR, betas=(0.9, 0.9)), 8),
    ("rmsprop", lambda p: torch.optim.RMSprop(p, lr=LR), 3),
    (
        "rmsprop_centered_momentum",
        lambda p: torch.optim.RMSprop(p, lr=LR, centered=True, momentum=0.5),
        3,
    ),
    ("adagrad", lambda p: torch.optim.Adagrad(p, lr=LR, lr_decay=0.05), 3),
    ("adadelta", lambda p: torch.optim.Adadelta(p, lr=LR), 3),
]


def _state_of(opt: torch.optim.Optimizer, p: nn.Parameter) -> dict:
    state = opt.state.get(p, {})
    return {
        k: (None if state.get(k) is None else state[k].clone())
        for k in ops.OPTIMIZER_STATE_KEYS[type(opt).__name__]
    }


class TestPrecondition:
    @pytest.mark.parametrize(("name", "make", "warmup"), OPTIMIZERS, ids=lambda x: x)
    def test_matches_a_real_optimizer_step(self, name, make, warmup):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(K))
        opt = make([p])
        for _ in range(warmup):
            p.grad = torch.randn(K)
            opt.step()
        state = _state_of(opt, p)
        hp = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
        g = torch.randn(K)
        before = p.detach().clone()
        p.grad = g.clone()
        opt.step()
        update = before - p.detach()
        gamma = ops.precondition(
            g[None], state, optimizer_type=type(opt).__name__, step=warmup + 1, **hp
        )
        assert gamma.shape == (1, K)
        assert torch.allclose(update, LR * gamma[0], atol=1e-6), name

    def test_rows_are_independent_samples(self):
        torch.manual_seed(1)
        g = torch.randn(B, K)
        state = {"exp_avg": torch.randn(K), "exp_avg_sq": torch.rand(K)}
        batched = ops.precondition(g, state, optimizer_type="Adam", step=3)
        for i in range(B):
            single = ops.precondition(
                g[i : i + 1], state, optimizer_type="Adam", step=3
            )
            assert torch.allclose(batched[i], single[0])

    def test_adam_is_the_less_gamma(self):
        torch.manual_seed(0)
        g, m, v = torch.randn(B, K), torch.randn(K), torch.rand(K)
        out = ops.precondition(
            g,
            {"exp_avg": m, "exp_avg_sq": v},
            optimizer_type="AdamW",
            step=3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        m_z = (0.9 * m + 0.1 * g) / (1 - 0.9**3)
        v_z = (0.999 * v + 0.001 * g * g) / (1 - 0.999**3)
        assert torch.allclose(out, m_z / (torch.sqrt(v_z) + 1e-8))

    def test_lion_is_a_sign_direction(self):
        g, m = torch.randn(B, K), torch.randn(K)
        out = ops.precondition(
            g, {"exp_avg": m}, optimizer_type="Lion", step=2, betas=(0.9, 0.99)
        )
        assert torch.equal(out, torch.sign(0.9 * m + 0.1 * g))

    def test_unknown_optimizer_and_bad_step_raise(self):
        with pytest.raises(NotImplementedError, match="no rule"):
            ops.precondition(torch.zeros(1, 2), {}, optimizer_type="Muon", step=1)
        with pytest.raises(ValueError, match="step must be >= 1"):
            ops.precondition(torch.zeros(1, 2), {}, optimizer_type="Adam", step=0)


def _dense_transition(d, s, *, lr, step, beta1, beta2, weight_decay):
    """``M_t`` as an explicit ``(3k, 3k)`` matrix (Eq. 11a)."""
    k = d.numel()
    eye, zero = torch.eye(k), torch.zeros(k, k)
    top = torch.cat(
        [
            (1 - lr * weight_decay) * eye,
            -lr * beta1 / (1 - beta1**step) * torch.diag(d),
            lr * beta2 / (1 - beta2**step) * torch.diag(s),
        ],
        dim=1,
    )
    mid = torch.cat([zero, beta1 * eye, zero], dim=1)
    bot = torch.cat([zero, zero, beta2 * eye], dim=1)
    return torch.cat([top, mid, bot], dim=0)


def _dense_coupling(d, s, g_t, *, lr, step, beta1, beta2):
    """``R_t`` as an explicit ``(3k, k)`` matrix (Eq. 11b)."""
    k = d.numel()
    top = -lr * (1 - beta1) / (1 - beta1**step) * torch.diag(d) + (
        2 * lr * (1 - beta2) / (1 - beta2**step)
    ) * torch.diag(s) @ torch.diag(g_t)
    return torch.cat(
        [top, (1 - beta1) * torch.eye(k), 2 * (1 - beta2) * torch.diag(g_t)]
    )


class TestAdamWInfluenceKernels:
    @staticmethod
    def _inputs():
        torch.manual_seed(1)
        d, s = torch.rand(K) + 0.5, torch.randn(K)
        g_t, g_z = torch.randn(K), torch.randn(B, K)
        w = torch.randn(P, 3 * K)
        return d, s, g_t, g_z, w

    def test_preconditioner_formula(self):
        m_hat, v_hat = torch.randn(K), torch.rand(K) + 0.1
        d, s = ops.adam_preconditioner(m_hat, v_hat, eps=1e-3)
        root = v_hat.sqrt()
        assert torch.allclose(d, 1 / (root + 1e-3))
        assert torch.allclose(s, m_hat / (2 * root * (root + 1e-3) ** 2))

    def test_transition_equals_dense_product(self):
        d, s, _, _, w = self._inputs()
        w_theta, w_m, w_v = w[:, :K], w[:, K : 2 * K], w[:, 2 * K :]
        got = torch.cat(
            ops.adamw_influence_transition(
                w_theta, w_m, w_v, d, s, weight_decay=0.1, **HP
            ),
            dim=1,
        )
        want = w @ _dense_transition(d, s, weight_decay=0.1, **HP)
        assert torch.allclose(got, want, atol=1e-5)

    def test_coupling_equals_dense_product(self):
        d, s, g_t, g_z, w = self._inputs()
        w_theta, w_m, w_v = w[:, :K], w[:, K : 2 * K], w[:, 2 * K :]
        got = ops.adamw_influence_coupling(w_theta, w_m, w_v, g_z, g_t, d, s, **HP)
        want = (w @ _dense_coupling(d, s, g_t, **HP) @ g_z.T).T
        assert torch.allclose(got, want, atol=1e-5)

    def test_push_blocks(self):
        d, s, g_t, g_z, _ = self._inputs()
        z = ops.adamw_influence_push(g_z, g_t, d, s, **HP)
        assert z.shape == (B, 3 * K)
        m_dot = -(1 - HP["beta1"]) * g_z
        v_dot = -2 * (1 - HP["beta2"]) * g_t * g_z
        theta_dot = -HP["lr"] * (
            d * m_dot / (1 - HP["beta1"] ** HP["step"])
            - s * v_dot / (1 - HP["beta2"] ** HP["step"])
        )
        assert torch.allclose(z[:, :K], theta_dot)
        assert torch.allclose(z[:, K : 2 * K], m_dot)
        assert torch.allclose(z[:, 2 * K :], v_dot)
