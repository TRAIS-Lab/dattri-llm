"""Optimizer maps on per-sample gradients.

Optimizer-aware attribution scores the *update* a sample induces rather than
its raw gradient.  For every coordinate-wise optimizer that update is a
function of the raw per-sample gradient and the optimizer's state at the same
coordinates, so once the gradient is available on a coordinate set -- the
whole layer, or a ``"subset_materialized"`` subset -- the map applies
exactly, entry by entry.  :func:`precondition` is that map for every
coordinate-wise optimizer in ``torch.optim`` (plus Lion), and the
``adamw_influence_*`` kernels are the pieces of AdamW-influence's trajectory
unrolling, which needs the optimizer's Jacobians rather than its update.

Every function takes the per-sample gradient entries as a dense ``(B, k)``
tensor and the optimizer quantities as ``(k,)`` tensors on the same
coordinates (a scalar state such as NAdam's ``mu_product`` is 0-d).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: The per-parameter state tensors each supported optimizer keeps, by the
#: names ``torch.optim`` uses (``state[param][key]``).  A key missing from a
#: parameter's state (before its first update, or an option that is off) is
#: passed to :func:`precondition` as ``None``.
OPTIMIZER_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "SGD": ("momentum_buffer",),
    "Adam": ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
    "AdamW": ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
    "Adamax": ("exp_avg", "exp_inf"),
    "NAdam": ("exp_avg", "exp_avg_sq", "mu_product"),
    "RAdam": ("exp_avg", "exp_avg_sq"),
    "RMSprop": ("square_avg", "grad_avg", "momentum_buffer"),
    "Adagrad": ("sum",),
    "Adadelta": ("square_avg", "acc_delta"),
    "Lion": ("exp_avg",),
}

# Hyperparameters each rule reads, with torch's defaults for a group that
# does not carry them (a foreign Lion has no torch defaults; these are the
# reference implementation's).
_HYPERPARAMETER_DEFAULTS: dict[str, dict[str, object]] = {
    "SGD": {"momentum": 0.0, "dampening": 0.0, "nesterov": False},
    "Adam": {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False},
    "AdamW": {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False},
    "Adamax": {"betas": (0.9, 0.999), "eps": 1e-8},
    "NAdam": {"betas": (0.9, 0.999), "eps": 1e-8, "momentum_decay": 4e-3},
    "RAdam": {"betas": (0.9, 0.999), "eps": 1e-8},
    "RMSprop": {"alpha": 0.99, "eps": 1e-8, "momentum": 0.0, "centered": False},
    "Adagrad": {"eps": 1e-10, "lr_decay": 0.0},
    "Adadelta": {"rho": 0.9, "eps": 1e-6},
    "Lion": {"betas": (0.9, 0.99)},
}


def _zeros_like_row(g: torch.Tensor, value: torch.Tensor | None) -> torch.Tensor:
    """*value* as a float ``(k,)`` tensor on *g*'s device; zeros when ``None``."""
    if value is None:
        return torch.zeros(g.shape[-1], device=g.device, dtype=g.dtype)
    return value.to(g.device, g.dtype)


def _sgd(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:  # noqa: ARG001
    momentum = float(hp["momentum"])
    if not momentum:
        return g
    buf = state.get("momentum_buffer")
    if buf is None:  # first step: the buffer is initialized to the gradient
        buf_new = g
    else:
        buf_new = momentum * buf.to(g.device, g.dtype) + (1.0 - hp["dampening"]) * g
    return g + momentum * buf_new if hp["nesterov"] else buf_new


def _adam_moments(
    g: torch.Tensor, state: Mapping, hp: Mapping
) -> tuple[torch.Tensor, torch.Tensor]:
    beta1, beta2 = hp["betas"]
    m = beta1 * _zeros_like_row(g, state.get("exp_avg")) + (1.0 - beta1) * g
    v = beta2 * _zeros_like_row(g, state.get("exp_avg_sq")) + (1.0 - beta2) * g * g
    return m, v


def _adam(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:
    beta1, beta2 = hp["betas"]
    m, v = _adam_moments(g, state, hp)
    if hp["amsgrad"]:
        v = torch.maximum(_zeros_like_row(g, state.get("max_exp_avg_sq")), v)
    denom = v.sqrt() / math.sqrt(1.0 - beta2**step) + hp["eps"]
    return m / (1.0 - beta1**step) / denom


def _adamax(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:
    beta1, beta2 = hp["betas"]
    m = beta1 * _zeros_like_row(g, state.get("exp_avg")) + (1.0 - beta1) * g
    u = torch.maximum(
        beta2 * _zeros_like_row(g, state.get("exp_inf")), g.abs() + hp["eps"]
    )
    return m / ((1.0 - beta1**step) * u)


def _nadam(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:
    beta1, beta2 = hp["betas"]
    decay = float(hp["momentum_decay"])
    mu = beta1 * (1.0 - 0.5 * 0.96 ** (step * decay))
    mu_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * decay))
    mu_product = state.get("mu_product")
    mu_product = (1.0 if mu_product is None else float(mu_product)) * mu
    m, v = _adam_moments(g, state, hp)
    denom = (v / (1.0 - beta2**step)).sqrt() + hp["eps"]
    return ((1.0 - mu) / (1.0 - mu_product)) * g / denom + (
        mu_next / (1.0 - mu_product * mu_next)
    ) * m / denom


def _radam(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:
    beta1, beta2 = hp["betas"]
    m, v = _adam_moments(g, state, hp)
    bias_correction2 = 1.0 - beta2**step
    m_hat = m / (1.0 - beta1**step)
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    rho_t = rho_inf - 2.0 * step * beta2**step / bias_correction2
    if rho_t <= 5.0:
        return m_hat
    rect = math.sqrt(
        (rho_t - 4.0)
        * (rho_t - 2.0)
        * rho_inf
        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
    )
    return m_hat * rect * math.sqrt(bias_correction2) / (v.sqrt() + hp["eps"])


def _rmsprop(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:  # noqa: ARG001
    alpha = float(hp["alpha"])
    v = alpha * _zeros_like_row(g, state.get("square_avg")) + (1.0 - alpha) * g * g
    if hp["centered"]:
        ga = alpha * _zeros_like_row(g, state.get("grad_avg")) + (1.0 - alpha) * g
        avg = (v - ga * ga).sqrt() + hp["eps"]
    else:
        avg = v.sqrt() + hp["eps"]
    direction = g / avg
    momentum = float(hp["momentum"])
    if momentum > 0.0:
        return momentum * _zeros_like_row(g, state.get("momentum_buffer")) + direction
    return direction


def _adagrad(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:
    total = _zeros_like_row(g, state.get("sum")) + g * g
    # torch scales the learning rate by 1 / (1 + (step - 1) lr_decay).
    return g / (total.sqrt() + hp["eps"]) / (1.0 + (step - 1) * hp["lr_decay"])


def _adadelta(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:  # noqa: ARG001
    rho, eps = float(hp["rho"]), float(hp["eps"])
    v = rho * _zeros_like_row(g, state.get("square_avg")) + (1.0 - rho) * g * g
    acc = _zeros_like_row(g, state.get("acc_delta"))
    return (acc + eps).sqrt() / (v + eps).sqrt() * g


def _lion(g: torch.Tensor, state: Mapping, step: int, hp: Mapping) -> torch.Tensor:  # noqa: ARG001
    beta1 = hp["betas"][0]
    return torch.sign(
        beta1 * _zeros_like_row(g, state.get("exp_avg")) + (1.0 - beta1) * g
    )


_RULES: dict[str, Callable[[torch.Tensor, Mapping, int, Mapping], torch.Tensor]] = {
    "SGD": _sgd,
    "Adam": _adam,
    "AdamW": _adam,
    "Adamax": _adamax,
    "NAdam": _nadam,
    "RAdam": _radam,
    "RMSprop": _rmsprop,
    "Adagrad": _adagrad,
    "Adadelta": _adadelta,
    "Lion": _lion,
}


def precondition(
    g: torch.Tensor,
    state: Mapping[str, torch.Tensor | None],
    *,
    optimizer_type: str,
    step: int,
    **hyperparameters: object,
) -> torch.Tensor:
    """The per-sample update direction an optimizer would take from ``g``.

    Each sample advances the optimizer's *pre-step* state as if it alone made
    up the step, and the direction of the update that would follow is its
    representation: the parameter moves by ``-lr * precondition(g)``.  For
    Adam this is LESS's ``Gamma(z)``; for plain SGD it is ``g`` itself.
    The learning rate is factored out and weight decay dropped, both being the
    same for every sample (a coupled L2 term is *not* folded into the moments).

    Args:
        g: ``(B, k)`` per-sample gradient entries.
        state: The optimizer's state tensors on the same ``k`` coordinates,
            keyed as in :data:`OPTIMIZER_STATE_KEYS`; ``None`` (or a missing
            key) for state the optimizer has not created yet.
        optimizer_type: A key of :data:`OPTIMIZER_STATE_KEYS`.
        step: Index of the update about to be applied, starting at 1 (the
            optimizer's ``state["step"] + 1``); drives the bias corrections.
        **hyperparameters: The group's hyperparameters (``betas``, ``eps``,
            ``momentum``, ...); torch's defaults fill in the missing ones.

    Returns:
        ``(B, k)`` float32 update directions.
    """
    try:
        rule = _RULES[optimizer_type]
    except KeyError:
        raise NotImplementedError(
            f"precondition has no rule for optimizer type {optimizer_type!r}; "
            f"supported: {sorted(_RULES)}.",
        ) from None
    if step < 1:
        raise ValueError(f"step must be >= 1 (the update about to run), got {step}.")
    hp = {**_HYPERPARAMETER_DEFAULTS[optimizer_type], **hyperparameters}
    return rule(g.float(), state, int(step), hp)


def adam_preconditioner(
    m_hat: torch.Tensor,
    v_hat: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The diagonals ``D_t`` and ``S_t`` of AdamW-influence (Eq. 11a).

    ``D_t = 1 / (sqrt(v_hat) + eps)`` is the Adam preconditioner and
    ``S_t = m_hat / (2 sqrt(v_hat) (sqrt(v_hat) + eps) ** 2)`` its sensitivity
    to the second moment, both from the *bias-corrected post-step* moments.
    """
    root = torch.sqrt(v_hat.double())
    d = 1.0 / (root + eps)
    # A coordinate whose gradient has been identically zero has v_hat == 0 (and
    # m_hat == 0): the update there does not depend on v, so its sensitivity is
    # zero -- not the 0/0 the formula gives.  float64 keeps the denominator
    # from underflowing for tiny but nonzero v_hat.
    denom = 2.0 * root * (root + eps) ** 2
    s = torch.where(root > 0, m_hat.double() / denom.clamp_min(1e-300), 0.0)
    return d.float(), s.float()


def adamw_influence_push(
    g_z: torch.Tensor,
    g_t: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor,
    *,
    lr: float,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.Tensor:
    """``Z_push(z) = (theta_dot_{t+1}, m_dot_t, v_dot_t)`` of AdamW-influence.

    The first-order response of the optimizer state to removing sample ``z``
    from the batch gradient at its step (``g_t(eps) = g_t - eps g_z``)::

        m_dot = -(1 - beta1) g_z
        v_dot = -2 (1 - beta2) g_t * g_z
        theta_dot = -lr (D_t m_dot / (1 - beta1 ** step)
                          - S_t v_dot / (1 - beta2 ** step))

    Args:
        g_z: ``(B, k)`` per-sample gradient entries.
        g_t: ``(k,)`` batch gradient the optimizer consumed at that step.
        d: ``(k,)`` ``D_t`` (see :func:`adam_preconditioner`).
        s: ``(k,)`` ``S_t``.
        lr: Learning rate applied at the step.
        step: Update count after the step.
        beta1: Adam's first-moment decay.
        beta2: Adam's second-moment decay.

    Returns:
        ``(B, 3k)``: the three blocks concatenated.
    """
    g_z = g_z.float()
    m_dot = -(1.0 - beta1) * g_z
    v_dot = -2.0 * (1.0 - beta2) * g_t.float() * g_z
    theta_dot = -lr * (
        d * m_dot / (1.0 - beta1**step) - s * v_dot / (1.0 - beta2**step)
    )
    return torch.cat([theta_dot, m_dot, v_dot], dim=-1)


def adamw_influence_transition(
    w_theta: torch.Tensor,
    w_m: torch.Tensor,
    w_v: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor,
    *,
    lr: float,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``W M_t`` for the block-diagonal transition ``M_t`` (Eq. 11a).

    ``W = [W_theta | W_m | W_v]`` is the ``(p, 3k)`` summary matrix; each
    block is ``(p, k)``.  ``M_t`` couples the parameter block to the moments
    through the diagonals ``D_t`` and ``S_t``, so the product is three column
    scalings::

        W_theta' = (1 - lr * wd) W_theta
        W_m'     = -lr beta1 / (1 - beta1 ** step) W_theta diag(D_t) + beta1 W_m
        W_v'     =  lr beta2 / (1 - beta2 ** step) W_theta diag(S_t) + beta2 W_v
    """
    c1 = beta1 / (1.0 - beta1**step)
    c2 = beta2 / (1.0 - beta2**step)
    new_theta = (1.0 - lr * weight_decay) * w_theta
    new_m = w_theta * (-lr * c1 * d)[None, :] + beta1 * w_m
    new_v = w_theta * (lr * c2 * s)[None, :] + beta2 * w_v
    return new_theta, new_m, new_v


def adamw_influence_coupling(
    w_theta: torch.Tensor,
    w_m: torch.Tensor,
    w_v: torch.Tensor,
    g_z: torch.Tensor,
    g_t: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor,
    *,
    lr: float,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.Tensor:
    """``W R_t g_z`` for every sample of a batch (Eq. 11b), ``(B, p)``.

    ``R_t g_z`` is the response of the next state to a change of the batch
    gradient along ``g_z``::

        r_theta = (-lr (1 - beta1) / (1 - beta1 ** step) D_t
                   + 2 lr (1 - beta2) / (1 - beta2 ** step) S_t g_t) * g_z
        r_m     = (1 - beta1) g_z
        r_v     = 2 (1 - beta2) g_t * g_z

    and ``W R_t g_z = W_theta r_theta + W_m r_m + W_v r_v``.
    """
    g_z = g_z.float()
    scale = (
        -lr * (1.0 - beta1) / (1.0 - beta1**step) * d
        + (2.0 * lr * (1.0 - beta2) / (1.0 - beta2**step)) * s * g_t.float()
    )
    r_theta = scale[None, :] * g_z
    r_m = (1.0 - beta1) * g_z
    r_v = 2.0 * (1.0 - beta2) * g_t.float()[None, :] * g_z
    return r_theta @ w_theta.T + r_m @ w_m.T + r_v @ w_v.T
