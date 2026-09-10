"""LESS against explicit oracles: the frozen-checkpoint form and the
per-step trajectory form.
"""

from __future__ import annotations

import pytest
import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.less import LESSAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.utils.hashing import hash_sample

IN, HID, OUT = 4, 6, 3
N_TRAIN, N_TEST = 6, 4
LAYERS = ["fc1", "fc2"]
BETAS, EPS = (0.8, 0.99), 1e-6


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def _flat_layer(grads: dict, layer: str) -> torch.Tensor:
    w, b = grads[f"{layer}.weight"], grads[f"{layer}.bias"]
    return torch.cat([w.reshape(w.shape[0], -1), b.reshape(-1, 1)], dim=1).reshape(-1)


def _data():
    g = torch.Generator().manual_seed(0)
    train = DictDataset(
        torch.randn(N_TRAIN, IN, generator=g), torch.randn(N_TRAIN, OUT, generator=g)
    )
    test = DictDataset(
        torch.randn(N_TEST, IN, generator=g), torch.randn(N_TEST, OUT, generator=g)
    )
    return train, test


def _task(model, checkpoints):
    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    return AttributionTask(loss_func=loss_func, model=model, checkpoints=checkpoints)


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = MLP()
    train, test = _data()
    ckpt0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Checkpoint 1: a few AdamW steps; its optimizer carries real moments.
    opt1 = torch.optim.AdamW(model.parameters(), lr=0.05, betas=BETAS, eps=EPS)
    for _ in range(3):
        opt1.zero_grad()
        ((model(train.x) - train.y) ** 2).sum().backward()
        opt1.step()
    ckpt1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt0 = torch.optim.AdamW(model.parameters(), lr=0.02, betas=BETAS, eps=EPS)
    task = _task(model, [ckpt0, ckpt1])
    return model, task, train, test, [ckpt0, ckpt1], [opt0, opt1]


def _per_sample_grads(model, x, y):
    rows = []
    for i in range(x.shape[0]):
        model.zero_grad()
        ((model(x[i : i + 1]) - y[i : i + 1]) ** 2).sum().backward()
        rows.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    return rows


def _moment(opt, wp, bp, key, rows):
    """One Adam moment of a linear layer in the ``[W[o,:], b[o]]`` row layout."""
    w = opt.state.get(wp, {}).get(key, torch.zeros_like(wp)).reshape(rows, -1)
    b = opt.state.get(bp, {}).get(key, torch.zeros_like(bp)).reshape(-1, 1)
    return torch.cat([w, b], dim=1).reshape(-1)


def _adam_step_count(opt, param) -> int:
    state = opt.state.get(param, {})
    return int(state["step"].item()) if "step" in state else 0


def _features(grads, model, opt, *, apply_map, subset):
    """Concatenated layer entries; the Adam map (torch's form) when asked."""
    params = dict(model.named_parameters())
    step = _adam_step_count(opt, params["fc1.weight"]) + 1
    parts = []
    for layer in LAYERS:
        g = _flat_layer(grads, layer)
        wp, bp = params[f"{layer}.weight"], params[f"{layer}.bias"]
        rows = HID if layer == "fc1" else OUT
        m = _moment(opt, wp, bp, "exp_avg", rows)
        v = _moment(opt, wp, bp, "exp_avg_sq", rows)
        if subset is not None:
            g, m, v = g[subset[layer]], m[subset[layer]], v[subset[layer]]
        if apply_map:
            m_z = (BETAS[0] * m + (1 - BETAS[0]) * g) / (1 - BETAS[0] ** step)
            v_z = (BETAS[1] * v + (1 - BETAS[1]) * g * g) / (1 - BETAS[1] ** step)
            g = m_z / (torch.sqrt(v_z) + EPS)
        parts.append(g)
    return torch.cat(parts)


def _cosine_block(model, opt, train_x, train_y, test_x, test_y, subset=None):
    """``cos(grad(z'), Gamma(z))`` at the model's current parameters."""
    tr = _per_sample_grads(model, train_x, train_y)
    te = _per_sample_grads(model, test_x, test_y)
    gamma = torch.stack(
        [_features(r, model, opt, apply_map=True, subset=subset) for r in tr]
    )
    query = torch.stack(
        [_features(r, model, opt, apply_map=False, subset=subset) for r in te]
    )
    gamma /= gamma.norm(dim=1, keepdim=True)
    query /= query.norm(dim=1, keepdim=True)
    return gamma @ query.T


def _frozen_oracle(model, ckpts, opts, train, test, weights, subset=None):
    scores = torch.zeros(N_TRAIN, N_TEST)
    for ckpt, opt, w in zip(ckpts, opts, weights, strict=True):
        model.load_state_dict(ckpt)
        scores += w * _cosine_block(
            model, opt, train.x, train.y, test.x, test.y, subset=subset
        )
    return scores


def _args(out_dir, batch=2, **kw):
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        use_cpu=True,
        dataloader_pin_memory=False,
        **kw,
    )


def _ordered(score, train, test):
    """Score matrix in dataset order (rows = train, cols = test)."""
    ids, matrix = score.agnostic_matrix()
    row = {h: i for i, h in enumerate(ids)}
    col = {h: i for i, h in enumerate(score.test_ids)}
    # Records are hashed on the model's inputs: the positional ``x`` only.
    tr = [hash_sample({"_arg0": train.x[i]}) for i in range(N_TRAIN)]
    te = [hash_sample({"_arg0": test.x[i]}) for i in range(N_TEST)]
    return matrix[[row[h] for h in tr]][:, [col[h] for h in te]]


HOOKS = HookManagerConfig(linear_io=[f"{n}$" for n in LAYERS])


class TestLESSFrozen:
    def test_matches_oracle_over_two_checkpoints(self, setup, tmp_path):
        model, task, train, test, ckpts, opts = setup
        weights = [0.02, 0.05]
        attr = LESSAttributor(
            _args(tmp_path), task=task, optimizers=opts, checkpoint_weights=weights
        )
        score = attr.attribute(train, test, hook_config=HOOKS)
        got = _ordered(score, train, test)
        want = _frozen_oracle(model, ckpts, opts, train, test, weights)
        assert torch.allclose(got, want, atol=1e-4), (
            f"max diff {(got - want).abs().max():.2e}"
        )

    def test_subset_capture_matches_oracle_on_the_subset(self, setup, tmp_path):
        model, task, train, test, ckpts, opts = setup
        projection = {
            "__default__": {
                "style": "subset_materialized",
                "proj_dim": 7,
                "proj_seed": 2,
            }
        }
        attr = LESSAttributor(_args(tmp_path), task=task, optimizers=opts)
        score = attr.attribute(
            train,
            test,
            hook_config=HookManagerConfig(
                linear_io=[f"{n}$" for n in LAYERS], projection=projection
            ),
        )
        proj = ops.DattriProjector()
        widths = {"fc1": HID * (IN + 1), "fc2": OUT * (HID + 1)}
        subset = {
            layer: proj.subset_indices(
                widths[layer], proj_dim=7, proj_seed=2, device=torch.device("cpu")
            )
            for layer in LAYERS
        }
        got = _ordered(score, train, test)
        want = _frozen_oracle(
            model, ckpts, opts, train, test, [1.0, 1.0], subset=subset
        )
        assert torch.allclose(got, want, atol=1e-4), (
            f"max diff {(got - want).abs().max():.2e}"
        )

    def test_from_cache_matches_live(self, setup, tmp_path):
        _, task, train, test, _, opts = setup
        weights = [0.02, 0.05]
        live = LESSAttributor(
            _args(tmp_path / "live"),
            task=task,
            optimizers=opts,
            checkpoint_weights=weights,
        ).attribute(train, test, hook_config=HOOKS)
        cached = LESSAttributor(_args(tmp_path / "cache"), task=task, optimizers=opts)
        pairs = cached.cache(train, test, hook_config=HOOKS)
        total = None
        for (train_dir, test_dir), w in zip(pairs, weights, strict=True):
            part = _ordered(
                cached.attribute_from_cache(train_dir, test_dir, checkpoint_weight=w),
                train,
                test,
            )
            total = part if total is None else total + part
        assert torch.allclose(total, _ordered(live, train, test), atol=1e-5)

    def test_optimizer_count_must_match_checkpoints(self, setup, tmp_path):
        _, task, train, test, _, opts = setup
        attr = LESSAttributor(_args(tmp_path), task=task, optimizers=opts[:1])
        with pytest.raises(ValueError, match="one optimizer per checkpoint"):
            attr.attribute(train, test)

    def test_rejects_logra_captures(self, setup, tmp_path):
        _, task, train, test, _, opts = setup
        projection = {
            "__default__": {
                "style": "logra_factorized",
                "proj_dim": 4,
                "proj_max_batch_size": 8,
            }
        }
        attr = LESSAttributor(_args(tmp_path), task=task, optimizers=opts)
        with pytest.raises(ValueError, match="cannot be preconditioned"):
            attr.attribute(
                train,
                test,
                hook_config=HookManagerConfig(
                    linear_io=[f"{n}$" for n in LAYERS], projection=projection
                ),
            )


def _trajectory_args(out_dir, batch=2):
    return _args(
        out_dir,
        batch=batch,
        learning_rate=0.05,
        weight_decay=0.01,
        adam_beta1=BETAS[0],
        adam_beta2=BETAS[1],
        adam_epsilon=EPS,
        max_grad_norm=None,
        lr_scheduler_type="constant",
    )


def _trajectory_oracle(ckpt, train, test, score, batch=2):
    """Replay the recorded trajectory (batch order from the score's rows) and
    accumulate ``lr * cos(grad(z'; theta_0), Gamma(z; theta_t))`` per step,
    with the query gradient taken once at the start (``loop_over_test=False``).
    """
    torch.manual_seed(0)
    model = MLP()
    model.load_state_dict(ckpt)
    params = dict(model.named_parameters())
    decay = [p for n, p in params.items() if "bias" not in n]
    no_decay = [p for n, p in params.items() if "bias" in n]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.05,
        betas=BETAS,
        eps=EPS,
    )
    index = {hash_sample({"_arg0": train.x[i]}): i for i in range(N_TRAIN)}
    order = {}
    for h, s in zip(score.row_train_ids, score.row_steps, strict=True):
        order.setdefault(s, []).append(index[h])
    # Query gradients at theta_0.
    query = torch.stack(
        [
            _features(r, model, opt, apply_map=False, subset=None)
            for r in _per_sample_grads(model, test.x, test.y)
        ]
    )
    query /= query.norm(dim=1, keepdim=True)
    scores = torch.zeros(N_TRAIN, N_TEST)
    for step in sorted(order):
        rows = order[step]
        assert len(rows) == batch
        x, y = train.x[rows], train.y[rows]
        gamma = torch.stack(
            [
                _features(r, model, opt, apply_map=True, subset=None)
                for r in _per_sample_grads(model, x, y)
            ]
        )
        gamma /= gamma.norm(dim=1, keepdim=True)
        scores[rows] += 0.05 * (gamma @ query.T)
        opt.zero_grad()
        ((model(x) - y) ** 2).sum().backward()
        opt.step()
    return scores


class TestLESSPerStep:
    def test_matches_a_replayed_trajectory(self, tmp_path):
        torch.manual_seed(0)
        model = MLP()
        train, test = _data()
        ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
        attr = LESSAttributor(_trajectory_args(tmp_path), task=_task(model, [ckpt]))
        score = attr.attribute(train, test, hook_config=HOOKS, enable_update=True)
        assert sorted(set(score.row_steps)) == list(range(N_TRAIN // 2))
        got = _ordered(score, train, test)
        want = _trajectory_oracle(ckpt, train, test, score)
        assert torch.allclose(got, want, atol=1e-4), (
            f"max diff {(got - want).abs().max():.2e}"
        )

    def test_from_cache_reads_the_recorded_schedule(self, tmp_path):
        torch.manual_seed(0)
        model = MLP()
        train, test = _data()
        ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
        task = _task(model, [ckpt])
        attr = LESSAttributor(_trajectory_args(tmp_path / "live"), task=task)
        torch.manual_seed(1)
        live = attr.attribute(train, test, hook_config=HOOKS, enable_update=True)
        task = _task(model, [ckpt])
        attr = LESSAttributor(_trajectory_args(tmp_path / "cache"), task=task)
        torch.manual_seed(1)
        ((train_dir, test_dir),) = attr.cache(
            train, test, hook_config=HOOKS, enable_update=True
        )
        cached = attr.attribute_from_cache(train_dir, test_dir)
        assert torch.allclose(
            _ordered(cached, train, test), _ordered(live, train, test), atol=1e-5
        )
        # An explicit constant overrides the recorded schedule.
        doubled = attr.attribute_from_cache(train_dir, test_dir, learning_rates=0.1)
        assert torch.allclose(
            _ordered(doubled, train, test), 2 * _ordered(cached, train, test), atol=1e-5
        )
