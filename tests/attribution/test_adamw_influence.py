"""AdamW-influence: the structured sweep against a dense Algorithm 1."""

from __future__ import annotations

import pytest
import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.adamw_influence import (
    AdamWInfluenceAttributor,
    _concat_layers,
)
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource
from dattri_llm.utils.hashing import hash_sample

IN, HID, OUT = 4, 5, 3
N_TRAIN, N_TEST, BATCH = 6, 3, 2
LAYERS = ["fc1", "fc2"]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def _setup():
    torch.manual_seed(0)
    model = MLP()
    g = torch.Generator().manual_seed(0)
    train = DictDataset(
        torch.randn(N_TRAIN, IN, generator=g), torch.randn(N_TRAIN, OUT, generator=g)
    )
    test = DictDataset(
        torch.randn(N_TEST, IN, generator=g), torch.randn(N_TEST, OUT, generator=g)
    )

    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[ckpt])
    return model, task, train, test


def _args(out_dir, **kw):
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        use_cpu=True,
        dataloader_pin_memory=False,
        learning_rate=0.05,
        weight_decay=0.01,
        adam_beta1=0.8,
        adam_beta2=0.99,
        max_grad_norm=None,
        lr_scheduler_type="constant",
        **kw,
    )


def _cat_state(d, which, side):
    """Concatenate one moment (0 = m, 1 = v) of one side over the layers."""
    return torch.cat([d[side][n][which].reshape(-1) for n in LAYERS])


def _dense_algorithm1(blocks, dynamics, test_rep):
    """Algorithm 1 with explicit (3p, 3p) / (3p, p) matrices, for a training
    loss summed over the batch (the GGN is the plain sum of outer products).
    """
    p = test_rep.shape[1]
    eye, zero = torch.eye(p), torch.zeros(p, p)
    W = torch.cat([eye, zero, zero], dim=1)
    rows = {}
    for step in sorted(blocks, reverse=True):
        G, ids = blocks[step]
        d = dynamics[step]
        b1, b2 = d["betas"]
        t, lr, wd, eps = (
            int(d["step"]),
            float(d["lr"]),
            float(d["weight_decay"]),
            float(d["eps"]),
        )
        m_pre = _cat_state(d, 0, "pre")
        m_post, v_post = _cat_state(d, 0, "post"), _cat_state(d, 1, "post")
        g_t = (m_post - b1 * m_pre) / (1 - b1)
        mh, vh = m_post / (1 - b1**t), v_post / (1 - b2**t)
        D = 1 / (vh.sqrt() + eps)
        S = mh / (2 * vh.sqrt() * (vh.sqrt() + eps) ** 2)
        M = torch.cat(
            [
                torch.cat(
                    [
                        (1 - lr * wd) * eye,
                        -lr * b1 / (1 - b1**t) * torch.diag(D),
                        lr * b2 / (1 - b2**t) * torch.diag(S),
                    ],
                    1,
                ),
                torch.cat([zero, b1 * eye, zero], 1),
                torch.cat([zero, zero, b2 * eye], 1),
            ],
            0,
        )
        R = torch.cat(
            [
                -lr * (1 - b1) / (1 - b1**t) * torch.diag(D)
                + 2 * lr * (1 - b2) / (1 - b2**t) * torch.diag(S) @ torch.diag(g_t),
                (1 - b1) * eye,
                2 * (1 - b2) * torch.diag(g_t),
            ],
            0,
        )
        step_rows = []
        dW = torch.zeros(p, 3 * p)
        for z in range(G.shape[0]):
            g_z = G[z]
            m_dot = -(1 - b1) * g_z
            v_dot = -2 * (1 - b2) * g_t * g_z
            th_dot = -lr * (D * m_dot / (1 - b1**t) - S * v_dot / (1 - b2**t))
            inf = W @ torch.cat([th_dot, m_dot, v_dot])
            step_rows.append(-(test_rep @ inf))
            v = W @ (R @ g_z)
            dW[:, :p] += torch.outer(v, g_z)
        rows[step] = (torch.stack(step_rows), ids)
        W = W @ M + dW
    return rows


def _load_blocks(train_dir, args):
    src = DiskGradientSource(GradientStorageManager(train_dir), args)
    return {
        step: (_concat_layers(block, LAYERS), list(hashes))
        for step, block, hashes in src
    }


class TestAdamWInfluence:
    @pytest.mark.parametrize(
        "projection",
        [
            None,
            {
                "__default__": {
                    "style": "subset_materialized",
                    "proj_dim": 4,
                    "proj_seed": 1,
                }
            },
        ],
    )
    def test_sweep_matches_dense_algorithm(self, tmp_path, projection):
        _, task, train, test = _setup()
        args = _args(tmp_path)
        attr = AdamWInfluenceAttributor(args, task=task)
        hook_config = HookManagerConfig(
            linear_io=[f"{n}$" for n in LAYERS], projection=projection
        )
        score = attr.attribute(
            train, test, hook_config=hook_config, loss_reduction="sum"
        )

        train_dir, test_dir = (
            str(tmp_path / "train_grads"),
            str(tmp_path / "test_grads"),
        )
        blocks = _load_blocks(train_dir, args)
        dynamics = torch.load(
            tmp_path / "train_grads" / "adamw_dynamics.pt", weights_only=False
        )
        test_src = DiskGradientSource(GradientStorageManager(test_dir), args)
        test_parts, test_ids = [], []
        for _s, block, hashes in test_src:
            test_parts.append(_concat_layers(block, LAYERS))
            test_ids.extend(hashes)
        want_rows = _dense_algorithm1(blocks, dynamics, torch.cat(test_parts))

        assert len(blocks) == N_TRAIN // BATCH
        col = {h: i for i, h in enumerate(score.test_ids)}
        cols = [col[h] for h in test_ids]
        rows = zip(score.row_train_ids, score.row_steps, strict=True)
        for r, (tid, step) in enumerate(rows):
            want, ids = want_rows[step]
            assert torch.allclose(
                score.scores[r][cols], want[ids.index(tid)], atol=1e-5
            ), (tid, step)

    def test_last_step_reduces_to_the_one_step_push(self, tmp_path):
        # At the final step W = [I 0 0], so the score is -grad(z')^T theta_dot.
        _, task, train, test = _setup()
        args = _args(tmp_path)
        attr = AdamWInfluenceAttributor(args, task=task)
        score = attr.attribute(
            train,
            test,
            hook_config=HookManagerConfig(linear_io=[f"{n}$" for n in LAYERS]),
        )
        blocks = _load_blocks(str(tmp_path / "train_grads"), args)
        dynamics = torch.load(
            tmp_path / "train_grads" / "adamw_dynamics.pt", weights_only=False
        )
        last = max(blocks)
        G, ids = blocks[last]
        d = dynamics[last]
        b1, b2 = d["betas"]
        t, lr, eps = int(d["step"]), float(d["lr"]), float(d["eps"])
        m_pre = _cat_state(d, 0, "pre")
        m_post, v_post = _cat_state(d, 0, "post"), _cat_state(d, 1, "post")
        g_t = (m_post - b1 * m_pre) / (1 - b1)
        # Dynamics sanity: the consumed batch gradient is the sum of the
        # per-sample gradients (sum loss, no clipping).
        assert torch.allclose(g_t, G.sum(0), atol=1e-4)
        mh, vh = m_post / (1 - b1**t), v_post / (1 - b2**t)
        D = 1 / (vh.sqrt() + eps)
        S = mh / (2 * vh.sqrt() * (vh.sqrt() + eps) ** 2)
        test_src = DiskGradientSource(
            GradientStorageManager(str(tmp_path / "test_grads")), args
        )
        test_rep = torch.cat([_concat_layers(b, LAYERS) for _s, b, _h in test_src])
        for z, tid in enumerate(ids):
            m_dot = -(1 - b1) * G[z]
            v_dot = -2 * (1 - b2) * g_t * G[z]
            th_dot = -lr * (D * m_dot / (1 - b1**t) - S * v_dot / (1 - b2**t))
            rows = zip(score.row_train_ids, score.row_steps, strict=True)
            r = next(i for i, (h, s) in enumerate(rows) if h == tid and s == last)
            assert torch.allclose(score.scores[r], -(test_rep @ th_dot), atol=1e-5)

    def test_rows_cover_every_sample_and_step(self, tmp_path):
        _, task, train, test = _setup()
        score = AdamWInfluenceAttributor(_args(tmp_path), task=task).attribute(
            train,
            test,
            hook_config=HookManagerConfig(linear_io=[f"{n}$" for n in LAYERS]),
        )
        assert score.scores.shape == (N_TRAIN, N_TEST)
        assert sorted(set(score.row_steps)) == list(range(N_TRAIN // BATCH))
        # Records are hashed on the model's inputs: the positional ``x`` only.
        expected = {hash_sample({"_arg0": train.x[i]}) for i in range(N_TRAIN)}
        assert set(score.row_train_ids) == expected

    def test_selected_steps_filter_rows(self, tmp_path):
        _, task, train, test = _setup()
        attr = AdamWInfluenceAttributor(_args(tmp_path), task=task)
        ((train_dir, test_dir),) = attr.cache(
            train,
            test,
            hook_config=HookManagerConfig(linear_io=[f"{n}$" for n in LAYERS]),
        )
        full = attr.attribute_from_cache(train_dir, test_dir)
        part = attr.attribute_from_cache(
            train_dir, test_dir, selected_training_steps=[1]
        )
        assert set(part.row_steps) == {1}
        keep = [i for i, s in enumerate(full.row_steps) if s == 1]
        assert torch.allclose(part.scores, full.scores[keep])
