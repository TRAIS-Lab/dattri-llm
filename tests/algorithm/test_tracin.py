"""Correctness tests for :class:`TracInAttributor` (on-disk workflow).

These collect real per-sample gradients to disk with the repo's
``HookManager`` + ``OffloadCallback`` pipeline, then check the attributor's
score against an independent autograd oracle.

The ``loop_over_test`` parametrization specifically guards the lazy
column-discovery optimisation: the test side must be scored once (no separate
hash pre-pass), and both the cached (``False``) and re-streamed (``True``)
paths must yield byte-identical results with the same column order as the
on-disk layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.tracin import TracInAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.utils import hash_sample

IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_TRAIN, N_TEST = 6, 4
SEED = 0
STEP_WEIGHTS = [1.0, 0.5]
WEIGHT_NAMES = ["mlp.fc1.weight", "mlp.fc2.weight"]
LAYER_NAMES = ["mlp.fc1", "mlp.fc2"]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID_DIM, OUT_DIM, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


def _make_data():
    g = torch.Generator().manual_seed(SEED)
    return (
        torch.randn(N_TRAIN, IN_DIM, generator=g),
        torch.randn(N_TRAIN, OUT_DIM, generator=g),
        torch.randn(N_TEST, IN_DIM, generator=g),
        torch.randn(N_TEST, OUT_DIM, generator=g),
    )


def _make_checkpoints(model):
    sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(SEED + 1)
    sd1 = {k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in sd0.items()}
    return [sd0, sd1]


def _per_sample_grads(model, x, y):
    params = [dict(model.named_parameters())[n] for n in WEIGHT_NAMES]
    rows = []
    for i in range(x.shape[0]):
        loss = ((model(x[i : i + 1]) - y[i : i + 1]) ** 2).sum()
        grads = torch.autograd.grad(loss, params)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)


def _oracle(model, checkpoints, x_tr, y_tr, x_te, y_te, *, normalized):
    total = torch.zeros(N_TRAIN, N_TEST)
    weights = [1.0, 1.0] if normalized else STEP_WEIGHTS
    for sd, w in zip(checkpoints, weights):
        model.load_state_dict(sd)
        g_tr = _per_sample_grads(model, x_tr, y_tr)
        g_te = _per_sample_grads(model, x_te, y_te)
        if normalized:
            g_tr = torch.nn.functional.normalize(g_tr, dim=-1)
            g_te = torch.nn.functional.normalize(g_te, dim=-1)
        total = total + w * (g_tr @ g_te.T)
    return total


def _collect_to_disk(model, checkpoints, x, y, out_dir: Path):
    fm = GradientFileManager(str(out_dir))
    offload = OffloadCallback(
        offload_interval=1, file_manager=fm, recording_type="per_sample"
    )
    hm = HookManager(
        model,
        config=HookManagerConfig(mlp_name_patterns=[r"mlp\."]),
        callbacks=[offload],
    )
    with hm.collect():
        for sd in checkpoints:
            model.load_state_dict(sd)
            model.zero_grad(set_to_none=True)
            loss = ((model(x=x, y=y) - y) ** 2).sum()
            loss.backward()
    hm.remove()
    model.zero_grad(set_to_none=True)


def _disk_test_column_order(test_dir: Path, step: int):
    """Reconstruct the expected column order directly from the on-disk index."""
    fm = GradientFileManager(str(test_dir))
    ids = []
    for file_rel, idxs in fm.iter_step(step):
        records = fm.load_records(file_rel)
        for idx in idxs:
            h = records[idx].input_hash
            ids.extend(h if isinstance(h, list) else [h])
    return ids


@pytest.fixture()
def collected(tmp_path):
    torch.manual_seed(SEED)
    model = MLP().eval()
    checkpoints = _make_checkpoints(model)
    x_tr, y_tr, x_te, y_te = _make_data()
    train_dir, test_dir = tmp_path / "train_g", tmp_path / "test_g"
    _collect_to_disk(model, checkpoints, x_tr, y_tr, train_dir)
    _collect_to_disk(model, checkpoints, x_te, y_te, test_dir)
    train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(N_TRAIN)]
    test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)]
    return dict(
        model=model, checkpoints=checkpoints,
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        train_dir=train_dir, test_dir=test_dir,
        train_hashes=train_hashes, test_hashes=test_hashes,
    )


def _make_attr(out_dir: Path, *, normalized):
    args = AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return TracInAttributor(
        args,
        layer_name=LAYER_NAMES,
        normalized_grad=normalized,
        weight_list=None if normalized else STEP_WEIGHTS,
        steps=[0, 1],
    )


class TestTracInOnDisk:
    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_matches_autograd_oracle(self, collected, tmp_path, normalized, loop_over_test):
        oracle = _oracle(
            collected["model"], collected["checkpoints"],
            collected["x_tr"], collected["y_tr"],
            collected["x_te"], collected["y_te"],
            normalized=normalized,
        )
        attr = _make_attr(tmp_path / f"out_{normalized}_{loop_over_test}", normalized=normalized)
        result = attr.attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=loop_over_test,
        )
        # Realign the hash-keyed score back to sample order for comparison.
        matrix = result.query(
            collected["train_hashes"], collected["test_hashes"], trajectory="agnostic"
        )
        assert torch.allclose(matrix, oracle, atol=1e-5, rtol=1e-4), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_loop_modes_and_column_order_agree(self, collected, tmp_path):
        """The double-load fix: cached vs re-streamed test paths must be
        identical, and the lazily discovered columns must equal disk order."""
        res_false = _make_attr(tmp_path / "a", normalized=False).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=False,
        )
        res_true = _make_attr(tmp_path / "b", normalized=False).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
            loop_over_test=True,
        )
        assert res_false.test_ids == res_true.test_ids
        assert res_false.row_train_ids == res_true.row_train_ids
        assert res_false.row_steps == res_true.row_steps
        assert torch.equal(res_false.scores, res_true.scores)

        # Columns discovered on-the-fly must match the on-disk step-0 order.
        assert res_false.test_ids == _disk_test_column_order(collected["test_dir"], 0)
