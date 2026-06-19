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

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import make_gradient_multistep_dataloader
from dattri_llm.algorithm.score import AttributionScore
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


def _mapped_oracle(model, checkpoints, x_tr, y_tr, x_te, y_te, step_map, weights):
    total = torch.zeros(N_TRAIN, N_TEST)
    for train_step, test_step in sorted(step_map.items()):
        model.load_state_dict(checkpoints[train_step])
        g_tr = _per_sample_grads(model, x_tr, y_tr)
        model.load_state_dict(checkpoints[test_step])
        g_te = _per_sample_grads(model, x_te, y_te)
        total = total + weights[train_step] * (g_tr @ g_te.T)
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


def _load_step_records(test_dir: Path, step: int):
    fm = GradientFileManager(str(test_dir))
    recs = []
    for file_rel, idxs in fm.iter_step(step):
        all_recs = fm.load_records(file_rel)
        recs.extend(all_recs[i] for i in idxs)
    return recs


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


def _args(out_dir: Path) -> AttributionArguments:
    return AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def _make_attr(out_dir: Path, *, normalized):
    args = _args(out_dir)
    return TracInAttributor(
        args,
        layer_name=LAYER_NAMES,
        normalized_grad=normalized,
        weight_list=None if normalized else STEP_WEIGHTS,
    )


class TestTracInOnDisk:
    def test_steps_are_auto_discovered_from_gradient_files(self, collected, tmp_path):
        """Attribution uses the common on-disk steps, with no constructor steps."""
        curated = tmp_path / "te_step0_only"
        fm_out = GradientFileManager(str(curated))
        fm_out.save_batch(_load_step_records(collected["test_dir"], 0))

        attr = self._make_weighted_attr(tmp_path / "o", weight_list=[1.0])
        res = attr.attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(curated),
        )

        assert res.algorithm_meta["steps"] == [0]
        assert res.algorithm_meta["weights"] == [1.0]
        assert sorted(set(res.row_steps)) == [0]

    def test_multistep_loader_loads_mixed_step_file_once(
        self, collected, tmp_path, monkeypatch
    ):
        mixed = tmp_path / "mixed"
        fm_out = GradientFileManager(str(mixed))
        fm_out.save_batch(
            _load_step_records(collected["train_dir"], 0)
            + _load_step_records(collected["train_dir"], 1)
        )
        reader = GradientFileManager(str(mixed))

        calls = []
        original = GradientFileManager.load_records

        def counted(self, file_relpath):
            calls.append(file_relpath)
            return original(self, file_relpath)

        monkeypatch.setattr(GradientFileManager, "load_records", counted)
        blocks = list(
            make_gradient_multistep_dataloader(reader, [0, 1], _args(tmp_path / "o"))
        )

        assert len(blocks) == 1
        assert set(blocks[0]) == {0, 1}
        assert len(calls) == 1

    def test_explicit_step_map_allows_different_test_steps(self, collected, tmp_path):
        curated = tmp_path / "te_step0_only"
        fm_out = GradientFileManager(str(curated))
        fm_out.save_batch(_load_step_records(collected["test_dir"], 0))

        step_map = {0: 0, 1: 0}
        attr = self._make_weighted_attr(
            tmp_path / "o", weight_list=STEP_WEIGHTS, step_map=step_map
        )
        res = attr.attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(curated),
        )

        matrix = res.query(
            collected["train_hashes"], collected["test_hashes"], trajectory="agnostic"
        )
        oracle = _mapped_oracle(
            collected["model"], collected["checkpoints"],
            collected["x_tr"], collected["y_tr"],
            collected["x_te"], collected["y_te"],
            step_map, STEP_WEIGHTS,
        )
        assert torch.allclose(matrix, oracle, atol=1e-5, rtol=1e-4)
        assert res.algorithm_meta["steps"] == [0, 1]
        assert res.algorithm_meta["train_steps"] == [0, 1]
        assert res.algorithm_meta["test_steps"] == [0, 0]
        assert res.algorithm_meta["step_map"] == [[0, 0], [1, 0]]

    def test_invalid_step_map_raises(self, collected, tmp_path):
        attr = self._make_weighted_attr(
            tmp_path / "o", weight_list=[1.0], step_map={99: 0}
        )
        with pytest.raises(ValueError, match="Invalid step_map"):
            attr.attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
            )

    def test_weight_list_matches_auto_discovered_steps(self, collected, tmp_path):
        attr = self._make_weighted_attr(tmp_path / "o", weight_list=[1.0])
        with pytest.raises(ValueError, match="weight_list length"):
            attr.attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
            )

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

    def test_score_metadata_uses_algorithm_meta(self, collected, tmp_path):
        """Method-specific metadata stays under algorithm_meta only."""
        out_dir = tmp_path / "o"
        res = _make_attr(out_dir, normalized=False).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )

        meta = json.loads((out_dir / "metadata.json").read_text())
        assert meta["algorithm_meta"] == {
            "steps": [0, 1],
            "train_steps": [0, 1],
            "test_steps": [0, 1],
            "step_map": [[0, 0], [1, 1]],
            "weights": STEP_WEIGHTS,
        }
        assert "steps" not in meta
        assert "weights" not in meta
        assert "token_reduction" not in meta

        loaded = AttributionScore.load(out_dir)
        assert loaded.algorithm_meta == res.algorithm_meta
        assert not hasattr(loaded, "token_reduction")

        meta.pop("algorithm_meta")
        (out_dir / "metadata.json").write_text(json.dumps(meta))
        assert AttributionScore.load(out_dir).algorithm_meta == {}

    def _make_weighted_attr(self, out_dir, weight_list, step_map=None):
        args = AttributionArguments(
            output_dir=str(out_dir),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        return TracInAttributor(
            args, layer_name=LAYER_NAMES, normalized_grad=False,
            weight_list=weight_list, step_map=step_map,
        )

    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_vanishing_test_column_raises(self, collected, tmp_path, loop_over_test):
        """A column defined at step 0 but absent at step 1 must error, not zero."""
        # Curate a test dir: step 0 keeps all samples, step 1 drops the last.
        curated = tmp_path / "te_curated"
        fm_out = GradientFileManager(str(curated))
        fm_out.save_batch(_load_step_records(collected["test_dir"], 0))
        fm_out.save_batch(_load_step_records(collected["test_dir"], 1)[:-1])

        attr = self._make_weighted_attr(tmp_path / "o", weight_list=[1.0, 0.5])
        with pytest.raises(KeyError, match=r"consistent across the trajectory"):
            attr.attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(curated),
                loop_over_test=loop_over_test,
            )

    def test_duplicate_content_test_samples_share_a_column(self, tmp_path):
        """Identical-content test samples collapse to one column (no zero column)."""
        torch.manual_seed(SEED)
        model = MLP().eval()
        checkpoints = _make_checkpoints(model)
        x_tr, y_tr, x_te, y_te = _make_data()
        x_te[1], y_te[1] = x_te[0].clone(), y_te[0].clone()  # duplicate sample 0

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        _collect_to_disk(model, checkpoints, x_tr, y_tr, train_dir)
        _collect_to_disk(model, checkpoints, x_te, y_te, test_dir)

        res = _make_attr(tmp_path / "o", normalized=False).attribute(
            train_gradients_dir=str(train_dir), test_gradients_dir=str(test_dir),
        )
        distinct = len({hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)})
        assert distinct == N_TEST - 1
        assert len(res.test_ids) == distinct          # collapsed, not N_TEST
        assert int((res.scores.abs().sum(0) == 0).sum()) == 0  # no spurious zero column
