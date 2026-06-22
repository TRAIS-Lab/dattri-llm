"""Correctness tests for :class:`TracInAttributor` (on-disk workflow).

These collect real per-sample gradients to disk with the repo's
``HookManager`` + ``OffloadCallback`` pipeline, then check the attributor's
score against an independent autograd oracle.

The simplified attributor computes the full ``(num_train, num_test)`` gradient
cross-gram — every train record against every test record, with no train/test
step alignment — structurally identical to the K-FAC family.  Rows are stamped
with the step each train gradient was recorded at, so a sample collected at
several checkpoints contributes one (trajectory-aware) row per checkpoint.

The ``loop_over_test`` parametrization guards the lazy column-discovery
optimisation: both the cached (``False``) and re-streamed (``True``) paths must
yield byte-identical results with the same column order as the on-disk layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import make_gradient_multistep_dataloader
from dattri_llm.algorithm.tracin import TracInAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.utils import hash_sample

IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_TRAIN, N_TEST = 6, 4
SEED = 0
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


def _grads_at(model, sd, x, y, *, normalized):
    model.load_state_dict(sd)
    g = _per_sample_grads(model, x, y)
    return torch.nn.functional.normalize(g, dim=-1) if normalized else g


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


def _args(out_dir: Path) -> AttributionArguments:
    return AttributionArguments(
        output_dir=str(out_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def _make_attr(out_dir: Path, *, normalized):
    return TracInAttributor(
        _args(out_dir), layer_name=LAYER_NAMES, normalized_grad=normalized
    )


@pytest.fixture()
def collected(tmp_path):
    """Single-checkpoint train + test dirs (mirrors the K-FAC fixture)."""
    torch.manual_seed(SEED)
    model = MLP().eval()
    sd = _make_checkpoints(model)[0]
    x_tr, y_tr, x_te, y_te = _make_data()
    train_dir, test_dir = tmp_path / "train_g", tmp_path / "test_g"
    _collect_to_disk(model, [sd], x_tr, y_tr, train_dir)
    _collect_to_disk(model, [sd], x_te, y_te, test_dir)
    train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(N_TRAIN)]
    test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)]
    return dict(
        model=model, sd=sd,
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        train_dir=train_dir, test_dir=test_dir,
        train_hashes=train_hashes, test_hashes=test_hashes,
    )


class TestTracInOnDisk:
    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_matches_autograd_oracle(self, collected, tmp_path, normalized, loop_over_test):
        """Full pairwise cross-gram matches an independent autograd oracle."""
        g_tr = _grads_at(collected["model"], collected["sd"],
                         collected["x_tr"], collected["y_tr"], normalized=normalized)
        g_te = _grads_at(collected["model"], collected["sd"],
                         collected["x_te"], collected["y_te"], normalized=normalized)
        oracle = g_tr @ g_te.T

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

    def test_algorithm_label_and_shape(self, collected, tmp_path):
        res = _make_attr(tmp_path / "a", normalized=False).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert res.algorithm == "TracIn"
        assert res.scores.shape == (N_TRAIN, N_TEST)
        assert res.algorithm_meta == {"selected_training_steps": [0]}

        gradcos = _make_attr(tmp_path / "b", normalized=True).attribute(
            train_gradients_dir=str(collected["train_dir"]),
            test_gradients_dir=str(collected["test_dir"]),
        )
        assert gradcos.algorithm == "GradCos"

    def test_loop_modes_and_column_order_agree(self, collected, tmp_path):
        """Cached vs re-streamed test paths must be identical, and the lazily
        discovered columns must equal the on-disk order."""
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

        assert res_false.test_ids == _disk_test_column_order(collected["test_dir"], 0)

    def test_missing_gradients_dir_raises(self, collected, tmp_path):
        attr = _make_attr(tmp_path / "o", normalized=False)
        with pytest.raises(ValueError, match=r"train_gradients_dir"):
            attr.attribute(test_gradients_dir=str(collected["test_dir"]))

    def test_multistep_loader_loads_mixed_step_file_once(
        self, tmp_path, monkeypatch
    ):
        """A file holding several steps is ``torch.load``-ed exactly once."""
        torch.manual_seed(SEED)
        model = MLP().eval()
        checkpoints = _make_checkpoints(model)
        x_tr, y_tr, _, _ = _make_data()
        raw = tmp_path / "raw"
        _collect_to_disk(model, checkpoints, x_tr, y_tr, raw)

        mixed = tmp_path / "mixed"
        fm_out = GradientFileManager(str(mixed))
        fm_out.save_batch(_load_step_records(raw, 0) + _load_step_records(raw, 1))
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

    def test_multistep_train_rows_are_trajectory_aware(self, tmp_path):
        """A train dir with two checkpoints yields one row per (sample, step),
        each stamped with its recorded step; the step-summed score equals the
        sum of per-checkpoint cross-grams against the (single-step) test set."""
        torch.manual_seed(SEED)
        model = MLP().eval()
        sd0, sd1 = _make_checkpoints(model)
        x_tr, y_tr, x_te, y_te = _make_data()

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        _collect_to_disk(model, [sd0, sd1], x_tr, y_tr, train_dir)
        _collect_to_disk(model, [sd0], x_te, y_te, test_dir)  # single-step test

        train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(N_TRAIN)]
        test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)]

        res = _make_attr(tmp_path / "o", normalized=False).attribute(
            train_gradients_dir=str(train_dir), test_gradients_dir=str(test_dir),
        )
        # Two steps per train sample → 2 * N_TRAIN rows, stamped {0, 1}.
        assert res.scores.shape[0] == 2 * N_TRAIN
        assert sorted(set(res.row_steps)) == [0, 1]

        g_te0 = _grads_at(model, sd0, x_te, y_te, normalized=False)
        oracle = (
            _grads_at(model, sd0, x_tr, y_tr, normalized=False) @ g_te0.T
            + _grads_at(model, sd1, x_tr, y_tr, normalized=False) @ g_te0.T
        )
        matrix = res.query(train_hashes, test_hashes, trajectory="agnostic")
        assert torch.allclose(matrix, oracle, atol=1e-5, rtol=1e-4), (
            f"max diff {(matrix - oracle).abs().max().item():.2e}"
        )

    def test_selected_training_steps_selects_checkpoints(self, tmp_path):
        """``selected_training_steps=[1]`` attributes only from checkpoint 1:
        rows are all stamped step 1 and the score equals that single
        checkpoint's cross-gram (no ensemble over the dropped step 0)."""
        torch.manual_seed(SEED)
        model = MLP().eval()
        sd0, sd1 = _make_checkpoints(model)
        x_tr, y_tr, x_te, y_te = _make_data()

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        _collect_to_disk(model, [sd0, sd1], x_tr, y_tr, train_dir)
        _collect_to_disk(model, [sd0], x_te, y_te, test_dir)

        train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(N_TRAIN)]
        test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)]

        res = _make_attr(tmp_path / "o", normalized=False).attribute(
            train_gradients_dir=str(train_dir), test_gradients_dir=str(test_dir),
            selected_training_steps=[1],
        )

        assert res.scores.shape[0] == N_TRAIN            # only one step's rows
        assert sorted(set(res.row_steps)) == [1]
        assert res.algorithm_meta["selected_training_steps"] == [1]

        oracle = (
            _grads_at(model, sd1, x_tr, y_tr, normalized=False)
            @ _grads_at(model, sd0, x_te, y_te, normalized=False).T
        )
        matrix = res.query(train_hashes, test_hashes, trajectory="agnostic")
        assert torch.allclose(matrix, oracle, atol=1e-5, rtol=1e-4)

    def test_unknown_steps_raise(self, collected, tmp_path):
        attr = _make_attr(tmp_path / "o", normalized=False)
        with pytest.raises(ValueError, match=r"requested steps"):
            attr.attribute(
                train_gradients_dir=str(collected["train_dir"]),
                test_gradients_dir=str(collected["test_dir"]),
                selected_training_steps=[99],
            )

    def test_duplicate_content_test_samples_share_a_column(self, tmp_path):
        """Identical-content test samples collapse to one column (no zero column)."""
        torch.manual_seed(SEED)
        model = MLP().eval()
        sd = _make_checkpoints(model)[0]
        x_tr, y_tr, x_te, y_te = _make_data()
        x_te[1], y_te[1] = x_te[0].clone(), y_te[0].clone()  # duplicate sample 0

        train_dir, test_dir = tmp_path / "tr", tmp_path / "te"
        _collect_to_disk(model, [sd], x_tr, y_tr, train_dir)
        _collect_to_disk(model, [sd], x_te, y_te, test_dir)

        res = _make_attr(tmp_path / "o", normalized=False).attribute(
            train_gradients_dir=str(train_dir), test_gradients_dir=str(test_dir),
        )
        distinct = len({hash_sample({"x": x_te, "y": y_te}, j) for j in range(N_TEST)})
        assert distinct == N_TEST - 1
        assert len(res.test_ids) == distinct          # collapsed, not N_TEST
        assert int((res.scores.abs().sum(0) == 0).sum()) == 0  # no spurious zero column
