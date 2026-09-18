"""The :class:`AttributionTask`: loss convention, checkpoints, wrappers, and
parity with a dattri task.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.kronecker import KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.task import AttributionTask, as_task, default_loss_func

IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_TRAIN, N_TEST = 6, 4
SEED = 0


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID_DIM, OUT_DIM, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def _loss(model: nn.Module, data: dict) -> torch.Tensor:
    return ((model(**data) - data["y"]) ** 2).sum()


def _data() -> tuple[DictDataset, DictDataset]:
    g = torch.Generator().manual_seed(SEED)
    return (
        DictDataset(
            torch.randn(N_TRAIN, IN_DIM, generator=g),
            torch.randn(N_TRAIN, OUT_DIM, generator=g),
        ),
        DictDataset(
            torch.randn(N_TEST, IN_DIM, generator=g),
            torch.randn(N_TEST, OUT_DIM, generator=g),
        ),
    )


def _model(seed: int = SEED) -> MLP:
    torch.manual_seed(seed)
    return MLP().eval()


def _args(out_dir) -> AttributionArguments:
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
    )


# --------------------------------------------------------------------------- #
# Construction and checkpoints                                                 #
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_defaults(self):
        model = _model()
        task = AttributionTask(_loss, model)
        assert task.model is model
        assert task.get_model() is model
        assert task.forward_model is None
        assert not task.is_wrapped
        assert task.checkpoints == [None]
        assert task.num_checkpoints() == 1
        assert task.target_func is task.loss_func
        assert task.load_checkpoint(0) is model
        assert task.current_checkpoint_idx == 0

    def test_default_loss_is_the_hf_convention(self):
        class WithLoss(nn.Module):
            def forward(self, x, labels):
                class Out:
                    loss = (x - labels).pow(2).sum()

                return Out()

        task = AttributionTask(None, WithLoss())
        assert task.loss_func is default_loss_func
        out = task.loss_func(task.model, {"x": torch.ones(2), "labels": torch.zeros(2)})
        assert out.item() == pytest.approx(2.0)
        with pytest.raises(TypeError, match="dict batch"):
            task.loss_func(task.model, (torch.ones(2), torch.zeros(2)))
        with pytest.raises(ValueError, match=r"no \.loss"):
            AttributionTask(None, _model()).loss_func(
                _model(), {"x": torch.ones(1, IN_DIM)}
            )

    def test_target_func_is_separate(self):
        def target(model, data):
            return model(**data).sum()

        task = AttributionTask(_loss, _model(), target_func=target)
        assert task.target_func is target
        assert task.loss_func is _loss

    def test_empty_checkpoint_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            AttributionTask(_loss, _model(), checkpoints=[])


class TestCheckpoints:
    def test_state_dict_and_path_checkpoints_load(self, tmp_path):
        a, b = _model(0), _model(1)
        path = tmp_path / "b.pt"
        torch.save(b.state_dict(), path)
        model = _model(2)
        task = AttributionTask(
            _loss, model, checkpoints=[a.state_dict(), str(path), None]
        )
        assert task.num_checkpoints() == 3

        assert task.load_checkpoint(0) is model
        for p, q in zip(model.parameters(), a.parameters(), strict=True):
            assert torch.equal(p, q)
        assert task.load_checkpoint(1) is model
        for p, q in zip(model.parameters(), b.parameters(), strict=True):
            assert torch.equal(p, q)
        # None: the parameters the model holds now (still b's).
        assert task.load_checkpoint(2) is model
        for p, q in zip(model.parameters(), b.parameters(), strict=True):
            assert torch.equal(p, q)

    def test_reloading_the_current_checkpoint_is_a_noop(self):
        calls = []

        def loader(model, checkpoint):
            calls.append(checkpoint)
            return model

        task = AttributionTask(
            _loss, _model(), checkpoints=["a", "b"], checkpoints_load_func=loader
        )
        task.load_checkpoint(0)
        task.load_checkpoint(0)
        task.load_checkpoint(1)
        task.load_checkpoint(0)
        assert calls == ["a", "b", "a"]

    def test_loader_may_rebuild_the_model(self):
        fresh = _model(7)

        def loader(model, checkpoint):  # noqa: ARG001
            return fresh

        task = AttributionTask(
            _loss, _model(), checkpoints=["x"], checkpoints_load_func=loader
        )
        assert task.load_checkpoint(0) is fresh
        assert task.model is fresh


# --------------------------------------------------------------------------- #
# Wrapped models                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def one_rank_group():
    """A single-rank gloo group, enough to construct a DDP wrapper on CPU."""
    import torch.distributed as dist

    fd, path = tempfile.mkstemp()
    os.close(fd)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        Path(path).unlink(missing_ok=True)


class TestWrappedModel:
    def test_ddp_wrapper_is_unwrapped_for_hooks(self, one_rank_group):
        model = _model()
        ddp = nn.parallel.DistributedDataParallel(model)
        task = AttributionTask(_loss, ddp)
        assert task.is_wrapped
        assert task.model is model
        assert task.forward_model is ddp
        assert task.load_checkpoint(0) is model

    def test_wrapped_model_refuses_default_state_dict_loading(self, one_rank_group):
        model = _model()
        ddp = nn.parallel.DistributedDataParallel(model)
        with pytest.raises(ValueError, match="wrapped"):
            AttributionTask(_loss, ddp, checkpoints=[model.state_dict()])
        # A loader that knows how to load through the wrapper is accepted.
        task = AttributionTask(
            _loss,
            ddp,
            checkpoints=[model.state_dict()],
            checkpoints_load_func=lambda m, c: m,
        )
        assert task.load_checkpoint(0) is model

    def test_attribute_on_wrapped_model_matches_unwrapped(
        self, one_rank_group, tmp_path
    ):
        train_ds, test_ds = _data()
        plain = TracInAttributor(
            _args(tmp_path / "plain"), task=AttributionTask(_loss, _model())
        )
        ref = plain.attribute(train_ds, test_ds).agnostic_matrix()[1]

        ddp = nn.parallel.DistributedDataParallel(_model())
        wrapped = TracInAttributor(
            _args(tmp_path / "ddp"), task=AttributionTask(_loss, ddp)
        )
        out = wrapped.attribute(train_ds, test_ds).agnostic_matrix()[1]
        assert torch.allclose(out, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# dattri interoperability                                                      #
# --------------------------------------------------------------------------- #


class TestDattriParity:
    def _dattri_task(self, model):
        from dattri.task import AttributionTask as DattriTask

        def loss_func(params, data):
            yhat = torch.func.functional_call(model, params, (data["x"],))
            return ((yhat - data["y"]) ** 2).sum()

        return DattriTask(
            loss_func=loss_func, model=model, checkpoints=[model.state_dict()]
        )

    def test_as_task_adapts_and_is_idempotent(self):
        model = _model()
        task = as_task(self._dattri_task(model))
        assert isinstance(task, AttributionTask)
        assert as_task(task) is task
        assert task.model is model
        batch = {"x": torch.ones(2, IN_DIM), "y": torch.zeros(2, OUT_DIM)}
        assert torch.allclose(task.loss_func(model, batch), _loss(model, batch))

    def test_as_task_rejects_other_objects(self):
        with pytest.raises(TypeError, match="dattri AttributionTask"):
            as_task(object())

    @pytest.mark.parametrize("cls", [TracInAttributor, KFACAttributor])
    def test_scores_match_dattri_task(self, cls, tmp_path):
        train_ds, test_ds = _data()
        kw = {} if cls is TracInAttributor else {"damping": 1e-3}
        ours = cls(_args(tmp_path / "ours"), task=AttributionTask(_loss, _model()))
        theirs = cls(_args(tmp_path / "dattri"), task=self._dattri_task(_model()))
        a = ours.attribute(train_ds, test_ds, **kw).agnostic_matrix()[1]
        b = theirs.attribute(train_ds, test_ds, **kw).agnostic_matrix()[1]
        assert torch.allclose(a, b, atol=1e-6)
