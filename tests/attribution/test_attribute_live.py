"""Live (on-the-fly) ``attribute()`` workflow tests.

The live workflow runs two ``GradientStreamer``s over one model with a
**shared** ``HookManager`` (the test streamer rides the train streamer's
hooks).  ``loop_over_test=True`` re-streams the test blocks between train
blocks, interleaving the two streamers' passes over the shared manager --
the step bookkeeping must survive that interleaving and produce scores
identical to the cached (``loop_over_test=False``) path.
"""

from __future__ import annotations

import pytest
import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.dvemb import DVEmbAttributor
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments

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


def _make_task_and_data():
    torch.manual_seed(SEED)
    model = MLP().eval()
    g = torch.Generator().manual_seed(SEED)
    train_ds = DictDataset(
        torch.randn(N_TRAIN, IN_DIM, generator=g),
        torch.randn(N_TRAIN, OUT_DIM, generator=g),
    )
    test_ds = DictDataset(
        torch.randn(N_TEST, IN_DIM, generator=g),
        torch.randn(N_TEST, OUT_DIM, generator=g),
    )

    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    checkpoint = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[checkpoint])
    return task, train_ds, test_ds


def _args(out_dir) -> AttributionArguments:
    # Batch sizes chosen so both sides span multiple blocks: 3 train blocks
    # and 2 test blocks.  loop_over_test=True re-streams the 2 test blocks
    # after every one of the 3 train blocks -- the interleaving under test.
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
    )


class TestLiveLoopOverTest:
    @pytest.mark.parametrize("normalized_grad", [False, True])
    def test_looped_matches_cached(self, tmp_path, normalized_grad):
        """loop_over_test=True must score identically to the cached path.

        Regression: with the shared HookManager, each test-source re-stream
        reset and advanced the shared step counter mid-train-pass, tripping
        the streamer's step-desync guard on the second train block.
        """
        task, train_ds, test_ds = _make_task_and_data()

        cached = TracInAttributor(_args(tmp_path / "cached"), task=task).attribute(
            train_ds,
            test_ds,
            loop_over_test=False,
            normalized_grad=normalized_grad,
        )
        looped = TracInAttributor(_args(tmp_path / "looped"), task=task).attribute(
            train_ds,
            test_ds,
            loop_over_test=True,
            normalized_grad=normalized_grad,
        )

        ids_c, m_c = cached.agnostic_matrix()
        ids_l, m_l = looped.agnostic_matrix()
        assert ids_c == ids_l
        assert cached.test_ids == looped.test_ids
        assert torch.allclose(m_c, m_l, atol=1e-5), (
            f"max diff {(m_c - m_l).abs().max():.2e}"
        )


class TestCachedGradientsMatchLive:
    """residency={memory,tiered} (collect the raw representations once into a
    store, then replay the Fisher sweeps) must score identically to the default
    residency=None re-streaming path.
    """

    @pytest.mark.parametrize("attr_cls", [KFACAttributor, EKFACAttributor])
    @pytest.mark.parametrize("residency", ["memory", "tiered", "disk"])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_cached_matches_restreamed(
        self,
        tmp_path,
        attr_cls,
        residency,
        loop_over_test,
    ):
        task, train_ds, test_ds = _make_task_and_data()

        # loop_over_test=True re-streams the test set per train block; the cached
        # run then caches the preconditioned test reps in the same residency, so
        # this also covers the residency-managed preconditioned store.
        live = attr_cls(_args(tmp_path / "live"), task=task).attribute(
            train_ds,
            test_ds,
            damping=1e-3,
            loop_over_test=loop_over_test,
        )
        cached = attr_cls(_args(tmp_path / "cached"), task=task).attribute(
            train_ds,
            test_ds,
            damping=1e-3,
            residency=residency,
            loop_over_test=loop_over_test,
        )

        ids_live, m_live = live.agnostic_matrix()
        ids_cached, m_cached = cached.agnostic_matrix()
        assert ids_live == ids_cached
        assert live.test_ids == cached.test_ids
        # loop_over_test=True caches *materialized* preconditioned test reps,
        # so KFAC's whiten-then-materialize adds float32 round-off vs the live
        # factor-domain contraction (EKFAC is already materialized -> exact).
        # loop=False stays in the factor domain and is bit-close.
        atol, rtol = (1e-4, 1e-3) if loop_over_test else (1e-5, 1e-5)
        assert torch.allclose(m_live, m_cached, atol=atol, rtol=rtol), (
            f"max diff {(m_live - m_cached).abs().max():.2e}"
        )


class TestDVEmbResidencyMatchesDisk:
    """DVEmb attribute() must score identically regardless of where the raw
    representations are stored (disk vs memory vs tiered) -- residency only
    relocates the trajectory store, never the sweep math.
    """

    @pytest.mark.parametrize("residency", ["memory", "tiered"])
    def test_residency_matches_disk(self, tmp_path, residency):
        # DVEmb collects an *updating* trajectory (enable_update=True) that
        # mutates the model in place, and its dataloader shuffle consumes global
        # RNG.  So each run needs a fresh task (model reset to init) and the same
        # seed, otherwise the two runs descend different trajectories.  With the
        # trajectory pinned, residency only relocates the store: scores are
        # bit-identical.
        task, train_ds, test_ds = _make_task_and_data()
        torch.manual_seed(SEED)
        disk = DVEmbAttributor(_args(tmp_path / "disk"), task=task).attribute(
            train_ds,
            test_ds,
            residency="disk",
            learning_rate=0.01,
        )
        task, train_ds, test_ds = _make_task_and_data()
        torch.manual_seed(SEED)
        ram = DVEmbAttributor(_args(tmp_path / "ram"), task=task).attribute(
            train_ds,
            test_ds,
            residency=residency,
            learning_rate=0.01,
        )
        ids_d, m_d = disk.agnostic_matrix()
        ids_r, m_r = ram.agnostic_matrix()
        assert ids_d == ids_r
        assert disk.test_ids == ram.test_ids
        assert torch.equal(m_d, m_r), f"max diff {(m_d - m_r).abs().max():.2e}"
