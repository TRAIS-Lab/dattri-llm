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
