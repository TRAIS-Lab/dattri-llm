"""This example runs every attributor beyond TracIn on one toy problem.

K-FAC, EK-FAC, LESS, DVEmb and AdamW-influence each score the same small MLP
and data through the one-call live ``attribute(...)`` workflow.  The last three
are trajectory methods: their ``attribute(...)`` trains the model for one
epoch, as configured by ``AttributionArguments``, while it captures, so every
method starts from a freshly built model.  Each prints the most influential training
sample per test sample.

Needs the ``transformers`` extra (the live ``attribute(...)`` path).
"""

from __future__ import annotations

import argparse
import tempfile

import torch
from torch import nn
from torch.utils.data import Dataset

from dattri_llm import (
    AdamWInfluenceAttributor,
    AttributionArguments,
    AttributionTask,
    DVEmbAttributor,
    EKFACAttributor,
    HookManagerConfig,
    KFACAttributor,
    LESSAttributor,
)

IN, HID, OUT = 8, 16, 4
LAYERS = ["fc1", "fc2"]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class DictDataset(Dataset):
    """Yields ``{"x", "y"}``; the task's loss runs the model on the batch."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def loss_func(model, batch):
    return ((model(batch["x"]) - batch["y"]) ** 2).sum()


def make_task() -> AttributionTask:
    # a fresh model per method: the trajectory methods update it in place
    torch.manual_seed(0)
    return AttributionTask(loss_func=loss_func, model=MLP())


def make_args(out_dir: str) -> AttributionArguments:
    # the optimizer fields describe the training run the trajectory methods
    # follow; K-FAC and EK-FAC ignore them
    return AttributionArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
        learning_rate=0.05,
        weight_decay=0.01,
        max_grad_norm=None,
        lr_scheduler_type="constant",
    )


def report(name: str, score) -> None:
    # agnostic_matrix() returns the (num_train, num_test) matrix; its rows
    # follow the order the samples were streamed in (score.query(...) looks a
    # sample up by content hash instead)
    _, matrix = score.agnostic_matrix()
    best = [int(matrix[:, j].argmax()) for j in range(matrix.shape[1])]
    cells = "  ".join(f"test{j}<-train{i}" for j, i in enumerate(best))
    print(f"{name:<18}{tuple(matrix.shape)!s:<10}{cells}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=3)
    cli = parser.parse_args()

    g = torch.Generator().manual_seed(0)
    train_ds = DictDataset(
        torch.randn(cli.n_train, IN, generator=g),
        torch.randn(cli.n_train, OUT, generator=g),
    )
    test_ds = DictDataset(
        torch.randn(cli.n_test, IN, generator=g),
        torch.randn(cli.n_test, OUT, generator=g),
    )
    hooks = HookManagerConfig(linear_io=[f"{n}$" for n in LAYERS])

    print(f"{'method':<18}{'shape':<10}most influential train sample per test")
    print("-" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        # Kronecker-factored influence at the model's current weights: the
        # covariances are fit from the training gradients, then damped
        for name, cls in (("K-FAC", KFACAttributor), ("EK-FAC", EKFACAttributor)):
            attr = cls(make_args(f"{tmp}/{name}"), task=make_task())
            report(name, attr.attribute(train_ds, test_ds, damping=1e-3))

        # LESS, trajectory form: trains for one epoch and scores each step's
        # Adam update direction against the query gradients
        attr = LESSAttributor(make_args(f"{tmp}/less"), task=make_task())
        report(
            "LESS",
            attr.attribute(train_ds, test_ds, hook_config=hooks, enable_update=True),
        )

        # DVEmb: data value embeddings along the training trajectory; the
        # learning-rate schedule matches the run (constant, args.learning_rate)
        attr = DVEmbAttributor(make_args(f"{tmp}/dvemb"), task=make_task())
        report("DVEmb", attr.attribute(train_ds, test_ds, learning_rate=0.05))

        # AdamW-influence: first-order influence unrolled through the AdamW
        # trajectory from the recorded optimizer moments
        attr = AdamWInfluenceAttributor(make_args(f"{tmp}/adamw"), task=make_task())
        report(
            "AdamW-influence",
            attr.attribute(train_ds, test_ds, hook_config=hooks, loss_reduction="sum"),
        )
    print("-" * 70)
