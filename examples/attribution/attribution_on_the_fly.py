"""This example shows on-the-fly attribution: score live, nothing written to disk."""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments

IN, HID, OUT = 8, 16, 4


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN, HID, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID, OUT, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
    """Yields ``{"x", "y"}``.  The task's loss runs the model on this batch, so
    the hooks capture each sample's gradient (and its content hash).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=6)
    parser.add_argument("--n_test", type=int, default=3)
    args_cli = parser.parse_args()
    n_train, n_test = args_cli.n_train, args_cli.n_test

    # generate the toy model and data
    torch.manual_seed(0)
    model = MLP().eval()
    g = torch.Generator().manual_seed(0)
    x_tr, y_tr = (
        torch.randn(n_train, IN, generator=g),
        torch.randn(n_train, OUT, generator=g),
    )
    x_te, y_te = (
        torch.randn(n_test, IN, generator=g),
        torch.randn(n_test, OUT, generator=g),
    )
    train_ds, test_ds = DictDataset(x_tr, y_tr), DictDataset(x_te, y_te)

    # describe the attribution target with a dattri AttributionTask: the loss is
    # functorch-style (params, data) -> loss; checkpoints is the list of model
    # states to score at (here just the current weights)
    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    checkpoint = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[checkpoint])

    with tempfile.TemporaryDirectory() as tmp:
        # attribute -- one call streams the gradients live and scores them;
        # nothing is persisted.  Pass normalized_grad=True for GradCos, or a
        # hook_config=HookManagerConfig(...) to control which layers are hooked
        # (and, optionally, per-layer random projection).
        attr_args = AttributionArguments(
            output_dir=tmp,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        attributor = TracInAttributor(attr_args, task=task)
        score = attributor.attribute(train_ds, test_ds)

        # agnostic_matrix() returns the full (num_train, num_test) matrix; its
        # rows/cols follow the order the samples were streamed in.  (To look a
        # sample up by identity instead, use score.query(train_hashes, ...).)
        train_ids, matrix = score.agnostic_matrix()

        print("TracIn score[train i, test j]")
        header = f"{'':<10}" + "".join(
            f"{f'test{j}':>12}" for j in range(matrix.shape[1])
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, row in enumerate(matrix):
            print(f"{f'train{i}':<10}" + "".join(f"{v:>+12.4f}" for v in row.tolist()))
        print("-" * len(header))

        # a typical TDA question: which training example most influences each
        # test prediction?  (largest inner product down each column)
        print(f"\n{'Test sample':<15}{'Most influential train':<25}{'Score':>10}")
        print("-" * 50)
        for j in range(matrix.shape[1]):
            i = int(matrix[:, j].argmax())
            print(f"{f'test{j}':<15}{f'train{i}':<25}{matrix[i, j]:>+10.3f}")
        print("-" * 50)
