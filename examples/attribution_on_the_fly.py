"""On-the-fly attribution (workflow 1): score directly, no gradient files.

One ``TracInAttributor.attribute(train_ds, test_ds)`` call runs forward/backward
live and scores the per-sample gradients as they stream — nothing is persisted.
The model, loss, and checkpoints come from a ``dattri`` ``AttributionTask``;
batch sizes and device from ``AttributionArguments``.

Use this when you attribute once; to re-query the same gradients (different
layers or methods) without recomputing, use ``attribution_from_disk.py``.

Run:
    python examples/attribution_on_the_fly.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dattri.task import AttributionTask

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.tracin import TracInAttributor


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
    the hooks capture each sample's gradient (and its content hash)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def main() -> None:
    # ── 1. Model + data ───────────────────────────────────────────────────────
    torch.manual_seed(0)
    model = MLP().eval()
    g = torch.Generator().manual_seed(0)
    x_tr, y_tr = torch.randn(6, IN, generator=g), torch.randn(6, OUT, generator=g)
    x_te, y_te = torch.randn(3, IN, generator=g), torch.randn(3, OUT, generator=g)
    train_ds, test_ds = DictDataset(x_tr, y_tr), DictDataset(x_te, y_te)

    # ── 2. Describe the attribution target with a dattri AttributionTask ───────
    # The loss is functorch-style ``(params, data) -> loss``; checkpoints is the
    # list of model states to score at (here just the current weights).
    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    checkpoint = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[checkpoint])

    with tempfile.TemporaryDirectory() as tmp:
        # ── 3. Attribute — one call collects gradients live and scores them ────
        args = AttributionArguments(
            output_dir=tmp,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        attr = TracInAttributor(args, layer_name=["mlp.fc1", "mlp.fc2"], task=task)
        score = attr.attribute(train_ds, test_ds)   # GradCos: normalized_grad=True

        # ── 4. Read the result ─────────────────────────────────────────────────
        # agnostic_matrix() returns the full (num_train, num_test) matrix; its
        # rows/cols follow the order the samples were streamed in.  (To look a
        # sample up by identity instead, use score.query(train_hashes, ...).)
        train_ids, matrix = score.agnostic_matrix()

        print(f"Influence matrix {tuple(matrix.shape)}  (num_train, num_test)")
        print("     " + "".join(f"  test{j:<5}" for j in range(matrix.shape[1])))
        for i, row in enumerate(matrix):
            print(f"  train{i}" + "".join(f"  {v:+8.3f}" for v in row.tolist()))

        # A typical TDA question: which training example most influences each
        # test prediction?  (Largest TracIn inner product down each column.)
        print("\nMost influential training example per test example:")
        for j in range(matrix.shape[1]):
            i = int(matrix[:, j].argmax())
            print(f"  test{j}  <-  train{i}  (score {matrix[i, j]:+.3f})")


if __name__ == "__main__":
    main()
