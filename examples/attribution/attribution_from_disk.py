"""This example shows store-then-attribute: collect gradients to disk, then score."""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.tracin import TracInAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, REGISTER_ALL
from dattri_llm.utils.hashing import hash_batch

IN, HID, OUT = 8, 16, 4


class MLP(nn.Module):
    """Bias-free 2-layer MLP; ``forward`` accepts (and ignores) ``y`` so the
    recorded input hash covers the same ``{x, y}`` pair the dataset yields."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN, HID, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID, OUT, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
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
    x_tr, y_tr = torch.randn(n_train, IN, generator=g), torch.randn(n_train, OUT, generator=g)
    x_te, y_te = torch.randn(n_test, IN, generator=g), torch.randn(n_test, OUT, generator=g)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        train_dir, test_dir = str(tmp / "train_grads"), str(tmp / "test_grads")

        # Stage 1 — collect per-sample gradients to disk.  HookManager captures
        # the factorized per-sample gradients during one full-batch backward;
        # OffloadCallback persists them via GradientFileManager.  The loss is a
        # SUM over samples so each captured gradient is that sample's own dL_i/dW.
        print("Stage 1 — collecting per-sample gradients to disk")
        print("-" * 50)
        for split, x, y, out_dir in [("train", x_tr, y_tr, train_dir),
                                     ("test", x_te, y_te, test_dir)]:
            fm = GradientFileManager(out_dir)
            hm = HookManager(
                model,
                config=HookManagerConfig(linear_io=REGISTER_ALL),
                callbacks=[OffloadCallback(offload_interval=1, file_manager=fm,
                                           recording_type="per_sample")],
            )
            with hm.collect():
                model.zero_grad(set_to_none=True)
                out = model(x=x, y=y)
                (((out - y) ** 2).sum()).backward()
            hm.remove()
            model.zero_grad(set_to_none=True)
            print(f"{split:<8}{len(fm.index)} sample records, steps={fm.available_steps()}")
        print("-" * 50)

        # Stage 2 — attribute from the cached gradients.  No model or backward
        # pass is needed; the score is <g_train_i, g_test_j> per pair.  Because
        # the gradients live on disk, this stage can be re-run with different
        # settings (e.g. layer_name=[...] or normalized_grad=True) for free.
        attr_args = AttributionArguments(
            output_dir=str(tmp / "scores"),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        attributor = TracInAttributor(attr_args)
        score = attributor.attribute_from_cache(train_dir, test_dir)

        # Rows/columns are keyed by content hash; realign them to the natural
        # 0..N sample order for printing.
        tr_hashes = hash_batch({"x": x_tr, "y": y_tr})
        te_hashes = hash_batch({"x": x_te, "y": y_te})
        matrix = score.query(tr_hashes, te_hashes, trajectory="agnostic")

        print("\nStage 2 — TracIn score[train i, test j]")
        header = f"{'':<10}" + "".join(f"{f'test{j}':>12}" for j in range(n_test))
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
        for j in range(n_test):
            i = int(matrix[:, j].argmax())
            print(f"{f'test{j}':<15}{f'train{i}':<25}{matrix[i, j]:>+10.3f}")
        print("-" * 50)
