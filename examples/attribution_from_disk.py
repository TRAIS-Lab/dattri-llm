"""Store-then-attribute (workflow 2): collect gradients to disk, then score.

Stage 1 — collect: ``HookManager`` + ``OffloadCallback`` run forward/backward and
persist the per-sample factorized gradients via ``GradientFileManager``.
Stage 2 — attribute: ``TracInAttributor.attribute_from_cache(train_dir, test_dir)``
scores them by inner product ``score[i, j] = <g_train_i, g_test_j>`` — no model
or backward pass needed.

Because the gradients live on disk, Stage 2 is re-runnable (different layers or
methods) without recomputing them — the advantage over on-the-fly.

Run:
    python examples/attribution_from_disk.py
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

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.tracin import TracInAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, REGISTER_ALL
from dattri_llm.gradient.utils import hash_sample


# --------------------------------------------------------------------------- #
# Model + data
# --------------------------------------------------------------------------- #

IN, HID, OUT = 8, 16, 4


class MLP(nn.Module):
    """Bias-free 2-layer MLP; ``forward`` accepts (and ignores) ``y`` so the
    input hash sees the same ``{x, y}`` dict the dataset yields."""

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


# --------------------------------------------------------------------------- #
# Stage 1 — collect per-sample gradients to disk
# --------------------------------------------------------------------------- #

def collect_to_disk(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                    out_dir: str) -> GradientFileManager:
    """One full-batch forward+backward, offloading per-sample gradients.

    The loss is a **sum over samples** so that each sample's captured
    ``grad_output`` is exactly its own gradient — the invariant that makes the
    factorized per-sample gradient equal ``dL_i/dW``.
    """
    fm = GradientFileManager(out_dir)
    offload = OffloadCallback(
        offload_interval=1, file_manager=fm, recording_type="per_sample",
    )
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[offload],
    )
    with hm.collect():
        model.zero_grad(set_to_none=True)
        out = model(x=x, y=y)
        (((out - y) ** 2).sum()).backward()
    hm.remove()
    model.zero_grad(set_to_none=True)
    return fm


def main() -> None:
    torch.manual_seed(0)
    model = MLP().eval()
    g = torch.Generator().manual_seed(0)
    x_tr, y_tr = torch.randn(6, IN, generator=g), torch.randn(6, OUT, generator=g)
    x_te, y_te = torch.randn(3, IN, generator=g), torch.randn(3, OUT, generator=g)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        train_dir, test_dir = str(tmp / "train_grads"), str(tmp / "test_grads")

        # ---- Stage 1: collect ------------------------------------------------
        print("Stage 1 — collecting gradients to disk …")
        fm_tr = collect_to_disk(model, x_tr, y_tr, train_dir)
        collect_to_disk(model, x_te, y_te, test_dir)
        print(f"  train: {len(fm_tr.index)} sample records on disk, "
              f"steps={fm_tr.available_steps()}")

        # ---- Stage 2: attribute (no model / no backward) ---------------------
        print("\nStage 2 — attributing from cached gradients …")
        args = AttributionArguments(
            output_dir=str(tmp / "scores"),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        attr = TracInAttributor(args)
        score = attr.attribute_from_cache(train_dir, test_dir)

        train_ids, matrix = score.agnostic_matrix()
        print(f"  influence matrix: {tuple(matrix.shape)}  (num_train, num_test)")

        # Rows/cols are keyed by content hash; realign to the natural 0..N order
        # via hash_sample so we can print an interpretable matrix.
        te_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(x_te.shape[0])]
        tr_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(x_tr.shape[0])]
        ordered = score.query(tr_hashes, te_hashes, trajectory="agnostic")

        print("\n  TracIn score[train i, test j]:")
        header = "        " + "".join(f"  test{j:<6}" for j in range(len(te_hashes)))
        print(header)
        for i, row in enumerate(ordered):
            cells = "".join(f"  {v:+9.4f}" for v in row.tolist())
            print(f"  train{i}{cells}")


if __name__ == "__main__":
    main()
