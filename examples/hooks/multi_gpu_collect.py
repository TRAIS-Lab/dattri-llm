"""This example shows per-sample gradient collection across multiple GPUs (DDP).

Launch with torchrun and each rank collects its own DistributedSampler shard;
rows are keyed by content hash, so the shards recombine into the full dataset
with no duplicates.  Pass the UNWRAPPED model — GradientStreamer registers the
hooks, places the model on its device, and wraps it in DDP/FSDP itself (set
fsdp="full_shard" in AttributionArguments to collect under FSDP instead).

Run:
    torchrun --nproc_per_node=2 examples/hooks/multi_gpu_collect.py          # GPUs
    torchrun --nproc_per_node=2 examples/hooks/multi_gpu_collect.py --cpu    # CPU (gloo)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.streaming import GradientStreamer
from dattri_llm.gradient.hooks import HookManagerConfig, REGISTER_ALL

IN, HID, OUT = 8, 16, 4
N_SAMPLES = 8


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
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Use the gloo backend on CPU (no GPU required).")
    args_cli = parser.parse_args()

    # initialize the process group when launched under torchrun
    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    use_cpu = args_cli.cpu or not torch.cuda.is_available()
    if distributed:
        dist.init_process_group("gloo" if use_cpu else "nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
        print("Not launched under torchrun (world_size=1); collecting one "
              "unsharded pass.  For the multi-process version, run:\n"
              f"    torchrun --nproc_per_node=2 {os.path.relpath(__file__)} --cpu\n")

    # the same dataset on every rank; the DistributedSampler gives each rank a
    # disjoint slice of it
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N_SAMPLES, IN, generator=g)
    y = torch.randn(N_SAMPLES, OUT, generator=g)
    dataset = DictDataset(x, y)
    model = MLP()

    def loss_fn(m, batch):
        out = m(x=batch["x"], y=batch["y"])
        return ((out - batch["y"]) ** 2).sum()

    with tempfile.TemporaryDirectory() as tmp:
        attr_args = AttributionArguments(
            output_dir=tmp,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            use_cpu=use_cpu,
            dataloader_pin_memory=False,
            # fsdp="full_shard",     # <- uncomment (multi-GPU) to collect under FSDP
        )

        # stream one frozen pass; each yielded block is (step, Gradient, hashes)
        # for this rank's shard
        hashes, grads = [], []
        streamer = GradientStreamer(
            model, dataset, attr_args,
            batch_size=attr_args.per_device_train_batch_size,
            enable_update=False, loss_fn=loss_fn,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
        )
        with streamer as s:
            for _step, grad, batch_hashes in s:
                hashes.extend(batch_hashes)
                grads.append(grad)

        print("-" * 50)
        print(f"{'Rank':<10}{'Shard size':<15}{'Dataset size'}")
        print("-" * 50)
        print(f"{f'{rank}/{world}':<10}{len(hashes):<15}{N_SAMPLES}")
        print("-" * 50)

        # gather the shards' hashes on rank 0 just to show they reassemble into
        # the whole dataset.  In practice each rank offloads its gradients to
        # disk and an attributor reads all ranks' directories together.
        if distributed:
            gathered: list = [None] * world
            dist.all_gather_object(gathered, hashes)
            if rank == 0:
                all_hashes = [h for shard in gathered for h in shard]
                print(f"[rank 0] all ranks together collected "
                      f"{len(set(all_hashes))} unique samples "
                      f"(the full dataset of {N_SAMPLES}).")
            dist.barrier()
            dist.destroy_process_group()
