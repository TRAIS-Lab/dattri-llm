"""This example shows how HookManagerConfig selects which layers to capture, and how."""

from __future__ import annotations

import pathlib
import sys

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, REGISTER_ALL


class TinyNet(nn.Module):
    """A small model with a few named sub-blocks to select between."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(8, 16))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(16, 8))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.mlp(h)
        h = self.norm(h)
        return self.head(h)


if __name__ == "__main__":
    # every layer of TinyNet that owns a trainable parameter
    trainable = [n for n, m in TinyNet().named_modules()
                 if any(p.requires_grad for p in m.parameters(recurse=False))]
    print(f"{'Layers with trainable parameters':<40}{trainable}")

    # Two hook families exist.  ``linear_io`` registers factorized ("ghost")
    # per-sample hooks on linear-family layers; ``param_grad`` registers
    # materialized ``param.grad`` hooks on any trainable layer.  A config picks
    # layers by explicit assignment (hook_types), regex selectors (linear_io /
    # param_grad), REGISTER_ALL, or the zero-argument default (linear_io
    # everywhere it applies, param_grad for the rest).
    configs = [
        ("1. explicit assignment (fc1/fc2 ghost, head param_grad)",
         HookManagerConfig(hook_types={
             "mlp.fc1": "linear_io",
             "mlp.fc2": "linear_io",
             "head": "param_grad",
         })),
        ("2. regex selector (linear_io on mlp.* only)",
         HookManagerConfig(linear_io=[r"mlp\."])),
        ("3. REGISTER_ALL (every linear-IO-capable layer)",
         HookManagerConfig(linear_io=REGISTER_ALL)),
        ("4. default config (ghost where possible, param_grad fallback)",
         HookManagerConfig()),
        ("5. mixed (head param_grad + regex linear_io on mlp.*)",
         HookManagerConfig(hook_types={"head": "param_grad"},
                           linear_io=[r"mlp\."])),
    ]

    print()
    print(f"{'Configuration':<62}{'linear_io (ghost)':<50}{'param_grad'}")
    print("-" * 124)
    for title, config in configs:
        hm = HookManager(TinyNet(), config=config)
        print(f"{title:<62}{str(hm.layer_names):<50}{hm.param_layer_names}")
        hm.remove()
    print("-" * 124)
