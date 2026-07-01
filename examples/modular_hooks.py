"""Modular hooks: choose which layers to capture, and how.

``HookManagerConfig`` composes the set of hooked layers per family:
``linear_io`` (factorized "ghost" hooks on linear-family layers) and
``param_grad`` (materialized ``param.grad`` hooks for any trainable layer).

Layers are selected by explicit assignment (``hook_types={"layer": "linear_io"}``),
a regex selector (``linear_io=[r"mlp\\."]``), or ``REGISTER_ALL``. The default
``HookManagerConfig()`` uses ``linear_io`` wherever it can and ``param_grad``
for the rest.

Run:
    python examples/modular_hooks.py
"""

from __future__ import annotations

import pathlib
import sys

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, REGISTER_ALL


# --------------------------------------------------------------------------- #
# A small model with a couple of named sub-blocks to select between.
# --------------------------------------------------------------------------- #

class TinyNet(nn.Module):
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


def show(title: str, config: HookManagerConfig) -> None:
    model = TinyNet()
    hm = HookManager(model, config=config)
    # linear_io (factorized) and param_grad (materialized) layers are reported
    # separately: `layer_names` vs `param_layer_names`.
    print(title)
    print(f"    linear_io (ghost)     -> {hm.layer_names}")
    print(f"    param_grad (materialized) -> {hm.param_layer_names}\n")
    hm.remove()


def main() -> None:
    print("All layers with trainable parameters:")
    print("   ", [n for n, m in TinyNet().named_modules()
                  if any(p.requires_grad for p in m.parameters(recurse=False))])
    print()

    # 1. Explicit per-layer assignment: capture only the two MLP linears,
    #    factorized; give the head a materialized param_grad hook instead.
    show(
        "1. Explicit assignment (hook_types) — fc1/fc2 ghost, head materialized:",
        HookManagerConfig(hook_types={
            "mlp.fc1": "linear_io",
            "mlp.fc2": "linear_io",
            "head": "param_grad",
        }),
    )

    # 2. Regex selector: linear_io on every layer whose name matches ``mlp.``.
    #    Selectors *extend* the explicit assignment without spelling out names.
    show(
        "2. Regex selector (linear_io=[r'mlp\\.']) — just the MLP sub-block:",
        HookManagerConfig(linear_io=[r"mlp\."]),
    )

    # 3. REGISTER_ALL: linear_io on every linear-IO-capable layer in the model
    #    (embedding + linears + norm are all factorizable).
    show(
        "3. REGISTER_ALL — every linear-IO-capable layer:",
        HookManagerConfig(linear_io=REGISTER_ALL),
    )

    # 4. The zero-argument default: linear_io everywhere it can, param_grad for
    #    anything left over.  Equivalent to HookManagerConfig().
    show(
        "4. Default HookManagerConfig() — ghost where possible, param_grad fallback:",
        HookManagerConfig(),
    )

    # 5. Mixed: an explicit param_grad assignment for the head, extended by a
    #    regex linear_io selector for the MLP.  Everything else stays unhooked.
    show(
        "5. Mixed — head=param_grad, plus regex linear_io on mlp.*:",
        HookManagerConfig(
            hook_types={"head": "param_grad"},
            linear_io=[r"mlp\."],
        ),
    )


if __name__ == "__main__":
    main()
