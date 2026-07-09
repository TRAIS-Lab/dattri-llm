"""This example shows random projection of per-sample gradients, end to end."""

from __future__ import annotations

import pathlib
import sys

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import CaptureCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.utils.module import rms_norm_module_kwargs

B, T, VOCAB, EMBED, HIDDEN, OUT = 8, 6, 128, 128, 256, 32
PROJ_DIM = 64


class TinyNet(nn.Module):
    """Embedding -> MLP -> LayerNorm -> head: one layer of every family."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, EMBED)
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(EMBED, HIDDEN))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HIDDEN, EMBED))
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, OUT)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.mlp(self.embed(input_ids))))


class HandRolledRMSNorm(nn.Module):
    """RMSNorm with the right math but non-standard attribute names -- the
    shape of HF's ``LlamaRMSNorm``, whose epsilon lives in ``variance_epsilon``.
    Automatic hyperparameter extraction cannot read it; stage 6 declares the
    type and kwargs instead.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.variance_epsilon)


class CustomNet(nn.Module):
    """Embedding -> Linear -> HandRolledRMSNorm (bias-free for easy checking)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, EMBED)
        self.fc = nn.Linear(EMBED, EMBED, bias=False)
        self.norm = HandRolledRMSNorm(EMBED, eps=1e-6)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(self.embed(input_ids)))


# Per-layer projection config.  Two styles exist, chosen per layer by
# ``factorize``: True (LoGRA) projects the two factors independently and the
# layer stays factorized at width proj_dim -- defined for outer-product
# gradients (linear/conv families, embeddings via one-hot inputs); False
# (TRAK) materializes the per-sample weight gradient and projects it to a
# dense (B, proj_dim) block -- required for norm layers.  "__default__"
# covers every hooked layer without its own entry.  Keep ``proj_seed`` fixed
# and the projection device consistent across everything scored together:
# different seeds (or dattri's CPU vs CUDA projectors) are different
# projections.
PROJ_KWARGS = {
    "__default__": {  # LoGRA: project both factors, stay factorized
        "factorize": True,
        "proj_dim": PROJ_DIM,
        "proj_max_batch_size": 8,
        "proj_type": "rademacher",
        "proj_seed": 7,
    },
    "norm": {  # TRAK: materialize the per-sample gradient, then project
        "factorize": False,
        "proj_dim": PROJ_DIM,
        "proj_max_batch_size": 8,
        "proj_type": "rademacher",
        "proj_seed": 7,
    },
}


def collect_one_step(
    config: HookManagerConfig,
    batch: torch.Tensor,
    model: nn.Module | None = None,
):
    """One forward+backward under *config*; returns the captured Gradient."""
    model = model if model is not None else TinyNet()
    capture = CaptureCallback()
    hm = HookManager(model, config=config, callbacks=[capture])
    with hm.collect():
        model(batch).pow(2).sum().backward()
    hm.remove()
    return capture.record.gradient


def describe(gradient, title: str) -> int:
    """Print a per-layer payload table; returns the total element count."""
    print(f"\n{title}")
    print("-" * 76)
    print(f"{'layer':<10}{'representation':<16}{'payload shape(s)':<38}numel")
    total = 0
    for name in sorted(gradient.layer_names):
        val = gradient.data[name]
        if hasattr(val, "activation"):  # Factorized
            shapes = (
                f"{tuple(val.activation.shape)} x "
                f"{tuple(val.pre_activation_grad.shape)}"
            )
            numel = val.activation.numel() + val.pre_activation_grad.numel()
        else:  # materialized tensor
            shapes = f"{tuple(val.shape)}"
            numel = val.numel()
        total += numel
        print(f"{name:<10}{gradient.representation[name]:<16}{shapes:<38}{numel}")
    print(f"{'total':<64}{total}")
    return total


if __name__ == "__main__":
    torch.manual_seed(1)
    batch = torch.randint(0, VOCAB, (B, T))

    # Stage 1 -- raw capture, the unprojected reference: per-layer factorized
    # payloads whose widths follow the layer dimensions.
    raw = collect_one_step(HookManagerConfig(linear_io=REGISTER_ALL), batch)
    n_raw = describe(raw, "1. Raw capture")

    # Stage 2 -- capture-time projection.  Each backward pass projects the
    # factors on the fly, so the raw factors are never buffered and every
    # projected layer's stored width becomes proj_dim regardless of its size.
    projected = collect_one_step(
        HookManagerConfig(linear_io=REGISTER_ALL, projection=PROJ_KWARGS),
        batch,
    )
    n_proj = describe(
        projected,
        "2. Capture-time projection (LoGRA default, TRAK norm)",
    )
    print(
        f"stored payload: {n_proj}/{n_raw} elements ({n_proj / n_raw:.0%}); "
        f"the ratio improves with layer width",
    )

    # Stage 3 -- post-hoc projection of the raw capture.  Same proj_seed and
    # device means the same projection, so gradients projected at collection
    # time and gradients projected later are directly comparable.
    from dattri.func.projection import random_project

    post_hoc = raw.project(random_project, PROJ_KWARGS)
    max_diff = 0.0
    for name in projected.layer_names:
        a, b = projected.data[name], post_hoc.data[name]
        if hasattr(a, "activation"):
            a = ops.materialize(a, projected.layer_types[name])
            b = ops.materialize(b, post_hoc.layer_types[name])
        max_diff = max(max_diff, (a - b).abs().max().item())
    print(f"\n3. capture-time vs post-hoc projection: max |diff| = {max_diff:.2e}")

    # Stage 4 -- why projection is sound for attribution: random projection
    # approximately preserves inner products (Johnson-Lindenstrauss), and
    # attribution scores are gradient dot products.
    sim_raw = raw.similarity(raw, metric="dot", reduce="all")
    sim_proj = projected.similarity(projected, metric="dot", reduce="all")
    corr = torch.corrcoef(
        torch.stack([sim_raw.reshape(-1), sim_proj.reshape(-1)]),
    )[0, 1]
    print(
        f"\n4. pairwise gradient dots, raw vs projected "
        f"({B}x{B} entries): Pearson r = {corr:.3f}",
    )

    # Stage 5 -- per-layer configuration.  The projection map is keyed by
    # layer name, so each layer gets its own budget; without "__default__",
    # layers with no entry are captured raw.
    mixed = collect_one_step(
        HookManagerConfig(
            linear_io=REGISTER_ALL,
            projection={
                "mlp.fc1": {  # wide layer, generous budget
                    "factorize": True,
                    "proj_dim": 128,
                    "proj_max_batch_size": 8,
                    "proj_type": "rademacher",
                    "proj_seed": 7,
                },
                "mlp.fc2": {  # same family, tighter budget
                    "factorize": True,
                    "proj_dim": 32,
                    "proj_max_batch_size": 8,
                    "proj_type": "rademacher",
                    "proj_seed": 7,
                },
                # embed / norm / head: no entry, no "__default__" -> raw
            },
        ),
        batch,
    )
    describe(mixed, "5. Per-layer projection (fc1 @128, fc2 @32, rest raw)")

    # Stage 6 -- custom layer classes.  layer_types declares what the layer
    # IS; module_kwargs supplies its hyperparameters, built with the
    # validated helpers from dattri_llm.utils.module (every field is
    # required, so a forgotten eps fails at config time instead of silently
    # corrupting every captured gradient).
    model = CustomNet()
    custom = collect_one_step(
        HookManagerConfig(
            hook_types={"fc": "linear_io", "norm": "linear_io"},
            layer_types={"norm": "nn.RMSNorm"},
            module_kwargs={
                "norm": rms_norm_module_kwargs(normalized_shape=EMBED, eps=1e-6),
            },
        ),
        batch,
        model=model,
    )
    describe(custom, "6. Custom layer declared via layer_types + module_kwargs")

    # The declaration is verifiably right: summing the captured per-sample
    # gradients reproduces autograd's param.grad (a wrong eps fails here).
    for name, param in (("fc", model.fc.weight), ("norm", model.norm.weight)):
        summed = (
            ops.materialize(custom.data[name], custom.layer_types[name])
            .sum(0)
            .reshape(param.grad.shape)
        )
        diff = (summed - param.grad).abs().max().item()
        print(f"{name:<6}sum of per-sample grads vs param.grad: |diff| = {diff:.2e}")

    # Declared layers flow through projection like any native layer.
    custom_projected = custom.project(random_project, {"norm": PROJ_KWARGS["norm"]})
    print(f"norm projected (TRAK): {tuple(custom_projected.data['norm'].shape)}")
