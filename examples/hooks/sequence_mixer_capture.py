"""This example shows per-sample gradient capture on a state-space (Mamba) model.

A Mamba mixer owns ordinary modules -- ``in_proj``, ``conv1d``, ``x_proj``,
``dt_proj``, ``out_proj`` -- but its forward applies some of their weights
functionally (``causal_conv1d_fn`` on ``conv1d.weight``, ``dt_proj.weight`` as
a bare matmul) or hands all of them to a fused kernel.  A module whose
forward is never called fires no hook, so such layers would be absent from
the record.  The manager recognises these mixers and, while collecting,
routes their forward through the modules; nothing is declared.  The same
mechanism serves the Mamba-2 family and the gated delta-net mixers of
Qwen3-Next and Kimi-Linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import MambaConfig, MambaForCausalLM

from dattri_llm import HookManager
from dattri_llm.gradient import ops

B, T, VOCAB = 3, 8, 64


def per_sample_loss(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Each sample's mean next-token loss; the batch loss is their sum."""
    logits = model(input_ids=ids).logits[:, :-1]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    return ce.view(ids.shape[0], -1).mean(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    config = MambaConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        state_size=4,
        num_hidden_layers=2,
        intermediate_size=32,
        conv_kernel=4,
        time_step_rank=2,
    )
    model = MambaForCausalLM(config).double()
    ids = torch.randint(0, VOCAB, (B, T))

    # The default manager: the mixers are detected and overridden automatically.
    with torch.no_grad():
        plain = model(input_ids=ids).logits
    hm = HookManager(model)
    with hm.collect():
        logits = model(input_ids=ids).logits
        per_sample_loss(model, ids).sum().backward()
    gradient = hm.get_gradient()
    hm.remove()

    # The override computes the mixer's own function: outputs are unchanged.
    print(
        f"Max |logits with override - plain logits|: {(logits - plain).abs().max():.1e}"
    )

    print("\nRecords of the first layer:")
    for name in sorted(n for n in gradient.data if "layers.0" in n):
        value = gradient.data[name]
        kind = "per-sample" if hasattr(value, "activation") else "batch-level"
        print(
            f"  {name.split('layers.0.')[1]:<22}{gradient.layer_types[name]:<20}{kind}"
        )

    # Every projection of the mixer matches the one-sample autograd gradient,
    # including conv1d (depthwise: groups == channels) and dt_proj.
    params = dict(model.named_parameters())
    worst = 0.0
    for i in range(B):
        model.zero_grad()
        per_sample_loss(model, ids[i : i + 1]).sum().backward()
        for name, value in gradient.data.items():
            if "layers.0.mixer" not in name or not hasattr(value, "activation"):
                continue
            weight = params[name + ".weight"]
            dense = ops.materialize(value, gradient.layer_types[name])[i]
            rows = dense.reshape(weight.shape[0], -1)[:, : weight[0].numel()]
            worst = max(
                worst, (rows.reshape(weight.shape) - weight.grad).abs().max().item()
            )
    print(f"\nMax |captured - autograd| over the mixer's projections: {worst:.1e}")
