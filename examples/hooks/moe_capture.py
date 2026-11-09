"""This example shows per-sample gradient capture on a Mixture-of-Experts model.

An MoE block hands its router the block input flattened to ``(B*T, d)`` and
each expert the ``(n_e, d)`` rows routed to it, so these layers do not carry
the batch axis the capture reads.  Naming the MoE blocks in
``HookManager(moe_routed_layers=...)`` makes the manager locate every row in
the block input and regroup it per sample.  The same call covers per-expert
``nn.Linear`` experts (Transformers 4) and fused 3-D expert parameters
(Transformers 5, gpt-oss), where parameter-free per-expert proxies are
installed while collecting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM

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


def build_model() -> MixtralForCausalLM:
    torch.manual_seed(0)
    config = MixtralConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model = MixtralForCausalLM(config).double()
    if hasattr(model.config, "_experts_implementation"):
        # Transformers 5 stores the experts fused and dispatches to a grouped
        # matmul kernel without a float64 path; the eager loop is exact.
        model.config._experts_implementation = "eager"  # noqa: SLF001
    return model


def reference_weight_grad(name: str, grads: dict[str, torch.Tensor]) -> torch.Tensor:
    """The autograd gradient of the weight a captured layer stands for.

    A per-expert ``nn.Linear`` has its own ``weight``; a fused-expert proxy
    ``<experts>.dattri_experts.<e>.<proj>`` stands for slice ``e`` of the 3-D
    parameter, and ``<router>.dattri_router`` for the router's weight.
    """
    if name + ".weight" in grads:
        return grads[name + ".weight"]
    if name.endswith(".dattri_router"):
        return grads[name.removesuffix(".dattri_router") + ".weight"]
    owner, expert, proj = name.rsplit(".", 3)[0], *name.rsplit(".", 2)[1:]
    param = "gate_up_proj" if proj == "gate_up" else "down_proj"
    return grads[f"{owner}.{param}"][int(expert)]


if __name__ == "__main__":
    model = build_model()
    ids = torch.randint(0, VOCAB, (B, T))
    block = (
        "block_sparse_moe"
        if hasattr(model.model.layers[0], "block_sparse_moe")
        else "mlp"
    )
    moe_blocks = [f"model.layers.{i}.{block}" for i in range(2)]

    # 1. Without the declaration the first backward fails, naming the layer whose
    #    rows do not line up with the batch (the router, fed B*T flattened rows).
    hm = HookManager(model)
    try:
        with hm.collect():
            per_sample_loss(model, ids).sum().backward()
    except ValueError as exc:
        print("Undeclared MoE block:\n  " + str(exc).split(". ")[0])
    finally:
        hm.remove()
        model.zero_grad()

    # 2. Declare the MoE blocks.  Everything else about the capture is unchanged.
    hm = HookManager(model, moe_routed_layers=moe_blocks)
    with hm.collect():
        per_sample_loss(model, ids).sum().backward()
    gradient = hm.get_gradient()
    hm.remove()

    print(f"\nStep batch size: {gradient.batch_size}")
    print("Records of the first MoE block (experts keep the full (B, T, d) layout):")
    for name in sorted(n for n in gradient.data if moe_blocks[0] in n):
        value = gradient.data[name]
        shape = tuple(value.activation.shape) if hasattr(value, "activation") else ()
        print(f"  {name.split(block + '.')[1]:<34}{gradient.indexing[name]:<14}{shape}")

    # 3. Every captured MoE weight gradient equals the one-sample autograd
    #    gradient; an expert a sample sent no token to is exactly zero.
    worst = 0.0
    for i in range(B):
        model.zero_grad()
        per_sample_loss(model, ids[i : i + 1]).sum().backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        for name, value in gradient.data.items():
            if moe_blocks[0] not in name or not hasattr(value, "activation"):
                continue
            ref = reference_weight_grad(name, grads)
            dense = ops.materialize(value, "nn.Linear")[i]
            mine = dense.reshape(ref.shape[0], -1)[:, : ref.shape[1]]
            worst = max(worst, (mine - ref).abs().max().item())
    print(f"\nMax |captured - autograd| over the block's weights: {worst:.1e}")

    # 4. Scores use every layer, experts included: the (B, B) gradient Gram matrix.
    gram = gradient.similarity(gradient, reduce="all")
    print("\nGradient inner products <g_i, g_j>:")
    for row in gram.tolist():
        print("  " + "".join(f"{v:>+12.4f}" for v in row))
