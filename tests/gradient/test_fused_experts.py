"""Fused Mixture-of-Experts capture (Transformers 5 layout).

Transformers 5 stores an MoE block's experts in one module with 3-D
parameters (``gate_up_proj``, ``down_proj``) and its router as a module with
a bare ``(E, d)`` weight; no ``nn.Linear`` exists to hook.  With the block
named in ``moe_routed_layers`` the manager installs parameter-free
per-expert proxies while collecting (see
:mod:`dattri_llm.gradient.hooks.override_layers.fused_experts`).  These tests
pin, on tiny Mixtral (concatenated gate/up, no bias) and gpt-oss (transposed slices with
biases) models in float64: the proxies' per-sample gradients equal those of
a per-sample autograd loop over the real parameters, the fused parameters
are no longer captured at batch level, training gradients are unchanged,
and removing the manager restores the model, including its fused kernel.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import HookManager
from dattri_llm.gradient.hooks.override_layers import find_fused_experts

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "MixtralConfig") or not hasattr(
    transformers.models.mixtral.modeling_mixtral, "MixtralExperts"
):
    pytest.skip(
        "fused experts need the Transformers 5 MoE layout", allow_module_level=True
    )

B, T, VOCAB = 3, 7, 64
COMMON = {
    "vocab_size": VOCAB,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}
BLOCK = "model.layers.0.mlp"


def _build(family: str) -> torch.nn.Module:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "mixtral":
            cfg = transformers.MixtralConfig(
                intermediate_size=32,
                num_local_experts=4,
                num_experts_per_tok=2,
                **COMMON,
            )
            model = transformers.MixtralForCausalLM(cfg)
        elif family == "gpt_oss":
            cfg = transformers.GptOssConfig(
                intermediate_size=32,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=2,
                sliding_window=16,
                layer_types=["full_attention"] * 2,
                **COMMON,
            )
            model = transformers.GptOssForCausalLM(cfg)
        else:
            raise ValueError(family)
    # The grouped-matmul kernel has no float64 path; the reference runs eager.
    model.config._experts_implementation = "eager"  # noqa: SLF001
    return model.double()


def _batch(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T), generator=gen)


def _per_sample_loss(model, ids):
    logits = model(input_ids=ids).logits[:, :-1]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    return ce.view(ids.shape[0], -1).mean(1)


def _reference(model, ids):
    out = []
    for i in range(ids.shape[0]):
        model.zero_grad()
        _per_sample_loss(model, ids[i : i + 1]).sum().backward()
        out.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    model.zero_grad()
    return out


def _collect(model, ids, hm):
    with hm.collect():
        model.zero_grad()
        _per_sample_loss(model, ids).sum().backward()
    return hm.get_gradient()


def _expert_reference(ref: dict, experts, name: str, e: int):
    """Reference ``(weight, bias)`` of expert *e*'s projection, in the proxy's
    ``(d_out, d_in)`` layout.
    """
    weight = ref[f"{BLOCK}.experts.{name}"][e]
    if getattr(experts, "is_transposed", False):
        weight = weight.T
    bias_name = f"{BLOCK}.experts.{name}_bias"
    bias = ref[bias_name][e] if bias_name in ref else None
    return weight, bias


def _split(dense_row: torch.Tensor, d_out: int, has_bias: bool):
    rows = dense_row.reshape(d_out, -1)
    if has_bias:
        return rows[:, :-1], rows[:, -1]
    return rows, None


@pytest.mark.parametrize("family", ["mixtral", "gpt_oss"])
class TestFusedExperts:
    def test_model_is_fused(self, family):
        model = _build(family)
        assert f"{BLOCK}.experts" in find_fused_experts(model)
        assert not any(
            isinstance(m, torch.nn.Linear)
            for m in model.get_submodule(f"{BLOCK}.experts").modules()
        )

    def test_proxies_match_per_sample_autograd(self, family):
        model = _build(family)
        ids = _batch()
        reference = _reference(model, ids)
        hm = HookManager(model, moe_routed_layers=[BLOCK])
        try:
            g = _collect(model, ids, hm)
        finally:
            hm.remove()
        assert g.batch_size == B
        experts = model.get_submodule(f"{BLOCK}.experts")
        checked = 0
        for name, value in g.data.items():
            if ".dattri_experts." not in name:
                continue
            e = int(name.split(".dattri_experts.")[1].split(".")[0])
            proj = "gate_up_proj" if name.endswith(".gate_up") else "down_proj"
            dense = ops.materialize(value, "nn.Linear")
            assert g.indexing[name] == "batch_token"
            assert value.activation.shape[0] == B
            for i, ref in enumerate(reference):
                weight, bias = _expert_reference(ref, experts, proj, e)
                mine_w, mine_b = _split(dense[i], weight.shape[0], bias is not None)
                torch.testing.assert_close(mine_w, weight, rtol=1e-10, atol=1e-12)
                if bias is not None:
                    torch.testing.assert_close(mine_b, bias, rtol=1e-10, atol=1e-12)
            checked += 1
        assert (
            checked == 2 * experts.gate_up_proj.shape[0]
        )  # two projections per expert

    def test_router_matches_per_sample_autograd(self, family):
        model = _build(family)
        ids = _batch()
        reference = _reference(model, ids)
        hm = HookManager(model, moe_routed_layers=[BLOCK])
        try:
            g = _collect(model, ids, hm)
        finally:
            hm.remove()
        (name,) = [n for n in g.data if n.endswith(".dattri_router")]
        router_name = name.rsplit(".", 1)[0]
        router = model.get_submodule(router_name)
        has_bias = getattr(router, "bias", None) is not None
        dense = ops.materialize(g.data[name], "nn.Linear")
        assert tuple(g.data[name].activation.shape) == (B, T, COMMON["hidden_size"])
        for i, ref in enumerate(reference):
            mine_w, mine_b = _split(dense[i], router.weight.shape[0], has_bias)
            torch.testing.assert_close(
                mine_w, ref[router_name + ".weight"], rtol=1e-10, atol=1e-12
            )
            if has_bias:
                torch.testing.assert_close(
                    mine_b, ref[router_name + ".bias"], rtol=1e-10, atol=1e-12
                )

    def test_fused_parameters_leave_the_batch_level_capture(self, family):
        model = _build(family)
        hm = HookManager(model, moe_routed_layers=[BLOCK])
        try:
            g = _collect(model, _batch(), hm)
        finally:
            hm.remove()
        param_grad = {n for n in g.data if g.layer_types[n] == ops.PARAM_GRAD_TYPES}
        # Declared block: no batch-level entry for its fused parameters.
        assert not any(n.startswith(BLOCK + ".") for n in param_grad)
        # Undeclared block (layer 1): still captured at batch level, as before.
        assert any(n.startswith("model.layers.1.mlp.experts.") for n in param_grad)

    def test_training_gradients_unchanged_and_model_restored(self, family):
        model = _build(family)
        ids = _batch()
        model.zero_grad()
        _per_sample_loss(model, ids).sum().backward()
        plain = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        experts = model.get_submodule(f"{BLOCK}.experts")
        hm = HookManager(model, moe_routed_layers=[BLOCK])
        try:
            assert hasattr(experts, "dattri_experts")
            _collect(model, ids, hm)
            for n, p in model.named_parameters():
                torch.testing.assert_close(p.grad, plain[n], rtol=0, atol=0)
        finally:
            hm.remove()
        assert not hasattr(experts, "dattri_experts")
        assert "forward" not in experts.__dict__
        # The fused kernel runs again after removal (no float64 support).
        model.config._experts_implementation = "grouped_mm"  # noqa: SLF001
        model.float()
        with torch.no_grad():
            assert model(input_ids=ids).logits.shape == (B, T, VOCAB)

    def test_no_proxies_without_declaration(self, family):
        model = _build(family)
        experts = model.get_submodule(f"{BLOCK}.experts")
        hm = HookManager(model)
        try:
            assert not hasattr(experts, "dattri_experts")
            g = _collect(model, _batch(), hm)
        finally:
            hm.remove()
        assert g.layer_types[f"{BLOCK}.experts.gate_up_proj"] == ops.PARAM_GRAD_TYPES


class TestDuplicateRows:
    @pytest.mark.parametrize("kind", ["repeated_sample", "shared_prefix"])
    def test_expert_gradients_stay_exact(self, kind):
        model = _build("mixtral")
        base = _batch()
        if kind == "repeated_sample":
            ids = torch.stack([base[0], base[0], base[1]])
        else:
            ids = torch.stack([base[0], torch.cat([base[0, :4], base[1, 4:]]), base[2]])
        reference = _reference(model, ids)
        hm = HookManager(model, moe_routed_layers=[BLOCK])
        try:
            g = _collect(model, ids, hm)
        finally:
            hm.remove()
        experts = model.get_submodule(f"{BLOCK}.experts")
        for name, value in g.data.items():
            if ".dattri_experts." not in name:
                continue
            e = int(name.split(".dattri_experts.")[1].split(".")[0])
            proj = "gate_up_proj" if name.endswith(".gate_up") else "down_proj"
            dense = ops.materialize(value, "nn.Linear")
            for i, ref in enumerate(reference):
                weight, bias = _expert_reference(ref, experts, proj, e)
                mine_w, _ = _split(dense[i], weight.shape[0], bias is not None)
                torch.testing.assert_close(mine_w, weight, rtol=1e-10, atol=1e-12)


class TestRouterVariants:
    """Routers that cast before their matmul (DeepSeek-V2/V3, GLM-4 MoE) and
    routers whose matmul is not ``F.linear`` at all.
    """

    def _router(self, model):
        block = model.get_submodule(BLOCK)
        (name,) = [n for n, m in block.named_children() if n != "experts"]
        return f"{BLOCK}.{name}", getattr(block, name)

    def test_casting_router_is_captured(self):
        model = _build("mixtral")
        router_name, router = self._router(model)
        original = type(router).forward

        def casting_forward(self, hidden_states):
            logits = F.linear(
                hidden_states.type(torch.float32), self.weight.type(torch.float32)
            )
            probs = torch.softmax(logits, dim=-1)
            top, idx = torch.topk(probs, self.top_k, dim=-1)
            top /= top.sum(dim=-1, keepdim=True)
            return logits, top.to(hidden_states.dtype), idx

        type(router).forward = casting_forward
        try:
            ids = _batch()
            reference = _reference(model, ids)
            hm = HookManager(model, moe_routed_layers=[BLOCK])
            try:
                g = _collect(model, ids, hm)
            finally:
                hm.remove()
        finally:
            type(router).forward = original
        name = f"{router_name}.dattri_router"
        assert name in g.data
        dense = ops.materialize(g.data[name], "nn.Linear")
        for i, ref in enumerate(reference):
            mine_w, _ = _split(dense[i], router.weight.shape[0], has_bias=False)
            torch.testing.assert_close(
                mine_w.double(), ref[router_name + ".weight"], rtol=1e-6, atol=1e-8
            )

    def test_router_bypassing_f_linear_warns(self):
        model = _build("mixtral")
        router_name, router = self._router(model)
        original = type(router).forward

        def matmul_forward(self, hidden_states):
            logits = hidden_states @ self.weight.T
            probs = torch.softmax(logits, dim=-1)
            top, idx = torch.topk(probs, self.top_k, dim=-1)
            return logits, top / top.sum(dim=-1, keepdim=True), idx

        type(router).forward = matmul_forward
        try:
            hm = HookManager(model, moe_routed_layers=[BLOCK])
            try:
                with pytest.warns(UserWarning, match="did not run in this step"):
                    g = _collect(model, _batch(), hm)
            finally:
                hm.remove()
        finally:
            type(router).forward = original
        assert f"{router_name}.dattri_router" not in g.data
