"""Capture on linear-attention / recurrent language models (Mamba, RWKV).

These architectures replace softmax attention with a recurrence whose
parameters are not linear-family modules (Mamba's ``A_log``/``D``, RWKV's
time-mix and decay vectors) while every projection stays an ``nn.Linear``
on ``(B, T, d)`` tokens.  The tests pin that the default ``HookManager``
captures such a model with no configuration: the projections, norms and
embeddings exactly match a per-sample autograd loop, and the recurrence
parameters are captured at batch level through the ``param_grad`` fallback.
Tiny Hugging Face models in float64; Mamba runs its sequential (pure
PyTorch) path, since the fused kernels are not installed on CPU.

A module applied *functionally* is invisible to module hooks: Transformers 5
runs Mamba's ``conv1d`` and ``dt_proj`` through ``causal_conv1d_fn`` /
``mamba_inner_fn`` on their weights, and the fused training kernel bypasses
``x_proj`` and ``out_proj`` too.  While collecting, the manager replaces a
Mamba mixer's forward with the reference computation through its modules
(:mod:`dattri_llm.gradient.hooks.override_layers.sequence_mixers`), so every
projection is captured whatever the version or backend, and the output is
the mixer's own.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import HookManager

transformers = pytest.importorskip("transformers")

B, T, VOCAB, D = 3, 7, 64, 16


def _build(family: str) -> torch.nn.Module:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Mamba's "fast path unavailable" notice
        if family == "mamba":
            cfg = transformers.MambaConfig(
                vocab_size=VOCAB,
                hidden_size=D,
                state_size=4,
                num_hidden_layers=2,
                intermediate_size=32,
                conv_kernel=4,
                time_step_rank=2,
            )
            model = transformers.MambaForCausalLM(cfg)
        elif family == "rwkv":
            cfg = transformers.RwkvConfig(
                vocab_size=VOCAB,
                hidden_size=D,
                num_hidden_layers=2,
                attention_hidden_size=D,
                intermediate_size=32,
                context_length=32,
            )
            model = transformers.RwkvForCausalLM(cfg)
        else:
            raise ValueError(family)
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


def _collect(model, ids, **hm_kwargs):
    hm = HookManager(model, **hm_kwargs)
    try:
        with hm.collect():
            model.zero_grad()
            _per_sample_loss(model, ids).sum().backward()
        return hm.get_gradient()
    finally:
        hm.remove()


def _per_sample(gradient, name, params):
    """A factorized layer's per-sample ``(weight, bias | None)`` gradients."""
    value = gradient.data[name]
    dense = ops.materialize(value, gradient.layer_types[name])
    weight = params[name + ".weight"]
    if gradient.layer_types[name] in ("nn.LayerNorm", "nn.RMSNorm"):
        n = weight.numel()
        return dense[:, :n].reshape(B, *weight.shape), dense[:, n:]
    has_bias = (name + ".bias") in params
    rows = dense.reshape(B, weight.shape[0], -1)
    if has_bias:
        return rows[:, :, :-1].reshape(B, *weight.shape), rows[:, :, -1]
    return rows.reshape(B, *weight.shape), None


@pytest.mark.parametrize("family", ["mamba", "rwkv"])
class TestLinearAttentionCapture:
    def test_batch_size_and_layer_kinds(self, family):
        model = _build(family)
        g = _collect(model, _batch())
        assert g.batch_size == B
        kinds = {g.layer_types[n] for n in g.data}
        assert "nn.Linear" in kinds
        assert "nn.Embedding" in kinds
        # The recurrence parameters are not linear-family modules: they are
        # captured at batch level by the param_grad fallback, not dropped.
        param_grad = {n for n in g.data if g.layer_types[n] == ops.PARAM_GRAD_TYPES}
        expected = {"mamba": "mixer.A_log", "rwkv": "attention.time_decay"}[family]
        assert any(n.endswith(expected) for n in param_grad)

    def test_projections_norms_match_per_sample_autograd(self, family):
        model = _build(family)
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids)
        params = dict(model.named_parameters())
        checked = 0
        for name in g.data:
            if g.layer_types[name] not in ("nn.Linear", "nn.LayerNorm"):
                continue
            if name + ".weight" not in params:
                continue  # a tied head: its parameter is the embedding's
            weight, bias = _per_sample(g, name, params)
            for i, ref in enumerate(reference):
                torch.testing.assert_close(
                    weight[i], ref[name + ".weight"], rtol=1e-10, atol=1e-12
                )
                if bias is not None:
                    torch.testing.assert_close(
                        bias[i], ref[name + ".bias"], rtol=1e-10, atol=1e-12
                    )
            checked += 1
        # Mamba: in_proj, x_proj, dt_proj and out_proj of both layers (the tied
        # head has no weight of its own); RWKV: more.
        assert checked >= 2 * 4

    def test_param_grad_layers_hold_the_batch_gradient(self, family):
        model = _build(family)
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids)
        for name, value in g.data.items():
            if g.layer_types[name] != ops.PARAM_GRAD_TYPES:
                continue
            total = sum(ref[name] for ref in reference)
            # The recurrence parameters' gradients pass through T sequential
            # steps; summation order differs between the batched and the
            # per-sample backward, so rounding is above fp64 tightness.
            torch.testing.assert_close(value, total, rtol=1e-6, atol=1e-11)


class TestMambaTiedHead:
    def test_tied_embedding_and_head_sum_to_the_parameter_gradient(self):
        model = _build("mamba")
        assert model.lm_head.weight is model.backbone.embeddings.weight
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids)
        emb = ops.materialize(g.data["backbone.embeddings"], "nn.Embedding")
        head = ops.materialize(g.data["lm_head"], "nn.Linear")
        shape = model.backbone.embeddings.weight.shape
        for i, ref in enumerate(reference):
            torch.testing.assert_close(
                (emb[i] + head[i]).reshape(shape),
                ref["backbone.embeddings.weight"],
                rtol=1e-10,
                atol=1e-12,
            )


def _conv_record(model):
    """The captured ``conv1d`` record and the per-sample reference."""
    ids = _batch()
    reference = _reference(model, ids)
    g = _collect(model, ids)
    name = "backbone.layers.0.mixer.conv1d"
    assert name in g.data  # the mixer override calls the module
    return g, reference, name


def _tiny(name):
    """A tiny model of a sequence-mixer family, or ``None`` when the installed
    Transformers lacks it.
    """
    torch.manual_seed(0)
    kw = {"vocab_size": VOCAB, "max_position_embeddings": 64}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == "mamba":
                return _build("mamba")
            if name == "falcon_mamba":
                cfg = transformers.FalconMambaConfig(
                    hidden_size=D,
                    state_size=4,
                    num_hidden_layers=2,
                    intermediate_size=32,
                    conv_kernel=4,
                    time_step_rank=2,
                    **kw,
                )
                model = transformers.FalconMambaForCausalLM(cfg)
            elif name == "mamba2":
                cfg = transformers.Mamba2Config(
                    hidden_size=D,
                    state_size=4,
                    num_hidden_layers=2,
                    num_heads=2,
                    head_dim=8,
                    expand=1,
                    n_groups=1,
                    chunk_size=4,
                    conv_kernel=4,
                    **kw,
                )
                model = transformers.Mamba2ForCausalLM(cfg)
            elif name == "jamba":
                cfg = transformers.JambaConfig(
                    hidden_size=D,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    attn_layer_period=2,
                    attn_layer_offset=1,
                    expert_layer_period=2,
                    expert_layer_offset=1,
                    num_experts=1,
                    num_experts_per_tok=1,
                    mamba_d_state=4,
                    mamba_d_conv=4,
                    mamba_expand=2,
                    mamba_dt_rank=2,
                    use_mamba_kernels=False,
                    **kw,
                )
                model = transformers.JambaForCausalLM(cfg)
            elif name == "qwen3_next":
                cfg = transformers.Qwen3NextConfig(
                    hidden_size=32,
                    intermediate_size=32,
                    moe_intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=16,
                    linear_num_value_heads=2,
                    linear_num_key_heads=2,
                    linear_key_head_dim=8,
                    linear_value_head_dim=8,
                    linear_conv_kernel_dim=4,
                    num_experts=2,
                    num_experts_per_tok=1,
                    shared_expert_intermediate_size=16,
                    layer_types=["linear_attention", "full_attention"],
                    **kw,
                )
                model = transformers.Qwen3NextForCausalLM(cfg)
            elif name == "kimi_linear":
                cfg = transformers.KimiLinearConfig(
                    hidden_size=32,
                    intermediate_size=32,
                    moe_intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=16,
                    num_experts=2,
                    num_experts_per_tok=1,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                    layer_types=["linear_attention", "full_attention"],
                    **kw,
                )
                model = transformers.KimiLinearForCausalLM(cfg)
            elif name == "lfm2":
                cfg = transformers.Lfm2Config(
                    hidden_size=32,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    layer_types=["conv", "full_attention"],
                    conv_L_cache=3,
                    **kw,
                )
                model = transformers.Lfm2ForCausalLM(cfg)
            else:
                raise ValueError(name)
        except AttributeError:
            return None  # family not in this Transformers version
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "eager"  # noqa: SLF001 - no fp64 kernel
    return model.double()


class TestSequenceMixerOverride:
    """The sequence-mixer overrides on every family the installed Transformers
    provides: the Mamba-1 reference forward (Mamba, Falcon-Mamba, Jamba) and
    the functional-conv interception (Mamba2, Qwen3-Next, Kimi-Linear, LFM2).
    Every ``nn.Linear`` / ``nn.Conv1d`` child of a mixer must be captured per
    sample and exactly, and the model's output must be its own.
    """

    # Families whose MoE blocks hand their sub-layers flattened/routed rows and
    # must be declared, as for any MoE model.
    MOE_BLOCKS: ClassVar[dict[str, list[str]]] = {
        "qwen3_next": [f"model.layers.{i}.mlp" for i in range(2)],
        "kimi_linear": [f"model.layers.{i}.mlp" for i in range(2)],
    }
    FAMILIES: ClassVar[dict[str, float]] = {  # name -> tolerance
        "mamba": 1e-12,
        "falcon_mamba": 1e-12,
        "jamba": 1e-12,
        "mamba2": 1e-9,  # its fallback chunk scan computes in float32
        "qwen3_next": 1e-12,
        "kimi_linear": 1e-12,
        "lfm2": 1e-12,
    }

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_mixer_children_captured_and_exact(self, family):
        from dattri_llm.gradient.hooks.override_layers import find_sequence_mixers

        model = _tiny(family)
        if model is None:
            pytest.skip(f"{family} is not in this Transformers version")
        mixers = find_sequence_mixers(model)
        assert mixers
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids, moe_routed_layers=self.MOE_BLOCKS.get(family))
        params = dict(model.named_parameters())
        tol = self.FAMILIES[family]
        checked = 0
        for mixer_name, mixer in mixers.items():
            for child, module in mixer.named_children():
                if not isinstance(module, (nn.Linear, nn.Conv1d)):
                    continue
                name = f"{mixer_name}.{child}"
                assert name in g.data, name
                weight = params[name + ".weight"]
                dense = ops.materialize(g.data[name], g.layer_types[name])
                k = weight[0].numel()
                rows = dense.reshape(B, weight.shape[0], -1)
                for i, ref in enumerate(reference):
                    torch.testing.assert_close(
                        rows[i, :, :k].reshape(weight.shape),
                        ref[name + ".weight"],
                        rtol=tol,
                        atol=tol,
                    )
                checked += 1
        assert checked >= 3

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_output_equals_the_original_forward(self, family):
        model = _tiny(family)
        if model is None:
            pytest.skip(f"{family} is not in this Transformers version")
        ids = _batch()
        with torch.no_grad():
            plain = model(input_ids=ids).logits
        hm = HookManager(model, moe_routed_layers=self.MOE_BLOCKS.get(family))
        try:
            with hm.collect():
                out = model(input_ids=ids).logits
                out.sum().backward()
        finally:
            hm.remove()
        tol = self.FAMILIES[family]
        torch.testing.assert_close(out.detach(), plain, rtol=tol, atol=tol)
        assert not [m for m in model.modules() if "forward" in m.__dict__]
