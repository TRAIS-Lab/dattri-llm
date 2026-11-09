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
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

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


def _collect(model, ids):
    hm = HookManager(model)
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
        assert checked >= 8

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


class TestMambaDepthwiseConv:
    """Mamba's causal ``conv1d`` is depthwise (``groups == channels``)."""

    @pytest.mark.xfail(
        reason="conv capture ignores `groups`: the per-sample gradient is "
        "materialized as an ungrouped (C_out, C_in*k + 1) block",
        strict=True,
    )
    def test_per_sample_gradient_has_the_grouped_shape(self):
        model = _build("mamba")
        g = _collect(model, _batch())
        name = "backbone.layers.0.mixer.conv1d"
        conv = model.backbone.layers[0].mixer.conv1d
        dense = ops.materialize(g.data[name], "nn.Conv1d")
        assert dense.shape[1] == conv.weight.numel() + conv.bias.numel()

    def test_ungrouped_block_holds_the_true_gradient_on_its_diagonal(self):
        # What the capture does hold is exact: the depthwise gradient is the
        # block diagonal of the ungrouped materialization.
        model = _build("mamba")
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids)
        name = "backbone.layers.0.mixer.conv1d"
        conv = model.backbone.layers[0].mixer.conv1d
        c, k = conv.weight.shape[0], conv.weight.shape[-1]
        dense = ops.materialize(g.data[name], "nn.Conv1d").reshape(B, c, -1)
        full = dense[:, :, :-1].reshape(B, c, c, k)
        diag = torch.stack([full[:, o, o, :] for o in range(c)], 1)
        for i, ref in enumerate(reference):
            torch.testing.assert_close(
                diag[i], ref[name + ".weight"][:, 0, :], rtol=1e-10, atol=1e-12
            )
            torch.testing.assert_close(
                dense[i, :, -1], ref[name + ".bias"], rtol=1e-10, atol=1e-12
            )
