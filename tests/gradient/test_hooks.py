"""Unit tests for dattri_llm.gradient.hooks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dattri_llm.gradient.hooks import (
    _is_hookable_layer,
    register_mlp_hooks,
    register_param_grad_hooks,
    remove_hooks,
)


# --------------------------------------------------------------------------- #
# _is_hookable_layer                                                            #
# --------------------------------------------------------------------------- #


class TestIsHookableLayer:
    # --- nn.Linear / Conv1D: require an MLP-family parent keyword ----------- #

    def test_linear_inside_mlp(self):
        assert _is_hookable_layer("transformer.h.0.mlp.fc1", nn.Linear(4, 4))

    def test_linear_inside_ffn(self):
        assert _is_hookable_layer("model.layers.1.ffn.gate_proj", nn.Linear(4, 4))

    def test_linear_not_in_mlp(self):
        # attn projection — parent path has no MLP keyword
        assert not _is_hookable_layer("transformer.h.0.attn.c_proj", nn.Linear(4, 4))

    def test_non_linear_inside_mlp(self):
        assert not _is_hookable_layer("model.mlp.act", nn.ReLU())

    def test_top_level_linear(self):
        # No parent path at all
        assert not _is_hookable_layer("lm_head", nn.Linear(4, 4))

    # --- nn.Embedding: always included regardless of parent path ------------ #

    def test_embedding_top_level(self):
        assert _is_hookable_layer("embedding", nn.Embedding(32, 8))

    def test_embedding_nested(self):
        assert _is_hookable_layer("model.transformer.wte", nn.Embedding(50257, 768))

    def test_embedding_inside_attn(self):
        # Even under a non-MLP parent, Embedding is hooked.
        assert _is_hookable_layer("model.attn.embed", nn.Embedding(16, 16))

    # --- nn.LayerNorm: always included regardless of parent path ------------ #

    def test_layernorm_top_level(self):
        assert _is_hookable_layer("ln_f", nn.LayerNorm(64))

    def test_layernorm_nested(self):
        assert _is_hookable_layer("model.layers.0.ln1", nn.LayerNorm(128))

    def test_layernorm_inside_attn(self):
        assert _is_hookable_layer("model.h.3.attn.ln", nn.LayerNorm(256))

    # --- Other module types are still excluded ------------------------------ #

    def test_relu_excluded(self):
        assert not _is_hookable_layer("model.mlp.act", nn.ReLU())

    def test_dropout_excluded(self):
        assert not _is_hookable_layer("model.dropout", nn.Dropout())


# --------------------------------------------------------------------------- #
# register_mlp_hooks / remove_hooks                                            #
# --------------------------------------------------------------------------- #


class TestRegisterMlpHooks:
    def test_hooks_registered_on_mlp_layers(self, tiny_model):
        buffers, handles = register_mlp_hooks(tiny_model)
        # TinyMLP has: embedding (Embedding — always hooked),
        # mlp.0 and mlp.2 (Linear inside an MLP block).
        # mlp.1 is ReLU → skipped; attn_proj and lm_head lack an MLP parent → skipped.
        assert len(buffers) == 3
        assert "embedding" in buffers
        assert "mlp.0" in buffers
        assert "mlp.2" in buffers
        # Each layer has 2 handles (forward + backward)
        assert len(handles) == 6
        remove_hooks(handles)

    def test_attn_proj_not_hooked(self, tiny_model):
        buffers, handles = register_mlp_hooks(tiny_model)
        assert "attn_proj" not in buffers
        remove_hooks(handles)

    def test_buffers_populated_after_forward_backward(self, tiny_model, tiny_batch):
        buffers, handles = register_mlp_hooks(tiny_model)
        input_ids = tiny_batch["input_ids"]
        logits = tiny_model(input_ids)
        loss = logits.mean()
        loss.backward()

        for name, buf in buffers.items():
            assert buf["activation"] is not None, f"{name}: activation is None"
            assert buf["grad_output"] is not None, f"{name}: grad_output is None"
        remove_hooks(handles)

    def test_activation_shape(self, tiny_model, tiny_batch):
        buffers, handles = register_mlp_hooks(tiny_model)
        B, T = tiny_batch["input_ids"].shape
        tiny_model(tiny_batch["input_ids"]).mean().backward()

        # mlp.0: Linear(16, 32) → activation shape (B, T, 16)
        act = buffers["mlp.0"]["activation"]
        assert act.shape == (B, T, 16)
        remove_hooks(handles)

    def test_grad_output_shape(self, tiny_model, tiny_batch):
        buffers, handles = register_mlp_hooks(tiny_model)
        B, T = tiny_batch["input_ids"].shape
        tiny_model(tiny_batch["input_ids"]).mean().backward()

        # mlp.0: output dim 32 → grad_output shape (B, T, 32)
        go = buffers["mlp.0"]["grad_output"]
        assert go.shape == (B, T, 32)
        remove_hooks(handles)

    def test_remove_hooks_clears_handles(self, tiny_model):
        buffers, handles = register_mlp_hooks(tiny_model)
        assert len(handles) > 0
        remove_hooks(handles)
        assert len(handles) == 0

    def test_hooks_do_not_fire_after_removal(self, tiny_model, tiny_batch):
        buffers, handles = register_mlp_hooks(tiny_model)
        remove_hooks(handles)
        # After removal buffers should remain None (no forward called yet)
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        for buf in buffers.values():
            assert buf["activation"] is None

    def test_custom_name_patterns(self, tiny_model, tiny_batch):
        # Hook only mlp.0 via a custom pattern
        buffers, handles = register_mlp_hooks(tiny_model, name_patterns=[r"mlp\.0"])
        assert list(buffers.keys()) == ["mlp.0"]
        remove_hooks(handles)

    def test_dataparallel_unwrapping(self, tiny_model):
        dp_model = nn.DataParallel(tiny_model)
        buffers, handles = register_mlp_hooks(dp_model)
        # Same 3 layers as the plain-model case: embedding, mlp.0, mlp.2
        assert len(buffers) == 3
        remove_hooks(handles)


# --------------------------------------------------------------------------- #
# register_param_grad_hooks                                                    #
# --------------------------------------------------------------------------- #


class TestRegisterParamGradHooks:
    def test_leaf_modules_hooked_by_default(self, tiny_model):
        # Default (no patterns): hooks all leaf modules with trainable params.
        # TinyMLP leaves: embedding, attn_proj, mlp.0, mlp.2, lm_head (mlp.1 is ReLU — no params)
        buffers, handles = register_param_grad_hooks(tiny_model)
        assert len(buffers) > 0
        assert all("weight" in buf or "bias" in buf for buf in buffers.values())
        remove_hooks(handles)

    def test_buffers_none_before_backward(self, tiny_model):
        buffers, handles = register_param_grad_hooks(tiny_model)
        for buf in buffers.values():
            for grad in buf.values():
                assert grad is None
        remove_hooks(handles)

    def test_buffers_populated_after_backward(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model)
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        for layer_name, buf in buffers.items():
            for pname, grad in buf.items():
                assert grad is not None, f"{layer_name}.{pname}: grad is None after backward"
                assert isinstance(grad, torch.Tensor)
        remove_hooks(handles)

    def test_grad_is_fresh_not_stale(self, tiny_model, tiny_batch):
        """Grad captured by hook matches param.grad immediately after backward."""
        buffers, handles = register_param_grad_hooks(tiny_model, name_patterns=[r"mlp\.0"])
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        # The hook-captured grad and param.grad should be identical tensors.
        hook_grad = buffers["mlp.0"]["weight"]
        param_grad = dict(tiny_model.named_parameters())["mlp.0.weight"].grad
        assert torch.allclose(hook_grad, param_grad.cpu())
        remove_hooks(handles)

    def test_grad_shape_matches_param(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model, name_patterns=[r"mlp\.0"])
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        weight_grad = buffers["mlp.0"]["weight"]
        weight = dict(tiny_model.named_parameters())["mlp.0.weight"]
        assert weight_grad.shape == weight.shape
        remove_hooks(handles)

    def test_grad_is_finite(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model)
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        for layer_name, buf in buffers.items():
            for pname, grad in buf.items():
                assert torch.isfinite(grad).all(), f"{layer_name}.{pname} has NaN/Inf"
        remove_hooks(handles)

    def test_custom_name_patterns(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model, name_patterns=[r"mlp\.[02]"])
        assert set(buffers.keys()) == {"mlp.0", "mlp.2"}
        remove_hooks(handles)

    def test_frozen_params_not_hooked(self, tiny_model):
        # Freeze mlp.0 and verify it produces no buffer entry.
        tiny_model.mlp[0].weight.requires_grad_(False)
        buffers, handles = register_param_grad_hooks(tiny_model, name_patterns=[r"mlp\.0"])
        # mlp.0 has no trainable params → no buffer created for it.
        assert "mlp.0" not in buffers
        tiny_model.mlp[0].weight.requires_grad_(True)  # restore
        remove_hooks(handles)

    def test_on_param_grad_callback(self, tiny_model, tiny_batch):
        fired: list[tuple[str, str]] = []

        def cb(layer_name, param_name, grad):
            fired.append((layer_name, param_name))

        buffers, handles = register_param_grad_hooks(
            tiny_model, name_patterns=[r"mlp\.0"], on_param_grad=cb
        )
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert ("mlp.0", "weight") in fired
        remove_hooks(handles)

    def test_remove_hooks_stops_updates(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model, name_patterns=[r"mlp\.0"])
        remove_hooks(handles)
        # Run backward after removal — buffer should stay None.
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert buffers["mlp.0"]["weight"] is None

    def test_dataparallel_unwrapping(self, tiny_model):
        dp_model = nn.DataParallel(tiny_model)
        buffers, handles = register_param_grad_hooks(dp_model)
        assert len(buffers) > 0
        remove_hooks(handles)
