"""Unit tests for dattri_llm.gradient.hooks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dattri_llm.gradient.hooks import _is_mlp_linear, register_mlp_hooks, remove_hooks


# --------------------------------------------------------------------------- #
# _is_mlp_linear                                                               #
# --------------------------------------------------------------------------- #


class TestIsMlpLinear:
    def test_linear_inside_mlp(self):
        assert _is_mlp_linear("transformer.h.0.mlp.fc1", nn.Linear(4, 4))

    def test_linear_inside_ffn(self):
        assert _is_mlp_linear("model.layers.1.ffn.gate_proj", nn.Linear(4, 4))

    def test_linear_not_in_mlp(self):
        # attn projection — parent path has no MLP keyword
        assert not _is_mlp_linear("transformer.h.0.attn.c_proj", nn.Linear(4, 4))

    def test_non_linear_inside_mlp(self):
        assert not _is_mlp_linear("model.mlp.act", nn.ReLU())

    def test_top_level_linear(self):
        # No parent path at all
        assert not _is_mlp_linear("lm_head", nn.Linear(4, 4))


# --------------------------------------------------------------------------- #
# register_mlp_hooks / remove_hooks                                            #
# --------------------------------------------------------------------------- #


class TestRegisterMlpHooks:
    def test_hooks_registered_on_mlp_layers(self, tiny_model):
        buffers, handles = register_mlp_hooks(tiny_model)
        # TinyMLP has mlp.0 (Linear) and mlp.2 (Linear); mlp.1 is ReLU → skipped
        assert len(buffers) == 2
        assert all(k.startswith("mlp.") for k in buffers)
        # Each layer has 2 handles (forward + backward)
        assert len(handles) == 4
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
        assert len(buffers) == 2
        remove_hooks(handles)
