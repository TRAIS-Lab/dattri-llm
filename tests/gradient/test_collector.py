"""Unit tests for dattri_llm.gradient.collector.GradientCollector."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dattri_llm.gradient.collector import GradientCollector


class TestGradientCollectorInit:
    def test_layer_names_populated(self, tiny_model):
        collector = GradientCollector(tiny_model)
        assert len(collector.layer_names) == 2
        collector.remove()

    def test_custom_patterns(self, tiny_model):
        collector = GradientCollector(tiny_model, name_patterns=[r"mlp\.0"])
        assert collector.layer_names == ["mlp.0"]
        collector.remove()


class TestGradientCollectorContextManager:
    def test_buffers_none_before_pass(self, tiny_model):
        collector = GradientCollector(tiny_model)
        with collector:
            pass  # no forward/backward
        for buf in collector.get_gradients().values():
            assert buf["activation"] is None
        collector.remove()

    def test_buffers_populated_after_pass(self, tiny_model, tiny_batch):
        collector = GradientCollector(tiny_model)
        with collector:
            logits = tiny_model(tiny_batch["input_ids"])
            logits.mean().backward()

        for name, buf in collector.get_gradients().items():
            assert buf["activation"] is not None, f"{name}: activation None"
            assert buf["grad_output"] is not None, f"{name}: grad_output None"
        collector.remove()

    def test_buffers_reset_between_passes(self, tiny_model, tiny_batch):
        collector = GradientCollector(tiny_model)
        with collector:
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        first_act = {
            k: v["activation"].clone() for k, v in collector.get_gradients().items()
        }

        # Second pass with a different seed should give different activations
        torch.manual_seed(99)
        new_ids = torch.randint(0, tiny_model.vocab_size, tiny_batch["input_ids"].shape)
        with collector:
            tiny_model(new_ids).mean().backward()
        second_act = {
            k: v["activation"] for k, v in collector.get_gradients().items()
        }

        for k in first_act:
            assert not torch.equal(first_act[k], second_act[k]), (
                f"Activations for {k} did not change between passes"
            )
        collector.remove()

    def test_collect_context_manager_alias(self, tiny_model, tiny_batch):
        collector = GradientCollector(tiny_model)
        with collector.collect():
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        for buf in collector.get_gradients().values():
            assert buf["activation"] is not None
        collector.remove()


class TestGetPerSampleGradients:
    def test_shape(self, tiny_model, tiny_batch):
        B = tiny_batch["input_ids"].shape[0]
        collector = GradientCollector(tiny_model)
        with collector:
            tiny_model(tiny_batch["input_ids"]).mean().backward()

        per_sample = collector.get_per_sample_gradients()
        # mlp.0: Linear(16, 32) → (B, 32, 16)
        assert per_sample["mlp.0"].shape == (B, 32, 16)
        # mlp.2: Linear(32, 16) → (B, 16, 32)
        assert per_sample["mlp.2"].shape == (B, 16, 32)
        collector.remove()

    def test_raises_before_pass(self, tiny_model):
        collector = GradientCollector(tiny_model)
        with collector:
            pass  # no backward
        with pytest.raises(RuntimeError, match="no buffered data"):
            collector.get_per_sample_gradients()
        collector.remove()

    def test_outer_product_correctness(self, tiny_model, tiny_batch):
        """Verify g ⊗ a matches manually computed outer product for one layer."""
        collector = GradientCollector(tiny_model, name_patterns=[r"mlp\.0"])
        with collector:
            tiny_model(tiny_batch["input_ids"]).mean().backward()

        per_sample = collector.get_per_sample_gradients()["mlp.0"]
        bufs = collector.get_gradients()["mlp.0"]
        a = bufs["activation"]  # (B, T, 16)
        g = bufs["grad_output"]  # (B, T, 32)
        B = a.shape[0]
        # Manual: sum over T
        expected = torch.einsum("bto,bti->boi", g, a)
        assert torch.allclose(per_sample, expected, atol=1e-6), (
            "Per-sample gradient does not match manual einsum"
        )
        collector.remove()

    def test_finite_gradients(self, tiny_model, tiny_batch):
        collector = GradientCollector(tiny_model)
        with collector:
            tiny_model(tiny_batch["input_ids"]).mean().backward()
        for name, grad in collector.get_per_sample_gradients().items():
            assert torch.isfinite(grad).all(), f"{name} contains NaN/Inf"
        collector.remove()


class TestGradientCollectorRemove:
    def test_remove_clears_layer_names(self, tiny_model):
        collector = GradientCollector(tiny_model)
        assert len(collector.layer_names) > 0
        collector.remove()
        assert collector.layer_names == []
