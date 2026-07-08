"""Tests for the module_kwargs builder helpers.

The contract under test: for every accepted layer type, the helper's output is
*identical* to what ``extract_module_kwargs`` reads off a real ``torch.nn``
module built with the same hyperparameters -- so a hand-built dict is a perfect
stand-in for automatic extraction.  All fields are keyword-only and required;
forgetting one raises immediately instead of silently assuming a default.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient.ops import extract_module_kwargs
from dattri_llm.utils.module import (
    bilinear_module_kwargs,
    conv1d_module_kwargs,
    conv2d_module_kwargs,
    conv3d_module_kwargs,
    conv_transpose1d_module_kwargs,
    conv_transpose2d_module_kwargs,
    conv_transpose3d_module_kwargs,
    embedding_bag_module_kwargs,
    embedding_module_kwargs,
    group_norm_module_kwargs,
    instance_norm1d_module_kwargs,
    instance_norm2d_module_kwargs,
    instance_norm3d_module_kwargs,
    layer_norm_module_kwargs,
    linear_module_kwargs,
    rms_norm_module_kwargs,
)

_HAS_RMSNORM = hasattr(nn, "RMSNorm")

# (helper output, real module, canonical type string) triples covering every
# accepted layer type; each helper call mirrors the module's constructor args.
_CASES = [
    (
        linear_module_kwargs(has_bias=True),
        nn.Linear(4, 8, bias=True),
        "nn.Linear",
    ),
    (
        linear_module_kwargs(has_bias=False),
        nn.Linear(4, 8, bias=False),
        "nn.Linear",
    ),
    (
        bilinear_module_kwargs(has_bias=False),
        nn.Bilinear(3, 4, 5, bias=False),
        "nn.Bilinear",
    ),
    (
        embedding_module_kwargs(),
        nn.Embedding(16, 4),
        "nn.Embedding",
    ),
    (
        embedding_bag_module_kwargs(mode="sum"),
        nn.EmbeddingBag(16, 4, mode="sum"),
        "nn.EmbeddingBag",
    ),
    (
        conv1d_module_kwargs(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            has_bias=True,
        ),
        nn.Conv1d(2, 4, 3, stride=2, padding=1, dilation=2),
        "nn.Conv1d",
    ),
    (
        conv2d_module_kwargs(
            kernel_size=(3, 5),
            stride=1,
            padding=(1, 2),
            dilation=1,
            has_bias=False,
        ),
        nn.Conv2d(2, 4, (3, 5), padding=(1, 2), bias=False),
        "nn.Conv2d",
    ),
    (
        conv3d_module_kwargs(
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            has_bias=True,
        ),
        nn.Conv3d(2, 4, 3),
        "nn.Conv3d",
    ),
    (
        conv_transpose1d_module_kwargs(
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=1,
            has_bias=True,
        ),
        nn.ConvTranspose1d(2, 4, 3, stride=2),
        "nn.ConvTranspose1d",
    ),
    (
        conv_transpose2d_module_kwargs(
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            has_bias=False,
        ),
        nn.ConvTranspose2d(2, 4, 3, padding=1, bias=False),
        "nn.ConvTranspose2d",
    ),
    (
        conv_transpose3d_module_kwargs(
            kernel_size=(2, 3, 2),
            stride=1,
            padding=0,
            dilation=1,
            has_bias=True,
        ),
        nn.ConvTranspose3d(2, 4, (2, 3, 2)),
        "nn.ConvTranspose3d",
    ),
    (
        layer_norm_module_kwargs(normalized_shape=8, eps=1e-6, has_bias=True),
        nn.LayerNorm(8, eps=1e-6),
        "nn.LayerNorm",
    ),
    (
        layer_norm_module_kwargs(normalized_shape=(4, 8), eps=1e-5, has_bias=False),
        nn.LayerNorm((4, 8), bias=False),
        "nn.LayerNorm",
    ),
    (
        group_norm_module_kwargs(num_groups=2, num_channels=8, eps=1e-5, has_bias=True),
        nn.GroupNorm(2, 8),
        "nn.GroupNorm",
    ),
    (
        instance_norm1d_module_kwargs(num_features=4, eps=1e-5, has_bias=True),
        nn.InstanceNorm1d(4, affine=True),
        "nn.InstanceNorm1d",
    ),
    (
        instance_norm2d_module_kwargs(num_features=4, eps=1e-5, has_bias=True),
        nn.InstanceNorm2d(4, affine=True),
        "nn.InstanceNorm2d",
    ),
    (
        instance_norm3d_module_kwargs(num_features=4, eps=1e-5, has_bias=True),
        nn.InstanceNorm3d(4, affine=True),
        "nn.InstanceNorm3d",
    ),
]
if _HAS_RMSNORM:
    _CASES.append(
        (
            rms_norm_module_kwargs(normalized_shape=8, eps=None),
            nn.RMSNorm(8, elementwise_affine=True),
            "nn.RMSNorm",
        ),
    )


class TestMatchesExtraction:
    @pytest.mark.parametrize(
        ("built", "module", "layer_type"),
        _CASES,
        ids=[f"{lt}-{i}" for i, (_, _, lt) in enumerate(_CASES)],
    )
    def test_equals_extract_module_kwargs(self, built, module, layer_type):
        assert built == extract_module_kwargs(module, layer_type)


class TestValidation:
    def test_conv_rank_mismatch_raises(self):
        with pytest.raises(ValueError, match="length-2"):
            conv2d_module_kwargs(
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0,
                dilation=1,
                has_bias=True,
            )

    def test_embedding_bag_max_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            embedding_bag_module_kwargs(mode="max")

    def test_missing_field_raises(self):
        # No silent defaults: forgetting any field is an immediate TypeError.
        with pytest.raises(TypeError):
            rms_norm_module_kwargs(normalized_shape=8)  # eps forgotten
        with pytest.raises(TypeError):
            linear_module_kwargs()  # has_bias forgotten
        with pytest.raises(TypeError):
            conv2d_module_kwargs(kernel_size=3, has_bias=True)  # geometry forgotten

    def test_positional_arguments_rejected(self):
        with pytest.raises(TypeError):
            linear_module_kwargs(True)  # noqa: FBT003 - asserting kw-only contract


@pytest.mark.skipif(not _HAS_RMSNORM, reason="nn.RMSNorm requires PyTorch >= 2.4")
class TestDeclaredTypeIntegration:
    def test_hf_style_rmsnorm_capture_is_exact(self):
        """End-to-end: a hand-rolled Llama-style norm declared via layer_types
        with a helper-built module_kwargs dict captures exact per-sample grads.
        """
        from dattri_llm.gradient.callbacks import CaptureCallback
        from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

        class MyRMSNorm(nn.Module):  # weight + variance_epsilon, no nn.RMSNorm base
            def __init__(self, d, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(d).abs() + 0.5)
                self.variance_epsilon = eps

            def forward(self, x):
                var = x.pow(2).mean(-1, keepdim=True)
                return self.weight * x * torch.rsqrt(var + self.variance_epsilon)

        torch.manual_seed(0)
        model = nn.Sequential()
        model.add_module("fc", nn.Linear(8, 8, bias=False))
        model.add_module("norm", MyRMSNorm(8))
        model = model.double()
        x = torch.randn(3, 8, dtype=torch.double)

        cfg = HookManagerConfig(
            hook_types={"fc": "linear_io", "norm": "linear_io"},
            layer_types={"norm": "nn.RMSNorm"},
            module_kwargs={
                "norm": rms_norm_module_kwargs(
                    normalized_shape=8,
                    eps=model.norm.variance_epsilon,
                ),
            },
        )
        cap = CaptureCallback()
        hm = HookManager(model, config=cfg, callbacks=[cap])
        with hm.collect():
            model.zero_grad(set_to_none=True)
            model(x).square().sum(dim=1).sum().backward()
        hm.remove()
        exp = cap.record.gradient.aggregate().materialize()

        for i in range(3):
            model.zero_grad(set_to_none=True)
            model(x[i : i + 1]).square().sum().backward()
            err = (exp.data["norm"][i] - model.norm.weight.grad).abs().max().item()
            assert err < 1e-6, f"sample {i}: max diff {err:.2e}"
