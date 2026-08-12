"""Unit tests for FSDP auto-wrap-policy construction from ``fsdp_config``.

``_build_auto_wrap_policy`` is a pure function of (model, config dict), so it
is testable without a process group: the returned policy is probed directly
with FSDP's ``(module, recurse, nonwrapped_numel)`` calling convention.
"""

from __future__ import annotations

import pytest
from torch import nn

from dattri_llm.gradient.streaming import _build_auto_wrap_policy


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.head = nn.Linear(4, 2)


class TestBuildAutoWrapPolicy:
    def test_no_policy_keys_returns_none(self):
        cfg = {"use_orig_params": True}
        assert _build_auto_wrap_policy(Model(), cfg) is None
        assert cfg == {"use_orig_params": True}  # untouched

    def test_transformer_cls_policy_matches_named_blocks(self):
        model = Model()
        cfg = {"transformer_layer_cls_to_wrap": ["Block"], "other": 1}
        policy = _build_auto_wrap_policy(model, cfg)
        assert policy is not None
        assert "transformer_layer_cls_to_wrap" not in cfg  # consumed
        assert cfg == {"other": 1}
        # FSDP probes with recurse=True while descending, then recurse=False
        # to decide wrapping; Block instances wrap, others do not.
        block = model.blocks[0]
        assert policy(module=block, recurse=False, nonwrapped_numel=0)
        assert not policy(module=model.head, recurse=False, nonwrapped_numel=0)
        assert policy(module=model, recurse=True, nonwrapped_numel=0)

    def test_single_class_name_string_accepted(self):
        policy = _build_auto_wrap_policy(
            Model(),
            {"transformer_layer_cls_to_wrap": "Block"},
        )
        assert policy is not None

    def test_unknown_class_raises(self):
        with pytest.raises(ValueError, match="NoSuchBlock"):
            _build_auto_wrap_policy(
                Model(),
                {"transformer_layer_cls_to_wrap": ["NoSuchBlock"]},
            )

    def test_min_num_params_builds_size_based_policy(self):
        model = Model()
        cfg = {"min_num_params": 1}
        policy = _build_auto_wrap_policy(model, cfg)
        assert policy is not None
        assert cfg == {}  # consumed
        # Size-based: wraps once the module holds >= min_num_params.
        assert policy(
            module=model.blocks[0],
            recurse=False,
            nonwrapped_numel=model.blocks[0].fc.weight.numel(),
        )

    def test_both_keys_raise(self):
        with pytest.raises(ValueError, match="not both"):
            _build_auto_wrap_policy(
                Model(),
                {"transformer_layer_cls_to_wrap": "Block", "min_num_params": 1},
            )

    def test_explicit_policy_passes_through(self):
        sentinel = object()
        cfg = {
            "auto_wrap_policy": sentinel,
            "transformer_layer_cls_to_wrap": "Block",
        }
        assert _build_auto_wrap_policy(Model(), cfg) is None
        assert cfg["auto_wrap_policy"] is sentinel  # untouched, wins as-is
