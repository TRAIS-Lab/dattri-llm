"""Unit tests for dattri_llm.gradient.hooks."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dattri_llm.gradient.callbacks import HookManagerCallback
from dattri_llm.gradient.hooks import (
    REGISTER_ALL,
    HookManager,
    HookManagerConfig,
    _is_linear_io_capable,
    default_hook_assignment,
    register_linear_io_hooks,
    register_param_grad_hooks,
    remove_hooks,
    resolve_hook_assignments,
)


# --------------------------------------------------------------------------- #
# _is_linear_io_capable — purely type-based, never name-based                  #
# --------------------------------------------------------------------------- #


class TestLinearIoCapable:
    def test_linear_is_capable(self):
        assert _is_linear_io_capable(nn.Linear(4, 4))

    def test_embedding_is_capable(self):
        assert _is_linear_io_capable(nn.Embedding(32, 8))

    def test_layernorm_is_capable(self):
        assert _is_linear_io_capable(nn.LayerNorm(64))

    def test_conv2d_is_capable(self):
        assert _is_linear_io_capable(nn.Conv2d(3, 3, 3))

    def test_relu_not_capable(self):
        assert not _is_linear_io_capable(nn.ReLU())

    def test_dropout_not_capable(self):
        assert not _is_linear_io_capable(nn.Dropout())


# --------------------------------------------------------------------------- #
# resolve_hook_assignments                                                      #
# --------------------------------------------------------------------------- #


class _TopLevelLinear(nn.Module):
    """Model whose only layer is a top-level ``nn.Linear`` named ``linear``.

    Regression guard: the old name-based heuristic skipped a top-level layer
    named ``linear`` and silently registered zero layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestResolveHookAssignments:
    def test_default_hooks_top_level_linear(self):
        # The reported bug: a top-level layer named "linear" must be hooked.
        model = _TopLevelLinear()
        assignment = resolve_hook_assignments(model, HookManagerConfig())
        assert assignment == {"linear": "linear_io"}

    def test_default_assigns_all_capable_to_linear_io(self, tiny_model):
        assignment = resolve_hook_assignments(tiny_model, HookManagerConfig())
        # Every linear-family layer is hooked, regardless of name.
        assert set(assignment) == {
            "embedding", "attn_proj", "mlp.0", "mlp.2", "lm_head",
        }
        assert all(t == "linear_io" for t in assignment.values())

    def test_register_all_linear_io(self, tiny_model):
        assignment = resolve_hook_assignments(
            tiny_model, HookManagerConfig(linear_io=REGISTER_ALL)
        )
        assert all(t == "linear_io" for t in assignment.values())
        assert "lm_head" in assignment

    def test_register_all_param_grad(self, tiny_model):
        assignment = resolve_hook_assignments(
            tiny_model, HookManagerConfig(param_grad=REGISTER_ALL)
        )
        assert all(t == "param_grad" for t in assignment.values())

    def test_pattern_selects_subset(self, tiny_model):
        assignment = resolve_hook_assignments(
            tiny_model, HookManagerConfig(linear_io=[r"mlp\."])
        )
        assert set(assignment) == {"mlp.0", "mlp.2"}

    def test_explicit_hook_types_assignment(self, tiny_model):
        assignment = resolve_hook_assignments(
            tiny_model,
            HookManagerConfig(
                hook_types={"mlp.0": "linear_io", "lm_head": "param_grad"}
            ),
        )
        assert assignment == {"mlp.0": "linear_io", "lm_head": "param_grad"}

    def test_selector_extends_explicit_assignment(self, tiny_model):
        assignment = resolve_hook_assignments(
            tiny_model,
            HookManagerConfig(
                hook_types={"lm_head": "param_grad"}, linear_io=[r"mlp\."]
            ),
        )
        assert assignment == {
            "lm_head": "param_grad", "mlp.0": "linear_io", "mlp.2": "linear_io",
        }

    def test_conflicting_assignment_raises(self, tiny_model):
        with pytest.raises(ValueError, match="conflicting hook types"):
            resolve_hook_assignments(
                tiny_model,
                HookManagerConfig(
                    hook_types={"mlp.0": "param_grad"}, linear_io=[r"mlp\."]
                ),
            )

    def test_explicit_incapable_layer_raises(self, tiny_model):
        # mlp.1 is a ReLU — cannot support linear_io.
        with pytest.raises(ValueError, match="does not support"):
            resolve_hook_assignments(
                tiny_model, HookManagerConfig(hook_types={"mlp.1": "linear_io"})
            )

    def test_explicit_missing_layer_raises(self, tiny_model):
        with pytest.raises(ValueError, match="does not exist"):
            resolve_hook_assignments(
                tiny_model, HookManagerConfig(hook_types={"nope": "linear_io"})
            )

    def test_zero_layers_warns(self):
        model = _TopLevelLinear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assignment = resolve_hook_assignments(
                model, HookManagerConfig(linear_io=[r"no_such_layer"])
            )
        assert assignment == {}
        assert any("zero layers" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# register_linear_io_hooks / remove_hooks                                       #
# --------------------------------------------------------------------------- #


class TestRegisterLinearIoHooks:
    def test_hooks_registered_on_all_linear_layers(self, tiny_model):
        buffers, handles = register_linear_io_hooks(tiny_model)
        # TinyMLP linear-family layers: embedding, attn_proj, mlp.0, mlp.2,
        # lm_head.  mlp.1 is ReLU → skipped.
        assert set(buffers) == {
            "embedding", "attn_proj", "mlp.0", "mlp.2", "lm_head",
        }
        # Each layer has 2 handles (forward + backward).
        assert len(handles) == 10
        remove_hooks(handles)

    def test_buffers_populated_after_forward_backward(self, tiny_model, tiny_batch):
        buffers, handles = register_linear_io_hooks(tiny_model)
        input_ids = tiny_batch["input_ids"]
        logits = tiny_model(input_ids)
        loss = logits.mean()
        loss.backward()

        for name, buf in buffers.items():
            assert buf["activation"] is not None, f"{name}: activation is None"
            assert buf["grad_output"] is not None, f"{name}: grad_output is None"
        remove_hooks(handles)

    def test_activation_shape(self, tiny_model, tiny_batch):
        buffers, handles = register_linear_io_hooks(tiny_model)
        B, T = tiny_batch["input_ids"].shape
        tiny_model(tiny_batch["input_ids"]).mean().backward()

        # mlp.0: Linear(16, 32) → activation shape (B, T, 16)
        act = buffers["mlp.0"]["activation"]
        assert act.shape == (B, T, 16)
        remove_hooks(handles)

    def test_grad_output_shape(self, tiny_model, tiny_batch):
        buffers, handles = register_linear_io_hooks(tiny_model)
        B, T = tiny_batch["input_ids"].shape
        tiny_model(tiny_batch["input_ids"]).mean().backward()

        # mlp.0: output dim 32 → grad_output shape (B, T, 32)
        go = buffers["mlp.0"]["grad_output"]
        assert go.shape == (B, T, 32)
        remove_hooks(handles)

    def test_remove_hooks_clears_handles(self, tiny_model):
        buffers, handles = register_linear_io_hooks(tiny_model)
        assert len(handles) > 0
        remove_hooks(handles)
        assert len(handles) == 0

    def test_hooks_do_not_fire_after_removal(self, tiny_model, tiny_batch):
        buffers, handles = register_linear_io_hooks(tiny_model)
        remove_hooks(handles)
        # After removal buffers should remain None (no forward called yet)
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        for buf in buffers.values():
            assert buf["activation"] is None

    def test_custom_layer_names(self, tiny_model, tiny_batch):
        # Hook only mlp.0 via an explicit layer-name set.
        buffers, handles = register_linear_io_hooks(
            tiny_model, layer_names={"mlp.0"}
        )
        assert list(buffers.keys()) == ["mlp.0"]
        remove_hooks(handles)

    def test_layer_names_filters_to_capable_only(self, tiny_model):
        # Requesting a non-existent / non-capable name yields nothing.
        buffers, handles = register_linear_io_hooks(
            tiny_model, layer_names={"mlp.1"}  # ReLU, not capable
        )
        assert buffers == {}
        remove_hooks(handles)

    def test_dataparallel_unwrapping(self, tiny_model):
        dp_model = nn.DataParallel(tiny_model)
        buffers, handles = register_linear_io_hooks(dp_model)
        assert set(buffers) == {
            "embedding", "attn_proj", "mlp.0", "mlp.2", "lm_head",
        }
        remove_hooks(handles)

    def test_type_overrides_relabels_class_name(self, tiny_model):
        # Override mlp.0's detected type; others keep canonical_class_name.
        buffers, handles = register_linear_io_hooks(
            tiny_model, type_overrides={"mlp.0": "nn.Bilinear"}
        )
        assert buffers["mlp.0"]["_class_name"] == "nn.Bilinear"
        assert buffers["mlp.2"]["_class_name"] == "nn.Linear"
        assert buffers["embedding"]["_class_name"] == "nn.Embedding"
        remove_hooks(handles)


# --------------------------------------------------------------------------- #
# register_param_grad_hooks                                                    #
# --------------------------------------------------------------------------- #


class TestRegisterParamGradHooks:
    def test_modules_hooked_by_default(self, tiny_model):
        # Default (no layer_names): hooks every module with trainable params.
        # TinyMLP: embedding, attn_proj, mlp.0, mlp.2, lm_head (mlp.1 is ReLU).
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
        buffers, handles = register_param_grad_hooks(tiny_model, layer_names={"mlp.0"})
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        # The hook-captured grad and param.grad should be identical tensors.
        hook_grad = buffers["mlp.0"]["weight"]
        param_grad = dict(tiny_model.named_parameters())["mlp.0.weight"].grad
        assert torch.allclose(hook_grad, param_grad.cpu())
        remove_hooks(handles)

    def test_grad_shape_matches_param(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model, layer_names={"mlp.0"})
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

    def test_custom_layer_names(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(
            tiny_model, layer_names={"mlp.0", "mlp.2"}
        )
        assert set(buffers.keys()) == {"mlp.0", "mlp.2"}
        remove_hooks(handles)

    def test_frozen_params_not_hooked(self, tiny_model):
        # Freeze mlp.0 and verify it produces no buffer entry.
        tiny_model.mlp[0].weight.requires_grad_(False)
        buffers, handles = register_param_grad_hooks(tiny_model, layer_names={"mlp.0"})
        # mlp.0 has no trainable params → no buffer created for it.
        assert "mlp.0" not in buffers
        tiny_model.mlp[0].weight.requires_grad_(True)  # restore
        remove_hooks(handles)

    def test_on_param_grad_callback(self, tiny_model, tiny_batch):
        fired: list[tuple[str, str]] = []

        def cb(layer_name, param_name, grad):
            fired.append((layer_name, param_name))

        buffers, handles = register_param_grad_hooks(
            tiny_model, layer_names={"mlp.0"}, on_param_grad=cb
        )
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert ("mlp.0", "weight") in fired
        remove_hooks(handles)

    def test_remove_hooks_stops_updates(self, tiny_model, tiny_batch):
        buffers, handles = register_param_grad_hooks(tiny_model, layer_names={"mlp.0"})
        remove_hooks(handles)
        # Run backward after removal — buffer should stay None.
        tiny_model(tiny_batch["input_ids"]).mean().backward()
        assert buffers["mlp.0"]["weight"] is None

    def test_dataparallel_unwrapping(self, tiny_model):
        dp_model = nn.DataParallel(tiny_model)
        buffers, handles = register_param_grad_hooks(dp_model)
        assert len(buffers) > 0
        remove_hooks(handles)


# --------------------------------------------------------------------------- #
# default_hook_assignment                                                       #
# --------------------------------------------------------------------------- #


class _GhostLinearModel(nn.Module):
    """A model whose ``ghost`` Linear is applied functionally, not as a module.

    ``ghost``'s forward/backward hooks therefore never fire — the situation that
    stalls HookManager step completion (cf. ``MultiheadAttention.out_proj``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(4, 4)
        self.ghost = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.used(x)
        # Apply ghost's weight functionally — its module hooks do not fire.
        return F.linear(x, self.ghost.weight, self.ghost.bias)


class TestDefaultHookAssignment:
    def test_skips_functionally_invoked_layer(self):
        model = _GhostLinearModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assignment = default_hook_assignment(model, torch.randn(2, 4))
        # 'used' is a real module call → kept; 'ghost' never fires → skipped.
        assert assignment == {"used": "linear_io"}
        assert any("ghost" in str(w.message) for w in caught)

    def test_default_model_keeps_all_layers(self, tiny_model, tiny_batch):
        assignment = default_hook_assignment(
            tiny_model, tiny_batch["input_ids"]
        )
        assert set(assignment) == {
            "embedding", "attn_proj", "mlp.0", "mlp.2", "lm_head",
        }
        assert all(t == "linear_io" for t in assignment.values())

    def test_result_is_usable_as_hook_types(self):
        model = _GhostLinearModel()
        assignment = default_hook_assignment(model, torch.randn(2, 4))
        # The discovered assignment resolves cleanly (ghost excluded).
        resolved = resolve_hook_assignments(
            model, HookManagerConfig(hook_types=assignment)
        )
        assert resolved == {"used": "linear_io"}

    def test_dict_input_unpacked(self, tiny_model, tiny_batch):
        assignment = default_hook_assignment(
            tiny_model, {"input_ids": tiny_batch["input_ids"]}
        )
        assert "mlp.0" in assignment

    def test_sequence_first_layer_is_discovered_without_shape_filter(self):
        class SequenceFirst(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                return self.proj(x.transpose(0, 1)).transpose(0, 1)

        model = SequenceFirst()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assignment = default_hook_assignment(model, torch.randn(2, 3, 4))
        assert assignment == {"proj": "linear_io"}

    def test_param_grad_layer_discovered_via_backward(self):
        class _CustomParam(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(4))
                self.lin = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x) * self.scale

        model = _CustomParam()
        assignment = default_hook_assignment(model, torch.randn(2, 4))
        # The container owns 'scale' directly (not linear-IO-capable) → param_grad;
        # the Linear is a real module call → linear_io.
        assert assignment[""] == "param_grad"
        assert assignment["lin"] == "linear_io"


# --------------------------------------------------------------------------- #
# HookManagerConfig.layer_types — manual layer-type overrides                  #
# --------------------------------------------------------------------------- #


class _CustomLinear(nn.Linear):
    """A user-defined linear layer whose class name is not recognised by
    ``canonical_class_name`` but which is still linear-IO-capable (it subclasses
    ``nn.Linear``)."""


class _CustomLinearModel(nn.Module):
    """``embedding → custom (CustomLinear) → lm_head``."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.custom = _CustomLinear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.custom(self.embedding(input_ids)))


class _Recording(HookManagerCallback):
    def __init__(self) -> None:
        self.records: list = []

    def on_step_end(self, record) -> None:
        self.records.append(record)


class TestLayerTypesConfig:
    def test_validation_rejects_non_dict(self):
        with pytest.raises(TypeError, match="layer_types must be a dict"):
            HookManagerConfig(layer_types=["mlp.0"])  # type: ignore[arg-type]

    def test_validation_rejects_non_str_values(self):
        with pytest.raises(TypeError, match="str type names"):
            HookManagerConfig(layer_types={"mlp.0": 123})  # type: ignore[dict-item]

    def test_none_defaults_to_empty(self):
        assert HookManagerConfig().layer_types == {}

    def test_layer_types_does_not_affect_is_default(self):
        # A pure relabel still uses the default hooking behaviour.
        assert HookManagerConfig(layer_types={"custom": "nn.Linear"}).is_default

    def test_override_relabels_captured_type(self):
        model = _CustomLinearModel()
        cb = _Recording()
        hm = HookManager(
            model,
            config=HookManagerConfig(layer_types={"custom": "nn.Linear"}),
            callbacks=[cb],
        )
        with hm.collect():
            model(torch.randint(0, 32, (2, 8))).mean().backward()
        types = cb.records[0].gradient.layer_types
        # Override applied to 'custom'; other layers keep their canonical type.
        assert types["custom"] == "nn.Linear"
        assert types["embedding"] == "nn.Embedding"
        assert types["lm_head"] == "nn.Linear"

    def test_without_override_custom_class_keeps_canonical_name(self):
        model = _CustomLinearModel()
        cb = _Recording()
        hm = HookManager(model, callbacks=[cb])
        with hm.collect():
            model(torch.randint(0, 32, (2, 8))).mean().backward()
        types = cb.records[0].gradient.layer_types
        # canonical_class_name returns the (unrecognised) custom path.
        assert types["custom"].endswith("_CustomLinear")

    def test_warns_when_named_layer_not_hooked(self):
        model = _CustomLinearModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HookManager(
                model,
                config=HookManagerConfig(layer_types={"nope": "nn.Linear"}),
            )
        assert any("were not hooked" in str(w.message) for w in caught)

    def test_no_warning_when_named_layer_hooked(self):
        model = _CustomLinearModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HookManager(
                model,
                config=HookManagerConfig(layer_types={"custom": "nn.Linear"}),
            )
        assert not any("were not hooked" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# linear_io hooks skip frozen (non-trainable) layers — LoRA / frozen backbones  #
# --------------------------------------------------------------------------- #


class TestTrainabilityFilter:
    def test_frozen_leading_layer_skipped_and_completes(self):
        # A frozen layer fed a leaf input never fires its backward (its output
        # needs no grad), which would stall completion if it were hooked.  It must
        # be skipped, leaving only the trainable layer — mirrors a LoRA base_layer.
        model = nn.Sequential()
        model.add_module("frozen", nn.Linear(8, 8, bias=False))
        model.add_module("trainable", nn.Linear(8, 4, bias=False))
        for p in model.frozen.parameters():
            p.requires_grad_(False)
        cb = _Recording()
        hm = HookManager(model, config=HookManagerConfig(linear_io=REGISTER_ALL),
                         callbacks=[cb])
        with hm.collect():
            model(torch.randn(4, 8)).sum().backward()
        hm.remove()
        assert len(cb.records) == 1                          # step completed
        assert set(cb.records[-1].gradient.layer_names) == {"trainable"}

    def test_lora_adapters_captured_exactly(self):
        peft = pytest.importorskip("peft")
        from dattri_llm.gradient import ops

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4, bias=False)

            def forward(self, x):
                return self.fc(x)

        torch.manual_seed(0)
        model = peft.get_peft_model(
            Wrap(), peft.LoraConfig(r=4, target_modules=["fc"], lora_alpha=8,
                                    bias="none", lora_dropout=0.0)
        ).eval()
        x = torch.randn(4, 8)

        cb = _Recording()
        # REGISTER_ALL now works out of the box: the frozen base_layer is skipped,
        # only the trainable lora_A / lora_B adapters are captured.
        hm = HookManager(model, config=HookManagerConfig(linear_io=REGISTER_ALL),
                         callbacks=[cb])
        with hm.collect():
            model(x).sum().backward()
        hm.remove()
        grad = cb.records[-1].gradient
        names = set(grad.layer_names)
        assert all("base_layer" not in n for n in names)
        assert any("lora_A" in n for n in names) and any("lora_B" in n for n in names)

        # The captured factorized adapter gradient equals the true per-sample one.
        a_name = next(n for n in names if "lora_A" in n)
        mat = ops.materialize(grad.data[a_name], "nn.Linear").float()
        module = dict(model.named_modules())[a_name]
        rows = []
        for i in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            model(x[i : i + 1]).sum().backward()
            rows.append(module.weight.grad.flatten().clone())
        assert torch.allclose(mat, torch.stack(rows), atol=1e-5)


# --------------------------------------------------------------------------- #
# HookManagerConfig(projection=...) — per-layer random projection on capture    #
# --------------------------------------------------------------------------- #


class TestProjectionConfig:
    def test_validation_rejects_non_dict(self):
        with pytest.raises(TypeError, match="projection must be a dict"):
            HookManagerConfig(projection=[{"proj_dim": 8}])  # type: ignore[arg-type]

    def test_validation_rejects_non_dict_values(self):
        with pytest.raises(TypeError, match="projection must be a dict"):
            HookManagerConfig(projection={"__default__": 8})  # type: ignore[dict-item]

    def test_none_keeps_projection_off(self):
        assert HookManagerConfig().projection is None

    def test_projection_does_not_affect_is_default(self):
        assert HookManagerConfig(projection={"__default__": {"proj_dim": 8}}).is_default

    def _run(self, projection, projector=None):
        torch.manual_seed(0)   # identical weights + input across calls, for comparisons
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        cb = _Recording()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL,
                                     projection=projection, projector=projector),
            callbacks=[cb],
        )
        with hm.collect():
            model(torch.randn(4, 16)).sum().backward()
        return cb.records[-1].gradient

    def test_trak_projection_materializes(self):
        g = self._run({"__default__": {"factorize": False, "proj_dim": 64,
                       "proj_max_batch_size": 8, "proj_type": "rademacher"}})
        for n in g.layer_names:
            assert g.representation[n] == "materialized"
            assert g.data[n].shape == (4, 64)

    def test_logra_projection_stays_factorized(self):
        g = self._run({"__default__": {"factorize": True, "proj_dim": 16,
                       "proj_max_batch_size": 8, "proj_type": "rademacher"}})
        for n in g.layer_names:
            assert g.representation[n] == "factorized"
            assert g.data[n].activation.shape[-1] == 16

    def test_off_by_default_keeps_factorized(self):
        g = self._run(None)
        assert all(g.representation[n] == "factorized" for n in g.layer_names)

    def test_only_configured_layers_projected(self):
        # Project just layer "0"; layer "2" has no entry and no __default__.
        g = self._run({"0": {"factorize": False, "proj_dim": 64,
                             "proj_max_batch_size": 8, "proj_type": "rademacher"}})
        assert g.representation["0"] == "materialized" and g.data["0"].shape == (4, 64)
        assert g.representation["2"] == "factorized"

    def test_custom_projector_is_used(self):
        # A sentinel projector (truncate-to-proj_dim) overrides the dattri default.
        called = {"n": 0}

        def fake_projector(feature, batch_size, *, proj_dim, **kw):
            called["n"] += 1
            return lambda x, ensemble_id=0: x[:, :proj_dim]

        g = self._run({"__default__": {"factorize": False, "proj_dim": 4}},
                      projector=fake_projector)
        assert called["n"] > 0
        for n in g.layer_names:
            assert g.data[n].shape == (4, 4)

    @pytest.mark.parametrize("factorize", [True, False])
    def test_capture_time_equals_assembly_time_projection(self, factorize):
        # Projecting at capture (per micro-batch) must be bit-identical to
        # projecting the fully-assembled gradient — project(cat) == cat(project).
        from dattri.func.projection import random_project
        from dattri_llm.gradient import ops
        from dattri_llm.gradient.gradient import Factorized
        cfg = {"factorize": factorize, "proj_dim": 12, "proj_max_batch_size": 8,
               "proj_type": "rademacher", "proj_seed": 7}
        raw = self._run(None)                                   # un-projected capture
        ref = raw.project(random_project, {"__default__": dict(cfg)})
        cap = self._run({"__default__": dict(cfg)})             # capture-time projection
        for n in cap.layer_names:
            a, b = cap.data[n], ref.data[n]
            if isinstance(a, Factorized):
                a = ops.materialize(a, "nn.Linear"); b = ops.materialize(b, "nn.Linear")
            assert torch.allclose(a, b, atol=1e-5), n

    def test_shared_layer_pairs_each_call(self):
        # A layer invoked twice per forward must project each call's (a, g) pair
        # (LIFO per-device stack), producing 2*B rows with no dropped call.
        class Shared(nn.Module):
            def __init__(self): super().__init__(); self.lin = nn.Linear(10, 6)
            def forward(self, x1, x2): return self.lin(x1) + self.lin(x2)
        m = Shared()
        cb = _Recording()
        hm = HookManager(m, config=HookManagerConfig(
            linear_io=REGISTER_ALL,
            projection={"__default__": {"factorize": False, "proj_dim": 12,
                        "proj_max_batch_size": 8, "proj_type": "rademacher"}}),
            callbacks=[cb])
        with hm.collect():
            m(torch.randn(4, 10), torch.randn(4, 10)).sum().backward()
        assert cb.records[-1].gradient.data["lin"].shape == (8, 12)   # 2*B rows


# --------------------------------------------------------------------------- #
# HookManager(non_batch_first_layers=...) — sequence-first captures            #
# --------------------------------------------------------------------------- #


class _SeqFirstModel(nn.Module):
    """``proj`` is applied to a sequence-first ``(T, B, d)`` tensor."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, d)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)        # (B, T, d)
        x = x.transpose(0, 1)                # (T, B, d) — sequence-first
        x = self.proj(x)
        return x.transpose(0, 1)


class TestNonBatchFirstLayers:
    def test_seq_first_layer_collects_without_error(self):
        # Regression: a (T, B, d) layer must not raise "same batch size".
        model = _SeqFirstModel()
        cb = _Recording()
        hm = HookManager(model, callbacks=[cb], non_batch_first_layers={"proj"})
        B, T = 3, 5
        with hm.collect():
            model(torch.randint(0, 16, (B, T))).sum().backward()
        g = cb.records[0].gradient
        assert g.batch_size == B
        assert g.data["proj"].batch_first is False
        # Raw capture stays sequence-first; batch axis is dim 1.
        assert g.data["proj"].activation.shape[1] == B

    def test_without_flag_raises_batch_mismatch(self):
        model = _SeqFirstModel()
        cb = _Recording()
        hm = HookManager(model, callbacks=[cb])  # proj NOT flagged
        with pytest.raises(ValueError, match="same batch size"):
            with hm.collect():
                model(torch.randint(0, 16, (3, 5))).sum().backward()

    def test_warns_when_named_layer_not_hooked(self):
        model = _SeqFirstModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HookManager(model, non_batch_first_layers={"nope"})
        assert any("had no effect" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# Unbatched positional embeddings (nanoGPT: pos = arange(T), shape (T,))       #
# --------------------------------------------------------------------------- #


class _NanoGPT(nn.Module):
    """Token + positional embeddings where ``wpe`` is fed an unbatched
    ``pos = arange(T)`` (shape ``(T,)``), as in nanoGPT."""

    def __init__(self, vocab: int = 16, d: int = 8, block: int = 5) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long)        # (T,) — no batch dim
        return self.head(self.wte(idx) + self.wpe(pos))   # broadcast add


class TestUnbatchedPositionalEmbedding:
    def test_collects_without_validation_error(self):
        # Regression: a (T,) positional-embedding input previously failed with
        # "invalid embedding factor dimensions".
        model = _NanoGPT()
        cb = _Recording()
        hm = HookManager(model, callbacks=[cb])
        B, T = 4, 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with hm.collect():
                model(torch.randint(0, 16, (B, T))).sum().backward()
        g = cb.records[0].gradient
        g.validate()
        # wpe is normalized to a single broadcast row; wte keeps the batch dim.
        assert g.data["wpe"].activation.shape == (1, T)
        assert g.data["wte"].activation.shape[0] == B
        assert g.batch_size == B

    def test_warns_about_broadcast_layer(self):
        # The broadcast (batch-collapsed) wpe gradient is not per-sample; the
        # user is warned once and told to exclude it.
        model = _NanoGPT()
        cb = _Recording()
        hm = HookManager(model, callbacks=[cb])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with hm.collect():
                for _ in range(3):  # only one warning across steps
                    model(torch.randint(0, 16, (4, 5))).sum().backward()
        broadcast = [str(w.message) for w in caught if "broadcast" in str(w.message)]
        assert len(broadcast) == 1
        assert "wpe" in broadcast[0]
