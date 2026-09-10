"""Optimizer state in materialized layout, the capture-side preconditioner,
and the per-step moments callback.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import CaptureCallback, OptimizerStateCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.gradient.optimizer_state import (
    GradientPreconditioner,
    OptimizerSnapshot,
    hf_trainer_param_names,
    optimizer_from_state_dict,
    optimizer_type,
)

IN, HID, OUT = 4, 8, 3
BATCH = 5


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.norm = nn.LayerNorm(HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.norm(self.fc1(x))))


def _model_and_optimizer(steps: int = 3, lr: float = 0.05):
    torch.manual_seed(0)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99), eps=1e-6)
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(BATCH, IN)).pow(2).sum().backward()
        opt.step()
    return model, opt


def _flat_linear(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([w.reshape(w.shape[0], -1), b.reshape(-1, 1)], dim=1).reshape(-1)


class TestOptimizerType:
    def test_resolves_torch_families_and_subclasses(self):
        p = [nn.Parameter(torch.zeros(2))]
        assert optimizer_type(torch.optim.AdamW(p)) == "AdamW"
        assert optimizer_type(torch.optim.SGD(p, lr=0.1)) == "SGD"

        class MyAdam(torch.optim.Adam):
            pass

        assert optimizer_type(MyAdam(p)) == "Adam"

    def test_rejects_non_coordinate_wise_optimizers(self):
        p = [nn.Parameter(torch.zeros(2))]
        with pytest.raises(NotImplementedError, match="not a supported optimizer"):
            optimizer_type(torch.optim.LBFGS(p))


class TestOptimizerSnapshot:
    def test_linear_layout_matches_weight_then_bias_per_row(self):
        model, opt = _model_and_optimizer()
        snap = OptimizerSnapshot(model, opt)
        w, b = model.fc1.weight, model.fc1.bias
        expected = _flat_linear(opt.state[w]["exp_avg"], opt.state[b]["exp_avg"])
        assert torch.equal(snap.state("fc1", "exp_avg"), expected)
        assert snap.width("fc1") == HID * (IN + 1)
        assert snap.width("fc1", include_bias=False) == HID * IN

    def test_norm_layout_is_gamma_then_beta(self):
        model, opt = _model_and_optimizer()
        snap = OptimizerSnapshot(model, opt)
        w, b = model.norm.weight, model.norm.bias
        expected = torch.cat([opt.state[w]["exp_avg_sq"], opt.state[b]["exp_avg_sq"]])
        assert torch.equal(snap.state("norm", "exp_avg_sq"), expected)

    def test_layout_agrees_with_materialized_capture(self):
        # The flat layout must index exactly as ops.materialize lays a layer's
        # per-sample gradient out: the captured gradients summed over the batch
        # equal the parameters' own .grad laid out the same way.
        torch.manual_seed(0)
        model = MLP()
        opt = torch.optim.AdamW(model.parameters())
        snap = OptimizerSnapshot(model, opt)
        cb = CaptureCallback()
        hm = HookManager(
            model, config=HookManagerConfig(linear_io=REGISTER_ALL), callbacks=[cb]
        )
        with hm.collect():
            model.zero_grad()
            model(torch.randn(BATCH, IN)).pow(2).sum().backward()
        hm.remove()
        grad = cb.record.gradient
        for layer in ("fc1", "norm", "fc2"):
            module = model.get_submodule(layer)
            flat_grad = snap.flatten(
                layer, {"weight": module.weight.grad, "bias": module.bias.grad}
            )
            captured = ops.materialize(grad.data[layer], grad.layer_types[layer]).sum(0)
            assert captured.shape == flat_grad.shape, layer
            assert torch.allclose(captured, flat_grad, atol=1e-5), layer

    def test_hf_conv1d_weight_is_transposed_into_the_gradient_layout(self):
        pytest.importorskip("transformers")
        from transformers.pytorch_utils import Conv1D

        torch.manual_seed(0)
        model = nn.Sequential(Conv1D(6, IN))  # weight (in, out) = (IN, 6)
        opt = torch.optim.AdamW(model.parameters())
        snap = OptimizerSnapshot(model, opt)
        cb = CaptureCallback()
        hm = HookManager(
            model, config=HookManagerConfig(linear_io=REGISTER_ALL), callbacks=[cb]
        )
        with hm.collect():
            model.zero_grad()
            model(torch.randn(BATCH, IN)).pow(2).sum().backward()
        hm.remove()
        grad = cb.record.gradient
        layer = model[0]
        flat = snap.flatten("0", {"weight": layer.weight.grad, "bias": layer.bias.grad})
        captured = ops.materialize(grad.data["0"], grad.layer_types["0"]).sum(0)
        assert captured.shape == flat.shape == (6 * (IN + 1),)
        assert torch.allclose(captured, flat, atol=1e-5)
        idx = torch.tensor([0, IN, IN + 1, 6 * (IN + 1) - 1])
        gathered = snap.flatten(
            "0", {"weight": layer.weight.grad, "bias": layer.bias.grad}, idx
        )
        assert torch.equal(gathered, flat[idx])

    @pytest.mark.parametrize("layer", ["fc1", "norm", "fc2"])
    @pytest.mark.parametrize("include_bias", [True, False])
    def test_gather_reads_the_same_entries_as_flatten_then_index(
        self, layer, include_bias
    ):
        model, opt = _model_and_optimizer(steps=2)
        snap = OptimizerSnapshot(model, opt)
        width = snap.width(layer, include_bias)
        idx = torch.randperm(width, generator=torch.Generator().manual_seed(1))[:7]
        full = snap.state(layer, "exp_avg", include_bias=include_bias)
        assert full.shape == (width,)
        got = snap.state(layer, "exp_avg", idx, include_bias=include_bias)
        assert torch.equal(got, full[idx])

    def test_state_is_none_before_the_first_update(self):
        torch.manual_seed(0)
        model = MLP()
        opt = torch.optim.AdamW(model.parameters())
        snap = OptimizerSnapshot(model, opt)
        assert snap.step_count("fc1") == 0
        assert snap.state("fc1", "exp_avg") is None
        assert snap.states("fc1") == {
            "exp_avg": None,
            "exp_avg_sq": None,
            "max_exp_avg_sq": None,
        }
        model_t, opt_t = _model_and_optimizer(steps=2)
        assert OptimizerSnapshot(model_t, opt_t).step_count("fc1") == 2

    def test_hyperparameters_come_from_the_layers_group(self):
        model, opt = _model_and_optimizer(lr=0.05)
        hp = OptimizerSnapshot(model, opt).hyperparameters("fc2")
        assert hp["lr"] == pytest.approx(0.05)
        assert hp["betas"] == (0.8, 0.99)
        assert hp["eps"] == pytest.approx(1e-6)
        assert hp["amsgrad"] is False  # from the optimizer's defaults

    def test_precondition_uses_the_gathered_state(self):
        model, opt = _model_and_optimizer(steps=2)
        snap = OptimizerSnapshot(model, opt)
        idx = torch.tensor([0, 5, 17, 33])
        g = torch.randn(3, 4)
        got = snap.precondition("fc1", g, idx)
        want = ops.precondition(
            g,
            snap.states("fc1", idx),
            optimizer_type="AdamW",
            step=3,
            **snap.hyperparameters("fc1"),
        )
        assert torch.allclose(got, want)

    def test_unsupported_layer_type_raises(self):
        model = nn.Sequential(nn.ConvTranspose1d(2, 2, 3))
        opt = torch.optim.AdamW(model.parameters())
        with pytest.raises(NotImplementedError, match="layout is not defined"):
            OptimizerSnapshot(model, opt).width("0")

    def test_optimizer_from_state_dict_roundtrips_a_trainer_style_optimizer(self):
        torch.manual_seed(0)
        model = MLP()
        names = hf_trainer_param_names(model)
        params = dict(model.named_parameters())
        decay = [params[n] for n in names if "bias" not in n and "norm" not in n]
        no_decay = [params[n] for n in names if "bias" in n or "norm" in n]
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=0.01,
        )
        for _ in range(2):
            opt.zero_grad()
            model(torch.randn(BATCH, IN)).pow(2).sum().backward()
            opt.step()
        saved = copy.deepcopy(opt.state_dict())
        live = OptimizerSnapshot(model, opt)
        loaded = OptimizerSnapshot(model, optimizer_from_state_dict(model, saved))
        for layer in ("fc1", "norm", "fc2"):
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(live.state(layer, key), loaded.state(layer, key)), (
                    layer,
                    key,
                )
            assert live.step_count(layer) == loaded.step_count(layer)
        assert loaded.hyperparameters("fc1")["weight_decay"] == pytest.approx(0.1)
        assert loaded.hyperparameters("norm")["weight_decay"] == pytest.approx(0.0)


class TestGradientPreconditioner:
    def test_subset_config_maps_the_kept_coordinates(self):
        model, opt = _model_and_optimizer(steps=2)
        snap = OptimizerSnapshot(model, opt)
        proj = ops.DattriProjector()
        precond = GradientPreconditioner(snap, proj)
        kw = {"style": "subset_materialized", "proj_dim": 6, "proj_seed": 3}
        idx = proj.subset_indices(
            snap.width("fc1"), proj_dim=6, proj_seed=3, device=torch.device("cpu")
        )
        entries = torch.randn(4, 6)
        assert torch.allclose(
            precond("fc1", entries, kw), snap.precondition("fc1", entries, idx)
        )

    def test_resolves_a_lazy_snapshot_once(self):
        model, opt = _model_and_optimizer(steps=1)
        calls = []

        def build():
            calls.append(1)
            return OptimizerSnapshot(model, opt)

        precond = GradientPreconditioner(build)
        entries = torch.randn(2, HID * (IN + 1))
        precond("fc1", entries)
        precond("fc1", entries)
        assert len(calls) == 1
        assert precond.enabled


class TestOptimizerStateCallback:
    @staticmethod
    def _run(projection=None, steps: int = 3):
        torch.manual_seed(0)
        model = MLP()
        opt = torch.optim.AdamW(model.parameters(), lr=0.05, betas=(0.8, 0.99))
        cb = OptimizerStateCallback(model, opt, projection=projection)
        capture = CaptureCallback()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL, projection=projection),
            callbacks=[capture, cb],
        )
        before, grads, records = [], [], []
        with hm.collect():
            for step in range(steps):
                opt.zero_grad()
                # What the optimizer holds right before this step's update.
                before.append(
                    {
                        n: (
                            opt.state.get(p, {})
                            .get("exp_avg", torch.zeros_like(p))
                            .clone(),
                            opt.state.get(p, {})
                            .get("exp_avg_sq", torch.zeros_like(p))
                            .clone(),
                        )
                        for n, p in model.named_parameters()
                    }
                )
                model(torch.randn(BATCH, IN)).pow(2).sum().backward()
                grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
                opt.step()
                cb.record_post(step)
                records.append(capture.record.gradient)
        hm.remove()
        return model, opt, cb, before, grads, records

    def test_pre_snapshot_is_the_state_before_the_step(self):
        _, _, cb, before, _, _ = self._run()
        for step in range(3):
            pre = cb.pre[step]
            assert pre["step"] == step
            m, v = pre["layers"]["fc1"]
            expected_m = _flat_linear(
                before[step]["fc1.weight"][0], before[step]["fc1.bias"][0]
            )
            expected_v = _flat_linear(
                before[step]["fc1.weight"][1], before[step]["fc1.bias"][1]
            )
            assert torch.allclose(m, expected_m)
            assert torch.allclose(v, expected_v)

    def test_dynamics_recover_the_consumed_batch_gradient(self):
        # g_t = (m_post - beta1 m_pre) / (1 - beta1) is exactly the gradient the
        # optimizer consumed (here: unclipped param.grad).
        _, _, cb, _, grads, _ = self._run()
        dyn = cb.dynamics()
        assert sorted(dyn) == [0, 1, 2]
        for step, d in dyn.items():
            beta1 = d["betas"][0]
            assert d["step"] == step + 1
            assert d["lr"] == pytest.approx(0.05)
            for layer in ("fc1", "fc2"):
                m_pre, _ = d["pre"][layer]
                m_post, _ = d["post"][layer]
                g_t = (m_post - beta1 * m_pre) / (1 - beta1)
                g = grads[step]
                expected = _flat_linear(g[f"{layer}.weight"], g[f"{layer}.bias"])
                assert torch.allclose(g_t, expected, atol=1e-5), (step, layer)

    def test_subset_projection_reads_the_captured_coordinates(self):
        projection = {
            "__default__": {
                "style": "subset_materialized",
                "proj_dim": 6,
                "proj_seed": 3,
            }
        }
        model, opt, cb, before, _, records = self._run(projection=projection)
        snap = OptimizerSnapshot(model, opt)
        proj = ops.DattriProjector()
        for layer in ("fc1", "fc2"):
            width = snap.width(layer)
            idx = proj.subset_indices(
                width, proj_dim=6, proj_seed=3, device=torch.device("cpu")
            )
            assert torch.equal(cb.coordinates(layer), idx)
            m, _ = cb.pre[1]["layers"][layer]
            full = _flat_linear(
                before[1][f"{layer}.weight"][0], before[1][f"{layer}.bias"][0]
            )
            assert torch.allclose(m, full[idx])
            assert records[1].data[layer].shape == (
                BATCH,
                6,
            )  # the capture kept the same S

    def test_post_before_pre_raises(self):
        model = MLP()
        cb = OptimizerStateCallback(model, torch.optim.AdamW(model.parameters()))
        with pytest.raises(KeyError, match="no pre-step snapshot"):
            cb.record_post(0)

    def test_non_adam_optimizer_raises(self):
        model = MLP()
        cb = OptimizerStateCallback(model, torch.optim.SGD(model.parameters(), lr=0.1))
        with pytest.raises(NotImplementedError, match="Adam-family"):
            cb.coordinates("fc1")
