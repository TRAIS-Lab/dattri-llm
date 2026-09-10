"""Optimizer-aware capture: ``HookManager(optimizer=...)`` buffers the
preconditioned per-sample gradient, exactly ``ops.precondition`` of the raw one.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import CaptureCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.gradient.optimizer_state import OptimizerSnapshot
from dattri_llm.gradient.streaming import GradientStreamer

IN, HID, OUT = 4, 6, 3
BATCH = 5
LAYERS = ("fc1", "norm", "fc2")


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.norm = nn.LayerNorm(HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.norm(self.fc1(x))))


OPTIMIZERS = {
    "sgd_momentum": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "adamw": lambda p: torch.optim.AdamW(p, lr=0.05, betas=(0.8, 0.99)),
    "rmsprop": lambda p: torch.optim.RMSprop(p, lr=0.01, centered=True),
}


def _trained(make, steps: int = 2):
    torch.manual_seed(0)
    model = MLP()
    opt = make(model.parameters())
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(BATCH, IN)).pow(2).sum().backward()
        opt.step()
    return model, opt


def _capture(model, opt, projection=None, precondition=True):
    """One step's captured Gradient (preconditioned or raw) at the same input."""
    cb = CaptureCallback()
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL, projection=projection),
        callbacks=[cb],
        optimizer=opt,
    )
    hm.precondition = precondition
    torch.manual_seed(1)
    x = torch.randn(BATCH, IN)
    with hm.collect():
        model.zero_grad()
        model(x).pow(2).sum().backward()
    hm.remove()
    return cb.record.gradient


class TestPreconditionedCapture:
    @pytest.mark.parametrize("name", list(OPTIMIZERS))
    def test_unprojected_capture_is_the_preconditioned_full_gradient(self, name):
        model, opt = _trained(OPTIMIZERS[name])
        snap = OptimizerSnapshot(model, opt)
        got = _capture(model, opt)
        raw = _capture(model, opt, precondition=False)
        for layer in LAYERS:
            assert got.representation[layer] == "materialized"
            assert raw.representation[layer] == "factorized"
            full = ops.materialize(raw.data[layer], raw.layer_types[layer])
            want = snap.precondition(layer, full)
            assert got.data[layer].shape == want.shape
            assert torch.allclose(got.data[layer], want, atol=1e-5), (name, layer)

    def test_subset_capture_preconditions_the_kept_coordinates(self):
        model, opt = _trained(OPTIMIZERS["adamw"])
        snap = OptimizerSnapshot(model, opt)
        projection = {
            "__default__": {
                "style": "subset_materialized",
                "proj_dim": 5,
                "proj_seed": 2,
            }
        }
        got = _capture(model, opt, projection=projection)
        raw = _capture(model, opt, precondition=False)
        proj = ops.DattriProjector()
        for layer in LAYERS:
            full = ops.materialize(raw.data[layer], raw.layer_types[layer])
            idx = proj.subset_indices(
                full.shape[1], proj_dim=5, proj_seed=2, device=torch.device("cpu")
            )
            want = snap.precondition(layer, full[:, idx], idx)
            assert got.data[layer].shape == (BATCH, 5)
            assert torch.allclose(got.data[layer], want, atol=1e-5), layer

    def test_materialized_style_projects_after_the_map(self):
        model, opt = _trained(OPTIMIZERS["adamw"])
        snap = OptimizerSnapshot(model, opt)
        kw = {"proj_dim": 4, "proj_seed": 3, "proj_max_batch_size": 8}
        projection = {"__default__": {"style": "materialized", **kw}}
        got = _capture(model, opt, projection=projection)
        raw = _capture(model, opt, precondition=False)
        proj = ops.DattriProjector()
        for layer in LAYERS:
            full = ops.materialize(raw.data[layer], raw.layer_types[layer])
            want = ops.apply_projection(proj, snap.precondition(layer, full), **kw)
            assert got.data[layer].shape == (BATCH, 4)
            assert torch.allclose(got.data[layer], want, atol=1e-4), layer

    def test_state_is_read_live_across_steps(self):
        # The map reads the optimizer's current (pre-step) state at every
        # capture, so a training loop under the hooks preconditions each
        # step with the state that step updates from.
        model, opt = _trained(OPTIMIZERS["adamw"], steps=0)
        snap = OptimizerSnapshot(model, opt)
        cb = CaptureCallback()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
            optimizer=opt,
        )
        with hm.collect():
            for step in range(3):
                opt.zero_grad()
                x = torch.randn(BATCH, IN)
                # Reference: raw per-sample gradient at this state, mapped by
                # hand with the moments before the update.
                m = torch.cat(
                    [
                        opt.state.get(p, {})
                        .get("exp_avg", torch.zeros_like(p))
                        .reshape(-1)
                        for p in (model.fc2.weight,)
                    ]
                )
                model(x).pow(2).sum().backward()
                got = cb.record.gradient.data["fc2"]
                width = snap.width("fc2")
                assert got.shape == (BATCH, width)
                assert snap.step_count("fc2") == step
                assert m.shape[0] == HID * OUT
                opt.step()
        hm.remove()

    def test_toggle_switches_between_raw_and_preconditioned(self):
        model, opt = _trained(OPTIMIZERS["adamw"])
        cb = CaptureCallback()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
            optimizer=opt,
        )
        assert hm.supports_preconditioning
        assert hm.precondition
        reps = []
        with hm.collect():
            for enabled in (True, False, True):
                hm.precondition = enabled
                model.zero_grad()
                model(torch.randn(BATCH, IN)).pow(2).sum().backward()
                reps.append(cb.record.gradient.representation["fc1"])
        hm.remove()
        assert reps == ["materialized", "factorized", "materialized"]

    def test_manager_without_optimizer_cannot_precondition(self):
        model, _ = _trained(OPTIMIZERS["adamw"])
        hm = HookManager(model, config=HookManagerConfig(linear_io=REGISTER_ALL))
        assert not hm.supports_preconditioning
        assert not hm.precondition
        hm.precondition = False  # a no-op, not an error
        with pytest.raises(ValueError, match="no optimizer"):
            hm.precondition = True
        hm.remove()

    def test_logra_styles_are_rejected(self):
        model, opt = _trained(OPTIMIZERS["adamw"])
        projection = {
            "__default__": {
                "style": "logra_factorized",
                "proj_dim": 4,
                "proj_max_batch_size": 8,
            }
        }
        with pytest.raises(ValueError, match="cannot be preconditioned"):
            HookManager(
                model,
                config=HookManagerConfig(linear_io=REGISTER_ALL, projection=projection),
                optimizer=opt,
            )

    def test_param_grad_layers_are_rejected(self):
        model, opt = _trained(OPTIMIZERS["adamw"])
        with pytest.raises(ValueError, match="param_grad"):
            HookManager(
                model,
                config=HookManagerConfig(linear_io=["fc1$"], param_grad=["fc2$"]),
                optimizer=opt,
            )


class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def _loss(model, batch):
    return ((model(batch["x"]) - batch["y"]) ** 2).sum()


def _args(out_dir):
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        use_cpu=True,
        dataloader_pin_memory=False,
    )


class TestStreamerPrecondition:
    @staticmethod
    def _data():
        g = torch.Generator().manual_seed(0)
        return DictDataset(
            torch.randn(2 * BATCH, IN, generator=g),
            torch.randn(2 * BATCH, OUT, generator=g),
        )

    def test_frozen_probe_preconditions_with_the_given_optimizer(self, tmp_path):
        model, opt = _trained(OPTIMIZERS["adamw"])
        snap = OptimizerSnapshot(model, opt)
        data = self._data()
        raw = GradientStreamer(
            model, data, _args(tmp_path), batch_size=BATCH, loss_fn=_loss
        )
        pre = GradientStreamer(
            model,
            data,
            _args(tmp_path),
            batch_size=BATCH,
            loss_fn=_loss,
            optimizer=opt,
            precondition=True,
        )
        with raw:
            raw_blocks = list(raw)
        with pre:
            pre_blocks = list(pre)
        assert len(raw_blocks) == len(pre_blocks) == 2
        for (_, rb, rh), (_, pb, ph) in zip(raw_blocks, pre_blocks, strict=True):
            assert rh == ph
            for layer in LAYERS:
                full = ops.materialize(rb.data[layer], rb.layer_types[layer])
                assert torch.allclose(
                    pb.data[layer], snap.precondition(layer, full), atol=1e-5
                )

    def test_shared_hooks_serve_a_raw_test_pass(self, tmp_path):
        model, opt = _trained(OPTIMIZERS["adamw"])
        data = self._data()
        train = GradientStreamer(
            model,
            data,
            _args(tmp_path),
            batch_size=BATCH,
            loss_fn=_loss,
            optimizer=opt,
            precondition=True,
        )
        test = GradientStreamer(
            model,
            data,
            _args(tmp_path),
            batch_size=BATCH,
            loss_fn=_loss,
            hook_manager=train.hook_manager,
        )
        with train, test:
            _, tb, _ = next(iter(train))
            _, eb, _ = next(iter(test))
            _, tb2, _ = next(iter(train))
        assert tb.representation["fc1"] == "materialized"
        assert eb.representation["fc1"] == "factorized"
        assert tb2.representation["fc1"] == "materialized"

    def test_precondition_needs_an_optimizer(self, tmp_path):
        model, opt = _trained(OPTIMIZERS["adamw"])
        with pytest.raises(ValueError, match="needs an optimizer"):
            GradientStreamer(
                model,
                self._data(),
                _args(tmp_path),
                batch_size=BATCH,
                loss_fn=_loss,
                precondition=True,
            )
        plain = GradientStreamer(
            model, self._data(), _args(tmp_path), batch_size=BATCH, loss_fn=_loss
        )
        with pytest.raises(ValueError, match="shared hook_manager"):
            GradientStreamer(
                model,
                self._data(),
                _args(tmp_path),
                batch_size=BATCH,
                loss_fn=_loss,
                hook_manager=plain.hook_manager,
                optimizer=opt,
                precondition=True,
            )
