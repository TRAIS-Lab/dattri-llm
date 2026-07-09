"""Multi-invocation capture: gradient checkpointing and custom module reuse.

The capture pipeline bracket-matches forward and backward fires per layer and
per device: forwards push onto a LIFO stack, each backward pops its own
forward (autograd runs a device's graph in reverse creation order), and the
end-of-backward engine callback discards forwards whose backward never came.

Regimes pinned here:

* **Gradient checkpointing, ``use_reentrant=False``** (PyTorch-recommended):
  the original forward *and* the recomputation fire grad-enabled, but only
  one backward comes.  The backward LIFO-matches the recomputed capture (the
  tensors the graph actually used) and the stale original is discarded at
  backward end -- records come out identical to an uncheckpointed run.
* **Gradient checkpointing, ``use_reentrant=True``** (legacy): the first pass
  runs under ``no_grad`` and is never captured; only the recomputation is.
  Already worked; pinned so it stays working.
* **Custom module reuse** (one layer invoked twice per step): each invocation
  is recorded as an independent virtual layer ``name`` / ``name@2`` with its
  own exactly-paired (a, g).
* **Detached / unused outputs**: a grad-enabled forward whose output never
  reaches the loss is discarded at backward end instead of stalling the step.
* **Orphan backwards** (e.g. a second ``backward(retain_graph=True)``): warn
  and discard.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import HookManagerCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

B, IN_DIM, HID, OUT_DIM = 3, 4, 8, 2
N_STEPS = 2


class Recorder(HookManagerCallback):
    def __init__(self) -> None:
        self.records = []

    def on_step_end(self, record) -> None:
        self.records.append(record)


def _segments() -> tuple[nn.Sequential, nn.Sequential]:
    torch.manual_seed(0)
    seg1 = nn.Sequential(nn.Linear(IN_DIM, HID), nn.ReLU(), nn.Linear(HID, HID))
    seg2 = nn.Sequential(nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, OUT_DIM))
    return seg1, seg2


class SegmentModel(nn.Module):
    """Two segments, optionally run through torch.utils.checkpoint."""

    def __init__(self, use_reentrant: bool | None) -> None:
        super().__init__()
        self.seg1, self.seg2 = _segments()
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_reentrant is None:
            return self.seg2(self.seg1(x))
        h = checkpoint(self.seg1, x, use_reentrant=self.use_reentrant)
        return checkpoint(self.seg2, h, use_reentrant=self.use_reentrant)


def _collect(model: nn.Module, batches: list[torch.Tensor]) -> Recorder:
    rec = Recorder()
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[rec],
    )
    try:
        with hm.collect():
            for x in batches:
                model.zero_grad()
                model(x).pow(2).sum().backward()
    finally:
        hm.remove()
    return rec


class TestGradientCheckpointing:
    @pytest.mark.parametrize("use_reentrant", [False, True])
    def test_records_match_uncheckpointed_run(self, use_reentrant):
        gen = torch.Generator().manual_seed(1)
        # requires_grad input: the reentrant variant needs at least one input
        # requiring grad for gradients to flow at all.
        batches = [
            torch.randn(B, IN_DIM, generator=gen).requires_grad_()
            for _ in range(N_STEPS)
        ]

        rec_ref = _collect(SegmentModel(None), batches)
        model_ckpt = SegmentModel(use_reentrant)
        rec_ckpt = _collect(model_ckpt, batches)

        assert [r.step for r in rec_ckpt.records] == list(range(N_STEPS))
        for r_ref, r_seen in zip(rec_ref.records, rec_ckpt.records, strict=True):
            # No virtual layers: checkpointing leaves one matched pair per
            # layer, so names and payloads are identical to the plain run.
            assert r_seen.gradient.layer_names == r_ref.gradient.layer_names
            for name in r_ref.gradient.layer_names:
                a = r_ref.gradient.data[name]
                b = r_seen.gradient.data[name]
                assert torch.allclose(a.activation, b.activation, atol=1e-6)
                assert torch.allclose(
                    a.pre_activation_grad,
                    b.pre_activation_grad,
                    atol=1e-6,
                )

    def test_param_grads_unaffected(self):
        """The capture machinery must not perturb training under checkpointing."""
        gen = torch.Generator().manual_seed(2)
        x = torch.randn(B, IN_DIM, generator=gen).requires_grad_()

        ref = SegmentModel(None)
        ref(x).pow(2).sum().backward()

        model = SegmentModel(use_reentrant=False)
        _collect(model, [x])

        for (n1, p1), (n2, p2) in zip(
            ref.named_parameters(),
            model.named_parameters(),
            strict=True,
        ):
            assert n1 == n2
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6), n1


class SharedLinear(nn.Module):
    """One Linear applied to two different inputs in a single forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(IN_DIM, IN_DIM)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.lin(x1) + self.lin(x2)


class TwinLinear(nn.Module):
    """Reference: two independent Linears initialised identically."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.lin_a = nn.Linear(IN_DIM, IN_DIM)
        self.lin_b = nn.Linear(IN_DIM, IN_DIM)
        for lin in (self.lin_a, self.lin_b):
            with torch.no_grad():
                lin.weight.copy_(weight)
                lin.bias.copy_(bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.lin_a(x1) + self.lin_b(x2)


class TestModuleReuse:
    def test_invocations_recorded_as_virtual_layers(self):
        """Lin / lin@2 must equal an untied twin model's lin_a / lin_b."""
        gen = torch.Generator().manual_seed(3)
        x1 = torch.randn(B, IN_DIM, generator=gen)
        x2 = torch.randn(B, IN_DIM, generator=gen)

        shared = SharedLinear()
        twins = TwinLinear(shared.lin.weight.detach(), shared.lin.bias.detach())

        def run(model):
            rec = Recorder()
            hm = HookManager(
                model,
                config=HookManagerConfig(linear_io=REGISTER_ALL),
                callbacks=[rec],
            )
            try:
                with hm.collect():
                    model(x1, x2).pow(2).sum().backward()
            finally:
                hm.remove()
            return rec.records[0].gradient

        # invocation 1 <-> lin_a (x1), invocation 2 <-> lin_b (x2)
        g_shared = run(shared)
        g_twins = run(twins)
        assert g_shared.layer_names == {"lin", "lin@2"}
        for vname, tname in (("lin", "lin_a"), ("lin@2", "lin_b")):
            a = g_shared.data[vname]
            b = g_twins.data[tname]
            assert torch.equal(a.activation, b.activation), vname
            assert torch.allclose(
                a.pre_activation_grad,
                b.pre_activation_grad,
                atol=1e-6,
            ), vname

    def test_virtual_layers_sum_to_true_weight_grad(self):
        gen = torch.Generator().manual_seed(4)
        x1 = torch.randn(B, IN_DIM, generator=gen)
        x2 = torch.randn(B, IN_DIM, generator=gen)
        model = SharedLinear()
        rec = Recorder()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[rec],
        )
        try:
            with hm.collect():
                model.zero_grad()
                model(x1, x2).sum().backward()
        finally:
            hm.remove()

        g = rec.records[0].gradient
        total = sum(
            ops.materialize(g.data[n], "nn.Linear", include_bias=False).sum(0)
            for n in ("lin", "lin@2")
        )
        assert torch.allclose(
            total.reshape_as(model.lin.weight.grad),
            model.lin.weight.grad,
            atol=1e-5,
        )


class DetachedBranchModel(nn.Module):
    """A layer whose grad-enabled output never reaches the loss."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.used = nn.Linear(IN_DIM, OUT_DIM)
        self.dangling = nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.dangling(x)  # computed, forward hook fires, never used
        return self.used(x)


class TestUnbackwardedForwards:
    def test_detached_branch_does_not_stall(self):
        gen = torch.Generator().manual_seed(5)
        batches = [torch.randn(B, IN_DIM, generator=gen) for _ in range(2)]
        rec = _collect(DetachedBranchModel(), batches)
        assert [r.step for r in rec.records] == [0, 1]
        for r in rec.records:
            assert r.gradient.layer_names == {"used"}

    def test_orphan_backward_warns_and_is_discarded(self):
        torch.manual_seed(6)
        model = nn.Sequential(nn.Linear(IN_DIM, OUT_DIM))
        rec = Recorder()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[rec],
        )
        try:
            with hm.collect():
                out = model(torch.randn(B, IN_DIM)).pow(2).sum()
                out.backward(retain_graph=True)  # completes step 0
                with pytest.warns(UserWarning, match="no matching forward"):
                    out.backward()  # orphan: step 0's captures were consumed
            steps = hm.steps_collected
        finally:
            hm.remove()
        assert steps == 1
        assert len(rec.records) == 1
