"""Step completion counts *observed* forward fires, not predicted replicas.

Regression under test: the manager used to fix the per-layer backward-hook
target to ``len(device_ids)`` at construction (``_n_replicas``).  Two ways
that prediction went wrong, in opposite directions:

* A trailing batch smaller than the device count makes ``DataParallel``'s
  scatter use fewer replicas -- fewer hook fires than predicted, so the step
  never completed (and its leftover counts poisoned the next batch).
* Wrapping in ``DataParallel`` *after* constructing the manager (the
  documented order for FSDP) left the prediction at 1 -- the step completed
  prematurely on the first replica's fires.

The fix derives the target per step and per layer from the forward hooks
actually observed (``_fwd_fires``): each grad-enabled forward produces
exactly one backward, and the forward pass finishes before any backward hook
runs, so the target is final by the time it is compared.

CPU emulation: ``ChunkedWrapper`` runs its submodule once per fixed-size
chunk of the batch -- per-layer hook counts are exactly those of a
``DataParallel`` scatter with that chunk size (N fires for a full batch,
fewer for a trailing short batch), without needing GPUs.  Unlike real
DataParallel (one fire per *device*, merged along the batch axis), these are
same-device multi-invocations, so each chunk is recorded as its own virtual
layer (``inner.X``, ``inner.X@2``, ...) with exact per-call (a, g) pairing.
A real ``nn.DataParallel`` test runs when 2+ CUDA devices are present.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks import HookManagerCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

IN_DIM, HID, OUT_DIM = 4, 8, 2
CHUNK = 2


def _inner() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN_DIM, HID), nn.ReLU(), nn.Linear(HID, OUT_DIM))


class ChunkedWrapper(nn.Module):
    """Run the submodule once per CHUNK-sized slice, like DataParallel scatter.

    Each hooked layer inside ``inner`` fires its forward/backward hooks once
    per chunk -- twice for a batch of 4, once for a trailing batch of 2.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = [self.inner(c) for c in x.split(CHUNK, dim=0)]
        return torch.cat(chunks, dim=0)


class Recorder(HookManagerCallback):
    def __init__(self) -> None:
        self.records = []

    def on_step_end(self, record) -> None:
        self.records.append(record)


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


class TestObservedFireCounting:
    def test_multi_fire_step_completes_once_with_full_batch(self):
        """Two chunks -> two fires per layer -> exactly one record: each chunk
        becomes a virtual layer whose (a, g) pair exactly matches the
        corresponding slice of the unchunked reference run.
        """
        batch = torch.randn(4, IN_DIM, generator=torch.Generator().manual_seed(1))

        rec_ref = _collect(_inner(), [batch])
        rec_dp = _collect(ChunkedWrapper(_inner()), [batch])

        assert len(rec_ref.records) == 1
        assert len(rec_dp.records) == 1
        g_ref, g_dp = rec_ref.records[0].gradient, rec_dp.records[0].gradient
        for name in g_ref.layer_names:
            ref = g_ref.data[name]
            for k, sl in enumerate((slice(0, CHUNK), slice(CHUNK, None))):
                vname = f"inner.{name}" if k == 0 else f"inner.{name}@{k + 1}"
                b = g_dp.data[vname]
                assert torch.allclose(ref.activation[sl], b.activation, atol=1e-6)
                assert torch.allclose(
                    ref.pre_activation_grad[sl],
                    b.pre_activation_grad,
                    atol=1e-6,
                )

    def test_trailing_short_batch_completes(self):
        """A final batch smaller than the chunk count still completes its step
        (the original hang), and the following state is clean: per-step layer
        sets reflect how many chunks actually ran.
        """
        gen = torch.Generator().manual_seed(2)
        batches = [
            torch.randn(4, IN_DIM, generator=gen),  # 2 chunks
            torch.randn(2, IN_DIM, generator=gen),  # 1 chunk -- "short" batch
            torch.randn(4, IN_DIM, generator=gen),  # 2 chunks again
        ]
        rec = _collect(ChunkedWrapper(_inner()), batches)

        assert [r.step for r in rec.records] == [0, 1, 2]
        two_chunk = {"inner.0", "inner.0@2", "inner.2", "inner.2@2"}
        one_chunk = {"inner.0", "inner.2"}
        assert [r.gradient.layer_names for r in rec.records] == [
            two_chunk,
            one_chunk,
            two_chunk,
        ]
        assert [r.gradient.batch_size for r in rec.records] == [CHUNK] * 3


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="real nn.DataParallel needs 2+ CUDA devices",
)
class TestRealDataParallel:
    def test_short_final_batch(self):
        """DataParallel over 2 GPUs with a final 1-sample batch: scatter uses
        a single replica, and the step must still complete.
        """
        inner = _inner().cuda()
        dp = nn.DataParallel(inner, device_ids=[0, 1])
        gen = torch.Generator().manual_seed(3)
        batches = [
            torch.randn(4, IN_DIM, generator=gen).cuda(),
            torch.randn(1, IN_DIM, generator=gen).cuda(),  # 1 < n_devices
        ]
        rec = _collect(dp, batches)
        assert [r.step for r in rec.records] == [0, 1]
        assert [r.gradient.batch_size for r in rec.records] == [4, 1]


class UnusedLayerModel(nn.Module):
    """A hooked-eligible layer that the forward never touches."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.used = nn.Linear(IN_DIM, OUT_DIM)
        self.unused = nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


class ConditionalModel(nn.Module):
    """Routes through branch_a or branch_b depending on a flag."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.stem = nn.Linear(IN_DIM, IN_DIM)
        self.branch_a = nn.Linear(IN_DIM, OUT_DIM)
        self.branch_b = nn.Linear(IN_DIM, OUT_DIM)
        self.use_a = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        return self.branch_a(h) if self.use_a else self.branch_b(h)


class TestUnusedRegisteredLayers:
    """Layers registered but never fired must not stall step completion.

    Regression: completion used to require *every* registered layer in
    ``_seen_bwd``, so a hooked layer off the execution path stalled the step
    forever (and its leftover state poisoned the following batches).  With
    observed-fire counting, only layers that participated in the step's
    forward are required, and they alone appear in the record.
    """

    def test_unused_layer_does_not_stall_steps(self):
        model = UnusedLayerModel()
        batches = [
            torch.randn(3, IN_DIM, generator=torch.Generator().manual_seed(i))
            for i in range(2)
        ]
        rec = _collect(model, batches)
        assert [r.step for r in rec.records] == [0, 1]
        for r in rec.records:
            assert set(r.gradient.layer_names) == {"used"}
            assert r.gradient.batch_size == 3

    def test_conditional_path_participation_is_per_step(self):
        model = ConditionalModel()
        rec = Recorder()
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[rec],
        )
        gen = torch.Generator().manual_seed(5)
        try:
            with hm.collect():
                for use_a in (True, False, True):
                    model.use_a = use_a
                    model.zero_grad()
                    x = torch.randn(3, IN_DIM, generator=gen)
                    model(x).pow(2).sum().backward()
        finally:
            hm.remove()

        assert [r.step for r in rec.records] == [0, 1, 2]
        expected = [
            {"stem", "branch_a"},
            {"stem", "branch_b"},
            {"stem", "branch_a"},
        ]
        for r, names in zip(rec.records, expected, strict=True):
            assert set(r.gradient.layer_names) == names
