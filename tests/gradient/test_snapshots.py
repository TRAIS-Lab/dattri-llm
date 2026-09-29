"""Trajectory snapshots and their replay: recomputed blocks equal stored ones."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.callbacks import (
    OffloadCallback,
    OptimizerStateCallback,
    ParameterSnapshotCallback,
)
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.snapshots import TrajectorySnapshots
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource, ReplayGradientSource

IN, HID, OUT = 4, 5, 3
N_STEPS, BATCH = 4, 3


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, OUT)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _loss(model, batch):
    return ((model(batch["x"]) - batch["y"]) ** 2).sum()


def _args(out_dir):
    return AttributionArguments(
        output_dir=str(out_dir), use_cpu=True, dataloader_pin_memory=False
    )


def _train_with_hooks(tmp_path, *, snapshots: bool):
    """A short AdamW run under hooks: gradients to a store, and (optionally)
    parameters, batches and moments to a snapshot store.
    """
    torch.manual_seed(0)
    model = MLP()
    g = torch.Generator().manual_seed(1)
    batches = [
        {
            "x": torch.randn(BATCH, IN, generator=g),
            "y": torch.randn(BATCH, OUT, generator=g),
        }
        for _ in range(N_STEPS)
    ]
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, betas=(0.8, 0.99))
    store = GradientStorageManager(str(tmp_path / "grads"))
    callbacks = [OffloadCallback(1, store)]
    snap = None
    if snapshots:
        snap = TrajectorySnapshots(tmp_path / "snaps")
        callbacks += [
            ParameterSnapshotCallback(model, snap),
            OptimizerStateCallback(model, opt, snapshots=snap),
        ]
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=["fc1$", "fc2$"]),
        callbacks=callbacks,
    )
    with hm.collect():
        for t, batch in enumerate(batches):
            if snap is not None:
                snap.save_batch(t, batch)
            opt.zero_grad(set_to_none=True)
            _loss(model, batch).backward()
            opt.step()
            if snap is not None:
                callbacks[2].record_post(t)
    hm.remove()
    return model, store, snap, batches


class TestTrajectorySnapshots:
    def test_roundtrip_and_layout(self, tmp_path):
        model, _store, snap, batches = _train_with_hooks(tmp_path, snapshots=True)
        assert TrajectorySnapshots.is_snapshot_dir(tmp_path / "snaps")
        assert not TrajectorySnapshots.is_snapshot_dir(tmp_path / "grads")
        assert snap.steps() == list(range(N_STEPS))
        assert snap.dynamics_steps() == list(range(N_STEPS))
        # The stored batch is the one consumed.
        loaded = snap.load_batch(2)
        assert torch.equal(loaded["x"], batches[2]["x"])
        # Loading a snapshot changes the model; the last step's parameters
        # are the ones before its update, not the trained ones.
        trained = {n: p.detach().clone() for n, p in model.named_parameters()}
        snap.load_parameters(N_STEPS - 1, model)
        assert not all(torch.equal(trained[n], p) for n, p in model.named_parameters())

    def test_reference_snapshots_resolve(self, tmp_path):
        model = MLP()
        snap = TrajectorySnapshots(tmp_path / "s")
        snap.save_parameters(0, model)
        snap.save_parameters(1, model, same_as=0)
        snap.save_batch(0, {"x": torch.zeros(1)})
        snap.save_batch(1, {"x": torch.zeros(1)})
        assert snap.steps() == [0, 1]
        other = MLP()
        snap.load_parameters(1, other)
        for (n, p), (_m, q) in zip(
            model.named_parameters(), other.named_parameters(), strict=True
        ):
            assert torch.equal(p, q), n


class TestReplayGradientSource:
    def test_replay_matches_stored_blocks(self, tmp_path):
        model, store, snap, _b = _train_with_hooks(tmp_path, snapshots=True)
        args = _args(tmp_path)
        stored = {
            step: (block.materialize(), hashes)
            for step, block, hashes in DiskGradientSource(store, args)
        }
        replay = ReplayGradientSource(
            model,
            args,
            snap,
            loss_fn=_loss,
            config=HookManagerConfig(linear_io=["fc1$", "fc2$"]),
        )
        with replay:
            assert replay.reusable
            assert replay.steps == list(range(N_STEPS))
            seen = []
            for step, block, hashes in replay:
                seen.append(step)
                want, want_hashes = stored[step]
                assert hashes == want_hashes
                got = block.materialize()
                for name in want.data:
                    assert torch.allclose(got.data[name], want.data[name], atol=1e-6)
            assert seen == list(range(N_STEPS))
            # Random access, in the order requested (the sweeps go backwards).
            desc = [s for s, _g, _h in replay.for_steps([2, 0])]
            assert desc == [2, 0]

    def test_close_restores_parameters_and_hooks(self, tmp_path):
        model, _store, snap, _b = _train_with_hooks(tmp_path, snapshots=True)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        replay = ReplayGradientSource(
            model,
            _args(tmp_path),
            snap,
            loss_fn=_loss,
            config=HookManagerConfig(linear_io=["fc1$", "fc2$"]),
        )
        list(replay.for_steps([0]))
        assert not all(torch.equal(before[n], p) for n, p in model.named_parameters())
        replay.close()
        assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
        assert all(p.grad is None for p in model.parameters())
        # A closed source re-arms itself on the next pass.
        assert [s for s, _g, _h in replay.for_steps([1])] == [1]
        replay.close()

    def test_layer_restriction_and_missing_steps(self, tmp_path):
        model, _store, snap, _b = _train_with_hooks(tmp_path, snapshots=True)
        replay = ReplayGradientSource(
            model,
            _args(tmp_path),
            snap,
            loss_fn=_loss,
            config=HookManagerConfig(linear_io=["fc1$", "fc2$"]),
            layer_name="fc2",
        )
        with replay:
            _s, block, _h = next(iter(replay))
            assert set(block.data) == {"fc2"}
        with pytest.raises(ValueError, match="none of the requested steps"):
            ReplayGradientSource(model, _args(tmp_path), snap, steps=[99])

    def test_moments_on_disk_match_in_memory(self, tmp_path):
        torch.manual_seed(0)
        model = MLP()
        opt = torch.optim.AdamW(model.parameters(), lr=0.05)
        snap = TrajectorySnapshots(tmp_path / "s")
        on_disk = OptimizerStateCallback(model, opt, snapshots=snap)
        in_memory = OptimizerStateCallback(model, opt)
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=["fc1$", "fc2$"]),
            callbacks=[on_disk, in_memory],
        )
        with hm.collect():
            for t in range(2):
                opt.zero_grad(set_to_none=True)
                _loss(
                    model, {"x": torch.randn(2, IN), "y": torch.randn(2, OUT)}
                ).backward()
                opt.step()
                on_disk.record_post(t)
                in_memory.record_post(t)
        hm.remove()
        lazy, ram = on_disk.dynamics(), in_memory.dynamics()
        assert list(lazy) == list(ram)
        for t in ram:
            for side in ("pre", "post"):
                for name in ram[t][side]:
                    for a, b in zip(
                        lazy[t][side][name], ram[t][side][name], strict=True
                    ):
                        assert torch.equal(a, b)
            assert lazy[t]["step"] == ram[t]["step"]
