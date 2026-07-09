"""Callback isolation tests: a callback must not perturb the HookManager.

The contract under test: attaching any callback may act on *its own* concerns
(persisting records, editing ``param.grad``), but must leave

* the **manager's internal state** -- step counter, completion flags, capture
  buffers, captured model inputs, last-gradient cache -- exactly as a run
  without the callback leaves it, step for step; and
* the **workflow of sibling callbacks** -- every other callback sees the same
  records (same steps, hashes, and gradient tensors), unmutated, in the same
  order.

The one documented exception is ``DataSelectionCallback(target="val_loader")``:
its secondary val backward completes a capture step of its own, so siblings
observe exactly one extra (val) record per training step.  That exception is
pinned precisely rather than ignored.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks import (
    CaptureCallback,
    DataSelectionCallback,
    HookManagerCallback,
    OffloadCallback,
)
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Factorized, Gradient
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.utils.hashing import hash_batch

SEED = 0
VOCAB = 24
EMBED = 6
HIDDEN = 12
OUT = 3
SEQ = 4
BATCH = 4
N_STEPS = 3


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, EMBED)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUT),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(token_ids))


def _batches(n: int = N_STEPS) -> list[torch.Tensor]:
    return [
        torch.randint(
            0,
            VOCAB,
            (BATCH, SEQ),
            generator=torch.Generator().manual_seed(100 + i),
        )
        for i in range(n)
    ]


VAL_BATCH = torch.randint(
    0,
    VOCAB,
    (2, SEQ),
    generator=torch.Generator().manual_seed(999),
)


def _val_loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch).sum()


class SpyCallback(HookManagerCallback):
    """Records every dispatched event; optionally deep-copies the records so a
    later in-place mutation by a sibling callback is detectable.
    """

    def __init__(self, deep: bool = False) -> None:
        self._deep = deep
        self.records = []
        self.layer_forward_counts: dict[str, int] = {}
        self.layer_backward_counts: dict[str, int] = {}

    def on_layer_forward(self, layer_name, activation):
        self.layer_forward_counts[layer_name] = (
            self.layer_forward_counts.get(layer_name, 0) + 1
        )

    def on_layer_backward(self, layer_name, grad_output):
        self.layer_backward_counts[layer_name] = (
            self.layer_backward_counts.get(layer_name, 0) + 1
        )

    def on_step_end(self, record):
        self.records.append(copy.deepcopy(record) if self._deep else record)


def _manager_fingerprint(hm: HookManager) -> dict:
    """The manager-internal state a callback must leave untouched."""
    return {
        "steps": hm.steps_collected,
        "bwd_done": hm._bwd_done,
        "grad_done": hm._grad_done,
        "mlp_param_hook_count": hm._mlp_param_hook_count,
        "param_hook_count": hm._param_hook_count,
        "seen_bwd": set(hm._seen_bwd),
        "replica_counts": dict(hm._bwd_replica_counts),
        "mlp_params_ready": hm._mlp_params_ready,
        "backward_end_scheduled": hm._backward_end_scheduled,
        "buffers_empty": all(
            not b["_act_parts"]
            and not b["_grad_parts"]
            and not b["_proj_parts"]
            and not any(b["_device_id"].values())
            for b in hm._buffers.values()
        ),
        "last_input_hashes": (
            tuple(hash_batch(hm._last_inputs)) if hm._last_inputs else None
        ),
    }


def _gradients_equal(g1: Gradient, g2: Gradient) -> bool:
    if g1.layer_names != g2.layer_names:
        return False
    for name in g1.layer_names:
        v1, v2 = g1.data[name], g2.data[name]
        if isinstance(v1, Factorized) != isinstance(v2, Factorized):
            return False
        if isinstance(v1, Factorized):
            if not torch.equal(v1.activation, v2.activation):
                return False
            if not torch.equal(v1.pre_activation_grad, v2.pre_activation_grad):
                return False
        elif not torch.equal(v1, v2):
            return False
    return True


def _records_equal(r1, r2) -> bool:
    return (
        r1.step == r2.step
        and r1.input_hash == r2.input_hash
        and _gradients_equal(r1.gradient, r2.gradient)
    )


def _run(callback_under_test=None, *, model_holder=None):
    """Run N deterministic steps; return per-step fingerprints + spy views.

    ``callback_under_test`` is a factory ``(model) -> callback`` placed
    *between* two spies, so both the before- and after-sibling views are
    captured.  The first spy deep-copies records, so any in-place mutation by
    the callback under test shows up as a mismatch with the second spy.
    """
    torch.manual_seed(SEED)
    model = TinyModel()
    if model_holder is not None:
        model_holder.append(model)
    spy_before = SpyCallback(deep=True)
    spy_after = SpyCallback(deep=False)
    callbacks = [spy_before]
    if callback_under_test is not None:
        callbacks.append(callback_under_test(model))
    callbacks.append(spy_after)

    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=callbacks,
    )
    fingerprints = []
    with hm.collect():
        for x in _batches():
            model.zero_grad()
            model(x).sum().backward()
            fingerprints.append(_manager_fingerprint(hm))
    last_gradient = hm.get_gradient()
    hm.remove()
    return fingerprints, spy_before, spy_after, last_gradient


def _fixed_target() -> Gradient:
    """A target gradient captured on an independent, identically-seeded model."""
    torch.manual_seed(SEED)
    model = TinyModel()
    cap = CaptureCallback()
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[cap],
    )
    with hm.collect():
        model(VAL_BATCH).sum().backward()
    hm.remove()
    return cap.record.gradient


def _callback_factory(kind: str, tmp_path):
    if kind == "capture":
        return lambda _model: CaptureCallback()
    if kind == "offload":
        fm = GradientFileManager(str(tmp_path / "grads"))
        return lambda _model: OffloadCallback(offload_interval=2, file_manager=fm)
    if kind == "ds_batch":
        return lambda model: DataSelectionCallback(
            model=model,
            threshold_mode="bottom_fraction",
            threshold=0.5,
            target="batch",
        )
    if kind == "ds_fixed":
        target = _fixed_target()
        return lambda model: DataSelectionCallback(
            model=model,
            threshold_mode="bottom_fraction",
            threshold=0.5,
            target="fixed",
            target_gradient=target,
        )
    raise ValueError(kind)


class TestCallbackNeutrality:
    """Any callback: manager state and sibling views match a control run."""

    @pytest.mark.parametrize("kind", ["capture", "offload", "ds_batch", "ds_fixed"])
    def test_manager_state_and_sibling_records_match_control(self, kind, tmp_path):
        ctrl_fps, ctrl_spy, _, ctrl_last = _run(None)
        fps, spy_before, spy_after, last = _run(_callback_factory(kind, tmp_path))

        # Manager internals identical to the control run after every step.
        assert fps == ctrl_fps

        # Sibling callbacks saw exactly the control run's records.
        assert len(spy_before.records) == len(ctrl_spy.records) == N_STEPS
        for r_ctrl, r_seen in zip(ctrl_spy.records, spy_before.records, strict=True):
            assert _records_equal(r_ctrl, r_seen)

        # The callback under test did not mutate records in flight: the
        # deep-copied before-view equals the by-reference after-view.
        for r_b, r_a in zip(spy_before.records, spy_after.records, strict=True):
            assert _records_equal(r_b, r_a)

        # Per-layer event dispatch is unchanged.
        assert spy_before.layer_forward_counts == ctrl_spy.layer_forward_counts
        assert spy_before.layer_backward_counts == ctrl_spy.layer_backward_counts

        # The manager's last-gradient cache is unchanged.
        assert _gradients_equal(last, ctrl_last)


class TestValLoaderException:
    """val_loader mode: state still restored; the extra val record is exactly
    the documented exception, nothing more.
    """

    def test_state_restored_and_exception_is_bounded(self):
        ctrl_fps, ctrl_spy, _, ctrl_last = _run(None)

        def factory(model):
            return DataSelectionCallback(
                model=model,
                threshold_mode="bottom_fraction",
                threshold=0.5,
                target="val_loader",
                val_loader=[VAL_BATCH],
                val_loss_fn=_val_loss_fn,
            )

        fps, spy_before, spy_after, last = _run(factory)

        # Manager internals identical to control after every step: the val
        # pass's save_state/clear_state/load_state must be a perfect undo --
        # including the step counter and the captured training inputs.
        assert fps == ctrl_fps

        # Siblings see train records interleaved with exactly one val record
        # per step.  The val pass is triggered from the DS callback's
        # on_step_end, so the interleaving depends on dispatch position:
        # callbacks *before* it get the train record first ([train0, val0,
        # ...]); callbacks *after* it get the nested val record first
        # ([val0, train0, ...]).
        assert len(spy_before.records) == 2 * N_STEPS
        assert len(spy_after.records) == 2 * N_STEPS
        before_train = spy_before.records[0::2]
        before_val = spy_before.records[1::2]
        after_val = spy_after.records[0::2]
        after_train = spy_after.records[1::2]
        for r_ctrl, r_seen in zip(ctrl_spy.records, before_train, strict=True):
            assert _records_equal(r_ctrl, r_seen)
        for i, r in enumerate(before_val):
            assert r.gradient.batch_size == VAL_BATCH.shape[0]
            # The training step is already finalized (counter advanced) when
            # the val pass runs, so val record i carries step i + 1.
            assert r.step == i + 1

        # No in-flight mutation of any record (train or val), pairing each
        # deep-copied before-view with its after-view by category.
        for r_b, r_a in zip(before_train, after_train, strict=True):
            assert _records_equal(r_b, r_a)
        for r_b, r_a in zip(before_val, after_val, strict=True):
            assert _records_equal(r_b, r_a)

        # Layer events fire twice per step (train + val pass) -- the hooks are
        # live during the val pass by design.
        for name, n in ctrl_spy.layer_forward_counts.items():
            assert spy_before.layer_forward_counts[name] == 2 * n

        assert _gradients_equal(last, ctrl_last)
