"""``on_step_end`` runs outside the manager's lock and supports reentrancy.

Regression under test: callbacks used to be dispatched while
``HookManager._step_lock`` (a non-reentrant ``threading.Lock``) was held, so
the most natural custom callback -- "run a validation backward when the step
completes" -- deadlocked silently on same-thread lock re-acquisition.  The
contract now (see :meth:`HookManagerCallback.on_step_end`): dispatch happens
with the per-step state already reset and the lock released, a reentrant
backward completes a capture step of its own, and
``save_state``/``clear_state``/``load_state`` keeps the training-facing state
pristine around it.
"""

from __future__ import annotations

import torch
from torch import nn

from dattri_llm.gradient.callbacks import HookManagerCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

B, IN_DIM, OUT_DIM = 3, 4, 2
VAL_B = 2
N_STEPS = 3


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN_DIM, 8), nn.ReLU(), nn.Linear(8, OUT_DIM))


class LockSpy(HookManagerCallback):
    """Records whether the manager's lock is held during on_step_end."""

    def __init__(self, hm: HookManager) -> None:
        self._hm = hm
        self.lock_held: list[bool] = []

    def on_step_end(self, record) -> None:
        self.lock_held.append(self._hm._step_lock.locked())


class ReentrantValCallback(HookManagerCallback):
    """The natural user callback: a val backward at step end, per contract."""

    def __init__(self, model: nn.Module, val_x: torch.Tensor) -> None:
        self._model = model
        self._val_x = val_x
        self._hm: HookManager | None = None
        self._in_val_pass = False
        self.val_gradients = []
        self.train_records = []

    def on_register(self, hook_manager: HookManager) -> None:
        self._hm = hook_manager

    def on_step_end(self, record) -> None:
        if self._in_val_pass:
            # Our own secondary step: consume it, don't recurse.
            self.val_gradients.append(record.gradient)
            return
        self.train_records.append(record)
        state = self._hm.save_state()
        self._hm.clear_state()
        self._in_val_pass = True
        try:
            with torch.enable_grad():  # hooks run with grad mode disabled
                self._model.zero_grad()
                self._model(self._val_x).pow(2).sum().backward()
        finally:
            self._in_val_pass = False
            self._hm.load_state(state)


class TestDispatchOutsideLock:
    def test_lock_is_released_during_on_step_end(self):
        model = _model()
        hm = HookManager(model, config=HookManagerConfig(linear_io=REGISTER_ALL))
        spy = LockSpy(hm)
        hm.add_callback(spy)
        try:
            with hm.collect():
                for _ in range(N_STEPS):
                    model.zero_grad()
                    model(torch.randn(B, IN_DIM)).pow(2).sum().backward()
        finally:
            hm.remove()
        assert spy.lock_held == [False] * N_STEPS


class TestReentrantBackward:
    def test_secondary_backward_completes_and_state_is_restored(self):
        """The previously-deadlocking scenario, done per contract: training
        steps keep consecutive numbering, each step yields exactly one val
        gradient, and the training records are byte-identical to a run without
        the reentrant callback.
        """
        # Control run: no reentrant callback.
        model = _model()
        control_records = []

        class Recorder(HookManagerCallback):
            def on_step_end(self, record) -> None:
                control_records.append(record)

        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[Recorder()],
        )
        batches = [
            torch.randn(B, IN_DIM, generator=torch.Generator().manual_seed(10 + i))
            for i in range(N_STEPS)
        ]
        try:
            with hm.collect():
                for x in batches:
                    model.zero_grad()
                    model(x).pow(2).sum().backward()
        finally:
            hm.remove()

        # Reentrant run over the same model/batches.
        model2 = _model()
        val_x = torch.randn(VAL_B, IN_DIM, generator=torch.Generator().manual_seed(99))
        cb = ReentrantValCallback(model2, val_x)
        hm2 = HookManager(
            model2,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
        )
        try:
            with hm2.collect():
                for x in batches:
                    model2.zero_grad()
                    model2(x).pow(2).sum().backward()
            steps_after = hm2.steps_collected
        finally:
            hm2.remove()

        # One val gradient per training step, with the val batch size.
        assert len(cb.val_gradients) == N_STEPS
        assert all(g.batch_size == VAL_B for g in cb.val_gradients)

        # Training records unaffected: consecutive steps, identical payloads.
        assert [r.step for r in cb.train_records] == list(range(N_STEPS))
        assert steps_after == N_STEPS  # load_state rolled back the val steps
        for r_ctrl, r_seen in zip(control_records, cb.train_records, strict=True):
            assert r_ctrl.step == r_seen.step
            assert r_ctrl.input_hash == r_seen.input_hash
            for name in r_ctrl.gradient.layer_names:
                a = r_ctrl.gradient.data[name]
                b = r_seen.gradient.data[name]
                assert torch.equal(a.activation, b.activation)
                assert torch.equal(a.pre_activation_grad, b.pre_activation_grad)
