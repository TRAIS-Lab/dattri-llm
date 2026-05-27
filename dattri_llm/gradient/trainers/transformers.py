"""HuggingFace Transformers Trainer integration.

Provides :class:`AttributionTrainerCallback`, a ``transformers.TrainerCallback``
that wires a :class:`~dattri_llm.gradient.hooks.HookManager` (with any number
of attached :class:`~dattri_llm.gradient.callbacks.HookManagerCallback` objects)
into the HuggingFace :class:`transformers.Trainer` training loop.

Usage
-----
::

    from transformers import Trainer, TrainingArguments
    from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
    from dattri_llm.gradient.callbacks import DataSelectionCallback
    from dattri_llm.gradient.trainers import AttributionTrainerCallback

    hook_cfg = HookManagerConfig(hook_types={"mlp_io": None})

    selection_cb = DataSelectionCallback(
        model=model,
        threshold=0.25,
        threshold_mode="bottom_fraction",
        target="batch",
        score_mode="ghost",
    )
    hook_manager = HookManager(model, config=hook_cfg, callbacks=[selection_cb])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[AttributionTrainerCallback(hook_manager)],
    )
    trainer.train()

How it works
------------
Our PyTorch forward and backward hooks are registered directly on
``nn.Module`` sub-layers.  They fire during **any** forward/backward
pass — including the one performed inside ``Trainer.training_step()``.
No Trainer method needs to be overridden.

:class:`AttributionTrainerCallback` only needs to:

1. Enter the :py:meth:`~dattri_llm.gradient.hooks.HookManager.collect`
   context manager when training begins, so the HookManager starts
   counting layer passes and assembling :class:`GradientRecord` objects.
2. Exit the context manager (and optionally remove all hooks) when
   training ends or is interrupted.

``DataSelectionCallback.on_step_end`` is triggered by a
``register_post_accumulate_grad_hook`` registered on model parameters.
That hook fires inside ``loss.backward()`` — which happens inside
``Trainer.training_step()`` *before* ``optimizer.step()`` — so gradient
modifications reach the optimizer correctly.

Gradient-accumulation note
--------------------------
When ``TrainingArguments.gradient_accumulation_steps > 1``, the Trainer
calls ``training_step`` multiple times before calling ``optimizer.step``.
Our parameter hook fires once per ``backward()`` call; the HookManager
fires ``on_step_end`` (and hence data-selection) at the end of *each*
micro-step, not only at the true optimizer step.  This is consistent
with the plain-loop usage and is correct for micro-batch selection.
If you want selection only at full steps, pass
``gradient_accumulation_steps=1`` or subclass and override the logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dattri_llm.gradient.hooks import HookManager


class AttributionTrainerCallback:
    """HuggingFace ``TrainerCallback`` that activates a :class:`HookManager`.

    This class follows the ``transformers.TrainerCallback`` interface without
    inheriting from it directly (to avoid making ``transformers`` a hard
    dependency of the core library).  If ``transformers`` is installed the
    class is automatically recognised as a valid callback by ``Trainer``
    because Trainer checks for the callback methods by duck-typing.

    Args:
        hook_manager: A fully-configured :class:`HookManager` with all
            desired callbacks already attached.  The caller is responsible
            for constructing the manager and its callbacks *before* passing
            it here.
        remove_hooks_on_end: If ``True`` (default), call
            :py:meth:`~dattri_llm.gradient.hooks.HookManager.remove` at
            the end of training to deregister all PyTorch hooks from the
            model.  Set to ``False`` if you intend to reuse the manager
            after training.
    """

    def __init__(
        self,
        hook_manager: "HookManager",
        remove_hooks_on_end: bool = True,
    ) -> None:
        self._hm = hook_manager
        self._remove_on_end = remove_hooks_on_end
        self._ctx: Optional[object] = None  # the context manager returned by .collect()

    # ------------------------------------------------------------------ #
    # TrainerCallback interface                                            #
    # ------------------------------------------------------------------ #

    def on_train_begin(self, args, state, control, **kwargs):
        """Enter the HookManager collect context at the start of training."""
        if self._ctx is None:
            self._ctx = self._hm.collect()
            self._ctx.__enter__()

    def on_train_end(self, args, state, control, **kwargs):
        """Exit the collect context and optionally remove all hooks."""
        self._teardown()

    def on_init_end(self, args, state, control, **kwargs):
        """No-op — required by some Trainer versions."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        """No-op."""

    def on_epoch_end(self, args, state, control, **kwargs):
        """No-op."""

    def on_step_begin(self, args, state, control, **kwargs):
        """No-op — hooks fire automatically during the forward pass."""

    def on_step_end(self, args, state, control, **kwargs):
        """No-op — data selection fires inside backward via param hook."""

    def on_substep_end(self, args, state, control, **kwargs):
        """No-op."""

    def on_evaluate(self, args, state, control, **kwargs):
        """No-op."""

    def on_save(self, args, state, control, **kwargs):
        """No-op."""

    def on_log(self, args, state, control, **kwargs):
        """No-op."""

    def on_prediction_step(self, args, state, control, **kwargs):
        """No-op."""

    def __getattr__(self, name: str):
        # Forward any callback method added in future transformers versions
        # (e.g. on_pre_optimizer_step) to a no-op so we stay compatible
        # without requiring an exact version pin.
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    # ------------------------------------------------------------------ #
    # Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Manually exit the collect context and remove hooks.

        Call this if you need to stop collection mid-training or if the
        Trainer did not call ``on_train_end`` (e.g. after an exception).
        """
        self._teardown()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _teardown(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
        if self._remove_on_end:
            self._hm.remove()
