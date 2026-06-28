"""Gradient collection callbacks.

Three built-in callbacks are provided:

``HookManagerCallback``
    Base class.  All methods are no-ops; subclass and override only what you
    need.

``OffloadCallback``
    Periodically flushes :class:`GradientRecord` objects to disk via
    :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.
    Supports both per-batch and per-sample recording granularity.

``DataSelectionCallback``
    Online data selection: computes per-sample influence scores at the end of
    each step, then removes low-influence samples' contributions from
    ``param.grad`` before ``optimizer.step()``.

    Two scoring modes are available via ``score_mode``:

    * ``"ghost"`` (default) — gram-matrix form, no weight-gradient
      materialisation. Cost O((B*T)^2 * (out + in)) per layer.
    * ``"materialized"`` — builds the explicit per-sample weight gradient
      and dots it against the batch gradient. Easier to verify but uses
      more memory.

    Both modes produce identical scores.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient import ops


# Per-parameter slice of an FSDP ``FlatParameter`` shard.  ``start``/``end`` are
# the inclusive indices into the *flattened, unsharded* parameter that this rank
# owns (i.e. ``param.grad`` equals ``full.reshape(-1)[start : end + 1]``);
# ``in_shard`` is False when this rank holds none of the parameter.
_ShardSpec = namedtuple("_ShardSpec", "orig_shape in_shard start end")


# --------------------------------------------------------------------------- #
# Base callback                                                                #
# --------------------------------------------------------------------------- #


class HookManagerCallback:
    """Base class for :class:`~dattri_llm.gradient.hooks.HookManager` callbacks.

    All methods are no-ops by default; subclasses override only what they need.
    ``on_layer_forward`` and ``on_layer_backward`` are wired directly into
    PyTorch's hook system and fire regardless of trainer callback support.
    """

    def on_layer_forward(self, layer_name: str, activation: torch.Tensor) -> None:
        """Called after each layer's forward hook captures the activation.

        Fires once per layer per forward pass.  ``activation`` is on CPU.

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            activation: Captured input activation, shape ``(B, *, in_features)``.
        """

    def on_layer_backward(self, layer_name: str, grad_output: torch.Tensor) -> None:
        """Called after each layer's backward hook captures the gradient.

        ``grad_output`` is on CPU.  Fires once per layer per backward pass
        (once per replica under DataParallel).

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            grad_output: Captured output gradient, shape ``(B, *, out_features)``.
        """

    def on_step_end(self, record: GradientRecord) -> None:
        """Called exactly once per batch step after the :class:`GradientRecord`
        is assembled.

        The record always contains the full-batch gradient (``input_hash`` is a
        list of B hashes).  Per-sample slicing is the callback's responsibility
        — see :class:`OffloadCallback` for an example.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
        """

    def on_context_end(self) -> None:
        """Called when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect`
        context closes.

        Use this to flush any remaining staged records to disk.
        """


# --------------------------------------------------------------------------- #
# CaptureCallback                                                              #
# --------------------------------------------------------------------------- #


class CaptureCallback(HookManagerCallback):
    """Holds the most recent per-step :class:`GradientRecord` in memory.

    The minimal callback: it persists nothing and simply stashes the latest
    step's record on :attr:`record`, so a caller can read it back immediately
    after the step (used by the live :class:`~dattri_llm.algorithm.streaming.\
GradientStreamer` to yield one block at a time).  Reset to ``None`` before each
    step by the consumer if a missed capture should be detectable.
    """

    def __init__(self) -> None:
        self.record: Optional[GradientRecord] = None

    def on_step_end(self, record: GradientRecord) -> None:
        self.record = record


# --------------------------------------------------------------------------- #
# OffloadCallback                                                              #
# --------------------------------------------------------------------------- #


class OffloadCallback(HookManagerCallback):
    """Periodically saves :class:`GradientRecord` objects to disk.

    Accumulates records in memory and writes them to a single batch file every
    ``offload_interval`` *batch steps*.  Any remaining staged records are written
    when the :meth:`~dattri_llm.gradient.hooks.HookManager.collect` context
    closes.

    Args:
        offload_interval: Number of batch steps to accumulate before writing a
            batch file.  Set to ``1`` for one file per step.
        file_manager: The :class:`GradientFileManager` to delegate saves to.
        recording_type: ``"per_batch"`` (default) stores one
            :class:`GradientRecord` per step.  ``"per_sample"`` slices the
            full-batch record into B individual records before staging — useful
            when downstream code looks up gradients by a single-sample hash.
    """

    def __init__(
        self,
        offload_interval: int,
        file_manager: GradientFileManager,
        recording_type: str = "per_batch",
    ) -> None:
        if recording_type not in ("per_sample", "per_batch"):
            raise ValueError(
                f"recording_type must be 'per_sample' or 'per_batch', "
                f"got {recording_type!r}."
            )
        self._offload_interval = offload_interval
        self.file_manager = file_manager
        self._recording_type = recording_type
        self._staged: List[GradientRecord] = []
        self._staged_steps: set = set()

    def on_step_end(self, record: GradientRecord) -> None:
        records = self._expand(record)
        if record.step not in self._staged_steps:
            if len(self._staged_steps) >= self._offload_interval:
                self._flush()
        self._staged.extend(records)
        self._staged_steps.add(record.step)

    def _expand(self, record: GradientRecord) -> List[GradientRecord]:
        """Slice into per-sample records when ``recording_type='per_sample'``."""
        if self._recording_type != "per_sample":
            return [record]
        hashes = (
            record.input_hash
            if isinstance(record.input_hash, list)
            else [record.input_hash]
        )
        batch_size = record.gradient.batch_size
        return [
            GradientRecord(
                step=record.step,
                input_hash=hashes[i],
                gradient=record.gradient.slice(dim="batch", index=i),
            )
            for i in range(batch_size)
        ]

    def on_context_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._staged:
            return
        self.file_manager.save_batch(self._staged)
        self._staged.clear()
        self._staged_steps.clear()

    @property
    def staged(self) -> List[GradientRecord]:
        """Records staged but not yet written to disk."""
        return list(self._staged)


# --------------------------------------------------------------------------- #
# DataSelectionCallback                                                        #
# --------------------------------------------------------------------------- #


_THRESHOLD_MODES = frozenset({"hard", "bottom_fraction", "negative_bottom_fraction"})
_SCORE_MODES = frozenset({"ghost", "materialized"})
_TARGET_MODES = frozenset({"batch", "fixed", "val_loader"})


class DataSelectionCallback(HookManagerCallback):
    """Online data selection via per-sample influence scoring.

    At the end of each batch step (inside ``loss.backward()``, before
    ``optimizer.step()``), this callback:

    1. Computes a per-sample influence score (see ``score_mode``).
    2. Identifies samples to drop according to ``threshold_mode``.
    3. Subtracts those samples' exact gradient contributions from every hooked
       layer's ``param.grad``, so the optimizer update proceeds as if those
       samples were never in the batch.

    **Score modes** (``score_mode`` argument):

    ``"ghost"`` *(default)*
        Ghost inner product — computes ``score[i] = <dL_i/dW, dL_target/dW>``
        via gram matrices on the factorised (g, a) factors without ever
        forming the (out x in) weight gradient.
        Cost O((B*T)^2 * (d_out + d_in)) per layer.

    ``"materialized"``
        Builds the full per-sample weight gradient
        ``dW_i = sum_t g_{i,t}^T x a_{i,t}`` and dots it against the target
        gradient ``dW_target = sum_{j,s} g_{target,j,s} x a_{target,j,s}``.
        Higher memory cost but easier to follow and useful for verification.

    Both modes produce identical scores because
    ``< g_i x a_i,  g_j x a_j >  =  (g_i . g_j) * (a_i . a_j)``.

    **Target modes** (``target`` argument):

    ``"batch"`` *(default)*
        Use the current training batch as the scoring target. Each sample's
        gradient is compared against the full-batch gradient (the sum over
        all samples in the same batch). This is a pure online method and
        requires no extra data.

    ``"fixed"``
        Use a pre-computed ``Gradient`` object (passed as ``target_gradient``)
        as the target at every step. The target is typically computed once
        on a reference or test set before training and then held fixed.
        ``target_gradient`` must be provided.

    ``"val_loader"``
        At each step, draw one batch from ``val_loader``, run
        ``val_loss_fn(model, batch).backward()`` to obtain a fresh target
        gradient, and use it for scoring. This mode re-uses the training
        ``HookManager``'s hooks to capture the val-batch factorised gradient
        via a re-entrancy guard, then restores ``param.grad`` so the training
        update is unaffected. ``val_loader`` and ``val_loss_fn`` must be
        provided.

        Note: the training ``HookManager``'s ``steps_collected`` counter
        increments once per training step *and* once per val pass (because
        the val backward triggers the same completion hooks). This is a
        known side-effect of the re-entrancy approach and does not affect
        gradient correctness or the training update.

    **Threshold modes**:

    ``"hard"`` *(default)*
        Drop every sample whose score is strictly below ``threshold``
        (a raw score value).  ``threshold=0.0`` removes only negatively
        influential samples; raise it to be more aggressive.

    ``"bottom_fraction"``
        Drop the bottom ``threshold`` fraction of the batch by score rank,
        regardless of sign.  ``threshold`` must be in ``[0, 1)``.
        E.g. ``threshold=0.1`` always drops the worst 10 % of each batch.

    ``"negative_bottom_fraction"``
        Drop the bottom ``threshold`` fraction of the batch by score rank,
        *but only if their score is negative*.  Samples that rank in the
        bottom k % yet have a non-negative score are kept.  This is the most
        conservative mode: it never removes a sample that contributes
        positively, and from the harmful set it only removes the most
        harmful fraction.  ``threshold`` must be in ``[0, 1)``.

    **Gradient removal** (all hooked layers):

    For each dropped sample *i* and each hooked layer *l*, the exact
    contribution is subtracted::

        W_l.grad  -=  sum_t  g_{i,t}^T  x  a_{i,t}

    The per-sample weight gradient comes from :func:`ops.materialize`, so every
    layer type supported by the library (linear, convolution, transposed
    convolution, normalization, and embedding families) is handled by the same
    code path; bias gradients are subtracted as the summed output gradient.

    **Distributed (FSDP):**

    ``FullyShardedDataParallel`` (``use_orig_params=True``) is supported.  Each
    original ``param.grad`` is then a 1-D *shard* holding FSDP's averaged
    gradient, and a dropped sample's weight elements may be owned by a different
    rank's shard, so removal is done collectively: every rank computes its
    dropped samples' full contribution, a single ``all_reduce`` sums them, the
    result is rescaled by ``1/world_size`` to match FSDP's averaging, and each
    rank subtracts the slice it owns.  This runs on every step (with zero
    contributions when a rank drops nothing) to keep ranks in lock-step, and
    assumes FSDP's default averaged-gradient reduction.

    Construction order under FSDP: build the :class:`HookManager` on the
    *unwrapped* model (so its hooks survive wrapping), wrap with FSDP, then pass
    the **wrapped** module here (needed to discover the shard layout) and attach
    via :meth:`HookManager.add_callback`::

        collector = HookManager(model, config=...)
        model = FSDP(model, use_orig_params=True)
        collector.add_callback(DataSelectionCallback(model, ...))

    Args:
        model: The model being trained.  ``DataParallel`` / ``DDP`` wrappers
            are unwrapped automatically via ``.module``; for FSDP, pass the
            FSDP-wrapped module (see "Distributed (FSDP)" above).
        threshold: Interpretation depends on ``threshold_mode``:

            * ``"hard"`` — score cutoff (any float, default ``0.0``).
            * ``"bottom_fraction"`` / ``"negative_bottom_fraction"`` — fraction
              of the batch to drop (float in ``[0, 1)``).

        threshold_mode: One of ``"hard"`` (default), ``"bottom_fraction"``,
            or ``"negative_bottom_fraction"``.  See class docstring for details.
        score_mode: Scoring algorithm. ``"ghost"`` (default) uses the ghost
            inner product (no weight-gradient materialisation); ``"materialized"``
            builds the full per-sample weight gradient and dots it against the
            target gradient.  Both produce identical scores.
        target: Target gradient source. ``"batch"`` (default), ``"fixed"``,
            or ``"val_loader"``.  See class docstring for details.
        target_gradient: Required when ``target="fixed"``.  A pre-computed
            :class:`~dattri_llm.gradient.gradient.Gradient` object used as
            the scoring target at every step.
        val_loader: Required when ``target="val_loader"``.  An iterable
            (e.g. ``torch.utils.data.DataLoader``) that yields batches for
            the validation target.  The loader is cycled automatically when
            exhausted.
        val_loss_fn: Required when ``target="val_loader"``.  A callable
            ``(model, batch) -> loss`` that computes a scalar loss on one
            validation batch.  The function must trigger a forward pass
            through all hooked layers so that factorised gradients can be
            captured.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.0,
        threshold_mode: str = "hard",
        score_mode: str = "ghost",
        target: str = "batch",
        target_gradient: Optional[Gradient] = None,
        val_loader: Optional[Any] = None,
        val_loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    ) -> None:
        if threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {sorted(_THRESHOLD_MODES)}, "
                f"got {threshold_mode!r}."
            )
        if threshold_mode in ("bottom_fraction", "negative_bottom_fraction"):
            if not (0.0 <= threshold < 1.0):
                raise ValueError(
                    f"threshold must be in [0, 1) for threshold_mode={threshold_mode!r}, "
                    f"got {threshold}."
                )
        if score_mode not in _SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {sorted(_SCORE_MODES)}, "
                f"got {score_mode!r}."
            )
        if target not in _TARGET_MODES:
            raise ValueError(
                f"target must be one of {sorted(_TARGET_MODES)}, "
                f"got {target!r}."
            )
        if target == "fixed" and target_gradient is None:
            raise ValueError(
                "target='fixed' requires target_gradient to be a Gradient object."
            )
        if target == "val_loader":
            if val_loader is None:
                raise ValueError("target='val_loader' requires val_loader.")
            if val_loss_fn is None:
                raise ValueError("target='val_loader' requires val_loss_fn.")

        self._root: nn.Module = getattr(model, "module", model)
        # Top-level model as passed in.  When the model is FSDP-wrapped this is
        # the FSDP module (``_root`` is its unwrapped ``.module``); it is the
        # entry point used to discover the FSDP ``FlatParameter`` shard layout.
        self._model: nn.Module = model
        # Lazily-built map ``id(param) -> _ShardSpec`` for FSDP gradient
        # correction; ``None`` until the first ``on_step_end`` (the model may be
        # wrapped in FSDP *after* this callback is constructed).  Empty dict ==
        # "not FSDP".  ``_fsdp_world_size`` is the process-group size used to
        # rescale all-reduced contributions to FSDP's averaged-gradient
        # convention.
        self._fsdp_shard_map: Optional[Dict[int, _ShardSpec]] = None
        self._fsdp_world_size: int = 1
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._score_mode = score_mode
        self._target = target
        self._target_gradient = target_gradient
        self._val_loader = val_loader
        self._val_loss_fn = val_loss_fn
        self._val_iter = iter(val_loader) if val_loader is not None else None

        # Reference to the training HookManager; set by on_register().
        # Used in _collect_val_gradient() to pause step-completion tracking
        # during the secondary (val) backward pass.
        self._hook_manager: Optional[Any] = None

        # Guard flag: True while the val backward is in flight.
        # Prevents on_layer_forward / on_step_end from re-entering val logic.
        self._in_val_pass: bool = False
        # Whether val target has already been collected for the current step.
        self._val_target_ready_for_step: bool = False
        # Most recently collected val gradient (used by _resolve_target).
        self._pending_val_gradient: Optional[Gradient] = None

        # Per-layer buffers for our own val-pass hooks.
        self._val_act_bufs: Dict[str, torch.Tensor] = {}
        self._val_grad_bufs: Dict[str, torch.Tensor] = {}
        self._val_hook_handles: List = []

        if target == "val_loader":
            self._register_val_hooks()

        # Exposed for inspection / debugging after each step.
        self.last_scores: Optional[torch.Tensor] = None
        self.last_dropped: List[int] = []

    # ---------------------------------------------------------------------- #
    # HookManager registration                                                #
    # ---------------------------------------------------------------------- #

    def on_register(self, hook_manager: Any) -> None:
        """Called by :class:`~dattri_llm.gradient.hooks.HookManager` when this
        callback is registered.

        Stores a reference to the HookManager so that
        :meth:`_collect_val_gradient` can call
        :meth:`~dattri_llm.gradient.hooks.HookManager.pause` during the
        secondary (val) backward pass.

        Args:
            hook_manager: The :class:`~dattri_llm.gradient.hooks.HookManager`
                that registered this callback.
        """
        self._hook_manager = hook_manager

    # ---------------------------------------------------------------------- #
    # Val-pass hook registration                                              #
    # ---------------------------------------------------------------------- #

    def _register_val_hooks(self) -> None:
        """Register lightweight forward+backward hooks for val-pass collection.

        These hooks are separate from the training HookManager's hooks and
        only capture data when ``_in_val_pass`` is ``True``.  They fire for
        both training and val passes (PyTorch does not allow selective hook
        suppression), but the flag ensures only val data is stored.
        """
        from dattri_llm.gradient.hooks import register_linear_io_hooks

        cb_self = self  # capture for closures

        def _val_fwd(layer_name: str, act: torch.Tensor) -> None:
            if cb_self._in_val_pass:
                cb_self._val_act_bufs[layer_name] = act

        def _val_bwd(layer_name: str, grad: torch.Tensor) -> None:
            if cb_self._in_val_pass:
                cb_self._val_grad_bufs[layer_name] = grad

        _, self._val_hook_handles = register_linear_io_hooks(
            model=self._root,
            on_layer_forward=_val_fwd,
            on_layer_backward=_val_bwd,
        )

    def on_context_end(self) -> None:
        """Remove val-pass hooks when the collect() context closes."""
        if self._val_hook_handles:
            from dattri_llm.gradient.hooks import remove_hooks
            remove_hooks(self._val_hook_handles)

    # ---------------------------------------------------------------------- #
    # Main entry points                                                        #
    # ---------------------------------------------------------------------- #

    def on_layer_forward(self, layer_name: str, activation: torch.Tensor) -> None:
        """Trigger val-target collection at the start of each training step.

        For ``target="val_loader"``, the first ``on_layer_forward`` call of
        each training step (detected via ``_val_target_ready_for_step``)
        kicks off :meth:`_collect_val_gradient`.  Subsequent calls within
        the same step, and all calls during the secondary val pass, are
        ignored.

        Args:
            layer_name: Fully-qualified name of the hooked layer.
            activation: Captured input activation (on CPU).
        """
        if self._target != "val_loader":
            return
        if self._in_val_pass:
            return  # inside val pass — no recursive collection
        if self._val_target_ready_for_step:
            return  # already collected for this step
        # First training-layer forward of this step → collect val target.
        self._val_target_ready_for_step = True
        self._collect_val_gradient()

    def on_step_end(self, record: GradientRecord) -> None:
        """Score and optionally drop samples at the end of each training step.

        When ``_in_val_pass`` is ``True`` the call originates from the
        val backward (triggered inside :meth:`_collect_val_gradient`) and is
        silently ignored — the val gradient has already been captured by the
        dedicated val hooks.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
        """
        if self._in_val_pass:
            # Val pass completed — data captured by our own val hooks.
            # Do not score; just return.
            return

        # Reset for the next step.
        self._val_target_ready_for_step = False

        target_grad = self._resolve_target()

        scores = self._compute_scores(record, target_grad)
        self.last_scores = scores.detach().cpu()
        dropped = self._select_dropped(self.last_scores)
        self.last_dropped = dropped

        self._ensure_fsdp_map()
        if self._fsdp_shard_map:
            # FSDP path runs every step (collectives must stay in lock-step
            # across ranks), even when this rank drops nothing.
            self._remove_contributions_fsdp(record, dropped)
        elif dropped:
            self._remove_contributions(record, dropped)

    # ---------------------------------------------------------------------- #
    # Target resolution                                                        #
    # ---------------------------------------------------------------------- #

    def _resolve_target(self) -> Optional[Gradient]:
        """Return the target Gradient for this step, or None for batch mode.

        Returns:
            * ``None`` when ``target="batch"`` (caller uses the training batch
              as its own scoring reference).
            * The pre-computed :class:`Gradient` when ``target="fixed"``.
            * The val-batch :class:`Gradient` collected at step start when
              ``target="val_loader"``.
        """
        if self._target == "batch":
            return None
        if self._target == "fixed":
            return self._target_gradient
        # val_loader: gradient was already collected in on_layer_forward.
        return self._pending_val_gradient

    def _collect_val_gradient(self) -> None:
        """Sample one val batch, run forward+backward, and store its Gradient.

        Called from :meth:`on_layer_forward` at the start of each training
        step (while the training forward pass is still in progress, *before*
        the training backward).  Running backward from here does **not**
        trigger re-entrancy in PyTorch's autograd engine.

        The training HookManager is paused via
        :meth:`~dattri_llm.gradient.hooks.HookManager.pause` so that:

        * The val backward does not trigger HookManager's step-completion
          logic (no spurious ``on_step_end`` call from the training
          HookManager).
        * Any training-pass activation data already captured in HookManager's
          buffers (e.g. the first layer's activation captured just before this
          call) is preserved and restored after the pause exits.

        Our own val hooks (registered in :meth:`_register_val_hooks`) capture
        the val factorised data into ``_val_act_bufs`` / ``_val_grad_bufs``
        independently of HookManager's buffers.

        ``param.grad`` is saved before the val pass and restored afterwards
        so the training gradient update is completely unaffected.
        """
        # Advance the val iterator, cycling when exhausted.
        try:
            batch = next(self._val_iter)  # type: ignore[arg-type]
        except StopIteration:
            self._val_iter = iter(self._val_loader)  # type: ignore[arg-type]
            batch = next(self._val_iter)  # type: ignore[arg-type]

        # Save parameter gradients (may be non-None under gradient accumulation).
        saved_grads: Dict[str, Optional[torch.Tensor]] = {
            n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in self._root.named_parameters()
        }

        self._val_act_bufs.clear()
        self._val_grad_bufs.clear()
        self._in_val_pass = True
        try:
            self._root.zero_grad()
            if self._hook_manager is not None:
                # Pause the training HookManager so val backward does not
                # trigger step completion or contaminate training buffers.
                with self._hook_manager.pause():
                    loss = self._val_loss_fn(self._root, batch)  # type: ignore[misc]
                    loss.backward()
            else:
                # No HookManager reference yet (e.g. standalone test).
                loss = self._val_loss_fn(self._root, batch)  # type: ignore[misc]
                loss.backward()
        finally:
            self._in_val_pass = False
            # Restore the training parameter gradients.
            for n, p in self._root.named_parameters():
                p.grad = saved_grads[n]

        self._pending_val_gradient = self._assemble_val_gradient()

    def _assemble_val_gradient(self) -> Optional[Gradient]:
        """Build a :class:`Gradient` from the captured val-pass buffers.

        Raw hook data is stored with the module reference so that
        :func:`~dattri_llm.gradient.ops.preprocess_factorized` is called
        lazily inside ops functions rather than at assembly time.

        Returns:
            A :class:`Gradient` with ``"factorized"`` representation for each
            layer that has both activation and grad_output data, or ``None``
            if no layers were captured (e.g. the model has no hooked layers).
        """
        data: Dict[str, Any] = {}
        representation: Dict[str, str] = {}
        indexing: Dict[str, str] = {}
        layer_types: Dict[str, str] = {}
        for layer_name, act in self._val_act_bufs.items():
            grad = self._val_grad_bufs.get(layer_name)
            if grad is None:
                continue
            try:
                module = self._root.get_submodule(layer_name)
                layer_type = ops.canonical_class_name(module)
                module_kwargs = ops.extract_module_kwargs(module, layer_type)
            except AttributeError:
                layer_type = "nn.Embedding" if not act.is_floating_point() else "nn.Linear"
                module_kwargs = None
            layer_types[layer_name] = layer_type
            data[layer_name] = Factorized(activation=act, pre_activation_grad=grad,
                                          module_kwargs=module_kwargs)
            representation[layer_name] = "factorized"
            indexing[layer_name] = (
                "batch_token"
                if (act.ndim >= 3 or not act.is_floating_point())
                else "batch"
            )
        if not data:
            return None
        return Gradient(
            representation=representation,
            data=data,
            layer_types=layer_types,
            indexing=indexing,
        )

    # ---------------------------------------------------------------------- #
    # Drop-set selection                                                       #
    # ---------------------------------------------------------------------- #

    def _select_dropped(self, scores: torch.Tensor) -> List[int]:
        """Return the list of batch indices to drop based on the configured mode.

        Args:
            scores: Float tensor of shape ``(B,)`` — one score per sample.

        Returns:
            Sorted list of batch indices to remove from ``param.grad``.
        """
        B = scores.shape[0]

        if self._threshold_mode == "hard":
            # Drop every sample strictly below the threshold value.
            return (scores < self._threshold).nonzero(as_tuple=True)[0].tolist()

        # Fraction-based modes: compute how many samples to consider.
        n_drop = round(B * self._threshold)
        if n_drop == 0:
            return []
        # Indices of the n_drop lowest-scored samples, in ascending score order.
        bottom_idx = scores.argsort()[:n_drop].tolist()

        if self._threshold_mode == "bottom_fraction":
            # Drop all n_drop regardless of sign.
            return bottom_idx

        # "negative_bottom_fraction": keep any candidate whose score ≥ 0.
        return [i for i in bottom_idx if scores[i] < 0]

    # ---------------------------------------------------------------------- #
    # Per-sample influence scoring                                             #
    # ---------------------------------------------------------------------- #

    def _compute_scores(
        self,
        record: GradientRecord,
        target: Optional[Gradient] = None,
    ) -> torch.Tensor:
        """Compute per-sample influence scores ``score[i] = ⟨∇W_i, ∇W_target⟩``.

        ``∇W_target`` is the sum of the target samples' gradients (the training
        batch itself when ``target`` is ``None``).  The per-layer inner products
        are delegated to :meth:`Gradient.similarity` — the single shared
        implementation of factorized gradient similarity — and this method only
        sums each layer's cross-gram over the target batch and accumulates
        across layers.

        ``score_mode`` selects the :meth:`~Gradient.similarity` ``mode``
        (``"ghost"`` → ``"factorized"``); both modes are numerically identical.

        Args:
            record: Full-batch GradientRecord for this step.
            target: Optional external target Gradient.  ``None`` → batch mode.

        Returns:
            Float tensor of shape (B,).
        """
        gradient = record.gradient
        other = gradient if target is None else target
        mode = "materialized" if self._score_mode == "materialized" else "factorized"

        # {layer: (B_layer, B_target)} cross-gram per selected scoring mode.
        per_layer = gradient.similarity(other, mode=mode)

        B = gradient.batch_size
        scores = torch.zeros(B)
        for matrix in per_layer.values():
            # Sum over target samples → ⟨∇W_i, Σ_j ∇W_target_j⟩.
            layer_scores = matrix.sum(1)
            # A layer whose gradient was summed over the batch during the forward
            # broadcast (e.g. GPT-2 wpe) yields fewer rows; every sample then
            # receives an equal contribution.
            if layer_scores.shape[0] < B:
                layer_scores = layer_scores.expand(B)
            scores += layer_scores
        return scores

    # ---------------------------------------------------------------------- #
    # Gradient correction                                                      #
    # ---------------------------------------------------------------------- #

    # Layers whose parameters are indexed channels-first (dim 1) rather than
    # channels-last (last dim).  Determines the reduction axis for bias grads.
    _CHANNELS_FIRST_NORMS = frozenset(
        {"nn.GroupNorm", "nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"}
    )

    def _remove_contributions(
        self,
        record: GradientRecord,
        dropped: List[int],
    ) -> None:
        """Subtract dropped samples' parameter-gradient contributions.

        The per-sample weight gradients are obtained from
        :func:`ops.materialize` (so every supported layer type is handled
        consistently with the rest of the library), and bias gradients are the
        summed output gradient.  The result is subtracted from each hooked
        layer's ``param.grad`` so the optimizer steps as if the dropped samples
        were never in the batch.

        Args:
            record: Full-batch :class:`GradientRecord` for this step.
            dropped: List of batch indices to remove.
        """
        B = record.gradient.batch_size
        for layer_name, val in record.gradient.data.items():
            if not isinstance(val, Factorized):
                continue
            # Normalise sequence-first captures so the batch axis is dim 0 before
            # we index samples / read the batch size below.
            val = val.as_batch_first()
            # Skip layers whose gradient was summed over the batch dim during
            # the forward broadcast (e.g. wpe in GPT-2, where position_ids has
            # shape (1, T)).  Per-sample contributions cannot be isolated.
            if val.pre_activation_grad.shape[0] < B:
                continue
            try:
                module = self._root.get_submodule(layer_name)
            except AttributeError:
                continue

            layer_type = ops.canonical_class_name(module)
            a_d = val.activation[dropped]            # (n, ...)
            g_d = val.pre_activation_grad[dropped]   # (n, ...)
            self._subtract_weight(module, layer_type, val.module_kwargs, a_d, g_d)
            self._subtract_bias(module, layer_type, g_d)

    def _subtract_weight(
        self,
        module: nn.Module,
        layer_type: str,
        module_kwargs: Optional[dict],
        a_d: torch.Tensor,
        g_d: torch.Tensor,
    ) -> None:
        """Subtract the dropped samples' weight-gradient contribution.

        Uses :func:`ops.materialize` (``include_bias=False``) for the per-sample
        weight gradient, then maps it onto ``weight.grad``'s natural shape — a
        vocab-row slice for embeddings, a position-sum for norms, a transpose
        for HuggingFace ``Conv1D``, and a plain ``reshape_as`` otherwise.
        """
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            return

        # (n, d_weight); sum over the dropped samples → (d_weight,).
        contrib = ops.materialize(
            Factorized(a_d, g_d, module_kwargs), layer_type, include_bias=False
        ).sum(0)

        if ops.is_embedding(layer_type):
            # materialize scatters into rows 0..max_token; subtract that slice.
            embed_dim = weight.shape[1]
            vocab_local = contrib.numel() // embed_dim
            weight.grad[:vocab_local] -= contrib.reshape(vocab_local, embed_dim).to(
                weight.grad.dtype
            )
            return

        if ops.is_norm(layer_type):
            # Per-position gradient flattened over positions; sum them.
            wnum = weight.numel()
            contrib = contrib.reshape(-1, wnum).sum(0)
        elif layer_type == "transformers.pytorch_utils.Conv1D":
            # materialize yields (out, in) order; HF Conv1D weight is (in, out).
            out_f, in_f = weight.shape[1], weight.shape[0]
            contrib = contrib.reshape(out_f, in_f).T

        weight.grad -= contrib.reshape_as(weight.grad).to(weight.grad.dtype)

    def _subtract_bias(
        self,
        module: nn.Module,
        layer_type: str,
        g_d: torch.Tensor,
    ) -> None:
        """Subtract the dropped samples' bias-gradient contribution.

        The bias gradient is the output gradient summed over every dimension
        except the channel dimension (last dim for linear/LayerNorm/RMSNorm,
        dim 1 for conv/transposed-conv/Group/Instance norm).
        """
        bias = getattr(module, "bias", None)
        if bias is None or bias.grad is None:
            return
        channels = bias.shape[0]
        g = g_d.float()
        if (
            ops.is_conv(layer_type)
            or ops.is_conv_transpose(layer_type)
            or layer_type in self._CHANNELS_FIRST_NORMS
        ):
            contrib = g.movedim(1, -1).reshape(-1, channels).sum(0)
        else:
            contrib = g.reshape(-1, channels).sum(0)
        bias.grad -= contrib.to(bias.grad.dtype)

    # ---------------------------------------------------------------------- #
    # FSDP gradient correction                                                 #
    # ---------------------------------------------------------------------- #
    #
    # Under ``FullyShardedDataParallel`` (``use_orig_params=True``) each
    # original ``param.grad`` is a 1-D *shard* — a contiguous slice of the
    # flattened, unsharded parameter — holding FSDP's *averaged* gradient
    # ``(1/world_size) · Σ_r G_r`` for that slice.  A dropped sample lives on a
    # single rank, but the weight elements its gradient touches may be owned by
    # a *different* rank's shard, so removal cannot be done rank-locally.
    #
    # Each rank instead computes the full-shaped gradient contribution of *its*
    # dropped samples, the contributions are summed across ranks with a single
    # ``all_reduce`` and rescaled by ``1/world_size`` to match FSDP's averaging,
    # and every rank then subtracts the slice covering the elements it owns.
    # The result equals ``(1/world_size) · Σ_r G_r^kept`` shard-for-shard.
    #
    # The collectives run on *every* step the callback is active (with zero
    # contributions when no rank dropped anything) so all ranks stay in
    # lock-step.  This assumes FSDP's default averaged-gradient reduction.

    def _ensure_fsdp_map(self) -> None:
        """Build the ``id(param) -> _ShardSpec`` map on first use (idempotent).

        The model may be FSDP-wrapped *after* this callback is constructed, so
        discovery is deferred to the first ``on_step_end``.  Leaves an empty
        map (and ``_fsdp_world_size == 1``) for non-FSDP models, which routes
        ``on_step_end`` through the rank-local :meth:`_remove_contributions`.
        """
        if self._fsdp_shard_map is not None:
            return
        shard_map: Dict[int, _ShardSpec] = {}
        for module in self._model.modules():
            flat_param = getattr(module, "_flat_param", None)
            if flat_param is None:
                continue
            infos = getattr(flat_param, "_shard_param_infos", None)
            params = getattr(flat_param, "_params", None)
            shapes = getattr(flat_param, "_shapes", None)
            if infos is None or params is None or shapes is None:
                continue
            for param, info, shape in zip(params, infos, shapes):
                shard_map[id(param)] = _ShardSpec(
                    orig_shape=tuple(shape),
                    in_shard=bool(info.in_shard),
                    start=info.intra_param_start_idx,
                    end=info.intra_param_end_idx,
                )
        self._fsdp_shard_map = shard_map
        if shard_map:
            import torch.distributed as dist

            self._fsdp_world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )

    def _remove_contributions_fsdp(
        self,
        record: GradientRecord,
        dropped: List[int],
    ) -> None:
        """Shard-aware, collective version of :meth:`_remove_contributions`.

        Runs on every rank in lock-step.  See the section comment above for the
        correctness argument.
        """
        import torch.distributed as dist

        world = self._fsdp_world_size
        use_dist = world > 1 and dist.is_available() and dist.is_initialized()

        # Skip the (expensive) contribution all-reduce when *no* rank dropped
        # anything this step.  The drop count itself is reduced first so the
        # decision is identical on every rank.
        if use_dist:
            count = torch.tensor([len(dropped)], dtype=torch.long)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            any_dropped = bool(count.item())
        else:
            any_dropped = bool(dropped)
        if not any_dropped:
            return

        entries = self._build_fsdp_contributions(record, dropped)
        if not entries:
            return

        if use_dist:
            buf = torch.cat([flat for _, flat in entries])
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf /= world
            offset = 0
            for param, flat in entries:
                self._subtract_shard(param, buf[offset : offset + flat.numel()])
                offset += flat.numel()
        else:
            # Single-rank FSDP (no sharding across ranks): grads are not
            # averaged, so subtract the local contribution directly.
            for param, flat in entries:
                self._subtract_shard(param, flat)

    def _build_fsdp_contributions(
        self,
        record: GradientRecord,
        dropped: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return ``[(param, full_flat_contribution)]`` for every sharded param.

        Each contribution is the *full*, unsharded parameter-gradient of this
        rank's dropped samples, flattened in the parameter's natural C-order
        (zeros when this rank dropped nothing).  The entry list — params and
        their full sizes — depends only on model structure and is therefore
        identical across ranks, which keeps the packed ``all_reduce`` aligned.
        """
        B = record.gradient.batch_size
        shard_map = self._fsdp_shard_map
        assert shard_map is not None
        entries: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_name, val in sorted(record.gradient.data.items()):
            if not isinstance(val, Factorized):
                continue
            # Normalise sequence-first captures to batch-first before indexing.
            val = val.as_batch_first()
            if val.pre_activation_grad.shape[0] < B:
                continue
            try:
                module = self._root.get_submodule(layer_name)
            except AttributeError:
                continue
            layer_type = ops.canonical_class_name(module)
            a_d = val.activation[dropped] if dropped else None
            g_d = val.pre_activation_grad[dropped] if dropped else None

            weight = getattr(module, "weight", None)
            if weight is not None and id(weight) in shard_map:
                full_numel = math.prod(shard_map[id(weight)].orig_shape)
                flat = torch.zeros(full_numel, dtype=torch.float32)
                if dropped:
                    contrib, off = self._weight_contrib_natural(
                        layer_type, val.module_kwargs, a_d, g_d,
                        shard_map[id(weight)].orig_shape,
                    )
                    flat[off : off + contrib.numel()] = contrib.float()
                entries.append((weight, flat))

            bias = getattr(module, "bias", None)
            if bias is not None and id(bias) in shard_map:
                channels = math.prod(shard_map[id(bias)].orig_shape)
                flat = torch.zeros(channels, dtype=torch.float32)
                if dropped:
                    flat = self._bias_contrib_natural(layer_type, g_d, channels).float()
                entries.append((bias, flat))
        return entries

    def _weight_contrib_natural(
        self,
        layer_type: str,
        module_kwargs: Optional[dict],
        a_d: torch.Tensor,
        g_d: torch.Tensor,
        full_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, int]:
        """Full weight-gradient contribution as a 1-D tensor in the parameter's
        natural C-order, plus the flat offset at which it begins.

        Mirrors the per-layer-type reshaping in :meth:`_subtract_weight` but
        uses the *unsharded* ``full_shape`` (FSDP shards expose only a 1-D
        slice, so ``weight.shape`` is unusable).  The offset is non-zero only
        for embeddings, whose contribution covers just rows ``0..max_token``.
        """
        contrib = ops.materialize(
            Factorized(a_d, g_d, module_kwargs), layer_type, include_bias=False
        ).sum(0)
        if ops.is_embedding(layer_type):
            # materialize scatters into rows 0..max_token, flattened row-major
            # — already aligned with the start of the flattened weight.
            return contrib, 0
        if ops.is_norm(layer_type):
            wnum = math.prod(full_shape)
            return contrib.reshape(-1, wnum).sum(0), 0
        if layer_type == "transformers.pytorch_utils.Conv1D":
            # materialize yields (out, in); HF Conv1D weight is (in, out).
            out_f, in_f = full_shape[1], full_shape[0]
            return contrib.reshape(out_f, in_f).T.reshape(-1), 0
        return contrib.reshape(-1), 0

    def _bias_contrib_natural(
        self,
        layer_type: str,
        g_d: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        """Full bias-gradient contribution as a 1-D ``(channels,)`` tensor."""
        g = g_d.float()
        if (
            ops.is_conv(layer_type)
            or ops.is_conv_transpose(layer_type)
            or layer_type in self._CHANNELS_FIRST_NORMS
        ):
            return g.movedim(1, -1).reshape(-1, channels).sum(0)
        return g.reshape(-1, channels).sum(0)

    def _subtract_shard(self, param: torch.Tensor, full_flat: torch.Tensor) -> None:
        """Subtract the slice of ``full_flat`` this rank owns from ``param.grad``.

        ``full_flat`` is the (already averaged) full-parameter contribution in
        natural C-order; the rank's shard is ``full.reshape(-1)[start:end+1]``.
        """
        spec = self._fsdp_shard_map[id(param)]  # type: ignore[index]
        if not spec.in_shard or param.grad is None:
            return
        seg = full_flat.reshape(-1)[spec.start : spec.end + 1]
        param.grad -= seg.to(param.grad.dtype)
