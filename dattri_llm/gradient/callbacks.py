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

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]


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

    Supports ``nn.Linear`` (weight ``(out, in)``), HuggingFace ``Conv1D``
    (weight ``(in, out)``), and ``nn.Embedding`` (weight ``(vocab, embed)``
    via scatter-sub).

    Args:
        model: The model being trained.  ``DataParallel`` / ``DDP`` wrappers
            are unwrapped automatically via ``.module``.
        threshold: Interpretation depends on ``threshold_mode``:

            * ``"hard"`` — score cutoff (any float, default ``0.0``).
            * ``"bottom_fraction"`` / ``"negative_bottom_fraction"`` — fraction
              of the batch to drop (float in ``[0, 1)``).

        threshold_mode: One of ``"hard"`` (default), ``"bottom_fraction"``,
            or ``"negative_bottom_fraction"``.  See class docstring for details.
        token_reduction: How to reduce the token dimension when computing
            scores.  ``"mean"`` (default) or ``"sum"``.  ``"mean"`` matches the
            per-token loss scale used by HuggingFace Transformers, verl, and
            OLMo-core, so scores are comparable across sequences of different
            lengths.  ``"sum"`` gives longer sequences proportionally higher
            scores.
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
        token_reduction: str = "mean",
        score_mode: str = "ghost",
        target: str = "batch",
        target_gradient: Optional[Gradient] = None,
        val_loader: Optional[Any] = None,
        val_loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    ) -> None:
        if token_reduction not in ("sum", "mean"):
            raise ValueError(
                f"token_reduction must be 'sum' or 'mean', got {token_reduction!r}."
            )
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
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._token_reduction = token_reduction
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
        from dattri_llm.gradient.hooks import register_mlp_hooks

        cb_self = self  # capture for closures

        def _val_fwd(layer_name: str, act: torch.Tensor) -> None:
            if cb_self._in_val_pass:
                cb_self._val_act_bufs[layer_name] = act

        def _val_bwd(layer_name: str, grad: torch.Tensor) -> None:
            if cb_self._in_val_pass:
                cb_self._val_grad_bufs[layer_name] = grad

        _, self._val_hook_handles = register_mlp_hooks(
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

        if self._score_mode == "ghost":
            scores = self._ghost_scores(record, target_grad)
        else:
            scores = self._materialized_scores(record, target_grad)
        self.last_scores = scores.detach().cpu()
        dropped = self._select_dropped(self.last_scores)
        self.last_dropped = dropped
        if dropped:
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

        Returns:
            A :class:`Gradient` with ``"factorized"`` representation for each
            layer that has both activation and grad_output data, or ``None``
            if no layers were captured (e.g. the model has no hooked layers).
        """
        data: Dict[str, Any] = {}
        representation: Dict[str, str] = {}
        for layer_name, act in self._val_act_bufs.items():
            grad = self._val_grad_bufs.get(layer_name)
            if grad is None:
                continue
            data[layer_name] = Factorized(activation=act, pre_activation_grad=grad)
            representation[layer_name] = "factorized"
        if not data:
            return None
        from dattri_llm.gradient.gradient import Gradient as _Gradient
        return _Gradient(representation=representation, data=data)

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
    # Scoring helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _per_sample_grad(self, val: Factorized, mat: torch.Tensor) -> torch.Tensor:
        """Apply token reduction to a materialised layer gradient.

        Args:
            val: The source :class:`Factorized` (used only to detect Embedding
                and recover T for mean normalisation).
            mat: Materialised gradient of shape ``(B, F)`` or ``(B, T, F)``.

        Returns:
            ``(B, F)`` tensor — per-sample gradient with token reduction applied.
        """
        if mat.ndim == 3:
            T = mat.shape[1]
            ps = mat.sum(1)  # (B, F)
            return ps / T if self._token_reduction == "mean" else ps
        # 2-D: no token dim, or Embedding (already summed over T internally).
        if val.is_embedding and self._token_reduction == "mean":
            T = val.activation.shape[1]
            return mat / T
        return mat

    def _batch_grad(self, mat: torch.Tensor) -> torch.Tensor:
        """Sum a materialised gradient over all samples (and tokens if present).

        The batch gradient is always a plain sum — token reduction is applied
        to the *per-sample* side, not here, so the scale is consistent.

        Args:
            mat: Materialised gradient of shape ``(B, F)`` or ``(B, T, F)``.

        Returns:
            ``(F,)`` tensor.
        """
        return mat.flatten(0, 1).sum(0) if mat.ndim == 3 else mat.sum(0)

    # ---------------------------------------------------------------------- #
    # Ghost inner product scoring                                              #
    # ---------------------------------------------------------------------- #

    def _ghost_scores(
        self,
        record: GradientRecord,
        target: Optional[Gradient] = None,
    ) -> torch.Tensor:
        """Compute per-sample influence scores via ghost inner product.

        Computes ``score[i] = <dL_i/dW, dL_target/dW>`` via gram (or cross-gram)
        matrices on the factorised (g, a) factors, avoiding the full weight
        gradient materialisation::

            score[i] = sum_{t, j, s}  (g_{i,t} . g_tgt_{j,s}) * (a_{i,t} . a_tgt_{j,s})

        When ``target`` is ``None`` the training batch acts as its own target
        (self-gram); otherwise a cross-gram against ``target`` is computed.

        Each (sample, token) pair is treated as an independent row on the
        flattened (B*T) axis.  Reducing tokens before gram formation gives
        wrong scores because the sum-of-products factorisation does not
        commute with token averaging.

        For ``nn.Embedding`` the activation cross-gram is computed in
        O(B*T * d_embed) via scatter-add instead of the O((B*T)^2) form.

        Args:
            record: Full-batch GradientRecord for this step.
            target: Optional external target Gradient.  ``None`` → batch mode.

        Returns:
            Float tensor of shape (B,).
        """
        B = record.gradient.batch_size
        scores = torch.zeros(B)

        for layer_name, val in record.gradient.data.items():
            if not isinstance(val, Factorized):
                continue

            # Resolve target factors for this layer.
            if target is None:
                tgt_val = val  # self-gram
            else:
                tgt_val = target.data.get(layer_name)
                if not isinstance(tgt_val, Factorized):
                    continue

            g = val.pre_activation_grad.float()          # (Bl,  T,  out) or (Bl,  out)
            g_tgt = tgt_val.pre_activation_grad.float()  # (Bt, Tt, out) or (Bt, out)

            # Bl may differ from B when a layer's gradient was summed over the
            # batch dim during the forward broadcast (e.g. GPT-2 wpe, where
            # position_ids has shape (1, T) shared across all samples).  In
            # that case all B samples receive an equal contribution.
            layer_B = g.shape[0]

            if g.ndim == 2:
                # No token dimension — cross-gram at sample level.
                a = val.reduce_activation(self._token_reduction)          # (Bl, in)
                a_tgt = tgt_val.reduce_activation(self._token_reduction)  # (Bt, in)
                G = g @ g_tgt.T     # (Bl, Bt)
                A = a @ a_tgt.T     # (Bl, Bt)
                layer_scores = (G * A).sum(-1)  # (Bl,)
                if layer_B < B:
                    layer_scores = layer_scores.expand(B)
                scores += layer_scores

            else:
                T = g.shape[1]
                g_flat = g.reshape(-1, g.shape[-1])               # (Bl*T,  out)
                g_tgt_flat = g_tgt.reshape(-1, g_tgt.shape[-1])   # (Bt*Tt, out)

                if val.is_embedding:
                    # A_cross[p, q] = 1[tok_train[p] == tok_tgt[q]].
                    # Efficient form: bt_scores[p] = g_flat[p] · G_tgt_sum[tok[p]]
                    # where G_tgt_sum[k] = sum_{q: tok_tgt[q]==k} g_tgt_flat[q].
                    tok_flat = val.activation.reshape(-1)          # (Bl*T,)  int
                    tok_tgt_flat = tgt_val.activation.reshape(-1)  # (Bt*Tt,) int
                    vocab = int(max(tok_flat.max().item(),
                                    tok_tgt_flat.max().item())) + 1
                    G_tgt_sum = torch.zeros(vocab, g_tgt_flat.shape[-1],
                                            dtype=g_tgt_flat.dtype)
                    G_tgt_sum.scatter_add_(
                        0,
                        tok_tgt_flat.unsqueeze(-1).expand_as(g_tgt_flat),
                        g_tgt_flat,
                    )
                    bt_scores = (g_flat * G_tgt_sum[tok_flat]).sum(-1)  # (Bl*T,)
                else:
                    a_flat = val.activation.float().reshape(-1, val.activation.shape[-1])
                    a_tgt_flat = tgt_val.activation.float().reshape(
                        -1, tgt_val.activation.shape[-1]
                    )
                    G_cross = g_flat @ g_tgt_flat.T    # (Bl*T, Bt*Tt)
                    A_cross = a_flat @ a_tgt_flat.T    # (Bl*T, Bt*Tt)
                    bt_scores = (G_cross * A_cross).sum(-1)  # (Bl*T,)

                layer_scores = bt_scores.reshape(layer_B, T).sum(1)  # (Bl,)
                if self._token_reduction == "mean":
                    layer_scores = layer_scores / T
                if layer_B < B:
                    layer_scores = layer_scores.expand(B)
                scores += layer_scores

        return scores

    # ---------------------------------------------------------------------- #
    # Materialized inner product scoring                                       #
    # ---------------------------------------------------------------------- #

    def _materialized_scores(
        self,
        record: GradientRecord,
        target: Optional[Gradient] = None,
    ) -> torch.Tensor:
        """Compute per-sample influence scores by materialising weight gradients.

        For each layer, builds the per-sample weight gradient explicitly::

            dW_i = sum_t  g_{i,t}^T  (x)  a_{i,t}

        and dots it against the target gradient::

            score[i] = < dW_i,  dW_target >
            dW_target = sum_{j,s}  g_tgt_{j,s}^T  (x)  a_tgt_{j,s}

        When ``target`` is ``None`` the training batch is the target
        (``dW_target = sum_j dW_j``).  This is identical to ``_ghost_scores``
        because ``< g_i x a_i, g_j x a_j > = (g_i . g_j)(a_i . a_j)``.

        Token reduction is applied to the per-sample gradient; the target
        gradient is always a plain sum for consistent scale.

        Args:
            record: Full-batch GradientRecord for this step.
            target: Optional external target Gradient.  ``None`` → batch mode.

        Returns:
            Float tensor of shape (B,).
        """
        B = record.gradient.batch_size
        scores = torch.zeros(B)

        for layer_name, val in record.gradient.data.items():
            if not isinstance(val, Factorized):
                continue

            mat = val.materialize().float()  # (B, F) or (B, T, F)
            per_sample = self._per_sample_grad(val, mat)  # (B, F) with reduction

            if target is None:
                batch_grad = self._batch_grad(mat)  # (F,)
            else:
                tgt_val = target.data.get(layer_name)
                if not isinstance(tgt_val, Factorized):
                    continue
                tgt_mat = tgt_val.materialize().float()

                # Embedding layers use vocab_size = max(token_ids) + 1, which
                # varies per batch.  Pad both sides to the same vocabulary
                # size so the dot product is well-defined.
                if val.is_embedding:
                    embed_dim = val.pre_activation_grad.shape[-1]
                    V_tr = per_sample.shape[1] // embed_dim
                    V_tgt = tgt_mat.shape[1] // embed_dim
                    V = max(V_tr, V_tgt)
                    if V_tr < V:
                        per_sample = torch.cat(
                            [
                                per_sample,
                                torch.zeros(
                                    per_sample.shape[0],
                                    (V - V_tr) * embed_dim,
                                    dtype=per_sample.dtype,
                                ),
                            ],
                            dim=1,
                        )
                    if V_tgt < V:
                        tgt_mat = torch.cat(
                            [
                                tgt_mat,
                                torch.zeros(
                                    tgt_mat.shape[0],
                                    (V - V_tgt) * embed_dim,
                                    dtype=tgt_mat.dtype,
                                ),
                            ],
                            dim=1,
                        )

                batch_grad = self._batch_grad(tgt_mat)  # (F,)

            scores += (per_sample * batch_grad).sum(-1)

        return scores

    # ---------------------------------------------------------------------- #
    # Gradient correction                                                      #
    # ---------------------------------------------------------------------- #

    def _remove_contributions(
        self,
        record: GradientRecord,
        dropped: List[int],
    ) -> None:
        """Subtract dropped samples' weight gradient contributions from param.grad.

        Args:
            record: Full-batch :class:`GradientRecord` for this step.
            dropped: List of batch indices to remove.
        """
        B = record.gradient.batch_size
        for layer_name, val in record.gradient.data.items():
            if not isinstance(val, Factorized):
                continue
            # Skip layers whose gradient was summed over the batch dim during
            # the forward broadcast (e.g. wpe in GPT-2, where position_ids has
            # shape (1, T)).  Per-sample contributions cannot be isolated.
            if val.pre_activation_grad.shape[0] < B:
                continue
            try:
                module = self._root.get_submodule(layer_name)
            except AttributeError:
                continue

            self._subtract_weight_contrib(module, val, dropped)
            self._subtract_bias_contrib(module, val, dropped)

    def _subtract_weight_contrib(
        self,
        module: nn.Module,
        val: Factorized,
        dropped: List[int],
    ) -> None:
        """Remove dropped samples' contributions from ``module.weight.grad``.

        Handles ``nn.Linear`` (weight ``(out, in)``), HuggingFace ``Conv1D``
        (weight ``(in, out)``), and ``nn.Embedding`` (weight ``(vocab, embed)``
        via scatter-add).

        ``nn.LayerNorm`` weight subtraction is skipped: the exact per-sample
        contribution requires the internally-normalised activation ``x_hat``
        which the forward hook does not capture (it sees the raw input ``x``).
        LayerNorm *bias* subtraction remains exact (see
        :meth:`_subtract_bias_contrib`).
        """
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            return

        # LayerNorm: weight grad = sum_{b,t} grad_out[b,t,k] * x_hat[b,t,k].
        # We only have the raw (un-normalised) activation, so we cannot form
        # the exact outer product.  Skip to avoid writing incorrect values.
        if isinstance(module, nn.LayerNorm):
            return

        is_conv1d = HF_Conv1D is not None and isinstance(module, HF_Conv1D)
        is_embedding = val.is_embedding

        for i in dropped:
            g_i = val.pre_activation_grad[i]  # (T, out) or (T, embed_dim)
            a_i = val.activation[i]            # (T, in) or (T,) integer for Embedding

            if is_embedding:
                # Embedding weight is (vocab_size, embed_dim).
                # Contribution: for each token t, weight[token_ids[t], :] += g_i[t, :].
                embed_dim = g_i.shape[-1]
                T = a_i.shape[0]
                idx = a_i.unsqueeze(-1).expand(T, embed_dim)       # (T, embed_dim)
                contrib = torch.zeros_like(weight.grad)
                contrib.scatter_add_(0, idx, g_i.to(contrib.dtype))
            elif is_conv1d:
                # Conv1D: weight is (in, out), grad_W = a^T @ g
                contrib = torch.einsum("ti,to->io", a_i, g_i)
            else:
                # nn.Linear: weight is (out, in), grad_W = g^T @ a
                contrib = torch.einsum("to,ti->oi", g_i, a_i)
            weight.grad -= contrib.to(weight.grad.dtype)

    def _subtract_bias_contrib(
        self,
        module: nn.Module,
        val: Factorized,
        dropped: List[int],
    ) -> None:
        """Remove dropped samples' contributions from ``module.bias.grad``."""
        bias = getattr(module, "bias", None)
        if bias is None or bias.grad is None:
            return
        for i in dropped:
            # bias grad = sum of grad_output over batch and tokens
            contrib = val.pre_activation_grad[i].sum(0)  # (out,)
            bias.grad -= contrib.to(bias.grad.dtype)
