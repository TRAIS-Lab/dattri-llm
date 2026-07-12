"""Online data-selection callback (per-sample influence scoring + gradient removal)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks.base import HookManagerCallback
from dattri_llm.gradient.gradient import (
    Factorized,
    Gradient,
    GradientRecord,
    base_layer_name,
)
from dattri_llm.utils.autograd import queue_after_backward_finalization
from dattri_llm.utils.distributed import dist_world_size, is_dist_initialized

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dattri_llm.gradient.hooks import HookManager


# Per-parameter slice of an FSDP ``FlatParameter`` shard.  ``start``/``end`` are
# the inclusive indices into the *flattened, unsharded* parameter that this rank
# owns (i.e. ``param.grad`` equals ``full.reshape(-1)[start : end + 1]``);
# ``in_shard`` is False when this rank holds none of the parameter.
class _ShardSpec(NamedTuple):
    orig_shape: tuple[int, ...]
    in_shard: bool
    start: int | None
    end: int | None


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
       layer's ``param.grad``, so the optimizer update proceeds without the
       dropped samples' influence (see **Renormalization** for the exact
       semantics under a mean-reduced loss).

    **Score modes** (``score_mode`` argument):

    ``"ghost"`` *(default)*
        Ghost inner product -- computes ``score[i] = <dL_i/dW, dL_target/dW>``
        from the captured (g, a) factors with no extra backward pass, routing
        each layer through the factorized-vs-materialized cost heuristic
        (:func:`ops.maybe_use_materialized_gram`): the pure factor contraction
        for short sequences, a token-contracted GEMM once its
        ``O((B*T)^2 (d_out + d_in))`` cost would dominate.  Both routes are
        numerically identical.

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
        At the end of each training step (from :meth:`on_step_end`, before
        scoring), draw one batch from ``val_loader``, run
        ``val_loss_fn(model, batch).backward()`` to obtain a fresh target
        gradient, and use it for scoring.  The val gradient is captured
        through the training ``HookManager``'s **own** hooks -- no second
        hook set -- so it is consistent with the scored training gradients by
        construction (same layer selection, projection, and device
        placement).  The manager's capture state is checkpointed around the
        val pass (``HookManager.save_state`` / ``clear_state`` /
        ``load_state``) and ``param.grad`` -- the just-completed training
        step's gradients -- is restored afterwards, so the training update
        is unaffected.
        ``val_loader`` and ``val_loss_fn`` must be provided, and the callback
        must be attached to the ``HookManager`` (via
        ``HookManager(callbacks=[...])`` or ``add_callback``).

        Two properties of this design to be aware of:

        * The val backward completes a capture step of its own, whose record
          is dispatched to **every** attached callback's ``on_step_end``
          (this callback consumes it as the target).  A persistence callback
          on the same manager (e.g. ``OffloadCallback``) would therefore
          store one extra (val) record per training step.
        * ``val_loss_fn`` must run a backward that reaches **all** hooked
          layers; otherwise the val capture step never completes and a
          ``RuntimeError`` is raised.

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

    **Renormalization** (``renormalize`` argument):

    For a mean-reduced loss ``L = (1/B) sum_i l_i``, subtracting the dropped
    contributions alone leaves each kept sample at weight ``1/B`` -- the
    dropped samples' loss terms are zeroed, but the batch denominator stays
    ``B``.  With ``renormalize=True`` (default ``False``) the kept samples'
    contribution is additionally rescaled by ``B/(B - k)``, giving exactly the
    gradient of the mean loss over a batch that never contained the ``k``
    dropped samples.  The rescale is applied as a per-step *correction*
    alongside the subtraction (never by scaling ``param.grad`` wholesale), so
    it is exact per micro-batch under gradient accumulation.

    Leave it ``False`` when the subtraction alone already has the semantics
    you want: a sum-reduced loss (each sample's contribution is independent of
    ``B``), or a loss whose per-sample weights are not uniform (e.g. a
    token-level mean over unequal sequence lengths, where the correct factor
    would be token-count based, not ``B/(B - k)``).  When *every* sample is
    dropped the batch's contribution is removed with no rescale (the
    empty-batch mean is undefined).  Distributed: the factor is rank-local
    (``k_r / (B_r - k_r)``), matching the average-of-local-means gradient
    semantics of DDP/FSDP.

    **Distributed (DDP / FSDP):**

    Both regimes hold the *averaged* global gradient
    ``(1/world_size) * sum_r G_r`` after backward -- DDP replicated in full on
    every rank, FSDP as a 1-D per-rank shard (``use_orig_params=True``) -- so
    a rank-local subtraction is wrong in both: it misses the ``1/world_size``
    scaling and (DDP) diverges the replicas / (FSDP) misses elements owned by
    other ranks' shards.  Removal is therefore collective in both: every rank
    computes its dropped samples' full contribution, a single ``all_reduce``
    sums them, the result is rescaled by ``1/world_size`` to match the
    averaged-gradient convention, and each rank subtracts the full tensor
    (DDP) or the slice it owns (FSDP).  The collectives run on every step
    (with zero contributions when a rank drops nothing) to keep ranks in
    lock-step, and assume the default averaged-gradient reduction.

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

            * ``"hard"`` -- score cutoff (any float, default ``0.0``).
            * ``"bottom_fraction"`` / ``"negative_bottom_fraction"`` -- fraction
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
        renormalize: When ``True``, rescale the kept samples' contribution by
            ``B/(B - k)`` after dropping ``k`` of ``B`` samples, so a
            mean-reduced loss behaves exactly as if the batch never contained
            the dropped samples.  Default ``False`` (kept samples stay at
            weight ``1/B``).  See **Renormalization** in the class docstring.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.0,
        threshold_mode: str = "hard",
        score_mode: str = "ghost",
        target: str = "batch",
        target_gradient: Gradient | None = None,
        val_loader: Iterable[object] | None = None,
        val_loss_fn: Callable[[nn.Module, Any], torch.Tensor] | None = None,
        renormalize: bool = False,
    ) -> None:
        if threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {sorted(_THRESHOLD_MODES)}, "
                f"got {threshold_mode!r}.",
            )
        if threshold_mode in (
            "bottom_fraction",
            "negative_bottom_fraction",
        ) and not (0.0 <= threshold < 1.0):
            raise ValueError(
                f"threshold must be in [0, 1) for "
                f"threshold_mode={threshold_mode!r}, "
                f"got {threshold}.",
            )
        if score_mode not in _SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {sorted(_SCORE_MODES)}, "
                f"got {score_mode!r}.",
            )
        if target not in _TARGET_MODES:
            raise ValueError(
                f"target must be one of {sorted(_TARGET_MODES)}, got {target!r}.",
            )
        if target == "fixed" and target_gradient is None:
            raise ValueError(
                "target='fixed' requires target_gradient to be a Gradient object.",
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
        self._fsdp_shard_map: dict[int, _ShardSpec] | None = None
        self._fsdp_world_size: int = 1
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._score_mode = score_mode
        self._target = target
        self._target_gradient = target_gradient
        self._val_loader = val_loader
        self._val_loss_fn = val_loss_fn
        self._val_iter = iter(val_loader) if val_loader is not None else None
        self._renormalize = renormalize

        # Reference to the training HookManager; set by on_register().
        # _collect_val_gradient() checkpoints its capture state (save_state /
        # clear_state / load_state) around the secondary (val) backward pass
        # and consumes the val step's record as the scoring target.
        self._hook_manager: Any | None = None

        # Guard flag: True while the val backward is in flight.  Routes
        # on_step_end to the val-record capture instead of scoring (the val
        # pass completes a capture step of its own).
        self._in_val_pass: bool = False
        # Most recently collected val gradient (used by _resolve_target).
        self._pending_val_gradient: Gradient | None = None

        # Exposed for inspection / debugging after each step.
        self.last_scores: torch.Tensor | None = None
        self.last_dropped: list[int] = []

    # ---------------------------------------------------------------------- #
    # HookManager registration                                                #
    # ---------------------------------------------------------------------- #

    def on_register(self, hook_manager: HookManager) -> None:
        """Called by :class:`~dattri_llm.gradient.hooks.HookManager` when this
        callback is registered.

        Stores a reference to the HookManager so that
        :meth:`_collect_val_gradient` can checkpoint its capture state
        (:meth:`~dattri_llm.gradient.hooks.HookManager.save_state` /
        :meth:`~dattri_llm.gradient.hooks.HookManager.load_state`) around the
        secondary (val) backward pass.

        Args:
            hook_manager: The :class:`~dattri_llm.gradient.hooks.HookManager`
                that registered this callback.
        """
        self._hook_manager = hook_manager

    # ---------------------------------------------------------------------- #
    # Main entry points                                                        #
    # ---------------------------------------------------------------------- #

    def on_step_end(self, record: GradientRecord) -> None:
        """Score and optionally drop samples at the end of each training step.

        For ``target="val_loader"`` the fresh val target is collected first,
        right here: :meth:`on_step_end` runs with the manager's per-step state
        already reset and its lock released (see
        :meth:`HookManagerCallback.on_step_end`), so
        :meth:`_collect_val_gradient` can run its secondary backward per the
        documented reentrancy contract.  When ``_in_val_pass`` is ``True`` the
        call carries that **val** step's record; its gradient is stashed as
        the scoring target and no scoring happens.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
        """
        if self._in_val_pass:
            # Val pass completed through the manager's own hooks -- the
            # record IS the val gradient.  Stash it; do not score.
            self._pending_val_gradient = record.gradient
            return

        if self._target == "val_loader":
            self._collect_val_gradient()

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
        elif is_dist_initialized() and dist_world_size() > 1:
            # Replicated (DDP) gradients: rank-local subtraction would be off
            # by 1/world and diverge the replicas -- remove collectively, in
            # lock-step every step, like the FSDP path.
            self._remove_contributions_ddp(record, dropped)
        elif dropped:
            self._remove_contributions(record, dropped)

    # ---------------------------------------------------------------------- #
    # Target resolution                                                        #
    # ---------------------------------------------------------------------- #

    def _resolve_target(self) -> Gradient | None:
        """Return the target Gradient for this step, or None for batch mode.

        Returns:
            * ``None`` when ``target="batch"`` (caller uses the training batch
              as its own scoring reference).
            * The pre-computed :class:`Gradient` when ``target="fixed"``.
            * The val-batch :class:`Gradient` collected just before scoring
              when ``target="val_loader"``.
        """
        if self._target == "batch":
            return None
        if self._target == "fixed":
            return self._target_gradient
        # val_loader: gradient was collected by _collect_val_gradient, called
        # from on_step_end right before target resolution.
        return self._pending_val_gradient

    def _collect_val_gradient(self) -> None:
        """Sample one val batch, run forward+backward, and store its Gradient.

        Called from :meth:`on_step_end` -- the secondary (val) backward is the
        reentrant pattern that dispatch explicitly supports (per-step state
        already reset, manager lock released; see
        :meth:`HookManagerCallback.on_step_end`).  ``on_step_end`` executes
        inside an autograd hook where gradient mode is disabled, so the val
        pass runs under ``torch.enable_grad()``.

        The val gradient is captured through the training HookManager's own
        hooks: the manager's capture state is checkpointed
        (:meth:`~dattri_llm.gradient.hooks.HookManager.save_state`) and
        cleared (:meth:`~dattri_llm.gradient.hooks.HookManager.clear_state`),
        the val backward then completes a capture step of its own whose
        record -- assembled from only the val captures, with the manager's
        full configuration (layer selection, projection, device placement) --
        is delivered back to :meth:`on_step_end` and stashed as the target.
        Afterwards :meth:`~dattri_llm.gradient.hooks.HookManager.load_state`
        puts the manager's training-facing state (step counter, last
        gradient, captured inputs) back exactly.  ``param.grad`` -- holding
        the just-completed training step's gradients, which the scoring and
        removal that follow depend on -- is saved before the val pass and
        restored after it.

        Raises:
            RuntimeError: If the callback is not attached to a
                :class:`~dattri_llm.gradient.hooks.HookManager`, or the val
                backward did not complete a capture step (``val_loss_fn``
                must reach every hooked layer).  # noqa: DAR402
        """
        if self._hook_manager is None:
            raise RuntimeError(
                "target='val_loader' requires this callback to be attached to "
                "a HookManager (via HookManager(callbacks=[...]) or "
                "add_callback): the val gradient is captured through the "
                "manager's own hooks.",
            )
        # Advance the val iterator, cycling when exhausted.
        try:
            batch = next(self._val_iter)  # type: ignore[arg-type]
        except StopIteration:
            self._val_iter = iter(self._val_loader)  # type: ignore[arg-type]
            batch = next(self._val_iter)  # type: ignore[arg-type]

        # Save the training step's parameter gradients (scoring/removal read
        # and edit them right after this returns).
        saved_grads: dict[str, torch.Tensor | None] = {
            n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in self._root.named_parameters()
        }

        state = self._hook_manager.save_state()
        self._hook_manager.clear_state()
        self._pending_val_gradient = None  # staleness guard (see below)
        self._in_val_pass = True
        try:
            with torch.enable_grad():  # hooks run with grad mode disabled
                self._root.zero_grad()
                loss = self._val_loss_fn(self._root, batch)  # type: ignore[misc]
                loss.backward()
        finally:
            self._in_val_pass = False
            self._hook_manager.load_state(state)
            # Restore the training parameter gradients.
            for n, p in self._root.named_parameters():
                p.grad = saved_grads[n]

        if self._pending_val_gradient is None:
            raise RuntimeError(
                "The val backward did not complete a capture step, so no val "
                "target was collected: val_loss_fn must run the model so that "
                "the backward reaches every hooked layer.",
            )

    # ---------------------------------------------------------------------- #
    # Drop-set selection                                                       #
    # ---------------------------------------------------------------------- #

    def _select_dropped(self, scores: torch.Tensor) -> list[int]:
        """Return the list of batch indices to drop based on the configured mode.

        Args:
            scores: Float tensor of shape ``(B,)`` -- one score per sample.

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

        # "negative_bottom_fraction": keep any candidate whose score >= 0.
        return [i for i in bottom_idx if scores[i] < 0]

    # ---------------------------------------------------------------------- #
    # Per-sample influence scoring                                             #
    # ---------------------------------------------------------------------- #

    def _compute_scores(
        self,
        record: GradientRecord,
        target: Gradient | None = None,
    ) -> torch.Tensor:
        """Compute per-sample influence scores ``score[i] = <dW_i, dW_target>``.

        ``dW_target`` is the sum of the target samples' gradients (the training
        batch itself when ``target`` is ``None``).  The per-layer inner products
        are delegated to :meth:`Gradient.similarity` -- the single shared
        implementation of factorized gradient similarity -- and this method only
        sums each layer's cross-gram over the target batch and accumulates
        across layers.

        ``score_mode`` selects the :meth:`~Gradient.similarity` ``mode``
        (``"ghost"`` -> ``"auto"``, the per-layer cost-optimal routing); both
        modes are numerically identical.

        Args:
            record: Full-batch GradientRecord for this step.
            target: Optional external target Gradient.  ``None`` -> batch mode.

        Returns:
            Float tensor of shape (B,).
        """
        gradient = record.gradient
        if target is not None and target.device != gradient.device:
            # Normalise the target to the captured gradient's device (e.g. a
            # CPU-precomputed fixed target scored against on-device captures).
            target = target.to(gradient.device)
            if self._target == "fixed":
                self._target_gradient = target  # cache the moved copy
        other = gradient if target is None else target
        # "ghost" routes through the per-layer cost heuristic ("auto"): the
        # factorized and materialized cross-grams are numerically identical,
        # and a fixed factorized path costs O(B^2 T^2 (d_in+d_out)) per layer
        # -- more than the training step itself at long sequence lengths.
        mode = "materialized" if self._score_mode == "materialized" else "auto"

        # {layer: (B_layer, B_target)} cross-gram per selected scoring mode.
        per_layer = gradient.similarity(other, mode=mode)

        B = gradient.batch_size
        scores = torch.zeros(B, device=gradient.device)
        for matrix in per_layer.values():
            # Sum over target samples -> <dW_i, sum_j dW_target_j>.
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
        {"nn.GroupNorm", "nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d"},
    )

    def _remove_contributions(
        self,
        record: GradientRecord,
        dropped: list[int],
    ) -> None:
        """Subtract dropped samples' parameter-gradient contributions.

        The per-sample weight gradients are obtained from
        :func:`ops.materialize` (so every supported layer type is handled
        consistently with the rest of the library), and bias gradients are the
        summed output gradient.  The result is subtracted from each hooked
        layer's ``param.grad``; with ``renormalize=True`` the kept samples'
        contribution is rescaled in the same pass (see
        :meth:`_renorm_weighted_factors`).

        Args:
            record: Full-batch :class:`GradientRecord` for this step.
            dropped: List of batch indices to remove.
        """
        B = record.gradient.batch_size
        renorm = self._renormalize and 0 < len(dropped) < B
        for layer_name, val in record.gradient.data.items():
            if not isinstance(val, Factorized):
                continue
            # Normalise sequence-first captures so the batch axis is dim 0 before
            # we index samples / read the batch size below.
            bf = val.as_batch_first()
            # Skip layers whose gradient was summed over the batch dim during
            # the forward broadcast (e.g. wpe in GPT-2, where position_ids has
            # shape (1, T)).  Per-sample contributions cannot be isolated.
            if bf.pre_activation_grad.shape[0] < B:
                continue
            try:
                # base_layer_name: a reused layer's extra invocations are
                # recorded as virtual layers "name@2", ... -- all of them
                # resolve to (and subtract from) the same real module.
                module = self._root.get_submodule(base_layer_name(layer_name))
            except AttributeError:
                continue

            # The record carries the layer type the HookManager captured
            # under -- including a layer_types declaration for classes whose
            # name is not recognisable (e.g. a hand-rolled RMSNorm declared
            # as "nn.RMSNorm").  Re-deriving from the module class here would
            # bypass exactly that declaration and mis-materialize the layer.
            layer_type = record.gradient.layer_types[layer_name]
            if renorm:
                a_d, g_d = self._renorm_weighted_factors(bf, dropped)
            else:
                a_d = bf.activation[dropped]  # (n, ...)
                g_d = bf.pre_activation_grad[dropped]  # (n, ...)
            self._subtract_weight(module, layer_type, bf.module_kwargs, a_d, g_d)
            self._subtract_bias(module, layer_type, g_d)

    @staticmethod
    def _renorm_weighted_factors(
        bf: Factorized,
        dropped: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-batch ``(a, g)`` with ``g`` weighted for renormalized removal.

        Every contribution helper is linear in ``g`` and *subtracts* the
        per-sample sum, so weighting the batch with ``w = 1`` on dropped
        samples and ``w = -k/(B-k)`` on kept ones subtracts the dropped
        contribution and adds back ``k/(B-k)`` times the kept one in a single
        pass -- rescaling each kept sample's weight from ``1/B`` to
        ``1/(B-k)`` of a mean-reduced loss.

        Only called with ``0 < k < B`` (with ``k == B`` the plain subtraction
        already removes everything, and the empty-batch rescale is undefined).
        """
        g = bf.pre_activation_grad
        B = g.shape[0]
        k = len(dropped)
        w = torch.full((B,), -k / (B - k), dtype=torch.float32, device=g.device)
        w[dropped] = 1.0
        return bf.activation, g * w.view(-1, *([1] * (g.dim() - 1)))

    @staticmethod
    def _subtract_weight(
        module: nn.Module,
        layer_type: str,
        module_kwargs: dict | None,
        a_d: torch.Tensor,
        g_d: torch.Tensor,
    ) -> None:
        """Subtract the dropped samples' weight-gradient contribution.

        Uses :func:`ops.materialize` (``include_bias=False``) for the per-sample
        weight gradient, then maps it onto ``weight.grad``'s natural shape -- a
        vocab-row slice for embeddings, a position-sum for norms, a transpose
        for HuggingFace ``Conv1D``, and a plain ``reshape_as`` otherwise.
        """
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            return

        # (n, d_weight); sum over the dropped samples -> (d_weight,).
        contrib = ops.materialize(
            Factorized(a_d, g_d, module_kwargs),
            layer_type,
            include_bias=False,
        ).sum(0)

        if ops.is_embedding(layer_type):
            # materialize scatters into rows 0..max_token; subtract that slice.
            embed_dim = weight.shape[1]
            vocab_local = contrib.numel() // embed_dim
            weight.grad[:vocab_local] -= contrib.reshape(vocab_local, embed_dim).to(
                device=weight.grad.device,
                dtype=weight.grad.dtype,
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

        weight.grad -= contrib.reshape_as(weight.grad).to(
            device=weight.grad.device,
            dtype=weight.grad.dtype,
        )

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
        bias.grad -= contrib.to(device=bias.grad.device, dtype=bias.grad.dtype)

    # ---------------------------------------------------------------------- #
    # FSDP gradient correction                                                 #
    # ---------------------------------------------------------------------- #
    #
    # Under ``FullyShardedDataParallel`` (``use_orig_params=True``) each
    # original ``param.grad`` is a 1-D *shard* -- a contiguous slice of the
    # flattened, unsharded parameter -- holding FSDP's *averaged* gradient
    # ``(1/world_size) * sum_r G_r`` for that slice.  A dropped sample lives on a
    # single rank, but the weight elements its gradient touches may be owned by
    # a *different* rank's shard, so removal cannot be done rank-locally.
    #
    # Each rank instead computes the full-shaped gradient contribution of *its*
    # dropped samples, the contributions are summed across ranks with a single
    # ``all_reduce`` and rescaled by ``1/world_size`` to match FSDP's averaging,
    # and every rank then subtracts the slice covering the elements it owns.
    # The result equals ``(1/world_size) * sum_r G_r^kept`` shard-for-shard.
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
        shard_map: dict[int, _ShardSpec] = {}
        for module in self._model.modules():
            flat_param = getattr(module, "_flat_param", None)
            if flat_param is None:
                continue
            infos = getattr(flat_param, "_shard_param_infos", None)
            params = getattr(flat_param, "_params", None)
            shapes = getattr(flat_param, "_shapes", None)
            if infos is None or params is None or shapes is None:
                continue
            for param, info, shape in zip(params, infos, shapes, strict=True):
                shard_map[id(param)] = _ShardSpec(
                    orig_shape=tuple(shape),
                    in_shard=bool(info.in_shard),
                    start=info.intra_param_start_idx,
                    end=info.intra_param_end_idx,
                )
        self._fsdp_shard_map = shard_map
        if shard_map:
            self._fsdp_world_size = dist_world_size()

    @staticmethod
    def _collective_device() -> torch.device:
        """Device for ``all_reduce`` buffers, matched to the process-group
        backend: NCCL reduces CUDA tensors only, gloo works on CPU.
        """
        import torch.distributed as dist

        if "nccl" in str(dist.get_backend()).lower():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _remove_contributions_fsdp(
        self,
        record: GradientRecord,
        dropped: list[int],
    ) -> None:
        """Shard-aware, collective version of :meth:`_remove_contributions`.

        Runs on every rank in lock-step.  See the section comment above for the
        correctness argument.
        """
        import torch.distributed as dist

        world = self._fsdp_world_size
        use_dist = world > 1 and is_dist_initialized()

        # Skip the (expensive) contribution all-reduce when *no* rank dropped
        # anything this step.  The drop count itself is reduced first so the
        # decision is identical on every rank.
        if use_dist:
            coll_device = self._collective_device()
            count = torch.tensor([len(dropped)], dtype=torch.long, device=coll_device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            any_dropped = bool(count.item())
        else:
            any_dropped = bool(dropped)
        if not any_dropped:
            return

        shard_map = self._fsdp_shard_map

        def _shard_shape(param: torch.Tensor) -> tuple[int, ...] | None:
            spec = shard_map.get(id(param))  # type: ignore[union-attr]
            return spec.orig_shape if spec is not None else None

        entries = self._build_full_contributions(record, dropped, _shard_shape)
        if not entries:
            return

        if use_dist:
            buf = torch.cat([flat for _, flat in entries]).to(coll_device)
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

    def _remove_contributions_ddp(
        self,
        record: GradientRecord,
        dropped: list[int],
    ) -> None:
        """Replicated-gradient (DDP) collective version of
        :meth:`_remove_contributions`.

        After DDP's allreduce every rank holds the same full
        ``param.grad = (1/world) * sum_r G_r``, so the removal must subtract
        ``(1/world) * sum_r C_r`` -- the *averaged* dropped contributions of
        **all** ranks -- identically on every rank.  Same recipe as the FSDP
        path (count gate, packed ``all_reduce``, ``1/world`` rescale), except
        the subtraction covers the whole replicated tensor instead of a shard.
        Runs on every rank in lock-step; assumes DDP's default
        averaged-gradient reduction.

        **Timing**: DDP's reducer copies each local gradient into its
        communication bucket the moment it is produced and overwrites
        ``param.grad`` with the averaged result only when the backward pass
        finalises -- *after* this method runs (``on_step_end`` fires from
        hooks inside the backward).  An immediate subtraction would be
        clobbered by that write-back, so the removal is queued on the autograd
        engine: queued callbacks run FIFO at finalisation, and DDP's own
        write-back callback was queued earlier (at its first gradient hook),
        so the removal lands on the final averaged gradients.
        """

        def _apply() -> None:
            import torch.distributed as dist

            world = dist_world_size()
            coll_device = self._collective_device()

            # Skip the contribution all-reduce when *no* rank dropped
            # anything; the count reduce keeps the decision identical on
            # every rank.
            count = torch.tensor([len(dropped)], dtype=torch.long, device=coll_device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            if not count.item():
                return

            entries = self._build_full_contributions(
                record,
                dropped,
                lambda param: tuple(param.shape),
            )
            if not entries:
                return

            buf = torch.cat([flat for _, flat in entries]).to(coll_device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf /= world
            offset = 0
            for param, flat in entries:
                n = flat.numel()
                if param.grad is not None:
                    seg = buf[offset : offset + n].reshape(param.grad.shape)
                    param.grad -= seg.to(
                        device=param.grad.device,
                        dtype=param.grad.dtype,
                    )
                offset += n

        # After-finalization queue: a plain backward-end callback may run
        # *before* DDP's averaged-gradient write-back (order among callbacks
        # queued during the backward is not observable); this variant is
        # guaranteed to run after it.
        if not queue_after_backward_finalization(_apply):
            # No engine queue on this build: subtract immediately (pre-2.x
            # fallback; may race DDP's write-back).
            _apply()

    def _build_full_contributions(
        self,
        record: GradientRecord,
        dropped: list[int],
        full_shape_of: Callable[[torch.Tensor], tuple[int, ...] | None],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return ``[(param, full_flat_contribution)]`` for every hooked param.

        Each contribution is the *full* (unsharded) parameter-gradient of this
        rank's dropped samples, flattened in the parameter's natural C-order
        (zeros when this rank dropped nothing).  With ``renormalize=True`` it
        additionally carries this rank's kept-sample rescale term (see
        :meth:`_renorm_weighted_factors`) -- the factor is rank-local, matching
        the average-of-local-means gradient semantics.  The entry list --
        params and their full sizes -- depends only on model structure and is
        therefore identical across ranks, which keeps the packed
        ``all_reduce`` aligned.

        *full_shape_of* resolves a parameter's full shape, or ``None`` to skip
        it: the FSDP path reads the shard map's ``orig_shape`` (a shard's own
        ``param.shape`` is a meaningless 1-D slice), the DDP path reads
        ``param.shape`` directly (replicated params keep their true shape).
        """
        B = record.gradient.batch_size
        renorm = self._renormalize and 0 < len(dropped) < B
        entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_name, val in sorted(record.gradient.data.items()):
            if not isinstance(val, Factorized):
                continue
            # Normalise sequence-first captures to batch-first before indexing.
            bf = val.as_batch_first()
            if bf.pre_activation_grad.shape[0] < B:
                continue
            try:
                # Virtual invocation layers ("name@2", ...) resolve to the
                # same real module as their base name.
                module = self._root.get_submodule(base_layer_name(layer_name))
            except AttributeError:
                continue
            # Captured (possibly declared-via-layer_types) type; never
            # re-derive from the module class (see _remove_contributions).
            layer_type = record.gradient.layer_types[layer_name]
            if renorm:
                a_d, g_d = self._renorm_weighted_factors(bf, dropped)
            elif dropped:
                a_d = bf.activation[dropped]
                g_d = bf.pre_activation_grad[dropped]
            else:
                a_d = g_d = None

            weight = getattr(module, "weight", None)
            weight_shape = full_shape_of(weight) if weight is not None else None
            if weight_shape is not None:
                full_numel = math.prod(weight_shape)
                # Contributions live on the captured gradients' device so the
                # materialize below is copy-free; identical on every rank.
                flat = torch.zeros(
                    full_numel,
                    dtype=torch.float32,
                    device=record.gradient.device,
                )
                if dropped:
                    contrib, off = self._weight_contrib_natural(
                        layer_type,
                        bf.module_kwargs,
                        a_d,
                        g_d,
                        weight_shape,
                    )
                    flat[off : off + contrib.numel()] = contrib.float()
                entries.append((weight, flat))

            bias = getattr(module, "bias", None)
            bias_shape = full_shape_of(bias) if bias is not None else None
            if bias_shape is not None:
                channels = math.prod(bias_shape)
                flat = torch.zeros(
                    channels,
                    dtype=torch.float32,
                    device=record.gradient.device,
                )
                if dropped:
                    flat = self._bias_contrib_natural(layer_type, g_d, channels).float()
                entries.append((bias, flat))
        return entries

    @staticmethod
    def _weight_contrib_natural(
        layer_type: str,
        module_kwargs: dict | None,
        a_d: torch.Tensor,
        g_d: torch.Tensor,
        full_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, int]:
        """Full weight-gradient contribution as a 1-D tensor in the parameter's
        natural C-order, plus the flat offset at which it begins.

        Mirrors the per-layer-type reshaping in :meth:`_subtract_weight` but
        uses the *unsharded* ``full_shape`` (FSDP shards expose only a 1-D
        slice, so ``weight.shape`` is unusable).  The offset is always 0 now
        that embedding materialization covers the full ``num_embeddings``
        width; it is kept for the partial-coverage case should one return.
        """
        contrib = ops.materialize(
            Factorized(a_d, g_d, module_kwargs),
            layer_type,
            include_bias=False,
        ).sum(0)
        if ops.is_embedding(layer_type):
            # materialize scatters into all num_embeddings rows, flattened
            # row-major -- exactly the flattened weight layout.
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
        param.grad -= seg.to(device=param.grad.device, dtype=param.grad.dtype)
