"""The HookManager: step-completion tracking, gradient assembly, and lifecycle."""

from __future__ import annotations

import operator
import threading
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import nn

from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.hooks.config import (
    LINEAR_IO,
    PARAM_GRAD,
    HookManagerConfig,
    _resolve_projector,
    resolve_hook_assignments,
)
from dattri_llm.gradient.hooks.hooks import (
    register_linear_io_hooks,
    register_linear_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.ops import PARAM_GRAD_TYPES, is_embedding
from dattri_llm.utils.autograd import queue_backward_end_callback
from dattri_llm.utils.hashing import hash_batch

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from dattri_llm.gradient.callbacks import HookManagerCallback


# Resolved once at import: the probe runs on every hooked layer's forward
# fire, so the per-call attribute lookup is hoisted out of the hot path.
_GRAPH_TASK_ID_PROBE = getattr(torch._C, "_current_graph_task_id", None)


def _current_graph_task_id() -> int:
    """Id of the autograd graph task executing on this thread, or ``-1``.

    A non-negative id inside a *forward* hook means the forward is a
    gradient-checkpointing recomputation (it runs during the backward).
    Wraps the private ``torch._C`` probe so older builds degrade to ``-1``
    (recompute detection off -- checkpointing was unsupported there anyway).
    """
    return _GRAPH_TASK_ID_PROBE() if _GRAPH_TASK_ID_PROBE is not None else -1


# --------------------------------------------------------------------------- #
# Hook manager                                                                 #
# --------------------------------------------------------------------------- #


class HookManager:
    """Collect per-sample or per-batch gradients via forward/backward hooks.

    Hooks are registered at construction (via :meth:`register`) and remain
    active until :meth:`remove` is called; a removed manager can be re-armed
    with another :meth:`register` call.  Collection is gated by
    :meth:`collect`, which can also take down the hooks itself when the
    context exits (``deregister_on_exit=True``) -- the safe form for a
    throwaway manager that is never referenced again::

        with HookManager(model, callbacks=[...]).collect(deregister_on_exit=True):
            trainer.train()

    .. warning::
        Gradient checkpointing (either ``use_reentrant`` variant) and
        ``nn.DataParallel`` are each supported, but their **combination** is
        untested territory: checkpoint recomputation would interleave with
        DataParallel's concurrent per-replica hook threads, and no test pins
        that interaction.  Prefer DDP when training with checkpointing.

    .. warning::
        Sample identifiers cover the **last** model forward of a step.  When
        the model is invoked more than once before a single backward (e.g. a
        hand-rolled Siamese / DPO-style two-forward loss), every invocation's
        gradients are captured (the extra calls appear as virtual ``name@k``
        layers), but the record's ``input_hash`` is derived from the final
        call's inputs only -- the earlier calls' samples are mislabelled.
        The same holds for a *grad-enabled* forward with no backward between
        steps: its captures are correctly discarded, but it overwrites the
        inputs the next record is labelled with (``no_grad`` / eval passes
        are safe).  Layer-level reuse *inside* one model forward is
        unaffected -- the model-level input capture fires once there.

    .. warning::
        A weight used in more than one place -- HF-style **tied parameters**
        (e.g. ``embed_tokens`` and ``lm_head`` sharing one weight) or
        repeated invocations of one module (the virtual ``name@k`` layers)
        -- is scored as if each use site held an independent weight,
        following the convention of existing attribution implementations.
        For per-sample contributions ``g_i`` (site 1) and ``h_i`` (site 2) of
        a shared weight, the true similarity ``<g_i + h_i, g_j + h_j>``
        includes the cross-site terms ``<g_i, h_j> + <h_i, g_j>``; per-layer
        scoring sums the sites independently and drops them -- the same
        block-diagonal treatment K-FAC applies across layers.  Both sites'
        factors are captured in full, so the cross terms remain computable
        downstream if exactness is ever required.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        config: :class:`HookManagerConfig` giving the per-layer ``hook_types``
            assignment (optionally extended by the ``linear_io`` / ``param_grad``
            regex selectors).  Defaults to ``HookManagerConfig()`` (linear_io on
            every capable layer, with a param_grad fallback for the rest).
        callbacks: List of :class:`HookManagerCallback` objects.
        sample_id_key: How each sample's **identifier** (the record's
            ``input_hash``) is derived.  ``None`` (default) content-hashes the
            captured model inputs via
            :func:`~dattri_llm.utils.hashing.hash_batch`.  When set, the
            identifiers are read directly from that input field instead: a
            ``str`` names a forward kwarg (e.g. ``"idx"`` for a dataset id
            column), an ``int`` indexes the positional forward arguments.
            The field must be present in every step's inputs with one value
            per sample (missing key or wrong length raises); values are
            stringified, so an ``idx`` column of ``[32, 42]`` yields
            identifiers ``["32", "42"]``.  The chosen key is stamped on every
            emitted record (:attr:`GradientRecord.sample_id_key`).
        non_batch_first_layers: Optional set/list of fully-qualified layer names
            whose captured activations are **sequence-first** (``(T, B, ...)``)
            rather than the default batch-first (``(B, T, ...)``) -- e.g. layers
            internal to a sequence-first model such as MusicTransformer.  The
            assembled :class:`~dattri_llm.gradient.gradient.Factorized` for these
            layers is tagged ``batch_first=False`` so the gradient machinery reads
            their batch axis from dim 1.  Names not actually hooked are ignored.
        offload_to_cpu: When ``True``, every buffered capture (raw factors,
            projected results, and ``param_grad`` tensors) is moved to CPU as
            it is captured, so the assembled
            :class:`~dattri_llm.gradient.gradient.Gradient` is CPU-resident
            and no device memory is held across the step.  Default ``False``:
            captures stay on their own (training) device -- no transfer
            happens, at the cost of keeping one step's captures in device
            memory until the next step completes.  A no-op either way when
            training on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        config: HookManagerConfig | None = None,
        callbacks: list[HookManagerCallback] | None = None,
        sample_id_key: str | int | None = None,
        non_batch_first_layers: Iterable[str] | None = None,
        offload_to_cpu: bool = False,
    ) -> None:
        self._model = model
        self._callbacks: list[HookManagerCallback] = callbacks or []
        self._sample_id_key = sample_id_key
        self._offload_to_cpu = offload_to_cpu
        self._non_batch_first_layers: set[str] = (
            set(non_batch_first_layers) if non_batch_first_layers is not None else set()
        )
        # Layers already warned about as broadcast (batch-collapsed) gradients,
        # so the warning fires at most once per layer.
        self._warned_broadcast: set[str] = set()

        self._config = config if config is not None else HookManagerConfig()

        self._step_count: int = 0
        self._collecting: bool = False

        # Single-slot cache holding the assembled gradient of the most recently
        # *completed* step.  The per-layer buffers are cleared as soon as a step
        # completes, so this is what :meth:`get_gradient` returns afterwards.
        # Only ever holds one step's gradient at a time (see _finalize_step).
        self._last_gradient: Gradient | None = None

        self._seen_bwd: set[str] = set()
        self._bwd_replica_counts: dict[str, int] = {}
        # This step's participation roster: layers with a grad-enabled
        # forward fire (``_fwd_fires > 0``), maintained incrementally so
        # completion checks are O(1).  Kept in sync wherever ``_fwd_fires``
        # changes: _dispatch_layer_forward (add), _reset_layer_buffers
        # (clear), _reconcile_unmatched_forwards (drop fully-orphaned).
        self._active_layers: set[str] = set()
        self._step_lock = threading.Lock()

        # Step-completion uses two composite barriers:
        #
        # ``_bwd_done`` -- True once *both* sub-conditions hold:
        #   (a) All MLP-layer full backward hooks have fired.
        #       Ensures ``_grad_parts`` buffers are populated for every
        #       hooked layer.
        #   (b) All MLP-layer trainable-parameter hooks have fired.
        #       Ensures ``weight.grad`` (and ``bias.grad``) are populated.
        #
        #   Sub-condition (b) is tracked via ``_mlp_param_hook_count``.
        #   Registering both types covers the two possible PyTorch orderings:
        #
        #   Case A -- module input *requires grad* (normal LLM training):
        #     param.register_hook fires first  -> weight.grad set
        #     register_full_backward_hook fires second -> _grad_parts set
        #     -> (b) done before (a); step triggered by (a)
        #
        #   Case B -- module input does *not* require grad (e.g. raw float
        #     tensor, no embedding):
        #     register_full_backward_hook fires first (PyTorch early-fire quirk)
        #     param.register_hook fires second -> weight.grad set
        #     -> (a) done before (b); step triggered by (b)
        #
        # ``_grad_done`` -- True once all user-specified ``param_grad`` hooks
        #   have fired (only relevant when ``param_grad`` layers are
        #   registered; starts True otherwise).
        self._bwd_done: bool = True
        self._mlp_param_hook_count: int = 0  # fires toward sub-cond (b)
        self._n_mlp_params: int = 0  # target for sub-cond (b)
        self._grad_done: bool = True

        # End-of-backward barrier for sub-cond (b).  Under FSDP the
        # per-parameter post-accumulate-grad hooks never fire (grads flow
        # through the flattened FlatParameter) and FSDP writes the (sharded)
        # ``param.grad`` back to each original parameter only *after* the whole
        # backward pass completes.  To cover this, every step queues a callback
        # on the autograd engine that fires once backward is fully done -- the
        # point at which all ``param.grad`` are guaranteed ready.  For non-FSDP
        # models the per-parameter hooks complete the step earlier (during
        # backward), so this callback is a harmless no-op.  ``_mlp_params_ready``
        # is the callback's signal; ``_backward_end_scheduled`` is the live
        # token for the in-flight step (guards against double-queuing and
        # against a stale callback leaking into the next step).
        self._mlp_params_ready: bool = False
        self._backward_end_scheduled: bool = False
        # True once a forward hook fired during an active backward this step
        # -- i.e. a gradient-checkpointing recomputation.  ``_outer_task_id``
        # is the autograd task id of the *outermost* backward (recorded at
        # the first queue of the end-of-backward callback); the callback only
        # acts when that task ends, never at a reentrant segment's nested
        # sub-backward end (see _dispatch_layer_forward / _on_backward_end).
        self._saw_recompute: bool = False
        self._outer_task_id: int = -1

        self._last_inputs: dict[str, torch.Tensor] = {}

        # Hook state -- populated by register(), emptied by remove().
        self._registered: bool = False
        self._model_fwd_handle: torch.utils.hooks.RemovableHandle | None = None
        self._has_linear_io: bool = False
        self._buffers: dict = {}
        self._handles: list = []
        self._mlp_weight_handles: list = []  # post-accumulate hooks for sub-cond (b)
        self._n_layers: int = 0
        self._has_param_grad: bool = False
        self._param_buffers: dict = {}
        self._param_handles: list = []
        self._n_params_hooked: int = 0
        self._param_hook_count: int = 0

        self.register()

        # Notify callbacks of this HookManager so they can reference it
        # (e.g. to checkpoint capture state around a secondary pass via
        # save_state / clear_state / load_state).
        for cb in self._callbacks:
            if hasattr(cb, "on_register"):
                cb.on_register(self)

    # ---------------------------------------------------------------------- #
    # Model-level pre-forward hook                                            #
    # ---------------------------------------------------------------------- #

    def _capture_model_input(
        self,
        _module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        if not self._collecting:
            return
        if torch.is_grad_enabled():
            # A new capture step is beginning, so the cached last-step
            # gradient is about to go stale: release it now.
            self._last_gradient = None
        captured: dict[str, torch.Tensor] = {
            k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        if not captured and args:
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    captured[f"_arg{i}"] = a
        # The sample-id field is captured whatever its type (an id column may
        # arrive as a plain list) and whichever calling convention delivered
        # it -- an int key indexes the positional arguments.
        key = self._sample_id_key
        if key is not None:
            if isinstance(key, int):
                if key < len(args):
                    captured[f"_arg{key}"] = args[key]
            elif key in kwargs:
                captured[key] = kwargs[key]
        self._last_inputs = captured

    # ---------------------------------------------------------------------- #
    # Per-layer hook dispatchers                                              #
    # ---------------------------------------------------------------------- #

    def _dispatch_layer_forward(
        self,
        layer_name: str,
        activation: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,
    ) -> None:
        # Maintain the step's participation roster incrementally: O(1) here
        # instead of an O(n_layers) rescan on every backward fire in
        # _bwd_hooks_done.  The unlocked membership pre-check makes repeat
        # fires free; set.add is idempotent, so the worst concurrent-replica
        # race is a harmless double add.
        if layer_name not in self._active_layers:
            with self._step_lock:
                self._active_layers.add(layer_name)
        # A forward hook firing while an autograd graph task is active is a
        # checkpoint *recomputation* (both use_reentrant variants), and it
        # executes inside the OUTERMOST backward's task -- the reentrant
        # variant's nested sub-backwards have not started yet.  Record that
        # task id (the true end-of-backward identity) and queue the
        # end-of-backward callback *here*, so it attaches to the outer task:
        # a callback queued from a backward hook inside a reentrant
        # sub-backward would fire at that segment's end, mid-step (see
        # _on_backward_end's nested-end guard).
        if self._collecting:
            task_id = _current_graph_task_id()
            if task_id != -1:
                with self._step_lock:
                    self._saw_recompute = True
                    if self._outer_task_id == -1:
                        self._outer_task_id = task_id
                    if (
                        self._n_mlp_params > 0
                        and not self._backward_end_scheduled
                        and queue_backward_end_callback(self._on_backward_end)
                    ):
                        self._backward_end_scheduled = True
        for cb in self._callbacks:
            cb.on_layer_forward(layer_name, activation, layer_type, module_kwargs)

    def _check_step_bwd_complete(
        self,
        layer_name: str,
        grad_output: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,
    ) -> None:
        """Fired by each MLP layer's full backward hook.

        Dispatches per-layer callbacks and, once all layers have reported,
        checks whether the composite ``_bwd_done`` condition is met.
        """
        for cb in self._callbacks:
            cb.on_layer_backward(layer_name, grad_output, layer_type, module_kwargs)

        if not self._collecting:
            return

        record = None
        with self._step_lock:
            # We are inside the backward pass here -- a valid place to queue
            # an end-of-backward callback.  Queue it once per step as the
            # FSDP-safe satisfier of sub-cond (b) (see ``__init__``), and
            # remember the queuing task's id: without checkpointing this hook
            # runs in the outermost (only) backward task, which is the end
            # the callback must act at.
            if (
                self._n_mlp_params > 0
                and not self._backward_end_scheduled
                and queue_backward_end_callback(self._on_backward_end)
            ):
                self._backward_end_scheduled = True
                if self._outer_task_id == -1:
                    self._outer_task_id = _current_graph_task_id()
            self._bwd_replica_counts[layer_name] = (
                self._bwd_replica_counts.get(layer_name, 0) + 1
            )
            # A layer's backward is complete when it has fired once per
            # *observed* grad-enabled forward this step (the forward pass is
            # fully done before any backward hook runs, so the target is
            # final here).  Deriving the target from observation -- instead
            # of predicting it from ``len(device_ids)`` at construction --
            # keeps the count right when DataParallel's scatter uses fewer
            # replicas than devices (trailing short batch), when the model is
            # wrapped after this manager is built, and when a layer is
            # invoked several times per step (weight tying).
            if (
                self._bwd_replica_counts[layer_name]
                >= self._buffers[layer_name]["_fwd_fires"]
            ):
                self._seen_bwd.add(layer_name)
            if self._bwd_hooks_done():
                record = self._check_mlp_done()
        if record is not None:
            self._dispatch_step_end(record)

    def _on_backward_end(self) -> None:
        """Fired once the backward pass fully completes (FSDP-safe barrier).

        By this point every ``param.grad`` is written -- including FSDP's
        (sharded) write-back to the original parameters -- so sub-cond (b) is
        satisfied and any callback that reads ``param.grad`` in ``on_step_end``
        sees ready gradients.  The ``_backward_end_scheduled`` guard ensures a
        callback that fires *after* its step already completed (the common
        non-FSDP case, where per-parameter hooks finished the step earlier) is
        ignored rather than leaking ``_mlp_params_ready`` into the next step.
        """
        if not self._collecting:
            return
        record = None
        with self._step_lock:
            if not self._backward_end_scheduled:
                return
            # Nested-end guard: under reentrant checkpointing each segment
            # runs its own sub-backward, whose end also fires engine
            # callbacks -- acting there would fake sub-condition (b) mid-step
            # (the not-yet-recomputed segments' params haven't fired).  The
            # true end is the task the callback was queued from (the
            # outermost task; see the queue sites).
            if (
                self._outer_task_id != -1
                and _current_graph_task_id() != self._outer_task_id
            ):
                return
            self._mlp_params_ready = True
            self._reconcile_unmatched_forwards()
            record = self._check_mlp_done()
        if record is not None:
            self._dispatch_step_end(record)

    def _reconcile_unmatched_forwards(self) -> None:
        """Discard forward captures whose backward never fired (backward is
        over -- an unmatched forward is now a definitive fact, not a guess).

        Two legitimate sources: non-reentrant **gradient checkpointing**
        (both the original forward and the recomputation fire grad-enabled,
        but only one backward comes -- the recomputed capture wins the LIFO
        match and the stale original is dropped here), and grad-enabled
        forwards whose output never reached the loss (detached branches).
        After dropping, ``_fwd_fires`` is deflated to the matched count and
        the layer's completion status re-evaluated, so steps whose counts
        could never equalise mid-backward complete here.

        Must be called with ``_step_lock`` held, from the end-of-backward
        engine callback only.
        """
        for layer_name, buf in self._buffers.items():
            with buf["_lock"]:
                had_orphans = any(buf["_unmatched"].values()) or any(
                    buf["_device_id"].values(),
                )
                if had_orphans:
                    self._drop_orphan_forwards(buf)
                buf["_unmatched"] = {}
                buf["_device_id"] = {}
                matched = len(buf["_pair_pos"])
                if buf["_fwd_fires"] != matched:
                    buf["_fwd_fires"] = matched
            if matched == 0:
                # Every forward was orphaned (detached branch): the layer did
                # not participate in this step after all.
                self._active_layers.discard(layer_name)
            elif self._bwd_replica_counts.get(layer_name, 0) >= matched:
                self._seen_bwd.add(layer_name)

    @staticmethod
    def _drop_orphan_forwards(buf: dict) -> None:
        """Drop a buffer's unmatched forward captures; re-rank pair positions.

        Must be called with the buffer's lock held.  The matched per-device
        invocation positions come from the pairs actually formed; positions
        are then re-ranked densely so orphans vanish from the numbering (a
        checkpoint recompute's surviving pair becomes invocation 0, keeping
        the layer's plain name).
        """
        matched_pos: dict[int, list[int]] = {}
        for dev, pos in buf["_pair_pos"]:
            matched_pos.setdefault(dev, []).append(pos)
        remap = {
            dev: {p: r for r, p in enumerate(sorted(ps))}
            for dev, ps in matched_pos.items()
        }
        if buf["_proj_kw"] is None:
            # Raw path: act parts append in forward order, so an entry's
            # per-device position is its rank among its device's entries;
            # keep only matched ones.
            seen: dict[int, int] = {}
            kept = []
            for dev, part in buf["_act_parts"]:
                pos = seen.get(dev, 0)
                seen[dev] = pos + 1
                if pos in remap.get(dev, {}):
                    kept.append((dev, part))
            buf["_act_parts"] = kept
        # Both paths: re-rank pair positions densely over the matched set.
        buf["_pair_pos"] = [(dev, remap[dev][pos]) for dev, pos in buf["_pair_pos"]]

    def _check_step_mlp_param_complete(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by :func:`register_linear_param_hooks` after each linear param's
        grad is accumulated.  Once all MLP param hooks have reported, checks
        whether the composite ``_bwd_done`` condition is met.
        """
        if not self._collecting:
            return
        record = None
        with self._step_lock:
            self._mlp_param_hook_count += 1
            if self._mlp_param_hook_count >= self._n_mlp_params:
                record = self._check_mlp_done()
        if record is not None:
            self._dispatch_step_end(record)

    def _bwd_hooks_done(self) -> bool:
        """True when every layer that participated in this step's forward has
        completed its backward fires.

        Registered layers that this step's execution path never reached
        (``_fwd_fires == 0`` -- conditional branches, unused submodules) are
        not required: the forward pass finishes before any backward hook
        runs, so a zero forward count here is definitive, and requiring such
        layers would stall the step forever.  At least one layer must have
        participated -- a step is never completed on no data.

        Must be called with ``_step_lock`` held.
        """
        if self._n_layers == 0:
            return True  # no linear_io hooks; sub-condition (a) is vacuous
        n_active = len(self._active_layers)
        return n_active > 0 and len(self._seen_bwd) == n_active

    def _check_mlp_done(self) -> GradientRecord | None:
        """Set ``_bwd_done`` and attempt step completion when both MLP
        sub-conditions (all participating bwd hooks fired, MLP param grads
        ready) hold.

        Must be called with ``_step_lock`` held.  Returns the finalized record
        when this call completed the step (the caller dispatches it *after*
        releasing the lock), else ``None``.
        """
        bwd_hooks_done = self._bwd_hooks_done()
        # Sub-condition (b): all MLP param grads are ready.  Satisfied when no
        # MLP params exist, when every per-parameter post-accumulate hook has
        # fired (non-FSDP), or when the FSDP end-of-backward callback has fired
        # (``_mlp_params_ready``).
        mlp_params_done = (
            self._n_mlp_params == 0
            or self._mlp_params_ready
            or self._mlp_param_hook_count >= self._n_mlp_params
        )
        if bwd_hooks_done and mlp_params_done:
            self._bwd_done = True
            return self._check_step_complete()
        return None

    def _check_step_grad_complete(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by each param's grad hook; marks grad done once all param
        hooks have reported.
        """
        if not self._collecting:
            return
        record = None
        with self._step_lock:
            self._param_hook_count += 1
            if self._param_hook_count >= self._n_params_hooked:
                self._param_hook_count = 0
                self._grad_done = True
                record = self._check_step_complete()
        if record is not None:
            self._dispatch_step_end(record)

    def _check_step_complete(self) -> GradientRecord | None:
        """Finalize the step iff both bwd and grad are done.

        Must be called with ``_step_lock`` held.  Returns the finalized record
        for the caller to dispatch after releasing the lock, else ``None``.
        """
        if self._bwd_done and self._grad_done:
            return self._finalize_step()
        return None

    # ---------------------------------------------------------------------- #
    # Step completion                                                          #
    # ---------------------------------------------------------------------- #

    def _extract_sample_ids(self, expected: int) -> list[str]:
        """Read this step's identifiers from the ``sample_id_key`` input field.

        No silent fallback: a missing field or a length that disagrees with
        the step's batch size raises, because falling back to content hashing
        on a typo would silently change every sample's identity.

        Args:
            expected: The step's batch size (one identifier per sample).

        Returns:
            The stringified per-sample identifier list, in batch order.

        Raises:
            KeyError: If the field is absent from the captured model inputs.
            ValueError: If the field does not yield ``expected`` values.
        """
        key = self._sample_id_key
        field = f"_arg{key}" if isinstance(key, int) else key
        if field not in self._last_inputs:
            raise KeyError(
                f"sample_id_key={key!r} was not found among the captured "
                f"model inputs {sorted(self._last_inputs)}; cannot assign "
                "sample identities.",
            )
        value = self._last_inputs[field]
        ids = (
            value.reshape(-1).tolist()
            if isinstance(value, torch.Tensor)
            else list(value)
        )
        if len(ids) != expected:
            raise ValueError(
                f"sample_id_key={key!r} yielded {len(ids)} identifiers but "
                f"the step's batch size is {expected}.",
            )
        return [str(i) for i in ids]

    def _get_input_batch_size(self) -> int:
        """First captured input tensor's leading dim -- a **fallback only**.

        The step batch size is normally read off the assembled gradient (see
        :meth:`_resolve_step_batch_size`), which is ground truth.  This probe
        covers the cases with no usable per-sample axis in the gradient; it is
        wrong whenever a broadcast field (e.g. ``position_ids`` of shape
        ``(1, T)``) happens to arrive first in the kwargs order.
        """
        for v in self._last_inputs.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                return v.shape[0]
        return 1

    def _resolve_step_batch_size(self, gradient: Gradient) -> int:
        """The number of samples this step's record is labelled with.

        Primary source: the assembled gradient's batch size -- ground truth
        read off the actual per-sample capture, immune to misleading input
        fields (a ``(1, T)`` ``position_ids`` broadcast arriving first in the
        kwargs order, a ``(B+1,)`` ``cu_seqlens``).  It is trusted whenever
        the captured inputs *corroborate* it -- some field actually has that
        leading dimension (or length), so identities can be sliced from it.

        Fallback: the first input tensor's leading dim, for the cases where
        the gradient cannot speak for the step's sample count -- a pure
        ``param_grad`` capture (no per-sample axis at all), and a
        multi-invocation capture whose per-layer batch axis is per
        *invocation* (e.g. a chunked scatter running the submodule once per
        batch slice), smaller than the step's input batch.
        """
        try:
            batch_size = gradient.batch_size
        except ValueError:
            return self._get_input_batch_size()
        for v in self._last_inputs.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch_size:
                return batch_size
            if isinstance(v, (list, tuple)) and len(v) == batch_size:
                return batch_size
        return self._get_input_batch_size()

    def _finalize_step(self) -> GradientRecord:
        """Assemble the completed step and reset every per-step state slot.

        Must be called with ``_step_lock`` held -- the assembly/reset must be
        atomic against concurrent hook threads (DataParallel replicas,
        single-process multi-device models).  The returned record is
        dispatched by the caller **after releasing the lock**, so
        ``on_step_end`` runs with the manager in a clean, unlocked "fresh
        step" state -- a callback may safely re-enter the manager (state
        checkpointing, even a secondary backward; see
        :meth:`HookManagerCallback.on_step_end`).
        """
        # Normally already None (released when this step's forward began, see
        # _capture_model_input); kept as a guard for forwards that bypass the
        # root pre-hook (e.g. a submodule invoked directly) so we never
        # transiently hold two full step gradients while assembling.
        self._last_gradient = None
        gradient = self._assemble_gradient()
        step = self._step_count

        self._step_count += 1
        self._seen_bwd.clear()
        self._bwd_replica_counts.clear()
        self._mlp_param_hook_count = 0
        self._param_hook_count = 0
        # Reset completion flags for the next step.
        self._mlp_params_ready = False
        self._backward_end_scheduled = False
        self._saw_recompute = False
        self._outer_task_id = -1
        self._bwd_done = not self._has_linear_io
        self._grad_done = not self._has_param_grad
        # The per-layer buffers are now cleared; the assembled ``gradient`` owns
        # the only copy of this step's tensors and becomes the last-step cache.
        self._reset_layer_buffers()
        self._reset_param_buffers()
        self._last_gradient = gradient

        batch_size = self._resolve_step_batch_size(gradient)
        if self._sample_id_key is not None:
            input_hash = self._extract_sample_ids(batch_size)
        else:
            input_hash = hash_batch(self._last_inputs, batch_size)
        return GradientRecord(
            step=step,
            input_hash=input_hash,
            gradient=gradient,
            sample_id_key=self._sample_id_key,
        )

    def _dispatch_step_end(self, record: GradientRecord) -> None:
        """Deliver a finalized record to every callback, outside the lock.

        Running user code while holding the non-reentrant ``_step_lock`` would
        turn any callback-triggered backward into a silent same-thread
        deadlock (the inner pass's hooks re-acquire the lock).  All per-step
        state was already reset by :meth:`_finalize_step`, so a reentrant
        backward here simply completes a capture step of its own.
        """
        for cb in self._callbacks:
            cb.on_step_end(record)

    @staticmethod
    def _concat_parts(parts: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Concatenate buffered ``(dev_idx, tensor)`` parts along the batch axis.

        Parts are ordered by source device index.  Under DataParallel with
        on-device buffering (``offload_to_cpu=False``) the replicas' parts live
        on different devices; they are gathered to the first part's device so
        the concatenation is well-defined (a no-op in the single-device and
        CPU-offload cases).
        """
        ordered = [t for _, t in sorted(parts, key=operator.itemgetter(0))]
        if len(ordered) == 1:
            # The common (non-DataParallel) case: torch.cat would copy the
            # whole tensor for nothing.
            return ordered[0]
        first_device = ordered[0].device
        if any(t.device != first_device for t in ordered[1:]):
            ordered = [t.to(first_device) for t in ordered]
        return torch.cat(ordered, dim=0)

    @staticmethod
    def _split_invocations(buf: dict) -> list[tuple[list, list, list]]:
        """Regroup a layer's matched parts by invocation.

        Returns one ``(act_parts, grad_parts, proj_parts)`` triple per
        per-device invocation position, in forward-invocation order.  The
        single-invocation case (the overwhelmingly common one) returns the
        buffers as-is.  For multi-fire layers the grouping undoes the fire
        *order*: raw activations buffer in forward order while gradients
        buffer in backward (reverse) order -- ``_pair_pos``, recorded when
        each backward LIFO-matched its forward, carries the pairing.
        """
        pair_pos = buf["_pair_pos"]
        n_inv = max((pos for _, pos in pair_pos), default=0) + 1
        if n_inv <= 1:
            return [(buf["_act_parts"], buf["_grad_parts"], buf["_proj_parts"])]
        acts: list[list] = [[] for _ in range(n_inv)]
        grads: list[list] = [[] for _ in range(n_inv)]
        projs: list[list] = [[] for _ in range(n_inv)]
        if buf["_proj_kw"] is None:
            # Raw path: an act entry's invocation is its rank among its own
            # device's entries (forward append order); grads carry pair_pos.
            seen: dict[int, int] = {}
            for dev, part in buf["_act_parts"]:
                pos = seen.get(dev, 0)
                seen[dev] = pos + 1
                acts[pos].append((dev, part))
            for (dev, part), (_, pos) in zip(
                buf["_grad_parts"],
                pair_pos,
                strict=True,
            ):
                grads[pos].append((dev, part))
        elif buf["_proj_kw"].get("style", "logra_factorized") == "logra_factorized":
            # Projected factorized: both factors were appended together at
            # match time, aligned with pair_pos.
            for (dev, part), (_, pos) in zip(
                buf["_act_parts"],
                pair_pos,
                strict=True,
            ):
                acts[pos].append((dev, part))
            for (dev, part), (_, pos) in zip(
                buf["_grad_parts"],
                pair_pos,
                strict=True,
            ):
                grads[pos].append((dev, part))
        else:
            for (dev, part), (_, pos) in zip(
                buf["_proj_parts"],
                pair_pos,
                strict=True,
            ):
                projs[pos].append((dev, part))
        return list(zip(acts, grads, projs, strict=True))

    def _assemble_gradient(self) -> Gradient:
        data: dict = {}
        representation: dict = {}
        layer_types: dict = {}
        indexing: dict = {}

        for layer_name, buf in self._buffers.items():
            # Registered but not on this step's execution path (conditional
            # branch, unused submodule): nothing was captured, and step
            # completion did not require it (see _bwd_hooks_done) -- the
            # layer is simply absent from this step's record.
            if buf["_fwd_fires"] == 0:
                continue
            batch_first = layer_name not in self._non_batch_first_layers
            proj_kw = buf["_proj_kw"]

            # A layer invoked more than once in a step (custom module reuse,
            # RNN unrolls, Siamese towers) is recorded as independent virtual
            # layers "name", "name@2", ... -- one per matched invocation.
            # This mirrors how parameter-tied modules are already treated:
            # per-layer scoring sums the invocations' contributions
            # (cross-invocation terms of the shared weight are not
            # represented, consistent with the layer-block-diagonal treatment
            # throughout).  Single-invocation layers keep their plain name.
            for inv_k, (act_parts, grad_parts, proj_parts) in enumerate(
                self._split_invocations(buf),
            ):
                out_name = layer_name if inv_k == 0 else f"{layer_name}@{inv_k + 1}"

                # Materialized styles ("materialized"/TRAK and
                # "logra_materialized"): the projected per-sample gradient was
                # assembled into _proj_parts at capture time.
                proj_style = (
                    proj_kw.get("style", "logra_factorized")
                    if proj_kw is not None
                    else "logra_factorized"
                )
                if proj_kw is not None and proj_style != "logra_factorized":
                    if not proj_parts:
                        raise RuntimeError(
                            f"Layer '{layer_name}' has no buffered data. "
                            "Ensure backward() was called inside the collect() "
                            "context.",
                        )
                    data[out_name] = self._concat_parts(proj_parts)
                    representation[out_name] = "materialized"
                    layer_types[out_name] = buf["_class_name"]
                    indexing[out_name] = "batch"
                    continue

                if not act_parts or not grad_parts:
                    raise RuntimeError(
                        f"Layer '{layer_name}' has no buffered data. "
                        "Ensure backward() was called inside the collect() context.",
                    )
                a = self._concat_parts(act_parts)
                g = self._concat_parts(grad_parts)

                # Factorized (LoGRA) projection: parts already hold the
                # projected, final factors (module_kwargs=None -> no
                # re-preprocessing).
                if proj_kw is not None:
                    data[out_name] = Factorized(
                        activation=a,
                        pre_activation_grad=g,
                        module_kwargs=None,
                        batch_first=batch_first,
                    )
                    representation[out_name] = "factorized"
                    layer_types[out_name] = "nn.Linear"
                    tokens = a.shape[1 if batch_first else 0]  # 1 => 2-D origin
                    indexing[out_name] = "batch_token" if tokens > 1 else "batch"
                    continue

                # Un-projected raw factorized capture.
                # A positional embedding fed an *unbatched* index tensor -- e.g.
                # nanoGPT's ``pos = arange(T)`` (shape ``(T,)``) added to every
                # sample -- is captured with no batch dim.  Add a length-1 batch
                # axis so it validates and materialises as a single broadcast row
                # (its gradient is already summed over the batch by the
                # broadcast add).
                if a.ndim == 1 and is_embedding(buf["_class_name"]):
                    a = a.unsqueeze(0)
                    g = g.unsqueeze(0)
                data[out_name] = Factorized(
                    activation=a,
                    pre_activation_grad=g,
                    module_kwargs=buf["_module_kwargs"],
                    batch_first=batch_first,
                )
                representation[out_name] = "factorized"
                layer_types[out_name] = buf["_class_name"]
                indexing[out_name] = (
                    "batch_token"
                    if a.ndim >= 3 or not a.is_floating_point()
                    else "batch"
                )

        for layer_name, buf in self._param_buffers.items():
            for pname, grad in buf.items():
                if grad is None:
                    continue
                key = f"{layer_name}.{pname}"
                data[key] = grad
                representation[key] = "materialized"
                layer_types[key] = PARAM_GRAD_TYPES
                indexing[key] = "batch"  # param_grad tensors have no token dim

        if not data:
            raise RuntimeError(
                "No gradient data assembled. "
                "Ensure backward() was called inside the collect() context.",
            )

        # Broadcast warning runs on the assembled (projected or raw) factors.
        self._warn_broadcast_layers(data, layer_types)

        return Gradient(
            representation=representation,
            data=data,
            layer_types=layer_types,
            indexing=indexing,
        )

    def _warn_broadcast_layers(self, data: dict, layer_types: dict) -> None:
        """Warn (once per layer) about broadcast / batch-collapsed gradients.

        A factorized layer whose batch dim is 1 while the step batch is larger --
        e.g. a positional embedding added to every sample -- carries a gradient
        that was *summed over the batch*, so it is **not** a per-sample gradient.
        Downstream per-sample attribution treats it as a single shared row, which
        is rarely what the user wants; surface it so they can exclude the layer.

        Operates on the raw ``data``/``layer_types`` dicts (before any projection)
        so it can run without materializing a full :class:`Gradient`.
        """
        # Step batch = largest per-layer sample dim (matches Gradient.batch_size);
        # param_grad tensors have no sample axis and are excluded.
        batch = 0
        for name, val in data.items():
            if layer_types.get(name) == PARAM_GRAD_TYPES:
                continue
            if isinstance(val, Factorized):
                batch = max(batch, val.activation.shape[0 if val.batch_first else 1])
            else:
                batch = max(batch, val.shape[0])
        if batch <= 1:
            return
        for name, val in data.items():
            if not isinstance(val, Factorized):
                continue
            layer_batch = val.activation.shape[0 if val.batch_first else 1]
            if layer_batch == 1 and name not in self._warned_broadcast:
                self._warned_broadcast.add(name)
                warnings.warn(
                    f"Layer '{name}' produced a broadcast (batch-collapsed) "
                    f"gradient: batch size 1 while the step batch is {batch} "
                    "(e.g. a positional embedding added to every sample).  Its "
                    "gradient is summed over the batch and is NOT per-sample; "
                    "per-sample attribution will treat it as a single shared row. "
                    f"Consider excluding '{name}' from gradient collection (e.g. "
                    "via HookManagerConfig hook selection or the attributor's "
                    "``layer_name``).",
                    stacklevel=3,
                )

    def _reset_layer_buffers(self) -> None:
        self._active_layers.clear()
        for buf in self._buffers.values():
            buf["activation"] = None
            buf["grad_output"] = None
            buf["_act_parts"] = []
            buf["_grad_parts"] = []
            buf["_proj_parts"] = []
            buf["_device_id"] = {}
            buf["_fwd_fires"] = 0
            buf["_unmatched"] = {}
            buf["_pair_pos"] = []

    def _reset_param_buffers(self) -> None:
        for buf in self._param_buffers.values():
            for pname in buf:
                buf[pname] = None

    # ---------------------------------------------------------------------- #
    # Capture-state checkpointing                                              #
    # ---------------------------------------------------------------------- #

    def save_state(self) -> dict:
        """Checkpoint the manager's capture state; undo with :meth:`load_state`.

        Together with :meth:`clear_state` and :meth:`load_state`, this lets a
        callback run a *secondary* forward+backward pass through the
        already-registered hooks without disturbing the surrounding training
        step -- e.g.
        :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`
        collecting a fresh validation target mid-step::

            state = hm.save_state()   # checkpoint the in-flight capture
            hm.clear_state()          # clean slate for the secondary pass
            try:
                # The secondary pass completes a capture step of its own:
                # its record (assembled from only the secondary captures)
                # is dispatched to on_step_end like any other.
                val_loss_fn(model, val_batch).backward()
            finally:
                hm.load_state(state)  # put the training step back exactly

        The secondary step's *external* effects (its ``on_step_end`` dispatch)
        are deliberately not undone -- the callback that triggered it consumes
        the record; sibling callbacks observe one extra record per secondary
        pass.

        Snapshots everything a completed (or in-flight) step touches: the
        per-layer capture buffers (raw and projected fields), the ``param_grad``
        buffers, the captured model inputs, the step counter, the last-step
        gradient cache, and the step-completion counters/flags.  Copies are
        shallow -- tensors are shared with the live buffers, which is safe
        because captures are only ever appended, never mutated in place.

        Returns:
            An opaque state dict to pass to :meth:`load_state`.
        """
        return {
            "layer_buffers": {
                name: {
                    "_act_parts": list(buf["_act_parts"]),
                    "_grad_parts": list(buf["_grad_parts"]),
                    "_proj_parts": list(buf["_proj_parts"]),
                    "_device_id": {k: list(v) for k, v in buf["_device_id"].items()},
                    "_fwd_fires": buf["_fwd_fires"],
                    "_unmatched": {k: list(v) for k, v in buf["_unmatched"].items()},
                    "_pair_pos": list(buf["_pair_pos"]),
                    "activation": buf["activation"],
                    "grad_output": buf["grad_output"],
                }
                for name, buf in self._buffers.items()
            },
            "param_buffers": {
                name: dict(buf) for name, buf in self._param_buffers.items()
            },
            "last_inputs": dict(self._last_inputs),
            "step_count": self._step_count,
            "last_gradient": self._last_gradient,
            "seen_bwd": set(self._seen_bwd),
            "active_layers": set(self._active_layers),
            "bwd_replica_counts": dict(self._bwd_replica_counts),
            "mlp_param_hook_count": self._mlp_param_hook_count,
            "param_hook_count": self._param_hook_count,
            "bwd_done": self._bwd_done,
            "grad_done": self._grad_done,
            "mlp_params_ready": self._mlp_params_ready,
            "backward_end_scheduled": self._backward_end_scheduled,
            "saw_recompute": self._saw_recompute,
            "outer_task_id": self._outer_task_id,
        }

    def clear_state(self) -> None:
        """Empty every capture buffer (a clean slate for a secondary pass).

        Only the buffers are cleared; counters, the step count, and the
        last-gradient cache are left alone -- :meth:`load_state` restores
        those wholesale.
        """
        self._reset_layer_buffers()
        self._reset_param_buffers()

    def load_state(self, state: dict) -> None:
        """Restore a checkpoint taken by :meth:`save_state`.

        Puts back the capture buffers, model inputs, step counter, last-step
        gradient cache, and completion counters exactly as they were, so a
        training step interrupted by a secondary pass resumes as if the
        secondary pass never ran.

        Args:
            state: The dict returned by :meth:`save_state`.
        """
        saved_layer = state["layer_buffers"]
        for name, buf in self._buffers.items():
            s = saved_layer.get(name, {})
            buf["_act_parts"] = s.get("_act_parts", [])
            buf["_grad_parts"] = s.get("_grad_parts", [])
            buf["_proj_parts"] = s.get("_proj_parts", [])
            buf["_device_id"] = s.get("_device_id", {})
            buf["_fwd_fires"] = s.get("_fwd_fires", 0)
            buf["_unmatched"] = s.get("_unmatched", {})
            buf["_pair_pos"] = s.get("_pair_pos", [])
            buf["activation"] = s.get("activation")
            buf["grad_output"] = s.get("grad_output")
        saved_param = state["param_buffers"]
        for name, buf in self._param_buffers.items():
            buf.clear()
            buf.update(saved_param.get(name, {}))
        self._last_inputs = state["last_inputs"]
        self._last_gradient = state["last_gradient"]
        with self._step_lock:
            self._step_count = state["step_count"]
            self._seen_bwd = state["seen_bwd"]
            self._active_layers = state.get("active_layers", set())
            self._bwd_replica_counts = state["bwd_replica_counts"]
            self._mlp_param_hook_count = state["mlp_param_hook_count"]
            self._param_hook_count = state["param_hook_count"]
            self._bwd_done = state["bwd_done"]
            self._grad_done = state["grad_done"]
            self._mlp_params_ready = state["mlp_params_ready"]
            self._backward_end_scheduled = state["backward_end_scheduled"]
            self._saw_recompute = state.get("saw_recompute", False)
            self._outer_task_id = state.get("outer_task_id", -1)

    # ---------------------------------------------------------------------- #
    # Primary API: collect()                                                   #
    # ---------------------------------------------------------------------- #

    @contextmanager
    def collect(
        self,
        deregister_on_exit: bool = False,
    ) -> Generator[HookManager, None, None]:
        """Enable gradient collection for the duration of the context.

        Works with any training loop::

            offload = OffloadCallback(offload_interval=32, file_manager=manager)
            collector = HookManager(model, callbacks=[offload])

            with collector.collect():
                trainer.train()

        Entering the context re-registers the hooks if they were previously
        removed, so a manager can alternate ``remove()`` / ``collect()`` (or
        chain ``deregister_on_exit=True`` contexts) freely.

        Args:
            deregister_on_exit: When ``True``, remove all hooks once the
                context exits (after the callbacks' ``on_context_end`` flush),
                leaving no trace on the model.  This makes the throwaway
                one-liner safe -- without it, an unreferenced manager's hooks
                stay registered on the model forever::

                    with HookManager(model, callbacks=[...]).collect(
                        deregister_on_exit=True,
                    ):
                        trainer.train()

                The last assembled gradient survives the teardown, so
                :meth:`get_gradient` still works after the context.  Defaults
                to ``False``: the hooks stay up until :meth:`remove`.

        Yields:
            This :class:`HookManager` instance.
        """
        self.register()
        self._reset_layer_buffers()
        self._reset_param_buffers()
        # Drop any cached gradient from a prior context before collecting anew.
        self._last_gradient = None
        self._param_hook_count = 0
        self._mlp_params_ready = False
        self._backward_end_scheduled = False
        self._seen_bwd.clear()
        self._bwd_replica_counts.clear()
        self._last_inputs = {}
        self._collecting = True
        try:
            yield self
        finally:
            self._collecting = False
            for cb in self._callbacks:
                cb.on_context_end()
            if deregister_on_exit:
                # remove() clears the cached gradient; preserve it so
                # get_gradient() remains valid after an auto-teardown context.
                last_gradient = self._last_gradient
                self.remove()
                self._last_gradient = last_gradient

    # ---------------------------------------------------------------------- #
    # Lifecycle and introspection                                              #
    # ---------------------------------------------------------------------- #

    def register(self) -> None:
        """Register all hooks on the model.

        Called automatically at construction.  Call it again to re-arm a
        manager whose hooks were taken down by :meth:`remove` (or by a
        ``collect(deregister_on_exit=True)`` context); no-op while the hooks
        are already registered.
        """
        if self._registered:
            return
        model = self._model
        root = getattr(model, "module", model)
        self._model_fwd_handle = root.register_forward_pre_hook(
            self._capture_model_input,
            with_kwargs=True,
        )

        # Resolve which hook family each layer is assigned to (one family per
        # layer), then register concrete layer-name sets.
        assignment = resolve_hook_assignments(root, self._config)
        linear_io_layers = {n for n, t in assignment.items() if t == LINEAR_IO}
        param_grad_layers = {n for n, t in assignment.items() if t == PARAM_GRAD}
        self._has_linear_io = bool(linear_io_layers)
        self._has_param_grad = bool(param_grad_layers)

        self._bwd_done = True
        self._n_mlp_params = 0
        self._n_layers = 0
        if self._has_linear_io:
            self._bwd_done = False
            self._buffers, self._handles = register_linear_io_hooks(
                model,
                layer_names=linear_io_layers,
                on_layer_forward=self._dispatch_layer_forward,
                on_layer_backward=self._check_step_bwd_complete,
                type_overrides=self._config.layer_types,
                kwargs_overrides=self._config.module_kwargs,
                projection=self._config.projection,
                projector=(
                    _resolve_projector(self._config.projector)
                    if self._config.projection is not None
                    else None
                ),
                offload_to_cpu=self._offload_to_cpu,
            )
            self._n_layers = len(self._buffers)
            self._n_mlp_params, self._mlp_weight_handles = register_linear_param_hooks(
                model,
                layer_names=linear_io_layers,
                on_linear_param_grad=self._check_step_mlp_param_complete,
            )

        self._grad_done = True
        self._n_params_hooked = 0
        if self._has_param_grad:
            self._grad_done = False
            self._param_buffers, self._param_handles = register_param_grad_hooks(
                model,
                layer_names=param_grad_layers,
                on_param_grad=self._check_step_grad_complete,
                offload_to_cpu=self._offload_to_cpu,
            )
            self._n_params_hooked = len(self._param_handles)

        # Warn about manual layer_types that never took effect because the named
        # layer was not hooked (e.g. a typo, or a layer excluded by the hook
        # selection).  Overrides only apply to factorized linear_io layers.
        if self._config.layer_types:
            hooked = set(self._buffers) | set(self._param_buffers)
            unused = sorted(set(self._config.layer_types) - hooked)
            if unused:
                warnings.warn(
                    "HookManagerConfig.layer_types designated a type for layers "
                    f"that were not hooked: {unused}. These overrides had no "
                    "effect.",
                    stacklevel=2,
                )

        # The sequence-first flag only applies to factorized (linear_io) layers;
        # warn about names that are not hooked that way (typo / excluded / a
        # param_grad layer, which is always batch-first).
        if self._non_batch_first_layers:
            unused = sorted(self._non_batch_first_layers - set(self._buffers))
            if unused:
                warnings.warn(
                    "non_batch_first_layers names layers not hooked with factorized "
                    f"(linear_io) capture; the flag had no effect: {unused}.",
                    stacklevel=2,
                )

        self._registered = True

    def remove(self) -> None:
        """Remove all registered hooks and clear all buffer dicts.

        Idempotent: safe to call on a manager whose hooks are already down.
        :meth:`register` re-arms the manager afterwards.
        """
        if not self._registered:
            return
        self._model_fwd_handle.remove()
        self._model_fwd_handle = None
        remove_hooks(self._handles)
        self._handles = []
        self._buffers.clear()
        remove_hooks(self._mlp_weight_handles)
        self._mlp_weight_handles = []
        remove_hooks(self._param_handles)
        self._param_handles = []
        self._param_buffers.clear()
        self._last_gradient = None
        self._registered = False

    def add_callback(self, callback: HookManagerCallback) -> None:
        """Attach a callback after construction and run its ``on_register``.

        Useful when a callback can only be built *after* the manager -- most
        notably :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`
        under FSDP, where the manager is created on the unwrapped model (so its
        hooks survive wrapping) but the callback needs the FSDP-wrapped module
        to discover the parameter shard layout.
        """
        self._callbacks.append(callback)
        if hasattr(callback, "on_register"):
            callback.on_register(self)

    @property
    def sample_id_key(self) -> str | int | None:
        """The identifier scheme this manager stamps on records: the input
        field sample ids are read from, or ``None`` for content hashing.
        """
        return self._sample_id_key

    @property
    def layer_names(self) -> list[str]:
        """Fully-qualified names of all hooked linear-IO layers."""
        return list(self._buffers.keys())

    @property
    def layer_name(self) -> list[str]:
        """The layer set an attributor scores: the hooked per-sample (linear-IO)
        layers.  Layer selection is decided here, at capture, via
        :class:`HookManagerConfig` -- attributors read it back through this
        property (e.g. for :class:`AttributionScore` metadata) instead of taking
        their own ``layer_name`` argument.
        """
        return list(self._buffers.keys())

    @property
    def param_layer_names(self) -> list[str]:
        """Fully-qualified names of all param-grad hooked layers."""
        return list(self._param_buffers.keys())

    @property
    def steps_collected(self) -> int:
        """Total number of batch steps collected since construction."""
        return self._step_count

    def reset_steps(self, to: int = 0) -> None:
        """Align the capture-step counter to a caller-chosen baseline.

        ``_step_count`` (stamped on each :attr:`GradientRecord.step` and exposed
        via :attr:`steps_collected`) is monotonic for the manager's lifetime by
        default.  A caller that restarts collection from a fresh logical baseline
        -- e.g. the live streamer beginning a new pass -- can reset it so
        ``record.step`` is pass-local and aligns with the caller's own per-pass
        index.  When several drivers share one manager (e.g. a train and a test
        :class:`~dattri_llm.gradient.streaming.GradientStreamer` riding one hook
        set, interleaved by ``loop_over_test``), each re-aligns the counter to
        its *own* next index (``to``) before every step, so records are always
        stamped with the owning driver's pass-local step.  Only the label
        counter is set; in-flight per-step buffers and completion flags are left
        untouched (callers align between steps, never mid-backward).

        Args:
            to: The value the next completed capture step is counted from.
        """
        with self._step_lock:
            self._step_count = to

    def get_gradient(self) -> Gradient:
        """Return the :class:`~dattri_llm.gradient.Gradient` of the most recently
        completed step::

            with collector.collect():
                loss = model(**inputs).loss
                loss.backward()
                grad = collector.get_gradient()

        Step completion happens inside the backward hooks, so by the time
        ``backward()`` returns the per-layer buffers have already been assembled
        and cleared.  The assembled gradient is retained in a single-slot cache
        (the *last-step gradient*), which is what this method returns.

        The cache lives only in the gap **between** steps: it is released the
        moment the next collected (grad-enabled) forward begins, so a full
        step of captures is never pinned through a training step's memory
        peak.  Read the gradient after ``backward()`` and before the next
        forward; ``no_grad`` forwards (e.g. evaluation) leave it intact.

        Returns:
            Gradient: The assembled gradient, with ``layer_types`` populated
            from the hooked module class names.

        Raises:
            RuntimeError: If no step has completed yet (no backward has run
                inside the collect context).
        """
        if self._last_gradient is None:
            raise RuntimeError(
                "No gradient available: no step has completed yet. "
                "Ensure backward() was called inside the collect() context "
                "before calling get_gradient().",
            )
        return self._last_gradient
