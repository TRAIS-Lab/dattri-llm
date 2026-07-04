"""The HookManager: step-completion tracking, gradient assembly, and lifecycle."""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Generator, Iterable, Optional

import torch
import torch.nn as nn

from dattri_llm.gradient.callbacks import HookManagerCallback
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.hooks.config import (
    LINEAR_IO,
    PARAM_GRAD,
    HookManagerConfig,
    _resolve_projector,
    resolve_hook_assignments,
)
from dattri_llm.gradient.hooks.hooks import (
    _queue_backward_end_callback,
    register_linear_io_hooks,
    register_linear_param_hooks,
    register_param_grad_hooks,
    remove_hooks,
)
from dattri_llm.gradient.ops import PARAM_GRAD_TYPES, is_embedding
from dattri_llm.utils.hashing import hash_batch


# --------------------------------------------------------------------------- #
# Hook manager                                                                 #
# --------------------------------------------------------------------------- #


class HookManager:
    """Collect per-sample or per-batch gradients via forward/backward hooks.

    Hooks are registered at construction and remain active until
    :meth:`remove` is called.  Collection is gated by :meth:`collect`.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        config: :class:`HookManagerConfig` giving the per-layer ``hook_types``
            assignment (optionally extended by the ``linear_io`` / ``param_grad``
            regex selectors).  Defaults to ``HookManagerConfig()`` (linear_io on
            every capable layer, with a param_grad fallback for the rest).
        callbacks: List of :class:`HookManagerCallback` objects.
        sample_id_key: Key in the model's forward kwargs used as a hint for
            batch size detection.  Defaults to ``"input_ids"``.
        non_batch_first_layers: Optional set/list of fully-qualified layer names
            whose captured activations are **sequence-first** (``(T, B, ...)``)
            rather than the default batch-first (``(B, T, ...)``) — e.g. layers
            internal to a sequence-first model such as MusicTransformer.  The
            assembled :class:`~dattri_llm.gradient.gradient.Factorized` for these
            layers is tagged ``batch_first=False`` so the gradient machinery reads
            their batch axis from dim 1.  Names not actually hooked are ignored.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[HookManagerConfig] = None,
        callbacks: Optional[list[HookManagerCallback]] = None,
        sample_id_key: str = "input_ids",
        non_batch_first_layers: Optional[Iterable[str]] = None,
    ) -> None:
        self._model = model
        self._callbacks: list[HookManagerCallback] = callbacks or []
        self._sample_id_key = sample_id_key
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
        # Only ever holds one step's gradient at a time (see _on_step_complete).
        self._last_gradient: Optional[Gradient] = None

        self._seen_bwd: set[str] = set()
        self._bwd_replica_counts: dict[str, int] = {}
        self._step_lock = threading.Lock()

        # Step-completion uses two composite barriers:
        #
        # ``_bwd_done`` — True once *both* sub-conditions hold:
        #   (a) All MLP-layer full backward hooks have fired.
        #       Ensures ``_grad_parts`` buffers are populated for every
        #       hooked layer.
        #   (b) All MLP-layer trainable-parameter hooks have fired.
        #       Ensures ``weight.grad`` (and ``bias.grad``) are populated.
        #
        #   Sub-condition (b) is tracked via ``_mlp_param_hook_count``.
        #   Registering both types covers the two possible PyTorch orderings:
        #
        #   Case A — module input *requires grad* (normal LLM training):
        #     param.register_hook fires first  → weight.grad set
        #     register_full_backward_hook fires second → _grad_parts set
        #     → (b) done before (a); step triggered by (a)
        #
        #   Case B — module input does *not* require grad (e.g. raw float
        #     tensor, no embedding):
        #     register_full_backward_hook fires first (PyTorch early-fire quirk)
        #     param.register_hook fires second → weight.grad set
        #     → (a) done before (b); step triggered by (b)
        #
        # ``_grad_done`` — True once all user-specified ``param_grad`` hooks
        #   have fired (only relevant when ``param_grad`` layers are
        #   registered; starts True otherwise).
        self._bwd_done: bool = True
        self._mlp_param_hook_count: int = 0   # fires toward sub-cond (b)
        self._n_mlp_params: int = 0            # target for sub-cond (b)
        self._grad_done: bool = True

        # End-of-backward barrier for sub-cond (b).  Under FSDP the
        # per-parameter post-accumulate-grad hooks never fire (grads flow
        # through the flattened FlatParameter) and FSDP writes the (sharded)
        # ``param.grad`` back to each original parameter only *after* the whole
        # backward pass completes.  To cover this, every step queues a callback
        # on the autograd engine that fires once backward is fully done — the
        # point at which all ``param.grad`` are guaranteed ready.  For non-FSDP
        # models the per-parameter hooks complete the step earlier (during
        # backward), so this callback is a harmless no-op.  ``_mlp_params_ready``
        # is the callback's signal; ``_backward_end_scheduled`` is the live
        # token for the in-flight step (guards against double-queuing and
        # against a stale callback leaking into the next step).
        self._mlp_params_ready: bool = False
        self._backward_end_scheduled: bool = False

        root = getattr(model, "module", model)
        self._n_replicas: int = (
            len(model.device_ids)
            if isinstance(model, nn.DataParallel)
            else 1
        )

        self._last_inputs: dict[str, torch.Tensor] = {}
        self._model_fwd_handle = root.register_forward_pre_hook(
            self._capture_model_input, with_kwargs=True
        )

        # Resolve which hook family each layer is assigned to (one family per
        # layer), then register concrete layer-name sets.
        assignment = resolve_hook_assignments(root, self._config)
        linear_io_layers = {n for n, t in assignment.items() if t == LINEAR_IO}
        param_grad_layers = {n for n, t in assignment.items() if t == PARAM_GRAD}
        self._has_linear_io = bool(linear_io_layers)
        self._has_param_grad = bool(param_grad_layers)

        self._buffers: dict = {}
        self._handles: list = []
        self._mlp_weight_handles: list = []   # post-accumulate hooks for sub-cond (b)
        self._n_layers: int = 0
        if self._has_linear_io:
            self._bwd_done = False
            self._buffers, self._handles = register_linear_io_hooks(
                model,
                layer_names=linear_io_layers,
                on_layer_forward=self._dispatch_layer_forward,
                on_layer_backward=self._check_step_bwd_complete,
                type_overrides=self._config.layer_types,
                projection=self._config.projection,
                projector=(
                    _resolve_projector(self._config.projector)
                    if self._config.projection is not None else None
                ),
            )
            self._n_layers = len(self._buffers)
            self._n_mlp_params, self._mlp_weight_handles = register_linear_param_hooks(
                model,
                layer_names=linear_io_layers,
                on_linear_param_grad=self._check_step_mlp_param_complete,
            )

        self._param_buffers: dict = {}
        self._param_handles: list = []
        self._n_params_hooked: int = 0
        self._param_hook_count: int = 0
        if self._has_param_grad:
            self._grad_done = False
            self._param_buffers, self._param_handles = register_param_grad_hooks(
                model,
                layer_names=param_grad_layers,
                on_param_grad=self._check_step_grad_complete,
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

        # Notify callbacks of this HookManager so they can reference it
        # (e.g. to call pause() during a secondary pass).
        for cb in self._callbacks:
            if hasattr(cb, "on_register"):
                cb.on_register(self)

    # ---------------------------------------------------------------------- #
    # Model-level pre-forward hook                                            #
    # ---------------------------------------------------------------------- #

    def _capture_model_input(self, _module, args: tuple, kwargs: dict) -> None:
        if not self._collecting:
            return
        captured: dict[str, torch.Tensor] = {
            k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        if not captured and args:
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    captured[f"_arg{i}"] = a
        self._last_inputs = captured

    # ---------------------------------------------------------------------- #
    # Per-layer hook dispatchers                                              #
    # ---------------------------------------------------------------------- #

    def _dispatch_layer_forward(self, layer_name: str, activation: torch.Tensor) -> None:
        for cb in self._callbacks:
            cb.on_layer_forward(layer_name, activation)

    def _check_step_bwd_complete(self, layer_name: str, grad_output: torch.Tensor) -> None:
        """Fired by each MLP layer's full backward hook.

        Dispatches per-layer callbacks and, once all layers have reported,
        checks whether the composite ``_bwd_done`` condition is met.
        """
        for cb in self._callbacks:
            cb.on_layer_backward(layer_name, grad_output)

        if not self._collecting:
            return

        with self._step_lock:
            # We are inside the backward pass here — the only valid place to
            # queue an end-of-backward callback.  Queue it once per step as the
            # FSDP-safe satisfier of sub-cond (b) (see ``__init__``).
            if self._n_mlp_params > 0 and not self._backward_end_scheduled:
                if _queue_backward_end_callback(self._on_backward_end):
                    self._backward_end_scheduled = True
            self._bwd_replica_counts[layer_name] = (
                self._bwd_replica_counts.get(layer_name, 0) + 1
            )
            if self._bwd_replica_counts[layer_name] >= self._n_replicas:
                self._seen_bwd.add(layer_name)
            if len(self._seen_bwd) == self._n_layers:
                self._check_mlp_done()

    def _on_backward_end(self) -> None:
        """Fired once the backward pass fully completes (FSDP-safe barrier).

        By this point every ``param.grad`` is written — including FSDP's
        (sharded) write-back to the original parameters — so sub-cond (b) is
        satisfied and any callback that reads ``param.grad`` in ``on_step_end``
        sees ready gradients.  The ``_backward_end_scheduled`` guard ensures a
        callback that fires *after* its step already completed (the common
        non-FSDP case, where per-parameter hooks finished the step earlier) is
        ignored rather than leaking ``_mlp_params_ready`` into the next step.
        """
        if not self._collecting:
            return
        with self._step_lock:
            if not self._backward_end_scheduled:
                return
            self._mlp_params_ready = True
            self._check_mlp_done()

    def _check_step_mlp_param_complete(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by :func:`register_linear_param_hooks` after each linear param's
        grad is accumulated.  Once all MLP param hooks have reported, checks
        whether the composite ``_bwd_done`` condition is met."""
        if not self._collecting:
            return
        with self._step_lock:
            self._mlp_param_hook_count += 1
            if self._mlp_param_hook_count >= self._n_mlp_params:
                self._check_mlp_done()

    def _check_mlp_done(self) -> None:
        """Set ``_bwd_done`` and attempt step completion when both MLP
        sub-conditions (all bwd hooks fired, all MLP param hooks fired) hold.

        Must be called with ``_step_lock`` held.
        """
        bwd_hooks_done = len(self._seen_bwd) == self._n_layers
        # Sub-condition (b): all MLP param grads are ready.  Satisfied when no
        # MLP params exist, when every per-parameter post-accumulate hook has
        # fired (non-FSDP), or when the FSDP end-of-backward callback has fired
        # (``_mlp_params_ready``).
        mlp_params_done = (self._n_mlp_params == 0
                           or self._mlp_params_ready
                           or self._mlp_param_hook_count >= self._n_mlp_params)
        if bwd_hooks_done and mlp_params_done:
            self._bwd_done = True
            self._check_step_complete()

    def _check_step_grad_complete(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by each param's grad hook; marks grad done once all param
        hooks have reported."""
        if not self._collecting:
            return
        with self._step_lock:
            self._param_hook_count += 1
            if self._param_hook_count >= self._n_params_hooked:
                self._param_hook_count = 0
                self._grad_done = True
                self._check_step_complete()

    def _check_step_complete(self) -> None:
        """Trigger _on_step_complete only when both bwd and grad are done.

        Must be called with ``_step_lock`` held.
        """
        if self._bwd_done and self._grad_done:
            self._on_step_complete()

    # ---------------------------------------------------------------------- #
    # Step completion                                                          #
    # ---------------------------------------------------------------------- #

    def _get_input_batch_size(self) -> int:
        for v in self._last_inputs.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                return v.shape[0]
        return 1

    def _on_step_complete(self) -> None:
        # Drop the previous step's cached gradient *before* assembling the new
        # one so we never transiently hold two full step gradients in memory.
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
        self._bwd_done = not self._has_linear_io
        self._grad_done = not self._has_param_grad
        # The per-layer buffers are now cleared; the assembled ``gradient`` owns
        # the only copy of this step's tensors and becomes the last-step cache.
        self._reset_layer_buffers()
        self._reset_param_buffers()
        self._last_gradient = gradient

        batch_size = self._get_input_batch_size()
        input_hash = hash_batch(self._last_inputs, batch_size)
        record = GradientRecord(step=step, input_hash=input_hash, gradient=gradient)
        for cb in self._callbacks:
            cb.on_step_end(record)

    def _assemble_gradient(self) -> Gradient:
        data: dict = {}
        representation: dict = {}
        layer_types: dict = {}
        indexing: dict = {}

        for layer_name, buf in self._buffers.items():
            batch_first = layer_name not in self._non_batch_first_layers
            proj_kw = buf["_proj_kw"]

            # Materialized (TRAK) projection: the projected per-sample gradient was
            # already assembled per micro-batch into _proj_parts at capture time.
            if proj_kw is not None and not proj_kw.get("factorize", True):
                parts = buf["_proj_parts"]
                if not parts:
                    raise RuntimeError(
                        f"Layer '{layer_name}' has no buffered data. "
                        "Ensure backward() was called inside the collect() context."
                    )
                data[layer_name] = torch.cat(
                    [t for _, t in sorted(parts, key=lambda x: x[0])], dim=0
                )
                representation[layer_name] = "materialized"
                layer_types[layer_name] = buf["_class_name"]
                indexing[layer_name] = "batch"
                continue

            act_parts = buf["_act_parts"]
            grad_parts = buf["_grad_parts"]
            if not act_parts or not grad_parts:
                raise RuntimeError(
                    f"Layer '{layer_name}' has no buffered data. "
                    "Ensure backward() was called inside the collect() context."
                )
            a = torch.cat([t for _, t in sorted(act_parts,  key=lambda x: x[0])], dim=0)
            g = torch.cat([t for _, t in sorted(grad_parts, key=lambda x: x[0])], dim=0)

            # Factorized (LoGRA) projection: _act_parts/_grad_parts already hold the
            # projected, final factors (module_kwargs=None → no re-preprocessing).
            if proj_kw is not None:
                data[layer_name] = Factorized(activation=a, pre_activation_grad=g,
                                              module_kwargs=None, batch_first=batch_first)
                representation[layer_name] = "factorized"
                layer_types[layer_name] = "nn.Linear"
                tokens = a.shape[1 if batch_first else 0]   # 1 ⇒ 2-D-origin layer
                indexing[layer_name] = "batch_token" if tokens > 1 else "batch"
                continue

            # Un-projected raw factorized capture (original behaviour).
            # A positional embedding fed an *unbatched* index tensor — e.g.
            # nanoGPT's ``pos = arange(T)`` (shape ``(T,)``) added to every
            # sample — is captured with no batch dim.  Add a length-1 batch axis
            # so it validates and materialises as a single broadcast row (its
            # gradient is already summed over the batch by the broadcast add).
            if a.ndim == 1 and is_embedding(buf["_class_name"]):
                a = a.unsqueeze(0)
                g = g.unsqueeze(0)
            data[layer_name] = Factorized(activation=a, pre_activation_grad=g,
                                          module_kwargs=buf["_module_kwargs"],
                                          batch_first=batch_first)
            representation[layer_name] = "factorized"
            layer_types[layer_name] = buf["_class_name"]
            indexing[layer_name] = (
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
                indexing[key] = "batch"   # param_grad tensors have no token dim

        if not data:
            raise RuntimeError(
                "No gradient data assembled. "
                "Ensure backward() was called inside the collect() context."
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

        A factorized layer whose batch dim is 1 while the step batch is larger —
        e.g. a positional embedding added to every sample — carries a gradient
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
                    "`layer_name`).",
                    stacklevel=3,
                )

    def _reset_layer_buffers(self) -> None:
        for buf in self._buffers.values():
            buf["activation"] = None
            buf["grad_output"] = None
            buf["_act_parts"] = []
            buf["_grad_parts"] = []
            buf["_proj_parts"] = []
            buf["_device_id"] = {}

    def _reset_param_buffers(self) -> None:
        for buf in self._param_buffers.values():
            for pname in buf:
                buf[pname] = None

    # ---------------------------------------------------------------------- #
    # Primary API: collect()                                                   #
    # ---------------------------------------------------------------------- #

    @contextmanager
    def collect(self) -> Generator["HookManager", None, None]:
        """Enable gradient collection for the duration of the context.

        Works with any training loop::

            offload = OffloadCallback(offload_interval=32, file_manager=manager)
            collector = HookManager(model, callbacks=[offload])

            with collector.collect():
                trainer.train()

        Yields:
            This :class:`HookManager` instance.
        """
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

    @contextmanager
    def pause(self) -> Generator["HookManager", None, None]:
        """Temporarily suspend step-completion tracking within ``collect()``.

        Use this when running a secondary forward+backward pass (e.g. a val
        pass triggered from ``on_layer_forward``) so that the secondary pass
        is not counted as a training step and does not contaminate the
        in-flight training-pass buffers.

        Behaviour during the pause:
        * ``_collecting`` is set to ``False``, so backward hooks do not
          trigger step completion.
        * Forward and backward hooks still fire and may append data to
          ``_act_parts`` / ``_grad_parts`` (PyTorch does not allow
          selectively suppressing individual hooks).
        * On exit, buffer state is fully *restored* to what it was before
          the pause, so any training-pass data captured before the pause
          (e.g. the first layer's activation in a mid-forward pause) is
          preserved intact for the continuing training pass.

        Example — val pass triggered from ``on_layer_forward``::

            def on_layer_forward(self, layer_name, activation):
                if self._first_layer and self._need_val_target:
                    with self._hook_manager.pause():
                        loss = val_loss_fn(model, val_batch)
                        loss.backward()   # does NOT trigger another on_step_end

        Yields:
            This :class:`HookManager` instance.
        """
        was_collecting = self._collecting
        self._collecting = False
        # Snapshot current buffer state so any training-pass data already
        # captured (e.g. activations for layers visited before the pause)
        # survives the secondary pass unmodified.
        saved_layer: dict = {
            name: {
                "_act_parts": list(buf["_act_parts"]),
                "_grad_parts": list(buf["_grad_parts"]),
                "activation": buf["activation"],
                "grad_output": buf["grad_output"],
            }
            for name, buf in self._buffers.items()
        }
        saved_param: dict = {
            name: dict(buf) for name, buf in self._param_buffers.items()
        }
        try:
            yield self
        finally:
            self._collecting = was_collecting
            # Restore buffer state, discarding any data accumulated during
            # the secondary pass.
            for name, buf in self._buffers.items():
                s = saved_layer.get(name, {})
                buf["_act_parts"] = s.get("_act_parts", [])
                buf["_grad_parts"] = s.get("_grad_parts", [])
                buf["activation"] = s.get("activation")
                buf["grad_output"] = s.get("grad_output")
            for name, buf in self._param_buffers.items():
                buf.clear()
                buf.update(saved_param.get(name, {}))

    # ---------------------------------------------------------------------- #
    # Lifecycle and introspection                                              #
    # ---------------------------------------------------------------------- #

    def remove(self) -> None:
        """Remove all registered hooks and clear all buffer dicts."""
        self._model_fwd_handle.remove()
        remove_hooks(self._handles)
        self._buffers.clear()
        remove_hooks(self._mlp_weight_handles)
        remove_hooks(self._param_handles)
        self._param_buffers.clear()
        self._last_gradient = None

    def add_callback(self, callback: HookManagerCallback) -> None:
        """Attach a callback after construction and run its ``on_register``.

        Useful when a callback can only be built *after* the manager — most
        notably :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`
        under FSDP, where the manager is created on the unwrapped model (so its
        hooks survive wrapping) but the callback needs the FSDP-wrapped module
        to discover the parameter shard layout.
        """
        self._callbacks.append(callback)
        if hasattr(callback, "on_register"):
            callback.on_register(self)

    @property
    def layer_names(self) -> list[str]:
        """Fully-qualified names of all hooked linear-IO layers."""
        return list(self._buffers.keys())

    @property
    def layer_name(self) -> list[str]:
        """The layer set an attributor scores: the hooked per-sample (linear-IO)
        layers.  Layer selection is decided here, at capture, via
        :class:`HookManagerConfig` — attributors read it back through this
        property (e.g. for :class:`AttributionScore` metadata) instead of taking
        their own ``layer_name`` argument."""
        return list(self._buffers.keys())

    @property
    def param_layer_names(self) -> list[str]:
        """Fully-qualified names of all param-grad hooked layers."""
        return list(self._param_buffers.keys())

    @property
    def steps_collected(self) -> int:
        """Total number of batch steps collected since construction."""
        return self._step_count

    def reset_steps(self) -> None:
        """Reset the capture-step counter to ``0``.

        ``_step_count`` (stamped on each :attr:`GradientRecord.step` and exposed
        via :attr:`steps_collected`) is monotonic for the manager's lifetime by
        default.  A caller that restarts collection from a fresh logical baseline
        — e.g. the live streamer beginning a new pass — can reset it so
        ``record.step`` is pass-local and aligns with the caller's own per-pass
        index.  Only the label counter is reset; in-flight per-step buffers and
        completion flags are left untouched (callers reset between steps, never
        mid-backward).
        """
        with self._step_lock:
            self._step_count = 0

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

        Returns:
            A :class:`~dattri_llm.gradient.Gradient` with ``layer_types``
            populated from the hooked module class names.

        Raises:
            RuntimeError: If no step has completed yet (no backward has run
                inside the collect context).
        """
        if self._last_gradient is None:
            raise RuntimeError(
                "No gradient available: no step has completed yet. "
                "Ensure backward() was called inside the collect() context "
                "before calling get_gradient()."
            )
        return self._last_gradient
