"""Core PyTorch hook utilities, HookManager, and related classes.

Two low-level hook families are provided:

MLP factorized hooks — ``register_mlp_hooks``
----------------------------------------------
Registers forward and backward hooks on MLP linear layers to capture the
input activations and output gradients needed for the outer-product identity:

    dL/dW ≈ g^T ⊗ a    (per sample)

DataParallel support: each replica fires its hooks in a separate thread.
The hooks accumulate all replica calls in ``_act_parts`` / ``_grad_parts``
(thread-safe, tagged by source device index) rather than overwriting a single
slot.  Single-GPU and DDP usage is unaffected — the lists always hold one
element in those cases.

Parameter gradient hooks — ``register_param_grad_hooks``
---------------------------------------------------------
Registers ``Tensor.register_hook`` on the *parameters* of general modules.
The hook fires during the backward pass immediately after each parameter's
gradient is freshly computed — not a post-backward ``.grad`` read, which can
be ``None`` on the first call or hold a stale accumulated value from the
previous step.

Under DataParallel the hook is placed on the *original* module's parameters,
which receive the gradient sum from all replicas before the hook fires.
Under DDP the hook fires after the allreduce.

These are **batch-level** gradients (one tensor per parameter per step),
not per-sample.  Use ``register_mlp_hooks`` when per-sample factorized
gradients are needed.

Per-layer callbacks
-------------------
Both families accept optional callables that fire inside each hook immediately
after capture, before the value is written to the buffer.  Because they execute
inside a PyTorch hook they work with any training loop.
"""

from __future__ import annotations

import re
import threading
import warnings
from contextlib import contextmanager
from typing import Callable, Generator, Optional

import torch
import torch.nn as nn

from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord, hash_sample
from dattri_llm.gradient.file_manager import GradientFileManager

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]

_LINEAR_TYPES: tuple[type, ...] = (nn.Linear,) + (
    (HF_Conv1D,) if HF_Conv1D is not None else ()
)

_MLP_PARENT_PATTERNS = re.compile(
    r"(mlp|ffn|feed_forward|feedforward|fc|dense)",
    re.IGNORECASE,
)

# Buffer type alias.
# Keys: "activation", "grad_output", "_act_parts", "_grad_parts", "_lock"
LayerBuffer = dict


def _device_sort_key(item: tuple[int, torch.Tensor]) -> int:
    return item[0]


def _is_mlp_linear(name: str, module: nn.Module) -> bool:
    """Return True if *module* is a linear projection inside an MLP block.

    Args:
        name: Fully-qualified module name (dot-separated path from root).
        module: The module to test.

    Returns:
        True if the module is an ``nn.Linear`` / ``Conv1D`` whose parent
        path contains an MLP keyword.
    """
    if not isinstance(module, _LINEAR_TYPES):
        return False
    parent_path = ".".join(name.split(".")[:-1])
    return bool(_MLP_PARENT_PATTERNS.search(parent_path))


def _make_layer_buffer() -> LayerBuffer:
    return {
        "activation": None,
        "grad_output": None,
        "_act_parts": [],
        "_grad_parts": [],
        "_lock": threading.Lock(),
    }


def register_mlp_hooks(
    model: nn.Module,
    name_patterns: Optional[list[str]] = None,
    on_layer_forward: Optional[Callable[[str, torch.Tensor], None]] = None,
    on_layer_backward: Optional[Callable[[str, torch.Tensor], None]] = None,
) -> tuple[dict[str, LayerBuffer], list[torch.utils.hooks.RemovableHook]]:
    """Register forward and backward hooks on MLP linear layers.

    For each qualifying layer the function registers:

    * A **forward hook** that captures ``input[0]`` (moved to CPU) and
      appends it to ``buffers[name]["_act_parts"]``.
    * A **backward hook** that captures ``grad_output[0]`` (moved to CPU)
      and appends it to ``buffers[name]["_grad_parts"]``.

    Optionally, user-supplied ``on_layer_forward`` and ``on_layer_backward``
    callables fire inside each hook with ``(layer_name, cpu_tensor)``
    immediately after capture.  Because these callables execute inside a
    PyTorch hook they are trainer-agnostic — no trainer callback system is
    required.

    Args:
        model: The PyTorch model (plain, ``DataParallel``, or
            ``DistributedDataParallel`` wrapped).
        name_patterns: Optional list of regex strings.  When provided, only
            layers whose fully-qualified name matches at least one pattern
            are hooked.  When ``None``, the MLP-keyword heuristic is used.
        on_layer_forward: Optional callable fired after each forward hook
            capture.  Signature: ``(layer_name: str, activation: Tensor)``.
            The tensor is on CPU.
        on_layer_backward: Optional callable fired after each backward hook
            capture.  Signature: ``(layer_name: str, grad_output: Tensor)``.
            The tensor is on CPU.

    Returns:
        ``(buffers, handles)`` where ``buffers`` maps layer name to a
        :data:`LayerBuffer` and ``handles`` is a list of removable hook
        objects.
    """
    root: nn.Module = getattr(model, "module", model)

    compiled: list[re.Pattern[str]] | None = None
    if name_patterns is not None:
        compiled = [re.compile(p) for p in name_patterns]

    buffers: dict[str, LayerBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        if compiled is not None:
            if not isinstance(module, _LINEAR_TYPES):
                continue
            if not any(p.search(name) for p in compiled):
                continue
        else:
            if not _is_mlp_linear(name, module):
                continue

        buffers[name] = _make_layer_buffer()

        def _make_forward_hook(layer_name: str):
            def _fwd(_module, inp, _out):
                t = inp[0].detach().cpu()
                dev_idx = inp[0].device.index if inp[0].is_cuda else 0
                buf = buffers[layer_name]
                with buf["_lock"]:
                    buf["_act_parts"].append((dev_idx, t))
                buf["activation"] = t
                if on_layer_forward is not None:
                    on_layer_forward(layer_name, t)
            return _fwd

        def _make_backward_hook(layer_name: str):
            def _bwd(_module, _grad_input, grad_output):
                t = grad_output[0].detach().cpu()
                dev_idx = grad_output[0].device.index if grad_output[0].is_cuda else 0
                buf = buffers[layer_name]
                with buf["_lock"]:
                    buf["_grad_parts"].append((dev_idx, t))
                buf["grad_output"] = t
                if on_layer_backward is not None:
                    on_layer_backward(layer_name, t)
            return _bwd

        handles.append(module.register_forward_hook(_make_forward_hook(name)))
        handles.append(module.register_full_backward_hook(_make_backward_hook(name)))

    return buffers, handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]) -> None:
    """Remove all registered hooks and clear the handle list.

    Args:
        handles: List of hook handles returned by :func:`register_mlp_hooks`
            or :func:`register_param_grad_hooks`.
    """
    for h in handles:
        h.remove()
    handles.clear()


# Buffer type alias for param grad hooks.
# Outer key: layer name.  Inner key: parameter name relative to that layer.
# Value: most recently computed gradient tensor (CPU), or None before the
# first backward pass.
ParamGradBuffer = dict  # {param_name: Tensor | None}


def register_param_grad_hooks(
    model: nn.Module,
    name_patterns: Optional[list[str]] = None,
    on_param_grad: Optional[Callable[[str, str, torch.Tensor], None]] = None,
) -> tuple[dict[str, ParamGradBuffer], list[torch.utils.hooks.RemovableHook]]:
    """Register parameter-gradient hooks on general module layers.

    For each qualifying module, a ``Tensor.register_hook`` is placed on every
    trainable parameter (``requires_grad=True``).  The hook fires during the
    backward pass at the moment the gradient for that parameter is freshly
    computed, writing it to the buffer.  This avoids two common pitfalls:

    * Reading ``.grad`` **after** ``backward()`` returns can yield ``None``
      on the first call if no gradient flowed to that parameter, or a stale
      accumulated value when gradient accumulation spans multiple steps.
    * Reading ``.grad`` **inside** a module backward hook may see a partially
      accumulated tensor when other parameters in the same module still have
      pending gradient contributions.

    Under ``DataParallel`` the hook is attached to the *original* module's
    parameters; replica gradients are summed back before the hook fires.
    Under DDP the hook fires after the allreduce, so the gradient is already
    the global average.

    These are **batch-level** gradients (one ``(out, in)`` tensor per
    parameter per step).  For per-sample factorized gradients use
    :func:`register_mlp_hooks` instead.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        name_patterns: Optional list of regex strings.  When provided, only
            modules whose fully-qualified name matches at least one pattern
            are hooked.  When ``None``, all leaf modules that have at least
            one trainable parameter are hooked.
        on_param_grad: Optional callback fired immediately when a parameter's
            gradient is computed.  Signature:
            ``(layer_name: str, param_name: str, grad: Tensor)``.
            The tensor is on CPU.

    Returns:
        ``(buffers, handles)`` where ``buffers`` maps layer name to a
        :data:`ParamGradBuffer` (``{param_name: grad_tensor}``) and
        ``handles`` is a list of removable hook objects.
    """
    root: nn.Module = getattr(model, "module", model)

    compiled: list[re.Pattern[str]] | None = None
    if name_patterns is not None:
        compiled = [re.compile(p) for p in name_patterns]

    buffers: dict[str, ParamGradBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer_name, module in root.named_modules():
        if compiled is not None:
            if not any(p.search(layer_name) for p in compiled):
                continue
        else:
            # Default: hook leaf modules (no child modules) with trainable params.
            if list(module.children()):
                continue

        trainable = [
            (pname, param)
            for pname, param in module.named_parameters(recurse=False)
            if param.requires_grad
        ]
        if not trainable:
            continue

        buffers[layer_name] = {pname: None for pname, _ in trainable}

        for pname, param in trainable:
            def _make_hook(ln: str, pn: str):
                def _hook(grad: torch.Tensor) -> None:
                    # grad is the freshly-computed gradient for this parameter.
                    # It arrives here before being written to param.grad, so
                    # there is no risk of reading a stale or None value.
                    g = grad.detach().cpu()
                    buffers[ln][pn] = g
                    if on_param_grad is not None:
                        on_param_grad(ln, pn, g)
                return _hook

            handles.append(param.register_hook(_make_hook(layer_name, pname)))

    return buffers, handles


# --------------------------------------------------------------------------- #
# Hook manager configuration                                                   #
# --------------------------------------------------------------------------- #


class HookManagerConfig:
    """Configuration for :class:`HookManager`.

    Args:
        recording_type: ``"per_sample"`` (default) or ``"per_batch"``.

            * ``"per_sample"`` — one :class:`GradientRecord` per sample per
              step; the gradient is sliced along the batch dimension.  Only
              ``"mlp_io"`` hooks are supported in this mode.  Specifying
              ``"param_grad"`` here raises a :exc:`ValueError` — use
              ``recording_type="per_batch"`` or run with ``batch_size=1`` to
              get per-sample parameter gradients.
            * ``"per_batch"`` — one :class:`GradientRecord` per batch step;
              the gradient retains the full batch dimension.  Both
              ``"mlp_io"`` and ``"param_grad"`` hook types are permitted.

        hook_types: One or more of ``"mlp_io"`` and ``"param_grad"``.

            * ``"mlp_io"`` — forward + backward hooks on MLP linear layers.
            * ``"param_grad"`` — ``Tensor.register_hook`` on trainable
              parameters; captures batch-level weight gradients.

            When ``None`` and ``recording_type="per_sample"``, defaults to
            ``["mlp_io"]`` and prints a notice that non-linear layers are
            skipped.  When ``None`` and ``recording_type="per_batch"``,
            defaults to ``["mlp_io"]`` silently.

        mlp_name_patterns: Regex patterns for :func:`register_mlp_hooks`.
            ``None`` uses the MLP-keyword heuristic.
        param_name_patterns: Regex patterns for
            :func:`register_param_grad_hooks`.  ``None`` hooks all leaf
            modules with trainable parameters.
    """

    def __init__(
        self,
        recording_type: str = "per_sample",
        hook_types: Optional[list[str]] = None,
        mlp_name_patterns: Optional[list[str]] = None,
        param_name_patterns: Optional[list[str]] = None,
    ) -> None:
        if recording_type not in ("per_sample", "per_batch"):
            raise ValueError(
                f"recording_type must be 'per_sample' or 'per_batch', got {recording_type!r}."
            )
        self.recording_type = recording_type
        self.mlp_name_patterns = mlp_name_patterns
        self.param_name_patterns = param_name_patterns

        if hook_types is None:
            self.hook_types: list[str] = ["mlp_io"]
            if recording_type == "per_sample":
                warnings.warn(
                    "hook_types not specified for per_sample recording; defaulting to "
                    "['mlp_io']. Non-linear layers (e.g. attention projections, "
                    "embeddings) are skipped. Pass hook_types explicitly to silence this.",
                    stacklevel=3,
                )
        else:
            unknown = set(hook_types) - {"mlp_io", "param_grad"}
            if unknown:
                raise ValueError(
                    f"Unknown hook_types: {unknown}. Choose from 'mlp_io', 'param_grad'."
                )
            if not hook_types:
                raise ValueError("hook_types must contain at least one entry.")
            if recording_type == "per_sample" and "param_grad" in hook_types:
                raise ValueError(
                    "'param_grad' hooks capture batch-level weight gradients and cannot "
                    "be used with recording_type='per_sample'. Use "
                    "recording_type='per_batch' instead, or set batch_size=1 and "
                    "recording_type='per_batch' to obtain per-sample gradients."
                )
            self.hook_types = list(hook_types)


# --------------------------------------------------------------------------- #
# Callback base class                                                          #
# --------------------------------------------------------------------------- #


class HookManagerCallback:
    """Base class for gradient collection event hooks.

    All methods are no-ops by default; subclasses override only what they need.
    ``on_layer_forward`` and ``on_layer_backward`` are wired into PyTorch's
    hook system and fire regardless of trainer callback support.
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

    def on_collect_end(self, record: GradientRecord) -> None:
        """Called once per record after its :class:`GradientRecord` is assembled.

        For ``per_sample`` recording with batch size B this fires B times per
        step.  For ``per_batch`` recording it fires once per step.

        Args:
            record: The assembled :class:`GradientRecord`.
        """

    def on_context_end(self) -> None:
        """Called when the :meth:`~HookManager.collect` context closes.

        Use this to flush any remaining staged records to disk.
        """


# --------------------------------------------------------------------------- #
# Built-in offload callback                                                    #
# --------------------------------------------------------------------------- #


class OffloadCallback(HookManagerCallback):
    """Built-in callback that periodically saves :class:`GradientRecord` objects.

    Accumulates records in memory and writes them to a single batch file every
    ``every_n_steps`` *batch steps*.  Any remaining staged records are written
    when the :meth:`~HookManager.collect` context closes.

    Args:
        every_n_steps: Number of **batch steps** to accumulate before writing
            a batch file.  Set to ``1`` for one file per step.
        file_manager: The :class:`GradientFileManager` to delegate saves to.
    """

    def __init__(self, every_n_steps: int, file_manager: GradientFileManager) -> None:
        self._every_n_steps = every_n_steps
        self.file_manager = file_manager
        self._staged: list[GradientRecord] = []
        self._staged_steps: set[int] = set()

    def on_collect_end(self, record: GradientRecord) -> None:
        # Flush *before* staging this record when we detect the start of a
        # new step and have already accumulated enough complete steps.
        if record.step not in self._staged_steps:
            if len(self._staged_steps) >= self._every_n_steps:
                self._flush()
        self._staged.append(record)
        self._staged_steps.add(record.step)

    def on_context_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._staged:
            return
        self.file_manager.save_batch(self._staged)
        self._staged.clear()
        self._staged_steps.clear()

    @property
    def staged(self) -> list[GradientRecord]:
        """Records staged but not yet written to disk."""
        return list(self._staged)


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
        config: :class:`HookManagerConfig` controlling ``recording_type``,
            ``hook_types``, and layer patterns.  Defaults to
            ``HookManagerConfig()`` (per-sample, mlp_io only).
        callbacks: List of :class:`HookManagerCallback` objects.
        sample_id_key: Key in the model's forward kwargs used as a hint for
            batch size detection.  Defaults to ``"input_ids"``.
        name_patterns: Kept for backward compatibility.  Forwarded as
            ``mlp_name_patterns`` when ``config`` is ``None``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[HookManagerConfig] = None,
        callbacks: Optional[list[HookManagerCallback]] = None,
        sample_id_key: str = "input_ids",
        # Kept for backward compatibility; ignored when config is provided.
        name_patterns: Optional[list[str]] = None,
    ) -> None:
        self._model = model
        self._callbacks: list[HookManagerCallback] = callbacks or []
        self._sample_id_key = sample_id_key

        if config is None:
            self._config = HookManagerConfig(
                hook_types=["mlp_io"],
                mlp_name_patterns=name_patterns,
            )
        else:
            self._config = config

        self._step_count: int = 0
        self._collecting: bool = False

        self._seen_bwd: set[str] = set()
        self._bwd_replica_counts: dict[str, int] = {}
        self._step_lock = threading.Lock()

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

        self._buffers: dict = {}
        self._handles: list = []
        self._n_layers: int = 0
        if "mlp_io" in self._config.hook_types:
            self._buffers, self._handles = register_mlp_hooks(
                model,
                name_patterns=self._config.mlp_name_patterns,
                on_layer_forward=self._dispatch_layer_forward,
                on_layer_backward=self._check_step_complete,
            )
            self._n_layers = len(self._buffers)

        self._param_buffers: dict = {}
        self._param_handles: list = []
        self._n_params_hooked: int = 0
        self._param_hook_count: int = 0
        if "param_grad" in self._config.hook_types:
            on_param_grad_cb = (
                self._on_param_grad_fired
                if "mlp_io" not in self._config.hook_types
                else None
            )
            self._param_buffers, self._param_handles = register_param_grad_hooks(
                model,
                name_patterns=self._config.param_name_patterns,
                on_param_grad=on_param_grad_cb,
            )
            self._n_params_hooked = len(self._param_handles)

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

    def _on_param_grad_fired(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by register_param_grad_hooks for each param during backward.

        Used in param_grad-only mode to detect step completion: when every
        expected parameter hook has fired exactly once, the step is complete.
        """
        if not self._collecting:
            return
        with self._step_lock:
            self._param_hook_count += 1
            if self._param_hook_count >= self._n_params_hooked:
                self._param_hook_count = 0
                self._on_param_step_complete()

    def _check_step_complete(self, layer_name: str, grad_output: torch.Tensor) -> None:
        for cb in self._callbacks:
            cb.on_layer_backward(layer_name, grad_output)

        if not self._collecting:
            return

        with self._step_lock:
            self._bwd_replica_counts[layer_name] = (
                self._bwd_replica_counts.get(layer_name, 0) + 1
            )
            if self._bwd_replica_counts[layer_name] >= self._n_replicas:
                self._seen_bwd.add(layer_name)
            if len(self._seen_bwd) == self._n_layers:
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
        gradient = self._assemble_gradient()
        step = self._step_count

        self._step_count += 1
        self._seen_bwd.clear()
        self._bwd_replica_counts.clear()
        self._reset_layer_buffers()
        self._reset_param_buffers()

        if self._config.recording_type == "per_sample":
            batch_size = gradient.batch_size
            for i in range(batch_size):
                input_hash = hash_sample(self._last_inputs, i)
                record = GradientRecord(
                    step=step,
                    input_hash=input_hash,
                    gradient=gradient.slice(dim="batch", index=i),
                )
                for cb in self._callbacks:
                    cb.on_collect_end(record)
        else:
            batch_size = self._get_input_batch_size()
            input_hash = [hash_sample(self._last_inputs, i) for i in range(batch_size)]
            record = GradientRecord(step=step, input_hash=input_hash, gradient=gradient)
            for cb in self._callbacks:
                cb.on_collect_end(record)

    def _on_param_step_complete(self) -> None:
        gradient = self._assemble_gradient()
        step = self._step_count
        self._step_count += 1
        self._reset_param_buffers()

        batch_size = self._get_input_batch_size()
        input_hash = [hash_sample(self._last_inputs, i) for i in range(batch_size)]
        record = GradientRecord(step=step, input_hash=input_hash, gradient=gradient)
        for cb in self._callbacks:
            cb.on_collect_end(record)

    def _assemble_gradient(self) -> Gradient:
        data: dict = {}
        representation: dict = {}
        layer_types: dict = {}

        for layer_name, buf in self._buffers.items():
            act_parts = buf["_act_parts"]
            grad_parts = buf["_grad_parts"]
            if not act_parts or not grad_parts:
                raise RuntimeError(
                    f"Layer '{layer_name}' has no buffered data. "
                    "Ensure backward() was called inside the collect() context."
                )
            a = torch.cat([t for _, t in sorted(act_parts,  key=lambda x: x[0])], dim=0)
            g = torch.cat([t for _, t in sorted(grad_parts, key=lambda x: x[0])], dim=0)
            data[layer_name] = Factorized(activation=a, pre_activation_grad=g)
            representation[layer_name] = "factorized"
            layer_types[layer_name] = "mlp_io"

        for layer_name, buf in self._param_buffers.items():
            for pname, grad in buf.items():
                if grad is None:
                    continue
                key = f"{layer_name}.{pname}"
                data[key] = grad
                representation[key] = "materialized"
                layer_types[key] = "param_grad"

        if not data:
            raise RuntimeError(
                "No gradient data assembled. "
                "Ensure backward() was called inside the collect() context."
            )

        indexing = "batch"
        for name, val in data.items():
            if layer_types.get(name) != "param_grad":
                indexing = "batch_token" if (
                    isinstance(val, Factorized) and val.activation.ndim == 3
                ) else "batch"
                break

        return Gradient(
            representation=representation,
            data=data,
            layer_types=layer_types,
            indexing=indexing,
        )

    def _reset_layer_buffers(self) -> None:
        for buf in self._buffers.values():
            buf["activation"] = None
            buf["grad_output"] = None
            buf["_act_parts"] = []
            buf["_grad_parts"] = []

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

            offload = OffloadCallback(every_n_steps=32, file_manager=manager)
            collector = HookManager(model, callbacks=[offload])

            with collector.collect():
                trainer.train()

        Yields:
            This :class:`HookManager` instance.
        """
        self._reset_layer_buffers()
        self._reset_param_buffers()
        self._param_hook_count = 0
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

    # ---------------------------------------------------------------------- #
    # Lifecycle and introspection                                              #
    # ---------------------------------------------------------------------- #

    def remove(self) -> None:
        """Remove all registered hooks and clear all buffer dicts."""
        self._model_fwd_handle.remove()
        remove_hooks(self._handles)
        self._buffers.clear()
        remove_hooks(self._param_handles)
        self._param_buffers.clear()

    @property
    def layer_names(self) -> list[str]:
        """Fully-qualified names of all hooked MLP layers (mlp_io)."""
        return list(self._buffers.keys())

    @property
    def param_layer_names(self) -> list[str]:
        """Fully-qualified names of all param-grad hooked layers."""
        return list(self._param_buffers.keys())

    @property
    def steps_collected(self) -> int:
        """Total number of batch steps collected since construction."""
        return self._step_count
