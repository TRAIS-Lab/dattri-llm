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

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]

_LINEAR_TYPES: tuple[type, ...] = (nn.Linear,) + (
    (HF_Conv1D,) if HF_Conv1D is not None else ()
)

# All layer types that can be registered by default.
# Embedding and LayerNorm are included because their weight gradients factorise
# in the same outer-product / per-sample form used by the influence-function
# methods this library targets:
#
#   Embedding  :  dL/dW[token_id, :] = Σ_t  g[t, :]  × 1[input[t] == token_id]
#                 ≡ one_hot(input)^T @ g  — same structure as Linear with
#                 a one-hot "activation".
#
#   LayerNorm  :  dL/dgamma = dL/d(output) ⊙ x_norm  (elementwise product
#                 summed over batch/tokens) — analogous to an MLP without the
#                 cross-feature interactions.
_HOOKABLE_TYPES: tuple[type, ...] = _LINEAR_TYPES + (nn.Embedding, nn.LayerNorm)

_MLP_PARENT_PATTERNS = re.compile(
    r"(mlp|ffn|feed_forward|feedforward|fc|dense)",
    re.IGNORECASE,
)

# Buffer type alias.
# Keys: "activation", "grad_output", "_act_parts", "_grad_parts", "_lock"
LayerBuffer = dict


def _device_sort_key(item: tuple[int, torch.Tensor]) -> int:
    return item[0]


def _is_hookable_layer(name: str, module: nn.Module) -> bool:
    """Return True if *module* should be registered by the default heuristic.

    * ``nn.Embedding`` and ``nn.LayerNorm`` are always included — they appear
      at the top level of transformer blocks, not nested inside an MLP sub-module,
      and their per-sample gradients factorise in a form compatible with the
      influence-function scoring used here.
    * ``nn.Linear`` and HuggingFace ``Conv1D`` are included only when the
      parent module path contains an MLP-family keyword (``mlp``, ``ffn``,
      ``feed_forward``, ``feedforward``, ``fc``, or ``dense``).

    Args:
        name: Fully-qualified module name (dot-separated path from root).
        module: The module to test.

    Returns:
        ``True`` when the module should be hooked by default.
    """
    if not isinstance(module, _HOOKABLE_TYPES):
        return False
    # Embedding and LayerNorm: hook regardless of parent path.
    if isinstance(module, (nn.Embedding, nn.LayerNorm)):
        return True
    # nn.Linear / Conv1D: require an MLP-family keyword in the parent path.
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
            if not isinstance(module, _HOOKABLE_TYPES):
                continue
            if not any(p.search(name) for p in compiled):
                continue
        else:
            if not _is_hookable_layer(name, module):
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
        handles: List of hook handles returned by :func:`register_mlp_hooks`,
            :func:`register_mlp_param_hooks`, or
            :func:`register_param_grad_hooks`.
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


def register_mlp_param_hooks(
    model: nn.Module,
    name_patterns: Optional[list[str]] = None,
    on_mlp_param_grad: Optional[Callable[[str, str, torch.Tensor], None]] = None,
) -> tuple[int, list[torch.utils.hooks.RemovableHook]]:
    """Register post-accumulate-grad hooks on hooked MLP layers' trainable params.

    Unlike :func:`register_param_grad_hooks`, which uses
    ``Tensor.register_hook`` (fires *before* ``param.grad`` is accumulated),
    this function uses ``Tensor.register_post_accumulate_grad_hook``
    (PyTorch ≥ 2.0) which fires *after* ``param.grad`` is written.

    This guarantees that when the callback runs, ``param.grad`` is non-None —
    a precondition for any callback that needs to read or modify weight
    gradients in-place (e.g. :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`).

    The same MLP-layer heuristic (or ``name_patterns`` list) as
    :func:`register_mlp_hooks` is used to identify qualifying layers.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        name_patterns: Optional list of regex strings.  When provided, only
            ``nn.Linear`` / ``Conv1D`` layers whose fully-qualified name
            matches at least one pattern are hooked.  When ``None``, the
            MLP-keyword heuristic is used.
        on_mlp_param_grad: Optional callback fired after each MLP parameter's
            gradient is accumulated.  Signature:
            ``(layer_name: str, param_name: str, grad: Tensor)`` where
            ``grad`` is ``param.grad.detach().cpu()``.

    Returns:
        ``(n_params, handles)`` where ``n_params`` is the total number of
        trainable parameters that were hooked and ``handles`` is a list of
        removable hook objects.
    """
    root: nn.Module = getattr(model, "module", model)

    compiled: list[re.Pattern[str]] | None = None
    if name_patterns is not None:
        compiled = [re.compile(p) for p in name_patterns]

    n_params: int = 0
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        if compiled is not None:
            if not isinstance(module, _HOOKABLE_TYPES):
                continue
            if not any(p.search(name) for p in compiled):
                continue
        else:
            if not _is_hookable_layer(name, module):
                continue

        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            n_params += 1

            def _make_hook(ln: str, pn: str):
                def _hook(p: torch.nn.Parameter) -> None:
                    if on_mlp_param_grad is not None:
                        g = p.grad.detach().cpu()
                        on_mlp_param_grad(ln, pn, g)
                return _hook

            handles.append(
                param.register_post_accumulate_grad_hook(_make_hook(name, pname))
            )

    return n_params, handles


# --------------------------------------------------------------------------- #
# Hook manager configuration                                                   #
# --------------------------------------------------------------------------- #

# Sentinel that distinguishes "caller did not pass this keyword" from
# "caller explicitly passed None (= use default heuristic)".
_UNSET: object = object()

_VALID_HOOK_TYPES = frozenset({"mlp_io", "param_grad"})


class HookManagerConfig:
    """Configuration for :class:`HookManager`.

    Can be constructed in **two equivalent forms**:

    **Dict form** — pass ``hook_types`` as a mapping whose *keys* name the
    active hook types and whose *values* give the optional regex name-pattern
    list (``None`` → use the default layer-identification heuristic):

    .. code-block:: python

        # mlp_io only, default heuristic
        HookManagerConfig(hook_types={"mlp_io": None})

        # mlp_io with custom pattern, param_grad with default heuristic
        HookManagerConfig(hook_types={"mlp_io": [r"mlp\\."], "param_grad": None})

        # param_grad only, custom pattern
        HookManagerConfig(hook_types={"param_grad": [r"lm_head"]})

    **Shorthand form** — pass ``mlp_name_patterns`` and/or
    ``param_name_patterns`` as keyword arguments.  *Passing* a keyword
    (even as ``None``) activates that hook type; *omitting* it deactivates
    it:

    .. code-block:: python

        # mlp_io only, default heuristic  (same as HookManagerConfig())
        HookManagerConfig(mlp_name_patterns=None)

        # param_grad only, default heuristic
        HookManagerConfig(param_name_patterns=None)

        # both, each with its own pattern
        HookManagerConfig(mlp_name_patterns=[r"mlp\\."], param_name_patterns=None)

    **Default** (no arguments) — equivalent to
    ``hook_types={"mlp_io": None}``: register mlp_io hooks on all
    default-heuristic layers.

    Mixing the dict form and the shorthand form in the same call raises
    :exc:`ValueError`.
    """

    def __init__(
        self,
        hook_types: Optional[dict[str, Optional[list[str]]]] = None,
        mlp_name_patterns: object = _UNSET,
        param_name_patterns: object = _UNSET,
    ) -> None:
        _has_shorthand = (
            mlp_name_patterns is not _UNSET or param_name_patterns is not _UNSET
        )

        if hook_types is not None and _has_shorthand:
            raise ValueError(
                "Provide either hook_types (dict form) or mlp_name_patterns / "
                "param_name_patterns (shorthand form), not both."
            )

        if hook_types is not None:
            # ── dict form ────────────────────────────────────────────────────
            if not isinstance(hook_types, dict):
                raise TypeError(
                    f"hook_types must be a dict mapping hook-type name to name "
                    f"patterns (or None), got {type(hook_types).__name__}. "
                    "Example: hook_types={'mlp_io': None, 'param_grad': None}"
                )
            if not hook_types:
                raise ValueError(
                    "hook_types must contain at least one entry. "
                    "Valid keys: 'mlp_io', 'param_grad'."
                )
            unknown = set(hook_types) - _VALID_HOOK_TYPES
            if unknown:
                raise ValueError(
                    f"Unknown hook_types key(s): {sorted(unknown)}. "
                    f"Valid keys: {sorted(_VALID_HOOK_TYPES)}."
                )
            self._hook_types: dict[str, Optional[list[str]]] = dict(hook_types)

        elif _has_shorthand:
            # ── shorthand form ───────────────────────────────────────────────
            self._hook_types = {}
            if mlp_name_patterns is not _UNSET:
                self._hook_types["mlp_io"] = mlp_name_patterns  # type: ignore[assignment]
            if param_name_patterns is not _UNSET:
                self._hook_types["param_grad"] = param_name_patterns  # type: ignore[assignment]

        else:
            # ── default: mlp_io with the default heuristic ───────────────────
            self._hook_types = {"mlp_io": None}

    # ---------------------------------------------------------------------- #
    # Read-only properties                                                     #
    # ---------------------------------------------------------------------- #

    @property
    def hook_types(self) -> list[str]:
        """Active hook-type names in insertion order."""
        return list(self._hook_types.keys())

    @property
    def mlp_name_patterns(self) -> Optional[list[str]]:
        """Regex patterns for :func:`register_mlp_hooks`, or ``None`` for the
        default layer-identification heuristic.

        Only meaningful when ``"mlp_io"`` is in :attr:`hook_types`.
        """
        return self._hook_types.get("mlp_io")

    @property
    def param_name_patterns(self) -> Optional[list[str]]:
        """Regex patterns for :func:`register_param_grad_hooks`, or ``None``
        to hook all leaf modules with trainable parameters.

        Only meaningful when ``"param_grad"`` is in :attr:`hook_types`.
        """
        return self._hook_types.get("param_grad")


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

    def on_step_end(self, record: GradientRecord) -> None:
        """Called exactly once per batch step after the :class:`GradientRecord` is assembled.

        The record always contains the full-batch gradient (``input_hash`` is a
        list of B hashes).  Per-sample slicing is the callback's responsibility
        — see :class:`OffloadCallback` for an example.

        Args:
            record: The assembled :class:`GradientRecord` for this step.
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
    ``offload_interval`` *batch steps*.  Any remaining staged records are written
    when the :meth:`~HookManager.collect` context closes.

    Args:
        offload_interval: Number of **batch steps** to accumulate before writing
            a batch file.  Set to ``1`` for one file per step.
        file_manager: The :class:`GradientFileManager` to delegate saves to.
        recording_type: ``"per_batch"`` (default) stores one
            :class:`GradientRecord` per step.  ``"per_sample"`` slices the
            full-batch record into B individual records (one per sample) before
            staging — useful when downstream code looks up gradients by a
            single-sample hash.
    """

    def __init__(
        self,
        offload_interval: int,
        file_manager: GradientFileManager,
        recording_type: str = "per_batch",
    ) -> None:
        if recording_type not in ("per_sample", "per_batch"):
            raise ValueError(
                f"recording_type must be 'per_sample' or 'per_batch', got {recording_type!r}."
            )
        self._offload_interval = offload_interval
        self.file_manager = file_manager
        self._recording_type = recording_type
        self._staged: list[GradientRecord] = []
        self._staged_steps: set[int] = set()

    def on_step_end(self, record: GradientRecord) -> None:
        records = self._expand(record)
        # Flush *before* staging when we detect the start of a new step and
        # have already accumulated enough complete steps.
        if record.step not in self._staged_steps:
            if len(self._staged_steps) >= self._offload_interval:
                self._flush()
        self._staged.extend(records)
        self._staged_steps.add(record.step)

    def _expand(self, record: GradientRecord) -> list[GradientRecord]:
        """Return per-sample slices when recording_type='per_sample', else [record]."""
        if self._recording_type != "per_sample":
            return [record]
        hashes = record.input_hash if isinstance(record.input_hash, list) else [record.input_hash]
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
        config: :class:`HookManagerConfig` controlling ``hook_types`` and
            layer patterns.  Defaults to ``HookManagerConfig()`` (mlp_io only).
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
            # Legacy path: name_patterns is forwarded as mlp_name_patterns.
            # HookManagerConfig(mlp_name_patterns=None) is the default (mlp_io,
            # default heuristic), same behaviour as before the refactor.
            self._config = HookManagerConfig(mlp_name_patterns=name_patterns)
        else:
            self._config = config

        self._step_count: int = 0
        self._collecting: bool = False

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
        #   have fired (only relevant for ``hook_types`` containing
        #   ``"param_grad"``; starts True otherwise).
        self._bwd_done: bool = True
        self._mlp_param_hook_count: int = 0   # fires toward sub-cond (b)
        self._n_mlp_params: int = 0            # target for sub-cond (b)
        self._grad_done: bool = True

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
        self._mlp_weight_handles: list = []   # post-accumulate hooks for sub-cond (b)
        self._n_layers: int = 0
        if "mlp_io" in self._config.hook_types:
            self._bwd_done = False
            self._buffers, self._handles = register_mlp_hooks(
                model,
                name_patterns=self._config.mlp_name_patterns,
                on_layer_forward=self._dispatch_layer_forward,
                on_layer_backward=self._check_step_bwd_complete,
            )
            self._n_layers = len(self._buffers)
            self._n_mlp_params, self._mlp_weight_handles = register_mlp_param_hooks(
                model,
                name_patterns=self._config.mlp_name_patterns,
                on_mlp_param_grad=self._check_step_mlp_param_complete,
            )

        self._param_buffers: dict = {}
        self._param_handles: list = []
        self._n_params_hooked: int = 0
        self._param_hook_count: int = 0
        if "param_grad" in self._config.hook_types:
            self._grad_done = False
            self._param_buffers, self._param_handles = register_param_grad_hooks(
                model,
                name_patterns=self._config.param_name_patterns,
                on_param_grad=self._check_step_grad_complete,
            )
            self._n_params_hooked = len(self._param_handles)

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
            self._bwd_replica_counts[layer_name] = (
                self._bwd_replica_counts.get(layer_name, 0) + 1
            )
            if self._bwd_replica_counts[layer_name] >= self._n_replicas:
                self._seen_bwd.add(layer_name)
            if len(self._seen_bwd) == self._n_layers:
                self._check_mlp_done()

    def _check_step_mlp_param_complete(
        self,
        _layer_name: str,
        _param_name: str,
        _grad: torch.Tensor,
    ) -> None:
        """Fired by :func:`register_mlp_param_hooks` after each MLP param's
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
        # Sub-condition (b): all MLP param hooks fired, or no MLP params exist.
        mlp_params_done = (self._n_mlp_params == 0
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
        gradient = self._assemble_gradient()
        step = self._step_count

        self._step_count += 1
        self._seen_bwd.clear()
        self._bwd_replica_counts.clear()
        self._mlp_param_hook_count = 0
        self._param_hook_count = 0
        # Reset completion flags for the next step.
        self._bwd_done = "mlp_io" not in self._config.hook_types
        self._grad_done = "param_grad" not in self._config.hook_types
        self._reset_layer_buffers()
        self._reset_param_buffers()

        batch_size = self._get_input_batch_size()
        input_hash = [hash_sample(self._last_inputs, i) for i in range(batch_size)]
        record = GradientRecord(step=step, input_hash=input_hash, gradient=gradient)
        for cb in self._callbacks:
            cb.on_step_end(record)

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

            offload = OffloadCallback(offload_interval=32, file_manager=manager)
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
