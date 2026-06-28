"""Core PyTorch hook utilities, HookManager, and related classes.

Two low-level hook families are provided:

Linear-IO factorized hooks — ``register_linear_io_hooks``
---------------------------------------------------------
Registers forward and backward hooks on linear-family layers (``nn.Linear``,
``nn.Conv*``, ``nn.Embedding``, norm layers, …) to capture the input
activations and output gradients needed for the outer-product identity:

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
not per-sample.  Use ``register_linear_io_hooks`` when per-sample factorized
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
from typing import Callable, Generator, Iterable, Optional

import torch
import torch.nn as nn

from dattri_llm.gradient.callbacks import HookManagerCallback, OffloadCallback
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.ops import (
    PARAM_GRAD_TYPES,
    canonical_class_name,
    extract_module_kwargs,
    is_embedding,
)
from dattri_llm.gradient.utils import hash_sample

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]


def _queue_backward_end_callback(fn: Callable[[], None]) -> bool:
    """Schedule ``fn`` to run once the in-flight backward pass fully completes.

    Must be called from *within* a backward pass (e.g. a module full-backward
    hook).  The callback fires after the autograd engine has finished — the
    point at which FSDP has written every (sharded) ``param.grad`` back to its
    original parameter, so reading ``param.grad`` inside ``fn`` is safe.

    Returns ``True`` if the callback was successfully queued, ``False`` if the
    autograd engine does not expose ``queue_callback`` on this build (in which
    case the caller should fall back to its existing behaviour).
    """
    try:
        torch.autograd.Variable._execution_engine.queue_callback(fn)
        return True
    except Exception:  # pragma: no cover - unexpected autograd internals
        return False

# ── Linear-IO-capable types ─────────────────────────────────────────────────
# Layers whose per-sample gradient factorises as an outer product of the input
# activation and the output gradient (``dL/dW ≈ g^T ⊗ a``).  These are the
# layers that can be hooked with the ``linear_io`` family.  Any layer (whether
# or not it appears here) can instead be hooked with the ``param_grad`` family,
# which materialises the batch-level parameter gradient directly.
#
# Membership is decided purely by type — never by the module's name in the
# graph, which is arbitrary.
_LINEAR_IO_TYPES: tuple[type, ...] = (
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Linear,
    nn.Bilinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
) + ((HF_Conv1D,) if HF_Conv1D is not None else ())
# RMSNorm was added in PyTorch 2.4 — guard for older versions.
if hasattr(nn, "RMSNorm"):
    _LINEAR_IO_TYPES = _LINEAR_IO_TYPES + (nn.RMSNorm,)  # type: ignore[assignment]

# Buffer type alias.
# Keys: "activation", "grad_output", "_act_parts", "_grad_parts", "_lock"
LayerBuffer = dict


def _device_sort_key(item: tuple[int, torch.Tensor]) -> int:
    return item[0]


def _is_linear_io_capable(module: nn.Module) -> bool:
    """Return ``True`` if *module*'s gradient factorises for ``linear_io`` hooks.

    Membership is decided purely by type (see :data:`_LINEAR_IO_TYPES`), never
    by the module's name in the graph.
    """
    return isinstance(module, _LINEAR_IO_TYPES)


def _has_trainable_params(module: nn.Module) -> bool:
    """Return ``True`` if *module* directly owns a trainable parameter.

    Only the module's own parameters are considered (``recurse=False``), so a
    parent is not credited with parameters that belong to its children.
    """
    return any(p.requires_grad for _, p in module.named_parameters(recurse=False))


def _make_layer_buffer() -> LayerBuffer:
    return {
        "activation": None,
        "grad_output": None,
        "_act_parts": [],
        "_grad_parts": [],
        "_lock": threading.Lock(),
    }


def register_linear_io_hooks(
    model: nn.Module,
    layer_names: Optional[set[str]] = None,
    on_layer_forward: Optional[Callable[[str, torch.Tensor], None]] = None,
    on_layer_backward: Optional[Callable[[str, torch.Tensor], None]] = None,
    type_overrides: Optional[dict[str, str]] = None,
) -> tuple[dict[str, LayerBuffer], list[torch.utils.hooks.RemovableHook]]:
    """Register forward and backward hooks on linear-family layers.

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
        layer_names: Optional set of fully-qualified module names to hook.
            When provided, only modules whose name is in the set *and* which
            are linear-IO-capable are hooked.  When ``None``, every
            linear-IO-capable layer is hooked (see :data:`_LINEAR_IO_TYPES`).
        on_layer_forward: Optional callable fired after each forward hook
            capture.  Signature: ``(layer_name: str, activation: Tensor)``.
            The tensor is on CPU.
        on_layer_backward: Optional callable fired after each backward hook
            capture.  Signature: ``(layer_name: str, grad_output: Tensor)``.
            The tensor is on CPU.
        type_overrides: Optional mapping from layer name to a layer-type string
            that overrides the type inferred by :func:`canonical_class_name`.
            Use this for user-defined layer classes that subclass a supported
            linear-family type but whose class name is not recognised (e.g. a
            custom ``MyLinear`` that should be treated as ``"nn.Linear"``).
            Layers absent from the mapping fall back to ``canonical_class_name``.

    Returns:
        ``(buffers, handles)`` where ``buffers`` maps layer name to a
        :data:`LayerBuffer` and ``handles`` is a list of removable hook
        objects.
    """
    root: nn.Module = getattr(model, "module", model)

    buffers: dict[str, LayerBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        if not _is_linear_io_capable(module):
            continue
        if layer_names is not None and name not in layer_names:
            continue

        buffers[name] = _make_layer_buffer()
        if type_overrides is not None and name in type_overrides:
            layer_type = type_overrides[name]
        else:
            layer_type = canonical_class_name(module)
        buffers[name]["_class_name"] = layer_type
        buffers[name]["_module_kwargs"] = extract_module_kwargs(module, layer_type)

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
        handles: List of hook handles returned by
            :func:`register_linear_io_hooks`,
            :func:`register_linear_param_hooks`, or
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
    layer_names: Optional[set[str]] = None,
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
    :func:`register_linear_io_hooks` instead.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        layer_names: Optional set of fully-qualified module names to hook.
            When provided, only modules whose name is in the set are hooked.
            When ``None``, every module that directly owns at least one
            trainable parameter is hooked.
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

    buffers: dict[str, ParamGradBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer_name, module in root.named_modules():
        if layer_names is not None and layer_name not in layer_names:
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


def register_linear_param_hooks(
    model: nn.Module,
    layer_names: Optional[set[str]] = None,
    on_linear_param_grad: Optional[Callable[[str, str, torch.Tensor], None]] = None,
) -> tuple[int, list[torch.utils.hooks.RemovableHook]]:
    """Register post-accumulate-grad hooks on linear layers' trainable params.

    Unlike :func:`register_param_grad_hooks`, which uses
    ``Tensor.register_hook`` (fires *before* ``param.grad`` is accumulated),
    this function uses ``Tensor.register_post_accumulate_grad_hook``
    (PyTorch ≥ 2.0) which fires *after* ``param.grad`` is written.

    This guarantees that when the callback runs, ``param.grad`` is non-None —
    a precondition for any callback that needs to read or modify weight
    gradients in-place (e.g. :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`).

    The same layer-selection rule as :func:`register_linear_io_hooks` is used
    to identify qualifying layers.

    Args:
        model: The model to hook (plain ``nn.Module``, ``DataParallel``, or
            ``DistributedDataParallel``).
        layer_names: Optional set of fully-qualified module names to hook.
            When provided, only linear-IO-capable modules whose name is in the
            set are hooked.  When ``None``, every linear-IO-capable layer is
            hooked.
        on_linear_param_grad: Optional callback fired after each parameter's
            gradient is accumulated.  Signature:
            ``(layer_name: str, param_name: str, grad: Tensor)`` where
            ``grad`` is ``param.grad.detach().cpu()``.

    Returns:
        ``(n_params, handles)`` where ``n_params`` is the total number of
        trainable parameters that were hooked and ``handles`` is a list of
        removable hook objects.
    """
    root: nn.Module = getattr(model, "module", model)

    n_params: int = 0
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        if not _is_linear_io_capable(module):
            continue
        if layer_names is not None and name not in layer_names:
            continue

        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            n_params += 1

            def _make_hook(ln: str, pn: str):
                def _hook(p: torch.nn.Parameter) -> None:
                    if on_linear_param_grad is not None:
                        g = p.grad.detach().cpu()
                        on_linear_param_grad(ln, pn, g)
                return _hook

            handles.append(
                param.register_post_accumulate_grad_hook(_make_hook(name, pname))
            )

    return n_params, handles


# --------------------------------------------------------------------------- #
# Hook manager configuration                                                   #
# --------------------------------------------------------------------------- #

class _RegisterAll:
    """Sentinel type for :data:`REGISTER_ALL`.

    A selector value of :data:`REGISTER_ALL` requests that *every* layer
    applicable to a given hook family be registered, regardless of name.
    """

    _instance: Optional["_RegisterAll"] = None

    def __new__(cls) -> "_RegisterAll":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "REGISTER_ALL"


REGISTER_ALL = _RegisterAll()
"""Importable selector meaning "register every applicable layer".

Pass it to :class:`HookManagerConfig` to register a hook family on all of its
applicable layers (linear-IO-capable layers for ``linear_io``; all layers with
trainable parameters for ``param_grad``)::

    from dattri_llm.gradient.hooks import HookManagerConfig, REGISTER_ALL

    # linear_io on every linear-family layer, nothing else
    HookManagerConfig(linear_io=REGISTER_ALL)

    # param_grad on every trainable layer
    HookManagerConfig(param_grad=REGISTER_ALL)
"""

# A hook-family selector is one of:
#   * ``None``        — not provided (the family is not requested explicitly).
#   * ``REGISTER_ALL``— register every applicable layer.
#   * ``list[str]``   — register applicable layers whose name matches a regex.
Selector = Optional[object]  # None | _RegisterAll | list[str]

# Hook-family names and the layer_types marker used for materialized grads.
LINEAR_IO = "linear_io"
PARAM_GRAD = "param_grad"


_VALID_HOOK_TYPES = frozenset({LINEAR_IO, PARAM_GRAD})


class HookManagerConfig:
    """Configuration for :class:`HookManager`.

    The core control is :attr:`hook_types`, an explicit **assignment** mapping
    each fully-qualified layer name to the hook family it should use:

    * ``linear_io`` — per-sample factorized hooks on linear-family layers
      (see :func:`register_linear_io_hooks`).
    * ``param_grad`` — batch-level materialized parameter-gradient hooks,
      available for *any* layer with trainable parameters
      (see :func:`register_param_grad_hooks`).

    .. code-block:: python

        # explicit per-layer assignment
        HookManagerConfig(hook_types={"mlp.0": "linear_io", "lm_head": "param_grad"})

    The regex **selectors** ``linear_io`` and ``param_grad`` are an add-on that
    *extends* the assignment without having to spell out every layer.  Each
    selector is one of:

    * ``None`` — add nothing.
    * :data:`REGISTER_ALL` — add every layer applicable to that family.
    * ``list[str]`` — add applicable layers whose fully-qualified name matches
      at least one of the given regex patterns.

    .. code-block:: python

        # linear_io on every linear-family layer
        HookManagerConfig(linear_io=REGISTER_ALL)

        # explicit assignment, extended by a regex add-on
        HookManagerConfig(
            hook_types={"lm_head": "param_grad"},
            linear_io=[r"mlp\\."],
        )

    The explicit assignment and the selector add-ons are merged into one final
    ``{layer_name: hook_type}`` map.  **If a layer is assigned two different
    hook families** (e.g. listed in ``hook_types`` as ``param_grad`` but also
    matched by the ``linear_io`` selector), a :exc:`ValueError` is raised — one
    layer may only be registered with one hook family.

    **Default** (no arguments) — register ``linear_io`` on every
    linear-IO-capable layer, and fall back to ``param_grad`` for any remaining
    layer that has trainable parameters but is not linear-IO-capable.

    **Manual layer types** — :attr:`layer_types` maps a layer name to a
    layer-type string that overrides the type inferred by
    :func:`canonical_class_name`.  This is for user-defined layer classes whose
    class name is not recognised by the built-in type detection (e.g. a custom
    ``MyLinear`` subclass that should be treated as ``"nn.Linear"``)::

        HookManagerConfig(layer_types={"mlp.0": "nn.Linear"})

    The mapping may be incomplete: layers it does not mention keep the
    automatically detected type.  It is orthogonal to which layers get hooked —
    a layer named here is only relabelled, not forced to be hooked.  If a named
    layer is not hooked in the end, :class:`HookManager` emits a warning.
    """

    def __init__(
        self,
        hook_types: Optional[dict[str, str]] = None,
        linear_io: Selector = None,
        param_grad: Selector = None,
        layer_types: Optional[dict[str, str]] = None,
    ) -> None:
        self.hook_types = self._validate_assignment(hook_types)
        self.linear_io = self._validate_selector(LINEAR_IO, linear_io)
        self.param_grad = self._validate_selector(PARAM_GRAD, param_grad)
        self.layer_types = self._validate_layer_types(layer_types)

    @staticmethod
    def _validate_assignment(
        hook_types: Optional[dict[str, str]],
    ) -> dict[str, str]:
        if hook_types is None:
            return {}
        if not isinstance(hook_types, dict):
            raise TypeError(
                "hook_types must be a dict mapping layer name to hook type "
                f"({sorted(_VALID_HOOK_TYPES)}), got "
                f"{type(hook_types).__name__}."
            )
        for layer_name, hook_type in hook_types.items():
            if hook_type not in _VALID_HOOK_TYPES:
                raise ValueError(
                    f"hook_types['{layer_name}'] = '{hook_type}' is not a valid "
                    f"hook type. Valid types: {sorted(_VALID_HOOK_TYPES)}."
                )
        return dict(hook_types)

    @staticmethod
    def _validate_layer_types(
        layer_types: Optional[dict[str, str]],
    ) -> dict[str, str]:
        if layer_types is None:
            return {}
        if not isinstance(layer_types, dict):
            raise TypeError(
                "layer_types must be a dict mapping layer name to a layer-type "
                f"string, got {type(layer_types).__name__}."
            )
        for layer_name, layer_type in layer_types.items():
            if not isinstance(layer_name, str) or not isinstance(layer_type, str):
                raise TypeError(
                    "layer_types must map str layer names to str type names, got "
                    f"{layer_name!r}: {layer_type!r}."
                )
        return dict(layer_types)

    @staticmethod
    def _validate_selector(name: str, selector: Selector) -> Selector:
        if selector is None or selector is REGISTER_ALL:
            return selector
        if isinstance(selector, (list, tuple)):
            if not all(isinstance(p, str) for p in selector):
                raise TypeError(
                    f"{name} pattern list must contain only regex strings."
                )
            return list(selector)
        raise TypeError(
            f"{name} must be None, REGISTER_ALL, or a list of regex strings, "
            f"got {type(selector).__name__}."
        )

    @property
    def is_default(self) -> bool:
        """True when nothing was requested (the auto fallback applies)."""
        return (
            not self.hook_types
            and self.linear_io is None
            and self.param_grad is None
        )


def _selector_matches(selector: Selector, name: str) -> bool:
    """Return ``True`` if *name* is selected by *selector*.

    ``None`` selects nothing, :data:`REGISTER_ALL` selects everything, and a
    list of regex strings selects names matching at least one pattern.
    """
    if selector is None:
        return False
    if selector is REGISTER_ALL:
        return True
    return any(re.search(p, name) for p in selector)  # type: ignore[union-attr]


def resolve_hook_assignments(
    root: nn.Module,
    config: HookManagerConfig,
) -> dict[str, str]:
    """Resolve the final ``{layer_name: hook_type}`` assignment.

    Resolution rules:

    * **Default** (``config.is_default``) — every linear-IO-capable layer is
      assigned ``linear_io``; every other layer that directly owns a trainable
      parameter falls back to ``param_grad``.
    * **Explicit assignment** (``config.hook_types``) — taken verbatim, after
      validating that each named layer exists and supports the requested family.
    * **Selector add-ons** (``config.linear_io`` / ``config.param_grad``) —
      extend the assignment with the applicable layers they match.

    A layer assigned two *different* hook families raises :exc:`ValueError`.  A
    warning is emitted if the resolution registers zero layers.
    """
    modules = dict(root.named_modules())
    assignment: dict[str, str] = {}

    def assign(layer_name: str, hook_type: str) -> None:
        existing = assignment.get(layer_name)
        if existing is not None and existing != hook_type:
            raise ValueError(
                f"Layer '{layer_name}' is assigned conflicting hook types: "
                f"'{existing}' and '{hook_type}'. A layer may only be "
                "registered with one hook family."
            )
        assignment[layer_name] = hook_type

    if config.is_default:
        for name, module in modules.items():
            if _is_linear_io_capable(module):
                assignment[name] = LINEAR_IO
            elif _has_trainable_params(module):
                assignment[name] = PARAM_GRAD
        # The default never produces conflicts, so skip the zero-layer warning
        # path below only if something was registered.
        if not assignment:
            _warn_zero_layers()
        return assignment

    # 1. Explicit per-layer assignment (validated against the model).
    for layer_name, hook_type in config.hook_types.items():
        module = modules.get(layer_name)
        if module is None:
            raise ValueError(
                f"hook_types names layer '{layer_name}', which does not exist "
                "in the model."
            )
        if hook_type == LINEAR_IO and not _is_linear_io_capable(module):
            raise ValueError(
                f"Layer '{layer_name}' was assigned 'linear_io' but its type "
                f"({canonical_class_name(module)}) does not support factorized "
                "linear-IO hooks."
            )
        if hook_type == PARAM_GRAD and not _has_trainable_params(module):
            raise ValueError(
                f"Layer '{layer_name}' was assigned 'param_grad' but has no "
                "trainable parameters."
            )
        assign(layer_name, hook_type)

    # 2. Selector add-ons extend the assignment with applicable layers only.
    for name, module in modules.items():
        if _is_linear_io_capable(module) and _selector_matches(
            config.linear_io, name
        ):
            assign(name, LINEAR_IO)
        if _has_trainable_params(module) and _selector_matches(
            config.param_grad, name
        ):
            assign(name, PARAM_GRAD)

    if not assignment:
        _warn_zero_layers()

    return assignment


def _warn_zero_layers() -> None:
    warnings.warn(
        "HookManager registered zero layers: no module matched the requested "
        "hook configuration. No gradients will be collected.",
        stacklevel=3,
    )


# --------------------------------------------------------------------------- #
# Active-layer discovery                                                        #
# --------------------------------------------------------------------------- #


def _invoke_model(model: nn.Module, sample_input: object) -> object:
    """Call *model* on *sample_input*, unpacking dicts / sequences."""
    if isinstance(sample_input, dict):
        return model(**sample_input)
    if isinstance(sample_input, (list, tuple)):
        return model(*sample_input)
    return model(sample_input)


def _derive_scalar_loss(output: object) -> Optional[torch.Tensor]:
    """Best-effort reduction of a model output to a scalar for ``backward()``.

    Prefers ``output.loss``; otherwise sums a floating-point output tensor
    (``output`` itself, ``output.logits``, or the first float tensor in a
    dict/sequence).  Returns ``None`` when no differentiable scalar is found.
    """
    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss

    if isinstance(output, torch.Tensor):
        candidate: Optional[torch.Tensor] = output
    else:
        candidate = getattr(output, "logits", None)
        if candidate is None:
            values: object = ()
            if isinstance(output, dict):
                values = output.values()
            elif isinstance(output, (list, tuple)):
                values = output
            candidate = next(
                (
                    v
                    for v in values
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                ),
                None,
            )

    if isinstance(candidate, torch.Tensor) and candidate.is_floating_point():
        return candidate.sum()
    return None


def default_hook_assignment(
    model: nn.Module,
    sample_input: object,
    loss_fn: Optional[Callable[[object], torch.Tensor]] = None,
) -> dict[str, str]:
    """Discover the default-style assignment for layers that actually fire.

    Some modules are registered as sub-modules but invoked *functionally*
    rather than called as modules — e.g. :class:`nn.MultiheadAttention` applies
    its ``out_proj`` weight through ``F.linear`` instead of ``out_proj(...)``.
    Such a module's forward/backward hooks never fire, so a :class:`HookManager`
    that waits on them can never complete a step.

    This helper runs one forward (and, when a scalar loss is available,
    backward) pass on *sample_input* with lightweight monitor hooks to find
    which layers are actually exercised, then returns the default assignment
    (:attr:`HookManagerConfig.is_default` behaviour) restricted to those
    layers.  Pass the result as ``hook_types``::

        assignment = default_hook_assignment(model, sample_input)
        hm = HookManager(model, config=HookManagerConfig(hook_types=assignment))

    A ``linear_io`` candidate is kept when its forward hook fires (the module is
    invoked as a module); a ``param_grad`` candidate is kept when one of its
    parameters receives a gradient.  Layers that never fire are skipped (with a
    warning) rather than left to stall step completion.

    Args:
        model: The model to inspect (plain, ``DataParallel``, or DDP wrapped).
        sample_input: A representative input.  A ``dict`` is passed as
            ``model(**sample_input)``, a ``list`` / ``tuple`` as
            ``model(*sample_input)``, anything else as ``model(sample_input)``.
        loss_fn: Optional callable mapping the model output to a scalar loss for
            the backward pass.  When omitted, a scalar is derived from the
            output; if none can be derived, only ``linear_io`` (forward-fired)
            layers are discovered.

    Returns:
        ``{layer_name: "linear_io" | "param_grad"}`` for the layers that fired.

    Note:
        This executes a real forward+backward pass, which may update stateful
        layers (e.g. BatchNorm running stats).  Run it on a throwaway batch.
    """
    root: nn.Module = getattr(model, "module", model)
    modules = dict(root.named_modules())

    # Default-style candidate assignment (same rule as the resolver's default).
    candidate: dict[str, str] = {}
    for name, module in modules.items():
        if _is_linear_io_capable(module):
            candidate[name] = LINEAR_IO
        elif _has_trainable_params(module):
            candidate[name] = PARAM_GRAD

    fired: set[str] = set()
    handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_forward_monitor(layer_name: str) -> Callable:
        def _hook(_module, _inp, _out) -> None:
            fired.add(layer_name)
        return _hook

    def _make_grad_monitor(layer_name: str) -> Callable:
        def _hook(_grad: torch.Tensor) -> None:
            fired.add(layer_name)
        return _hook

    for name, module in modules.items():
        kind = candidate.get(name)
        if kind == LINEAR_IO:
            handles.append(module.register_forward_hook(_make_forward_monitor(name)))
        elif kind == PARAM_GRAD:
            for _, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    handles.append(param.register_hook(_make_grad_monitor(name)))

    try:
        with torch.enable_grad():
            output = _invoke_model(model, sample_input)
            try:
                loss = (
                    loss_fn(output)
                    if loss_fn is not None
                    else _derive_scalar_loss(output)
                )
            except Exception as exc:  # noqa: BLE001 - user loss_fn may fail
                loss = None
                warnings.warn(
                    f"default_hook_assignment could not compute a loss "
                    f"({exc!r}); param_grad layers may be missed. Pass loss_fn.",
                    stacklevel=2,
                )
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                model.zero_grad(set_to_none=True)
                loss.backward()
            elif any(t == PARAM_GRAD for t in candidate.values()):
                warnings.warn(
                    "default_hook_assignment could not derive a differentiable "
                    "scalar loss; param_grad layers may be missed. Pass loss_fn.",
                    stacklevel=2,
                )
    finally:
        remove_hooks(handles)
        model.zero_grad(set_to_none=True)

    assignment = {n: t for n, t in candidate.items() if n in fired}

    skipped = sorted(set(candidate) - set(assignment))
    if skipped:
        warnings.warn(
            "default_hook_assignment skipped layers that did not fire during "
            f"the sample pass (e.g. invoked functionally): {skipped}.",
            stacklevel=2,
        )
    if not assignment:
        _warn_zero_layers()

    return assignment


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
            # A positional embedding fed an *unbatched* index tensor — e.g.
            # nanoGPT's ``pos = arange(T)`` (shape ``(T,)``) added to every
            # sample — is captured with no batch dim.  Add a length-1 batch axis
            # so it validates and materialises as a single broadcast row (its
            # gradient is already summed over the batch by the broadcast add).
            if a.ndim == 1 and is_embedding(buf["_class_name"]):
                a = a.unsqueeze(0)
                g = g.unsqueeze(0)
            batch_first = layer_name not in self._non_batch_first_layers
            data[layer_name] = Factorized(activation=a, pre_activation_grad=g,
                                          module_kwargs=buf["_module_kwargs"],
                                          batch_first=batch_first)
            representation[layer_name] = "factorized"
            layer_types[layer_name] = buf["_class_name"]

        for layer_name, buf in self._param_buffers.items():
            for pname, grad in buf.items():
                if grad is None:
                    continue
                key = f"{layer_name}.{pname}"
                data[key] = grad
                representation[key] = "materialized"
                layer_types[key] = PARAM_GRAD_TYPES

        if not data:
            raise RuntimeError(
                "No gradient data assembled. "
                "Ensure backward() was called inside the collect() context."
            )

        indexing: dict = {}
        for name, val in data.items():
            if layer_types[name] == PARAM_GRAD_TYPES:
                # param_grad tensors are always (B, …) without a token dim
                indexing[name] = "batch"
            else:
                indexing[name] = (
                    "batch_token"
                    if isinstance(val, Factorized) and (
                        val.activation.ndim >= 3   # 3D seq, 4D Conv2d, 5D Conv3d
                        or not val.activation.is_floating_point()  # Embedding int
                    )
                    else "batch"
                )

        gradient = Gradient(
            representation=representation,
            data=data,
            layer_types=layer_types,
            indexing=indexing,
        )
        self._warn_broadcast_layers(gradient)
        return gradient

    def _warn_broadcast_layers(self, gradient: Gradient) -> None:
        """Warn (once per layer) about broadcast / batch-collapsed gradients.

        A factorized layer whose batch dim is 1 while the step batch is larger —
        e.g. a positional embedding added to every sample — carries a gradient
        that was *summed over the batch*, so it is **not** a per-sample gradient.
        Downstream per-sample attribution treats it as a single shared row, which
        is rarely what the user wants; surface it so they can exclude the layer.
        """
        batch = gradient.batch_size
        if batch <= 1:
            return
        for name, val in gradient.data.items():
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
