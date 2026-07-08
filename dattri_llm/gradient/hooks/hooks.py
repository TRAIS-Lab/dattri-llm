"""Low-level hook registration for gradient capture.

Two low-level hook families are provided:

Linear-IO factorized hooks -- ``register_linear_io_hooks``
---------------------------------------------------------
Registers forward and backward hooks on linear-family layers (``nn.Linear``,
``nn.Conv*``, ``nn.Embedding``, norm layers, ...) to capture the input
activations and output gradients needed for the outer-product identity:

    dL/dW ~ g^T x a    (per sample)

DataParallel support: each replica fires its hooks in a separate thread.
The hooks accumulate all replica calls in ``_act_parts`` / ``_grad_parts``
(thread-safe, tagged by source device index) rather than overwriting a single
slot.  Single-GPU and DDP usage is unaffected -- the lists always hold one
element in those cases.

Parameter gradient hooks -- ``register_param_grad_hooks``
---------------------------------------------------------
Registers ``Tensor.register_hook`` on the *parameters* of general modules.
The hook fires during the backward pass immediately after each parameter's
gradient is freshly computed -- not a post-backward ``.grad`` read, which can
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

import threading
from typing import TYPE_CHECKING

import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.ops import (
    canonical_class_name,
    extract_module_kwargs,
    is_embedding,
)

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]


def _queue_backward_end_callback(fn: Callable[[], None]) -> bool:
    """Schedule ``fn`` to run once the in-flight backward pass fully completes.

    Must be called from *within* a backward pass (e.g. a module full-backward
    hook).  The callback fires after the autograd engine has finished -- the
    point at which FSDP has written every (sharded) ``param.grad`` back to its
    original parameter, so reading ``param.grad`` inside ``fn`` is safe.

    Returns ``True`` if the callback was successfully queued, ``False`` if the
    autograd engine does not expose ``queue_callback`` on this build (in which
    case the caller should fall back to its existing behaviour).
    """
    try:
        torch.autograd.Variable._execution_engine.queue_callback(fn)
    except Exception:  # noqa: BLE001 - guarded probe of autograd internals
        return False
    return True


# -- Linear-IO-capable types -------------------------------------------------
# Layers whose per-sample gradient factorises as an outer product of the input
# activation and the output gradient (``dL/dW ~ g^T x a``).  These are the
# layers that can be hooked with the ``linear_io`` family.  Any layer (whether
# or not it appears here) can instead be hooked with the ``param_grad`` family,
# which materialises the batch-level parameter gradient directly.
#
# Membership is decided purely by type -- never by the module's name in the
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
# RMSNorm was added in PyTorch 2.4 -- guard for older versions.
if hasattr(nn, "RMSNorm"):
    _LINEAR_IO_TYPES += (nn.RMSNorm,)  # type: ignore[assignment]

# Buffer type alias.
# Keys: "activation", "grad_output", "_act_parts", "_grad_parts", "_lock"
LayerBuffer = dict


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
        # Projected per-sample gradient parts for a materialized (TRAK) layer;
        # for factorized (LoGRA) layers the projected factors live in
        # _act_parts/_grad_parts (already projected).  ``_proj_kw`` is the layer's
        # resolved proj_kwargs (or None to capture raw factors).
        # ``_device_id`` maps each replica's device id to a stack of on-device
        # forward activations, so the backward hook can pair (a, g) *per device*
        # (LIFO) before projecting -- a single shared slot would race across DDP
        # replica threads and drop calls for layers invoked more than once.
        "_proj_parts": [],
        "_proj_kw": None,
        "_device_id": {},
        "_lock": threading.Lock(),
    }


def register_linear_io_hooks(
    model: nn.Module,
    layer_names: set[str] | None = None,
    on_layer_forward: Callable[[str, torch.Tensor], None] | None = None,
    on_layer_backward: Callable[[str, torch.Tensor], None] | None = None,
    type_overrides: dict[str, str] | None = None,
    kwargs_overrides: dict[str, dict] | None = None,
    projection: dict[str, dict] | None = None,
    projector: Callable | None = None,
    offload_to_cpu: bool = False,
) -> tuple[dict[str, LayerBuffer], list[torch.utils.hooks.RemovableHook]]:
    """Register forward and backward hooks on linear-family layers.

    For each qualifying layer the function registers:

    * A **forward hook** that captures ``input[0]`` and appends it to
      ``buffers[name]["_act_parts"]``.
    * A **backward hook** that captures ``grad_output[0]`` and appends it to
      ``buffers[name]["_grad_parts"]``.

    Captures stay on their own (training) device by default so no transfer
    happens; ``offload_to_cpu=True`` moves each buffered tensor to CPU (a
    no-op when training on CPU).  For a projected layer the flag applies to
    the *projected* result -- the raw factors are never buffered either way.

    Optionally, user-supplied ``on_layer_forward`` and ``on_layer_backward``
    callables fire inside each hook with ``(layer_name, tensor)``
    immediately after capture.  Because these callables execute inside a
    PyTorch hook they are trainer-agnostic -- no trainer callback system is
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
            The tensor is on the capture device (CPU iff *offload_to_cpu*
            or CPU training).
        on_layer_backward: Optional callable fired after each backward hook
            capture.  Signature: ``(layer_name: str, grad_output: Tensor)``.
            The tensor is on the capture device (CPU iff *offload_to_cpu*
            or CPU training).
        projection: Optional per-layer proj_kwargs map (a ``"__default__"``
            entry covers unlisted layers); ``None`` captures raw factors.
        projector: Projection factory following dattri's ``random_project``
            protocol; required when *projection* is given.
        offload_to_cpu: When ``True``, move every buffered capture (the raw
            factors, or the projected result for a projected layer) to CPU.
            Default ``False``: buffers stay on the tensors' own device to
            avoid device transfers, at the cost of holding one step's
            captures in that device's memory.
        type_overrides: Optional mapping from layer name to a layer-type string
            that overrides the type inferred by :func:`canonical_class_name`.
            Use this for user-defined layer classes that subclass a supported
            linear-family type but whose class name is not recognised (e.g. a
            custom ``MyLinear`` that should be treated as ``"nn.Linear"``).
            Layers absent from the mapping fall back to ``canonical_class_name``.
        kwargs_overrides: Optional mapping from layer name to the hyperparameter
            dict normally produced by :func:`extract_module_kwargs`.  A listed
            layer uses the provided dict verbatim (no extraction) -- for
            declared-type overrides on classes whose attributes do not follow
            the standard names (e.g. HF ``LlamaRMSNorm``).  Layers absent from
            the mapping are extracted as usual.

    Returns:
        ``(buffers, handles)`` where ``buffers`` maps layer name to a
        :data:`LayerBuffer` and ``handles`` is a list of removable hook
        objects.
    """
    root: nn.Module = getattr(model, "module", model)

    buffers: dict[str, LayerBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        # An explicit type_overrides entry vouches for the layer's math (e.g.
        # a hand-rolled HF RMSNorm declared as "nn.RMSNorm"), so it bypasses
        # the type-based capability check.
        if (
            type_overrides is None or name not in type_overrides
        ) and not _is_linear_io_capable(module):
            continue
        if layer_names is not None and name not in layer_names:
            continue

        if not _has_trainable_params(module):
            continue

        buffers[name] = _make_layer_buffer()
        if type_overrides is not None and name in type_overrides:
            layer_type = type_overrides[name]
        else:
            layer_type = canonical_class_name(module)
        buffers[name]["_class_name"] = layer_type
        if kwargs_overrides is not None and name in kwargs_overrides:
            buffers[name]["_module_kwargs"] = dict(kwargs_overrides[name])
        else:
            buffers[name]["_module_kwargs"] = extract_module_kwargs(module, layer_type)
        if projection is not None:
            buffers[name]["_proj_kw"] = projection.get(
                name,
                projection.get("__default__"),
            )

        def _make_forward_hook(layer_name: str) -> Callable:
            def _fwd(_module: nn.Module, inp: tuple, _out: object) -> None:
                # A forward under no_grad / inference_mode can never produce a
                # backward, so capturing its activation would only contaminate
                # the step buffers (e.g. an RL log-prob or eval pass between
                # training steps).
                if not torch.is_grad_enabled():
                    return
                a = inp[0].detach()
                dev_idx = inp[0].device.index if inp[0].is_cuda else 0
                buf = buffers[layer_name]
                if buf["_proj_kw"] is None:
                    if offload_to_cpu:
                        a = a.cpu()
                    with buf["_lock"]:
                        buf["_act_parts"].append((dev_idx, a))
                    buf["activation"] = a
                else:
                    # Projected layer: keep the raw activation on-device so the
                    # backward hook projects (a, g) together; the full factor is
                    # never buffered.  A per-device *stack* pairs each call's a
                    # with its g even when a layer is invoked multiple times per
                    # forward (weight tying / RNN unroll) -- backward pops in the
                    # reverse (LIFO) order the forwards ran.
                    with buf["_lock"]:
                        buf["_device_id"].setdefault(dev_idx, []).append(a)
                if on_layer_forward is not None:
                    on_layer_forward(layer_name, a)

            return _fwd

        def _make_backward_hook(layer_name: str) -> Callable:
            def _bwd(
                _module: nn.Module,
                _grad_input: tuple,
                grad_output: tuple,
            ) -> None:
                g = grad_output[0].detach()
                dev_idx = grad_output[0].device.index if grad_output[0].is_cuda else 0
                buf = buffers[layer_name]
                if buf["_proj_kw"] is None:
                    if offload_to_cpu:
                        g = g.cpu()
                    with buf["_lock"]:
                        buf["_grad_parts"].append((dev_idx, g))
                    buf["grad_output"] = g
                else:
                    _capture_projected(buf, g, dev_idx, projector, offload_to_cpu)
                if on_layer_backward is not None:
                    on_layer_backward(layer_name, g)

            return _bwd

        handles.extend(
            (
                module.register_forward_hook(_make_forward_hook(name)),
                module.register_full_backward_hook(_make_backward_hook(name)),
            ),
        )

    return buffers, handles


def _capture_projected(
    buf: LayerBuffer,
    g: torch.Tensor,
    dev_idx: int,
    projector: Callable,
    offload_to_cpu: bool = False,
) -> None:
    """Project one micro-batch's ``(activation, grad_output)`` into the buffer.

    Called from the backward hook of a projected layer, pairing ``g`` with the
    matching per-replica forward activation.  For a **factorized** (LoGRA) layer
    the projected factors are appended to ``_act_parts``/``_grad_parts``; for a
    **materialized** (TRAK) layer the projected per-sample gradient is appended to
    ``_proj_parts``.  Only the small projected result is retained (moved to CPU
    when *offload_to_cpu*) -- the raw factors are discarded here, so the buffer
    never holds the full gradient.
    """
    with buf["_lock"]:
        stack = buf["_device_id"].get(dev_idx)
        a = stack.pop() if stack else None
    if a is None:
        return  # backward without a matching forward capture -- nothing to project

    kw = dict(buf["_proj_kw"])
    factorize = kw.pop("factorize", True)
    layer_type = buf["_class_name"]
    module_kwargs = buf["_module_kwargs"]
    if a.ndim == 1 and is_embedding(layer_type):
        a, g = a.unsqueeze(0), g.unsqueeze(0)

    if factorize:
        a_p, g_p = ops._project_factorized(
            a,
            g,
            layer_type,
            projector,
            module_kwargs,
            **kw,
        )
        if offload_to_cpu:
            a_p, g_p = a_p.cpu(), g_p.cpu()
        with buf["_lock"]:
            buf["_act_parts"].append((dev_idx, a_p))
            buf["_grad_parts"].append((dev_idx, g_p))
    else:
        mat = ops._project_materialized(
            a,
            g,
            layer_type,
            projector,
            module_kwargs,
            **kw,
        )
        if offload_to_cpu:
            mat = mat.cpu()
        with buf["_lock"]:
            buf["_proj_parts"].append((dev_idx, mat))


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
# Value: most recently computed gradient tensor (on the parameter's device,
# or CPU under offload_to_cpu), or None before the first backward pass.
ParamGradBuffer = dict  # {param_name: Tensor | None}


def register_param_grad_hooks(
    model: nn.Module,
    layer_names: set[str] | None = None,
    on_param_grad: Callable[[str, str, torch.Tensor], None] | None = None,
    offload_to_cpu: bool = False,
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
            The tensor is on the capture device (CPU iff *offload_to_cpu*
            or CPU training).
        offload_to_cpu: When ``True``, move each buffered gradient to CPU.
            Default ``False``: the buffer keeps the tensor on the parameter's
            own device (no transfer).

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

            def _make_hook(ln: str, pn: str) -> Callable:
                def _hook(grad: torch.Tensor) -> None:
                    # grad is the freshly-computed gradient for this parameter.
                    # It arrives here before being written to param.grad, so
                    # there is no risk of reading a stale or None value.
                    g = grad.detach()
                    if offload_to_cpu:
                        g = g.cpu()
                    buffers[ln][pn] = g
                    if on_param_grad is not None:
                        on_param_grad(ln, pn, g)

                return _hook

            handles.append(param.register_hook(_make_hook(layer_name, pname)))

    return buffers, handles


def register_linear_param_hooks(
    model: nn.Module,
    layer_names: set[str] | None = None,
    on_linear_param_grad: Callable[[str, str, torch.Tensor], None] | None = None,
) -> tuple[int, list[torch.utils.hooks.RemovableHook]]:
    """Register post-accumulate-grad hooks on linear layers' trainable params.

    Unlike :func:`register_param_grad_hooks`, which uses
    ``Tensor.register_hook`` (fires *before* ``param.grad`` is accumulated),
    this function uses ``Tensor.register_post_accumulate_grad_hook``
    (PyTorch >= 2.0) which fires *after* ``param.grad`` is written.

    This guarantees that when the callback runs, ``param.grad`` is non-None --
    a precondition for any callback that needs to read or modify weight
    gradients in-place
    (e.g. :class:`~dattri_llm.gradient.callbacks.DataSelectionCallback`).

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
            ``grad`` is ``param.grad.detach()`` -- left on the parameter's
            own device (no copy is made; consumers that need CPU move it
            themselves).

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

            def _make_hook(ln: str, pn: str) -> Callable:
                def _hook(p: torch.nn.Parameter) -> None:
                    if on_linear_param_grad is not None:
                        # No .cpu() here: the only in-tree consumer is the
                        # HookManager's step-completion counter, which ignores
                        # the value -- a device transfer would be pure waste.
                        on_linear_param_grad(ln, pn, p.grad.detach())

                return _hook

            handles.append(
                param.register_post_accumulate_grad_hook(_make_hook(name, pname)),
            )

    return n_params, handles
