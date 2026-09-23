"""Low-level hook registration for gradient capture.

Two low-level hook families are provided:

Linear-IO factorized hooks -- ``register_linear_io_hooks``
---------------------------------------------------------
Registers forward and backward hooks on linear-family layers (``nn.Linear``,
``nn.Conv*``, ``nn.Embedding``, norm layers, ...) to capture the input
activations and output gradients needed for the outer-product identity:

    dL/dW ~ g^T x a    (per sample)

Under DataParallel each replica fires its hooks in a separate thread.  The
hooks append every replica's call to ``_act_parts`` / ``_grad_parts``
(thread-safe, tagged by source device index).  Single-device and DDP usage
appends one element per call.

Parameter gradient hooks -- ``register_param_grad_hooks``
---------------------------------------------------------
Registers ``Tensor.register_hook`` on the *parameters* of general modules.
The hook fires during the backward pass immediately after each parameter's
gradient is computed, so the captured value is that step's gradient: never
``None`` and never a stale accumulated value from an earlier step.

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
import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.ops import (
    canonical_class_name,
    dtypes,
    extract_module_kwargs,
    is_embedding,
    to_3d,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dattri_llm.gradient.optimizer_state import GradientPreconditioner

try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]


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


def _is_invasive_capable(module: nn.Module) -> bool:
    """Return ``True`` if *module* supports ``invasive_linear_io`` hooks.

    The invasive variant replaces the layer's forward with a custom autograd
    ``Function`` whose forward is exactly ``F.linear``, so it is restricted to
    plain :class:`torch.nn.Linear`.  (Unlike :func:`_is_linear_io_capable`,
    which also admits conv/embedding/norm layers whose forwards differ.)
    """
    return isinstance(module, nn.Linear)


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
        # forward activations, so the backward hook pairs (a, g) *per device*
        # (LIFO) before projecting; the per-device stacks stay correct across
        # DataParallel replica threads and for layers invoked more than once.
        "_proj_parts": [],
        "_proj_kw": None,
        # The layer's LoGRA projections resolved once per side ("a" / "g"):
        # matrix, bias row, dtype and device (see _projection_matrix).
        "_proj_matrix": {},
        "_capture_style": "factorized",
        "_device_id": {},
        # Grad-enabled forward invocations observed this step.  Each forward
        # produces exactly one backward, so this is the per-step target for
        # the layer's backward count -- it adapts to however many DataParallel
        # replicas actually ran (a trailing batch smaller than the device
        # count uses fewer) and to repeated invocations (weight tying).
        # Forwards left unmatched at the end of the backward pass (see below)
        # are discarded and deducted by the manager's reconciliation.
        "_fwd_fires": 0,
        # Bracket matching of forward and backward fires, per device.
        # ``_unmatched[dev]`` stacks the indices (into ``_act_parts``, raw
        # path) or a sentinel (projected path -- the activation itself lives
        # on the ``_device_id`` stack) of forwards awaiting their backward;
        # each backward pops its own forward LIFO (autograd runs a device's
        # graph in reverse creation order).  ``_pair_pos``, aligned with the
        # appended grad (or projected) parts, records each matched pair's
        # per-device *invocation* position, so assembly can regroup repeated
        # invocations of one layer (checkpoint recomputation leaves an
        # unmatched forward instead -- discarded at backward end).
        "_unmatched": {},
        "_pair_pos": [],
        "_lock": threading.Lock(),
    }


def register_linear_io_hooks(
    model: nn.Module,
    layer_names: set[str] | None = None,
    on_layer_forward: Callable[[str, torch.Tensor], None] | None = None,
    on_layer_backward: Callable[[str, torch.Tensor], None] | None = None,
    type_overrides: dict[str, str] | None = None,
    kwargs_overrides: dict[str, dict] | None = None,
    projection_kwargs: dict[str, dict] | None = None,
    capture_style: str = "factorized",
    projector: ops.DattriProjector | None = None,
    offload_to_cpu: bool = False,
    preconditioner: GradientPreconditioner | None = None,
    include_frozen: bool = False,
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
        projection_kwargs: Optional per-layer proj_kwargs map (a ``"__default__"``
            entry covers unlisted layers); ``None`` captures raw factors.
        projector: The :class:`~dattri_llm.gradient.ops.DattriProjector`
            applying the projection (it owns the projection-matrix cache);
            required when *projection_kwargs* is given.
        capture_style: The representation a layer is buffered in where there
            is a choice -- ``"factorized"`` (the raw or logra-projected
            factors), ``"materialized"`` (the dense per-sample gradient, formed
            in the backward hook), or ``"auto"`` (the cheaper of the two by
            :func:`~dattri_llm.gradient.ops.should_materialize`, per layer and
            micro-batch; a layer's choice is fixed by its first micro-batch of
            a step).
        include_frozen: Hook a layer even when none of its parameters
            requires grad (see :class:`HookManagerConfig`).
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
        preconditioner: Optional
            :class:`~dattri_llm.gradient.optimizer_state.GradientPreconditioner`.
            While it is ``enabled``, every layer's capture is the optimizer's
            per-sample update direction instead of the raw gradient: the
            per-sample gradient is formed on the captured coordinates (the
            whole layer when unprojected, the kept subset under ``"mask"``,
            the whole layer *before* a ``"dense"`` projection) and mapped
            through the optimizer
            state, and only that dense result is buffered.  Like projection,
            it runs inside the backward hook, so the state read is the one
            the coming ``optimizer.step()`` updates from.  The ``"logra"``
            style cannot be preconditioned (the map does not commute with a
            factor-side projection).  The flag must not change between a
            layer's forward and its backward.

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

        if not include_frozen and not _has_trainable_params(module):
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
        if projection_kwargs is not None:
            buffers[name]["_proj_kw"] = projection_kwargs.get(
                name,
                projection_kwargs.get("__default__"),
            )
        buffers[name]["_capture_style"] = capture_style

        def _make_forward_hook(layer_name: str) -> Callable:
            def _fwd(_module: nn.Module, inp: tuple, _out: object) -> None:
                # A forward under no_grad / inference_mode produces no
                # backward (e.g. an RL log-prob or eval pass between training
                # steps); its activation is not part of the step.
                if not torch.is_grad_enabled():
                    return
                a = inp[0].detach()
                dev_idx = inp[0].device.index if inp[0].is_cuda else 0
                buf = buffers[layer_name]
                emit_type, emit_kwargs = buf["_class_name"], buf["_module_kwargs"]
                if _raw_capture(buf, preconditioner):
                    if offload_to_cpu:
                        a = a.cpu()
                    with buf["_lock"]:
                        buf["_unmatched"].setdefault(dev_idx, []).append(
                            len(buf["_act_parts"]),
                        )
                        buf["_act_parts"].append((dev_idx, a))
                        buf["_fwd_fires"] += 1
                    buf["activation"] = a
                else:
                    # Projected and/or preconditioned layer: the dense result is
                    # formed at backward.  For linear (a-side independent of g)
                    # under a LoGRA style the activation is projected *here*, so
                    # the buffer holds only the small a_p and the callback sees
                    # the projected factor; other cases keep the raw activation
                    # for the joint backward capture.  A per-device *stack* pairs
                    # each call's a with its g even when a layer is invoked
                    # multiple times per forward (weight tying / RNN unroll) --
                    # backward pops LIFO.
                    if _preprojects_activation(
                        buf["_class_name"], _projection_style(buf)
                    ):
                        a = _projection_matrix(buf, "a", a, projector)(
                            to_3d(dtypes.align(a)[0])
                        )
                        emit_type, emit_kwargs = "nn.Linear", None
                    with buf["_lock"]:
                        buf["_device_id"].setdefault(dev_idx, []).append(a)
                        buf["_fwd_fires"] += 1
                if on_layer_forward is not None:
                    on_layer_forward(layer_name, a, emit_type, emit_kwargs)

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
                emit_g, emit_type, emit_kwargs = (
                    g,
                    buf["_class_name"],
                    buf["_module_kwargs"],
                )
                if _raw_capture(buf, preconditioner):
                    if offload_to_cpu:
                        g = g.cpu()
                    with buf["_lock"]:
                        stack = buf["_unmatched"].get(dev_idx)
                        if not stack:
                            _warn_orphan_backward(layer_name)
                            return
                        stack.pop()
                        # Per-device invocation position of the matched
                        # forward: forwards pushed 0..n-1 in order, backwards
                        # pop LIFO, so the popped one sat at len(stack).
                        buf["_pair_pos"].append((dev_idx, len(stack)))
                        buf["_grad_parts"].append((dev_idx, g))
                    buf["grad_output"] = g
                    emit_g = g
                else:
                    matched, g_p = _capture_projected(
                        buf,
                        g,
                        dev_idx,
                        projector,
                        offload_to_cpu,
                        preconditioner=preconditioner,
                        layer_name=layer_name,
                    )
                    if not matched:
                        _warn_orphan_backward(layer_name)
                        return
                    if g_p is not None:
                        # Pre-projected linear: hand the callback the projected
                        # gradient factor (paired with the forward's a_p).
                        emit_g, emit_type, emit_kwargs = g_p, "nn.Linear", None
                if on_layer_backward is not None:
                    on_layer_backward(layer_name, emit_g, emit_type, emit_kwargs)

            return _bwd

        handles.extend(
            (
                module.register_forward_hook(_make_forward_hook(name)),
                module.register_full_backward_hook(_make_backward_hook(name)),
            ),
        )

    return buffers, handles


class _InvasiveLinearFunction(torch.autograd.Function):
    """``nn.Linear`` op whose backward skips the weight/bias gradient.

    ``forward`` is identical to ``F.linear`` (same output).  ``backward`` returns
    the gradient w.r.t. the **input** (``grad_output @ weight`` -- keeping
    backprop to earlier layers bit-identical to a normal linear) but returns
    ``None`` for the weight and bias, which tells autograd **not** to run the
    ``d_out x d_in`` weight-gradient matmul.  The factorized ``linear_io``
    capture only reads the input activation and output gradient (grabbed by the
    ordinary forward/backward hooks), never ``weight.grad``, so the captured
    factors are unchanged -- only the wasted weight-gradient work is removed.

    .. warning::
        A layer running this op does **not** accumulate ``weight.grad`` /
        ``bias.grad`` and therefore cannot be updated by an optimizer while it
        is active.  It is meant for gradient *capture* (attribution), not
        training.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        return nn.functional.linear(tensor_input, weight, bias)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        (weight,) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Same as nn.Linear's input gradient; keeps backprop flowing.
            grad_input = grad_output @ weight.to(grad_output.dtype)
        # None for weight and bias -> autograd skips the weight-gradient matmul.
        return grad_input, None, None


def install_invasive_forward(
    model: nn.Module,
    layer_names: set[str],
    should_intervene: Callable[[], bool],
) -> list[Callable[[], None]]:
    """Override each named ``nn.Linear``'s forward to skip its weight/bias grad.

    Routes the layer's forward through :class:`_InvasiveLinearFunction` so the
    ``d_out x d_in`` weight-gradient matmul is never computed (the factorized
    capture doesn't need it).  ``should_intervene`` is consulted on **every**
    forward: when it returns ``False`` (e.g. the manager is not collecting) the
    layer runs its **original** forward, so ordinary training -- which needs
    ``weight.grad`` -- is unaffected outside a collection context.

    The override is per-instance and pairs the ordinary ``register_linear_io_hooks``
    capture (whose forward/backward hooks still fire and capture identical
    factors); only the weight-gradient computation changes.  Restrict the caller
    to layers that are genuinely ``nn.Linear``.

    Args:
        model: The model (plain, ``DataParallel``, or DDP-wrapped).
        layer_names: Fully-qualified names of the ``nn.Linear`` layers to patch.
        should_intervene: Zero-arg predicate consulted per forward; ``True``
            routes through the weight-grad-skipping op, ``False`` runs the
            original forward.

    Returns:
        A list of zero-arg revert closures that restore each module's original
        forward exactly (idempotent -- safe to call once).

    Raises:
        ValueError: if a named layer is not a plain ``nn.Linear``.
    """
    root: nn.Module = getattr(model, "module", model)
    reverts: list[Callable[[], None]] = []
    for name, module in root.named_modules():
        if name not in layer_names:
            continue
        if not _is_invasive_capable(module):
            msg = (
                f"Layer '{name}' was assigned 'invasive_linear_io' but is "
                f"{canonical_class_name(module)}, not nn.Linear. The invasive "
                "hook overrides the linear forward and only supports nn.Linear."
            )
            raise ValueError(msg)

        original_forward = module.forward
        had_own_forward = "forward" in vars(module)

        def _make_forward(mod: nn.Module, orig: Callable) -> Callable:
            def _invasive_forward(
                tensor_input: torch.Tensor,
                *args: object,
                **kwargs: object,
            ) -> torch.Tensor:
                if should_intervene():
                    return _InvasiveLinearFunction.apply(
                        tensor_input,
                        mod.weight,
                        mod.bias,
                    )
                return orig(tensor_input, *args, **kwargs)

            return _invasive_forward

        module.forward = _make_forward(module, original_forward)

        def _make_revert(
            mod: nn.Module,
            orig: Callable,
            had_own: bool,
        ) -> Callable[[], None]:
            def _revert() -> None:
                if had_own:
                    mod.forward = orig
                else:
                    # Drop the instance override -> back to the class method.
                    vars(mod).pop("forward", None)

            return _revert

        reverts.append(_make_revert(module, original_forward, had_own_forward))

    return reverts


def _warn_orphan_backward(layer_name: str) -> None:
    """Warn about a backward fire with no unconsumed forward capture.

    Every grad-enabled forward owes exactly one backward, so an orphan means
    something abnormal produced *extra* backwards -- e.g. a second
    ``backward(retain_graph=True)`` over an already-consumed step.  The fire
    is discarded (not buffered, not counted) rather than corrupting the step.
    """
    warnings.warn(
        f"Discarding a backward fire on layer '{layer_name}' with no matching "
        "forward capture (e.g. a repeated backward over a retained graph); "
        "this is not a normal forward+backward step.",
        stacklevel=2,
    )


def _projection_matrix(
    buf: LayerBuffer,
    side: str,
    x: torch.Tensor,
    projector: ops.DattriProjector | None,
) -> ops.ProjectionMatrix:
    """The layer's LoGRA projection of *side* (``"a"``: the activation, with
    the bias ones-column and dattri's ``proj_seed + 1``; ``"g"``: the output
    gradient, ``proj_seed``), resolved on first use and reused while the
    feature keeps its width, dtype and device.  It is the same map
    :func:`~dattri_llm.gradient.ops.project_activation` /
    :func:`~dattri_llm.gradient.ops.project_gradient` apply.
    """
    matrix = buf["_proj_matrix"].get(side)
    (x,) = dtypes.align(x)
    if matrix is not None and matrix.matches(x):
        return matrix
    kw = {k: v for k, v in buf["_proj_kw"].items() if k != "style"}
    seed = kw.pop("proj_seed", 0)
    module_kwargs = buf["_module_kwargs"]
    with_bias = side == "a" and module_kwargs is not None and module_kwargs["has_bias"]
    matrix = ops.DattriProjector.coerce(projector).resolve(
        x.shape[-1],
        proj_seed=seed + 1 if side == "a" else seed,
        include_bias=with_bias,
        device=x.device,
        dtype=x.dtype,
        **kw,
    )
    buf["_proj_matrix"][side] = matrix
    return matrix


def _preprojects_activation(layer_type: str, style: str | None) -> bool:
    """Whether a projected layer's activation is projected in the *forward* hook.

    ``True`` for linear layers under the double-sided (``"logra"``) style: the
    a-side projection does not depend on the gradient, so it can run at forward
    -- the buffer then holds the small ``(B, T, proj_dim)`` factor instead of
    the full activation, and the projected factors reach the per-layer
    callbacks.  Other types stay on the joint backward projection (e.g. an
    embedding's g-masking reads the raw activation's padding ids, which are
    gone once projected).
    """
    return ops.is_linear(layer_type) and style == "logra"


def _preconditioning(preconditioner: GradientPreconditioner | None) -> bool:
    """Whether captures are currently mapped through the optimizer."""
    return preconditioner is not None and preconditioner.enabled


def _projection_style(buf: LayerBuffer) -> str | None:
    """The layer's projection style, ``None`` for an unprojected layer."""
    if buf["_proj_kw"] is None:
        return None
    return buf["_proj_kw"].get("style", "logra")


def _raw_capture(
    buf: LayerBuffer, preconditioner: GradientPreconditioner | None
) -> bool:
    """Whether the layer takes the raw path: factors buffered as they appear,
    with nothing to decide at backward -- unprojected, ``"factorized"`` capture
    style, and no optimizer map.
    """
    return (
        buf["_proj_kw"] is None
        and buf["_capture_style"] == "factorized"
        and not _preconditioning(preconditioner)
    )


def _keep_factors(buf: LayerBuffer, seq_len: int, k_a: int, k_g: int) -> bool:
    """Whether this micro-batch keeps the factors, under the layer's capture
    style -- fixed by the step's first micro-batch so a step never mixes the
    two representations.
    """
    if buf["_act_parts"]:
        return True
    if buf["_proj_parts"]:
        return False
    return not ops.should_materialize(buf["_capture_style"], seq_len, k_a, k_g)


def _capture_projected(
    buf: LayerBuffer,
    g: torch.Tensor,
    dev_idx: int,
    projector: ops.DattriProjector | None,
    offload_to_cpu: bool = False,
    *,
    preconditioner: GradientPreconditioner | None = None,
    layer_name: str = "",
) -> tuple[bool, torch.Tensor | None]:
    """Reduce one micro-batch's ``(activation, grad_output)`` into the buffer.

    Called from the backward hook of every layer that is not on the raw path
    (see :func:`_raw_capture`), pairing ``g`` with the matching per-replica
    forward activation.  The projection ``style`` and the capture style decide
    what is buffered:

    * ``"logra"`` -- the projected factors go to ``_act_parts``/``_grad_parts``
      when the factors are kept, else their token-summed outer product (formed
      in the small projected space) is appended to ``_proj_parts``.
    * ``"dense"`` -- the materialize-then-project per-sample block goes to
      ``_proj_parts``.
    * ``"mask"`` -- the kept coordinates, gathered from the raw factors
      without materializing, go to ``_proj_parts``.
    * no projection -- the raw factors are kept, or (``"materialized"``, or
      ``"auto"`` when the dense gradient is smaller) the materialized
      per-sample gradient goes to ``_proj_parts``; a preconditioned layer is
      always materialized.

    With an enabled *preconditioner* the per-sample entries are mapped through
    the optimizer before buffering -- after the mask gather, and before a
    ``"dense"`` projection.

    When the layer pre-projected its activation at forward
    (:func:`_preprojects_activation`), the popped ``a`` is already the projected
    factor ``a_p``, so only ``g`` is projected here.

    Only the reduced result is retained (moved to CPU when *offload_to_cpu*),
    so a materializing layer never holds its factors beyond this call.

    Returns:
        ``(matched, g_p)`` -- ``matched`` is ``False`` for an orphan backward
        (nothing buffered).  ``g_p`` is the projected gradient factor when the
        activation was pre-projected (so the caller can emit the projected
        factors to callbacks), else ``None``.
    """
    with buf["_lock"]:
        stack = buf["_device_id"].get(dev_idx)
        a = stack.pop() if stack else None
        pair_pos = (dev_idx, len(stack)) if a is not None else None
    if a is None:
        return False, None

    proj_kw = buf["_proj_kw"]
    kw = dict(proj_kw or {})
    style = kw.pop("style", "logra") if proj_kw is not None else None
    precondition = preconditioner if _preconditioning(preconditioner) else None
    layer_type = buf["_class_name"]
    module_kwargs = buf["_module_kwargs"]

    def keep_factors(a_f: torch.Tensor, g_f: torch.Tensor) -> None:
        if offload_to_cpu:
            a_f, g_f = a_f.cpu(), g_f.cpu()
        with buf["_lock"]:
            buf["_act_parts"].append((dev_idx, a_f))
            buf["_grad_parts"].append((dev_idx, g_f))
            buf["_pair_pos"].append(pair_pos)

    def keep_dense(mat: torch.Tensor) -> None:
        if offload_to_cpu:
            mat = mat.cpu()
        with buf["_lock"]:
            buf["_proj_parts"].append((dev_idx, mat))
            buf["_pair_pos"].append(pair_pos)

    if style == "logra":
        # Activation pre-projected at forward: ``a`` is already ``a_p``;
        # project only ``g``.  The stored factors are identical to the joint
        # path (same seeds).
        if _preprojects_activation(layer_type, style):
            a_p = a
            g_p = _projection_matrix(buf, "g", g, projector)(to_3d(dtypes.align(g)[0]))
            emit = g_p
        else:
            if a.ndim == 1 and is_embedding(layer_type):
                a, g = a.unsqueeze(0), g.unsqueeze(0)
            a_p, g_p = ops.project_factors(
                a, g, layer_type, projector, module_kwargs, **kw
            )
            emit = None
        seq_len = a_p.shape[1] if a_p.ndim == 3 else 1
        if _keep_factors(buf, seq_len, a_p.shape[-1], g_p.shape[-1]):
            keep_factors(a_p, g_p)
        else:
            # Projected outer-product factors behave as a plain linear layer;
            # module_kwargs=None avoids re-preprocessing.
            keep_dense(ops.materialize_factors(a_p, g_p, "nn.Linear"))
        return True, emit

    if a.ndim == 1 and is_embedding(layer_type):
        a, g = a.unsqueeze(0), g.unsqueeze(0)

    if style == "mask":
        mat = ops.mask_factors(a, g, layer_type, projector, module_kwargs, **kw)
        if precondition is not None:
            mat = precondition(layer_name, mat, proj_kw)
        keep_dense(mat)
        return True, None
    if style == "dense":
        if precondition is not None:
            # Precondition the whole gradient, then project it.
            include_bias = bool(kw.pop("include_bias", True))
            mat = ops.materialize_factors(a, g, layer_type, module_kwargs, include_bias)
            mat = precondition(layer_name, mat, proj_kw)
            mat = ops.apply_projection(projector, mat, **kw)
        else:
            mat = ops.project_materialized_factors(
                a, g, layer_type, projector, module_kwargs, **kw
            )
        keep_dense(mat)
        return True, None

    # Unprojected.  Preconditioned: the whole per-sample gradient, mapped.
    if precondition is not None:
        mat = ops.materialize_factors(a, g, layer_type, module_kwargs)
        keep_dense(precondition(layer_name, mat, None))
        return True, None
    # Otherwise the capture style decides: the raw factors, or the dense
    # per-sample gradient when it is the smaller (or requested) form.
    seq_len = a.shape[1] if a.ndim == 3 else 1
    n_in = a.shape[-1] + 1  # the bias column materialize adds
    n_out = g.shape[-1]
    if _keep_factors(buf, seq_len, n_in, n_out):
        keep_factors(a, g)
    else:
        keep_dense(ops.materialize_factors(a, g, layer_type, module_kwargs))
    return True, None


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
    backward pass at the moment the gradient for that parameter is computed,
    writing it to the buffer.  The captured value is therefore that step's
    gradient for the parameter: never ``None``, never a value accumulated
    from an earlier step, and complete even while other parameters of the
    same module still have pending contributions.

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
                    # grad is this parameter's gradient for the step; it
                    # arrives here before being written to param.grad.
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
                        # The gradient is handed over on its own device; no
                        # copy is made.
                        on_linear_param_grad(ln, pn, p.grad.detach())

                return _hook

            handles.append(
                param.register_post_accumulate_grad_hook(_make_hook(name, pname)),
            )

    return n_params, handles
