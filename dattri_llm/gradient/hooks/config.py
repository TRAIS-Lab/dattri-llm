"""HookManager configuration, layer selectors, and hook-assignment resolution."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from dattri_llm.gradient.hooks.hooks import (
    _has_trainable_params,
    _is_linear_io_capable,
    remove_hooks,
)
from dattri_llm.gradient.ops import ALL_LAYER_TYPES, canonical_class_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

# --------------------------------------------------------------------------- #
# Hook manager configuration                                                   #
# --------------------------------------------------------------------------- #


class _RegisterAll:
    """Sentinel type for :data:`REGISTER_ALL`.

    A selector value of :data:`REGISTER_ALL` requests that *every* layer
    applicable to a given hook family be registered, regardless of name.
    """

    _instance: _RegisterAll | None = None

    def __new__(cls) -> Self:
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
#   * ``None``        -- not provided (the family is not requested explicitly).
#   * ``REGISTER_ALL``-- register every applicable layer.
#   * ``list[str]``   -- register applicable layers whose name matches a regex.
Selector = _RegisterAll | list[str] | None

# Hook-family names and the layer_types marker used for materialized grads.
LINEAR_IO = "linear_io"
PARAM_GRAD = "param_grad"


_VALID_HOOK_TYPES = frozenset({LINEAR_IO, PARAM_GRAD})


class HookManagerConfig:
    r"""Configuration for :class:`HookManager`.

    The core control is :attr:`hook_types`, an explicit **assignment** mapping
    each fully-qualified layer name to the hook family it should use:

    * ``linear_io`` -- per-sample factorized hooks on linear-family layers
      (see :func:`register_linear_io_hooks`).
    * ``param_grad`` -- batch-level materialized parameter-gradient hooks,
      available for *any* layer with trainable parameters
      (see :func:`register_param_grad_hooks`).

    .. code-block:: python

        # explicit per-layer assignment
        HookManagerConfig(hook_types={"mlp.0": "linear_io", "lm_head": "param_grad"})

    The regex **selectors** ``linear_io`` and ``param_grad`` are an add-on that
    *extends* the assignment without having to spell out every layer.  Each
    selector is one of:

    * ``None`` -- add nothing.
    * :data:`REGISTER_ALL` -- add every layer applicable to that family.
    * ``list[str]`` -- add applicable layers whose fully-qualified name matches
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
    matched by the ``linear_io`` selector), a :exc:`ValueError` is raised -- one
    layer may only be registered with one hook family.

    **Default** (no arguments) -- register ``linear_io`` on every
    linear-IO-capable layer, and fall back to ``param_grad`` for any remaining
    layer that has trainable parameters but is not linear-IO-capable.

    **Manual layer types** -- :attr:`layer_types` maps a layer name to a
    layer-type string that overrides the type inferred by
    :func:`canonical_class_name`.  This is for user-defined layer classes whose
    class name is not recognised by the built-in type detection (e.g. a custom
    ``MyLinear`` subclass that should be treated as ``"nn.Linear"``)::

        HookManagerConfig(layer_types={"mlp.0": "nn.Linear"})

    The mapping may be incomplete: layers it does not mention keep the
    automatically detected type.  It is orthogonal to which layers get hooked --
    a layer named here is only relabelled, not forced to be hooked.  If a named
    layer is not hooked in the end, :class:`HookManager` emits a warning.

    **Manual module kwargs** -- :attr:`module_kwargs` maps a layer name to the
    hyperparameter dict normally produced by
    :func:`~dattri_llm.gradient.ops.extract_module_kwargs`, and the provided
    dict is used verbatim for that layer (no extraction).  Use it together
    with :attr:`layer_types` when the declared class stores its
    hyperparameters under non-standard attribute names (e.g. an HF
    ``LlamaRMSNorm``, whose epsilon is ``variance_epsilon``).  Layers it does
    not mention are extracted as usual.

    Every provided dict must contain ``has_bias`` (``bool``, whether the layer
    has a bias parameter).  The remaining required keys depend on the layer's
    (declared) type:

    * ``nn.Linear``, ``nn.Bilinear``, ``nn.Embedding``,
      ``transformers.pytorch_utils.Conv1D`` -- no additional keys.
    * ``nn.Conv1d/2d/3d``, ``nn.ConvTranspose1d/2d/3d`` -- ``kernel_size``,
      ``stride``, ``padding``, ``dilation`` (the module-attribute tuples,
      e.g. ``kernel_size=(3, 3)`` for a Conv2d).
    * ``nn.LayerNorm``, ``nn.RMSNorm`` -- ``normalized_shape`` (tuple) and
      ``eps`` (float).
    * ``nn.GroupNorm`` -- ``num_groups`` (int), ``num_channels`` (int),
      ``eps`` (float).
    * ``nn.InstanceNorm1d/2d/3d`` -- ``num_features`` (int), ``eps`` (float).
    * ``nn.EmbeddingBag`` -- ``mode`` (``"sum"`` or ``"mean"``).

    Example -- per-sample capture of a Llama-style RMSNorm::

        HookManagerConfig(
            hook_types={"model.norm": "linear_io"},
            layer_types={"model.norm": "nn.RMSNorm"},
            module_kwargs={"model.norm": {
                "has_bias": False, "normalized_shape": (4096,), "eps": 1e-6,
            }},
        )

    **Per-layer projection** -- :attr:`projection` enables capture-time random
    projection: instead of buffering a layer's raw factors, each backward pass
    projects them down to ``proj_dim`` on the training device, and only the
    small projected result is kept (on CPU).  It maps a layer name -- or
    ``"__default__"``, covering every hooked layer without its own entry -- to
    that layer's ``proj_kwargs`` dict.  A layer with neither an entry nor a
    ``"__default__"`` is captured raw (unprojected), so mixed configs are
    fine.  :attr:`projector` is the projection factory, following dattri's
    ``random_project`` protocol; ``None`` (the default) lazily imports
    dattri's ``random_project``.

    Keys consumed by the library:

    * ``proj_dim`` (int, **required**) -- target width of the projection.
    * ``factorize`` (bool, default ``True``) -- ``True`` projects the two
      factors independently (LoGRA style): the layer stays *factorized* at
      width ``proj_dim`` and is relabelled ``"nn.Linear"``.  This is defined
      for outer-product gradients: the linear / conv families, and the
      embedding family (whose integer ids are expanded to one-hot inputs
      first).  **Norm layers must use** ``factorize=False`` (TRAK style:
      materialize the per-sample weight gradient, then project it to a dense
      ``(B, proj_dim)`` block).
    * ``proj_seed`` (int, default ``0``) -- base seed.  Factorized projection
      uses ``proj_seed`` for the output-gradient factor and ``proj_seed + 1``
      for the activation factor (dattri's LoGRA convention).  Keep it fixed
      per layer so gradients captured at different steps stay comparable.
    * ``device`` -- where the projection runs.  The factors are moved to this
      device before projecting (dattri builds a device-specific projector for
      it); the small projected result is then buffered to CPU as usual.
      Unset, it defaults to the tensors' own (training) device.
      **Caution**: dattri's CPU and CUDA projectors do not
      produce the same projection for the same ``proj_seed``, and the valid
      ``proj_type`` sets differ (``"sjlt"``/``"grass"`` are CUDA-only) -- use
      one device consistently across every gradient that will be compared
      (train and test, capture-time and post-hoc).

    Every remaining key is forwarded verbatim to the projector factory.  With
    the default dattri projector that means:

    * ``proj_max_batch_size`` (int, **required** -- dattri's
      ``random_project`` has no default for it; must be a multiple of 8 for
      the CUDA projector).
    * ``proj_type`` (str, default ``"normal"``) -- e.g. ``"rademacher"`` or
      ``"sjlt"``.

    Example -- LoGRA projection on a GPT-2-style model.  The regexes hook the
    attention/MLP linears plus the token embedding; both fall through to
    ``"__default__"`` (embeddings project factorized too, via one-hot inputs),
    while a per-layer entry demonstrates overriding one layer to the
    materialize-then-project (TRAK) style::

        HookManagerConfig(
            linear_io=[r"transformer\.h\.\d+\.(attn|mlp)\.", r"wte$"],
            projection={
                "__default__": {          # project both factors (LoGRA)
                    "factorize": True,
                    "proj_dim": 512,
                    "proj_max_batch_size": 8,
                    "proj_type": "rademacher",
                    "device": "cuda",
                },
                "transformer.wte": {      # materialize-then-project (TRAK)
                    "factorize": False,
                    "proj_dim": 512,
                    "proj_max_batch_size": 8,
                    "device": "cuda",
                },
            },
        )

    Note that a ``"__default__"`` entry with ``factorize=True`` combined with
    a hook selection that includes norm layers (e.g.
    ``linear_io=REGISTER_ALL``) raises inside the first backward pass -- give
    those layers explicit ``factorize=False`` entries, or exclude them from
    hooking.
    """

    def __init__(
        self,
        hook_types: dict[str, str] | None = None,
        linear_io: Selector = None,
        param_grad: Selector = None,
        layer_types: dict[str, str] | None = None,
        module_kwargs: dict[str, dict] | None = None,
        projection: dict[str, dict] | None = None,
        projector: Callable | None = None,
    ) -> None:
        self.hook_types = self._validate_assignment(hook_types)
        self.linear_io = self._validate_selector(LINEAR_IO, linear_io)
        self.param_grad = self._validate_selector(PARAM_GRAD, param_grad)
        self.layer_types = self._validate_layer_types(layer_types)
        self.module_kwargs = self._validate_module_kwargs(module_kwargs)
        # Optional per-layer random projection applied to every assembled step
        # gradient (see :meth:`Gradient.project`).  ``projection`` is the per-layer
        # proj_kwargs map ``{layer_name: {factorize, proj_dim, ...}}`` (a
        # ``"__default__"`` entry covers unlisted layers); ``projector`` is the
        # projection factory, defaulting to dattri's ``random_project``.
        self.projection = self._validate_projection(projection)
        self.projector = projector

    @staticmethod
    def _validate_assignment(
        hook_types: dict[str, str] | None,
    ) -> dict[str, str]:
        if hook_types is None:
            return {}
        if not isinstance(hook_types, dict):
            raise TypeError(
                "hook_types must be a dict mapping layer name to hook type "
                f"({sorted(_VALID_HOOK_TYPES)}), got "
                f"{type(hook_types).__name__}.",
            )
        for layer_name, hook_type in hook_types.items():
            if hook_type not in _VALID_HOOK_TYPES:
                raise ValueError(
                    f"hook_types['{layer_name}'] = '{hook_type}' is not a valid "
                    f"hook type. Valid types: {sorted(_VALID_HOOK_TYPES)}.",
                )
        return dict(hook_types)

    @staticmethod
    def _validate_projection(
        projection: dict[str, dict] | None,
    ) -> dict[str, dict] | None:
        if projection is None:
            return None
        if not isinstance(projection, dict) or not all(
            isinstance(v, dict) for v in projection.values()
        ):
            raise TypeError(
                "projection must be a dict mapping layer name (or '__default__') "
                "to a proj_kwargs dict, e.g. "
                "{'__default__': {'factorize': True, 'proj_dim': 512}}.",
            )
        return {k: dict(v) for k, v in projection.items()}

    @staticmethod
    def _validate_layer_types(
        layer_types: dict[str, str] | None,
    ) -> dict[str, str]:
        if layer_types is None:
            return {}
        if not isinstance(layer_types, dict):
            raise TypeError(
                "layer_types must be a dict mapping layer name to a layer-type "
                f"string, got {type(layer_types).__name__}.",
            )
        for layer_name, layer_type in layer_types.items():
            if not isinstance(layer_name, str) or not isinstance(layer_type, str):
                raise TypeError(
                    "layer_types must map str layer names to str type names, got "
                    f"{layer_name!r}: {layer_type!r}.",
                )
        return dict(layer_types)

    @staticmethod
    def _validate_module_kwargs(
        module_kwargs: dict[str, dict] | None,
    ) -> dict[str, dict]:
        if module_kwargs is None:
            return {}
        if not isinstance(module_kwargs, dict) or not all(
            isinstance(k, str) and isinstance(v, dict) for k, v in module_kwargs.items()
        ):
            raise TypeError(
                "module_kwargs must be a dict mapping layer name to the layer's "
                "hyperparameter dict (see extract_module_kwargs), e.g. "
                "{'model.norm': {'has_bias': False, 'normalized_shape': (4096,), "
                "'eps': 1e-6}}.",
            )
        return {k: dict(v) for k, v in module_kwargs.items()}

    @staticmethod
    def _validate_selector(name: str, selector: Selector) -> Selector:
        if selector is None or selector is REGISTER_ALL:
            return selector
        if isinstance(selector, (list, tuple)):
            if not all(isinstance(p, str) for p in selector):
                raise TypeError(
                    f"{name} pattern list must contain only regex strings.",
                )
            return list(selector)
        raise TypeError(
            f"{name} must be None, REGISTER_ALL, or a list of regex strings, "
            f"got {type(selector).__name__}.",
        )

    @property
    def is_default(self) -> bool:
        """True when nothing was requested (the auto fallback applies)."""
        return (
            not self.hook_types and self.linear_io is None and self.param_grad is None
        )


def _resolve_projector(projector: Callable | None) -> Callable:
    """Return *projector*, or lazily fall back to dattri's ``random_project``.

    The import is deferred so configuring projection is the only thing that pulls
    in dattri's projection backend.
    """
    if projector is not None:
        return projector
    from dattri.func.projection import random_project

    return random_project


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

    * **Default** (``config.is_default``) -- every linear-IO-capable layer is
      assigned ``linear_io``; every other layer that directly owns a trainable
      parameter falls back to ``param_grad``.
    * **Explicit assignment** (``config.hook_types``) -- taken verbatim, after
      validating that each named layer exists and supports the requested family.
    * **Selector add-ons** (``config.linear_io`` / ``config.param_grad``) --
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
                "registered with one hook family.",
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
                "in the model.",
            )
        # A layer_types override declaring a supported factorizable type
        # (e.g. a hand-rolled HF RMSNorm declared as "nn.RMSNorm") makes
        # the layer eligible; without one it is only param_grad material.
        if (
            hook_type == LINEAR_IO
            and not _is_linear_io_capable(module)
            and config.layer_types.get(layer_name) not in ALL_LAYER_TYPES
        ):
            raise ValueError(
                f"Layer '{layer_name}' was assigned 'linear_io' but its type "
                f"({canonical_class_name(module)}) does not support factorized "
                "linear-IO hooks. If the layer computes the same math as a "
                "supported type, declare it via layer_types "
                f"(e.g. layer_types={{'{layer_name}': 'nn.RMSNorm'}}).",
            )
        if hook_type == PARAM_GRAD and not _has_trainable_params(module):
            raise ValueError(
                f"Layer '{layer_name}' was assigned 'param_grad' but has no "
                "trainable parameters.",
            )
        assign(layer_name, hook_type)

    # 2. Selector add-ons extend the assignment with applicable layers only.
    for name, module in modules.items():
        if _is_linear_io_capable(module) and _selector_matches(
            config.linear_io,
            name,
        ):
            assign(name, LINEAR_IO)
        if _has_trainable_params(module) and _selector_matches(
            config.param_grad,
            name,
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


def _derive_scalar_loss(output: object) -> torch.Tensor | None:
    """Best-effort reduction of a model output to a scalar for ``backward()``.

    Prefers ``output.loss``; otherwise sums a floating-point output tensor
    (``output`` itself, ``output.logits``, or the first float tensor in a
    dict/sequence).  Returns ``None`` when no differentiable scalar is found.
    """
    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss

    if isinstance(output, torch.Tensor):
        candidate: torch.Tensor | None = output
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
    loss_fn: Callable[[object], torch.Tensor] | None = None,
) -> dict[str, str]:
    """Discover the default-style assignment for layers that actually fire.

    Some modules are registered as sub-modules but invoked *functionally*
    rather than called as modules -- e.g. :class:`nn.MultiheadAttention` applies
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
        def _hook(_module: nn.Module, _inp: object, _out: object) -> None:
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
            except Exception as exc:  # noqa: BLE001 - user loss_fn may raise anything
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
