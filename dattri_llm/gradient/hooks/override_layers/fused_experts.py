"""Per-sample capture of fused Mixture-of-Experts modules.

Current model code stores all experts of an MoE block in one module holding
3-D parameters -- ``gate_up_proj`` of shape ``(E, 2*d_ff, d)`` and
``down_proj`` of shape ``(E, d, d_ff)`` (Hugging Face Transformers 5,
gpt-oss, Megatron-style trainers) -- and the router as a module with a bare
``(E, d)`` weight.  Neither contains an ``nn.Linear``: the expert matmuls
are functional calls on slices of the 3-D tensor, often dispatched to a
grouped-matmul kernel, so module hooks find nothing to attach to and the
default capture falls back to a batch-level ``param_grad`` of the whole
tensor.

This module gives such a block the shape the hooks understand without
changing the model: while a :class:`~dattri_llm.gradient.hooks.HookManager`
is collecting, the experts module's ``forward`` is replaced by the
reference per-expert loop (the same computation the eager implementation
performs), written so that each expert's two projections go through a
parameter-free proxy :class:`SliceLinear` that applies ``F.linear`` with
the expert's slice of the 3-D parameter.  The proxies are ordinary hooked
layers of type ``nn.Linear``; their inputs are the rows routed to the
expert, which :class:`~dattri_llm.gradient.hooks.routing.RoutedTokenTracker`
regroups by sample.  The router's ``F.linear`` call is likewise routed
through a proxy during its forward.  Autograd flows through the slicing
into the real parameters, so weight gradients, the optimizer and the
model's state dict are untouched; outside ``collect()`` the original
forwards run (including any fused kernel).  This is the fused-expert
analogue of :mod:`~dattri_llm.gradient.hooks.override_layers.invasive_linear`.

Records are named ``<experts>.dattri_experts.<e>.gate_up`` and ``.down``
(and ``<router>.dattri_router``), one factorized layer per expert and
projection, in the ``(d_out, d_in)`` layout of the proxy's logical weight
(the transpose of the stored slice for a transposed layout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ``F.linear`` as imported: the router override patches the attribute for
# the duration of the router's forward, and the proxies must not recurse
# into the patched version.
_linear = F.linear

GATE_UP = "gate_up_proj"
DOWN = "down_proj"
PROXY_EXPERTS = "dattri_experts"
PROXY_ROUTER = "dattri_router"
_INDEX_KWARGS = ("top_k_index", "router_indices", "selected_experts", "topk_indices")
_WEIGHT_KWARGS = ("top_k_weights", "routing_weights", "router_scores", "topk_weights")


class SliceLinear(nn.Module):
    """``F.linear`` with one expert's slice of a 3-D parameter as weight.

    Owns no parameters: ``weight`` and ``bias`` are read from *owner* at
    call time, so the module is a view of the fused parameters for the
    hooks and autograd reaches the real tensors.

    Args:
        owner: The fused experts module.
        weight_name: Name of the 3-D weight parameter on *owner*.
        index: The expert index.
        bias_name: Name of the ``(E, d_out)`` bias parameter, or ``None``.
        transposed: Whether the stored slice is ``(d_in, d_out)`` rather
            than ``(d_out, d_in)``.
    """

    def __init__(
        self,
        owner: nn.Module,
        weight_name: str,
        index: int,
        bias_name: str | None = None,
        *,
        transposed: bool = False,
    ) -> None:
        super().__init__()
        # Not a submodule: the owner holds the proxies, not the reverse.
        object.__setattr__(self, "_owner", owner)
        self._weight_name = weight_name
        self._bias_name = bias_name
        self._index = index
        self._transposed = transposed

    @property
    def weight(self) -> torch.Tensor:
        """The expert's ``(d_out, d_in)`` weight."""
        w = getattr(self._owner, self._weight_name)[self._index]
        return w.transpose(0, 1) if self._transposed else w

    @property
    def bias(self) -> torch.Tensor | None:
        """The expert's bias, or ``None``."""
        if self._bias_name is None:
            return None
        return getattr(self._owner, self._bias_name)[self._index]

    def proxied_parameters(self) -> Iterator[nn.Parameter]:
        """The real parameters this proxy reads (for trainability checks).

        Yields:
            The 3-D weight parameter, then the bias parameter if any.
        """
        yield getattr(self._owner, self._weight_name)
        if self._bias_name is not None:
            yield getattr(self._owner, self._bias_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert's linear map to the rows *x*."""
        return _linear(x, self.weight, self.bias)


class RouterLinear(nn.Module):
    """``F.linear`` with a router module's ``weight`` (and ``bias``)."""

    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_router", router)

    @property
    def weight(self) -> torch.Tensor:
        """The router's ``(E, d)`` weight."""
        return self._router.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """The router's bias, or ``None``."""
        return getattr(self._router, "bias", None)

    def proxied_parameters(self) -> Iterator[nn.Parameter]:
        """The router parameters this proxy reads.

        Yields:
            The weight, then the bias if any.
        """
        yield self._router.weight
        bias = getattr(self._router, "bias", None)
        if bias is not None:
            yield bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The router logits of the rows *x*, in *x*'s dtype.

        Routers that cast to float32 before the matmul (DeepSeek-V2/V3, GLM-4
        MoE and others) pass a cast *x*; the weight follows it so the logits
        are the model's own.
        """
        bias = self.bias
        return _linear(
            x, self.weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def is_fused_experts(module: nn.Module) -> bool:
    """Whether *module* stores its experts as 3-D ``gate_up_proj``/``down_proj``."""
    gate_up = getattr(module, GATE_UP, None)
    down = getattr(module, DOWN, None)
    return (
        isinstance(gate_up, torch.Tensor)
        and isinstance(down, torch.Tensor)
        and gate_up.ndim == 3
        and down.ndim == 3
        and gate_up.shape[0] == down.shape[0]
    )


def is_fused_router(module: nn.Module, num_experts: int) -> bool:
    """Whether *module* is a router holding a bare ``(E, d)`` weight."""
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, nn.Parameter)
        and weight.ndim == 2
        and weight.shape[0] == num_experts
        and not isinstance(module, nn.Linear)
        and not any(True for _ in module.children())
    )


def find_fused_experts(root: nn.Module) -> dict[str, nn.Module]:
    """``{name: module}`` of every fused experts module under *root*."""
    return {n: m for n, m in root.named_modules() if is_fused_experts(m)}


def _routing_args(args: tuple, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(top_k_index, top_k_weights)`` of an experts forward call."""
    if len(args) >= 2:
        return args[0], args[1]
    index = next((kwargs[k] for k in _INDEX_KWARGS if k in kwargs), None)
    weights = next((kwargs[k] for k in _WEIGHT_KWARGS if k in kwargs), None)
    if len(args) == 1:
        if index is None:
            index = args[0]
        else:
            weights = args[0]
    if index is None or weights is None:
        raise TypeError(
            "fused experts forward called without recognisable routing "
            f"arguments: positional {len(args)}, keywords {sorted(kwargs)}.",
        )
    return index, weights


def _experts_forward(
    experts: nn.Module,
    proxies: nn.ModuleList,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """The reference per-expert loop, through the proxies."""
    num_experts = len(proxies)
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=num_experts)
        mask = mask.permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert in hit:
        e = int(expert[0])
        if e == num_experts:  # masking index
            continue
        top_k_pos, token_idx = torch.where(mask[e])
        rows = hidden_states[token_idx]
        gate_up = proxies[e]["gate_up"](rows)
        gated = experts._apply_gate(gate_up)  # noqa: SLF001 - the model's own hook
        out = proxies[e]["down"](gated)
        # Out of place: ``out`` is the output of a hooked module, a view the
        # backward hook machinery forbids modifying in place.
        out = out * top_k_weights[token_idx, top_k_pos, None]  # noqa: PLR6104
        final.index_add_(0, token_idx, out.to(final.dtype))
    return final


def install_fused_experts(
    root: nn.Module,
    experts_names: list[str],
    should_intervene: Callable[[], bool],
) -> tuple[dict[str, str], list[Callable[[], None]]]:
    """Attach proxies to fused experts (and their routers) and override
    their forwards while *should_intervene* holds.

    Args:
        root: The hooked model.
        experts_names: Names of fused experts modules under *root*.
        should_intervene: Returns whether the override is active (the
            manager's collecting flag).

    Returns:
        ``(proxy_types, reverts)``: the ``{proxy layer name: "nn.Linear"}``
        type declarations to hook the proxies with, and closures restoring
        the modules.

    Raises:
        ValueError: If a named module is not a fused experts module.
    """
    modules = dict(root.named_modules())
    proxy_types: dict[str, str] = {}
    reverts: list[Callable[[], None]] = []
    for name in experts_names:
        experts = modules[name]
        if not is_fused_experts(experts):
            raise ValueError(f"{name} is not a fused experts module.")
        if hasattr(experts, PROXY_EXPERTS):
            raise RuntimeError(f"{name} already carries fused-expert proxies.")
        num_experts = getattr(experts, GATE_UP).shape[0]
        transposed = bool(getattr(experts, "is_transposed", False))
        has_bias = bool(getattr(experts, "has_bias", False)) and hasattr(
            experts, GATE_UP + "_bias"
        )
        proxies = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "gate_up": SliceLinear(
                            experts,
                            GATE_UP,
                            e,
                            GATE_UP + "_bias" if has_bias else None,
                            transposed=transposed,
                        ),
                        "down": SliceLinear(
                            experts,
                            DOWN,
                            e,
                            DOWN + "_bias" if has_bias else None,
                            transposed=transposed,
                        ),
                    },
                )
                for e in range(num_experts)
            ],
        )
        experts.add_module(PROXY_EXPERTS, proxies)
        for e in range(num_experts):
            proxy_types[f"{name}.{PROXY_EXPERTS}.{e}.gate_up"] = "nn.Linear"
            proxy_types[f"{name}.{PROXY_EXPERTS}.{e}.down"] = "nn.Linear"

        original = experts.forward  # bound; the class forward may dispatch kernels

        def forward(
            *args: object,
            _experts: nn.Module = experts,
            _proxies: nn.ModuleList = proxies,
            _original: Callable = original,
            **kwargs: object,
        ) -> torch.Tensor:
            if not should_intervene():
                return _original(*args, **kwargs)
            hidden_states, rest = args[0], args[1:]
            index, weights = _routing_args(rest, kwargs)
            return _experts_forward(_experts, _proxies, hidden_states, index, weights)

        experts.forward = forward

        def revert_experts(_experts: nn.Module = experts) -> None:
            del _experts.forward  # back to the class attribute
            delattr(_experts, PROXY_EXPERTS)

        reverts.append(revert_experts)

        # The router: a sibling of the experts module with a bare (E, d) weight.
        parent_name = name.rpartition(".")[0]
        parent = modules.get(parent_name, root if not parent_name else None)
        if parent is None:
            continue
        for child_name, child in parent.named_children():
            if not is_fused_router(child, num_experts) or hasattr(child, PROXY_ROUTER):
                continue
            router_proxy = RouterLinear(child)
            child.add_module(PROXY_ROUTER, router_proxy)
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            proxy_types[f"{full}.{PROXY_ROUTER}"] = "nn.Linear"
            reverts.append(_override_router(child, router_proxy, should_intervene))
            break
    return proxy_types, reverts


def _override_router(
    router: nn.Module,
    proxy: RouterLinear,
    should_intervene: Callable[[], bool],
) -> Callable[[], None]:
    """Route the router's ``F.linear(x, self.weight[, self.bias])`` through
    *proxy* while collecting, leaving the rest of its forward as written.
    """
    original = router.forward

    def patched_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The router's own matmul is the one ``F.linear`` in its forward with
        # a weight of its weight's shape -- matched by shape, not identity,
        # since some routers cast the weight (``self.weight.type(float32)``)
        # before the call.
        if weight.shape == router.weight.shape:
            return proxy(x)
        return _linear(x, weight, bias)

    def forward(*args: object, **kwargs: object) -> object:
        if not should_intervene():
            return original(*args, **kwargs)
        F.linear = patched_linear
        try:
            return original(*args, **kwargs)
        finally:
            F.linear = _linear

    router.forward = forward

    def revert() -> None:
        del router.forward
        delattr(router, PROXY_ROUTER)

    return revert
