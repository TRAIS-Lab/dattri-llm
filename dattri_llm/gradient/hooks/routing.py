"""Per-sample layout recovery for layers fed flattened or routed token rows.

A Mixture-of-Experts block hands its sub-layers 2-D token matrices instead of
the ``(B, T, d)`` tensors the capture assumes: the router sees the block
input flattened to ``(B*T, d)``, and each expert sees the ``(n_e, d)`` rows
of the tokens routed to it, gathered from arbitrary samples in an
implementation-defined order.  Read as-is, such a tensor claims ``B*T`` or
``n_e`` samples, and the per-sample gradient of an expert cannot be formed
at all without knowing which sample each row came from.

:class:`RoutedTokenTracker` recovers that map.  It records the ``(B, T, d)``
input of the enclosing block, and when a module under a declared MoE
module receives a 2-D float matrix of the same width it locates every row
in the block input -- by an exact bit-level fingerprint of the row,
verified by an exact comparison afterwards -- so the row's sample is
``token // T``.  The rows are then regrouped by sample into a zero-padded
``(B, T_e, d)`` tensor, ``T_e`` the largest number of rows any sample sent
to the layer, and the hooks buffer that tensor exactly as they would an
ordinary sequence layer: the padding rows carry an exactly-zero output
gradient, which every downstream operation treats as inert (materialization
and inner products sum over positions; the K-FAC factors drop
gradient-free rows).  The regrouping is independent of the order in which
a model lays out the routed rows, so it does not depend on the model
implementation; a flattened ``(B*T, d)`` input is recognised directly and
reshaped without a search.

The map is computed once per 2-D input tensor (an expert's inner linears
receive the same tensor object the expert received) and reused by every
hooked layer under the module whose row count matches, so an expert's
down projection -- whose rows are not block rows -- inherits the layout of
its up projection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

# Rows of the block input processed per chunk when fingerprinting, so the
# int64 temporaries stay around 64 MB at hidden sizes of a few thousand.
_FINGERPRINT_CHUNK_ELEMENTS = 8 * 1024 * 1024
_FINGERPRINT_SEED = 0x5EED
_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


@dataclass(frozen=True)
class RowLayout:
    """Where each row of a routed 2-D capture sits in the ``(B, T_e)`` grid.

    ``sample is None`` marks the flattened case: the rows are the block's
    tokens in order, and :meth:`regroup` is a reshape.
    """

    batch: int
    tokens: int
    rows: int
    sample: torch.Tensor | None = None
    position: torch.Tensor | None = None

    def regroup(self, t: torch.Tensor) -> torch.Tensor:
        """Return *t*'s ``(rows, ...)`` rows laid out as ``(batch, tokens, ...)``.

        Raises:
            ValueError: If *t* does not have ``rows`` rows.
        """
        if t.shape[0] != self.rows:
            raise ValueError(
                f"routed capture has {t.shape[0]} rows but the layout holds "
                f"{self.rows}.",
            )
        if self.sample is None:
            return t.reshape(self.batch, self.tokens, *t.shape[1:])
        out = t.new_zeros((self.batch, self.tokens, *t.shape[1:]))
        out[self.sample.to(t.device), self.position.to(t.device)] = t
        return out


@dataclass
class _Source:
    """The most recent ``(B, T, d)`` input of a declared module's ancestry."""

    flat: torch.Tensor  # (B*T, d) view
    batch: int
    tokens: int
    keys: torch.Tensor | None = None  # per-token fingerprints, sorted
    order: torch.Tensor | None = None  # argsort of the fingerprints


class RoutedTokenTracker:
    """Recover per-sample layouts for hooked layers under MoE modules.

    Args:
        root: The hooked model (the module whose ``named_modules`` names the
            layers).
        module_names: Fully-qualified names of the modules whose hooked
            sub-layers receive flattened or routed token rows -- typically
            each MoE block (``model.layers.3.mlp``), or the experts container
            or a single expert or the router under it.  A name under another
            declared name is redundant and allowed.

    Raises:
        ValueError: If a name is not a module of *root*.
    """

    def __init__(self, root: nn.Module, module_names: Iterable[str]) -> None:
        modules = dict(root.named_modules())
        self._declared: tuple[str, ...] = tuple(sorted(set(module_names)))
        missing = [n for n in self._declared if n not in modules]
        if missing:
            raise ValueError(
                f"moe_routed_layers names modules that do not exist: {missing}.",
            )
        self._root = root
        self._modules = modules
        self.layers: set[str] = set()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._lock = threading.Lock()
        # Per device: the enclosing block input, the layouts established for
        # the 2-D inputs seen since it, and the latest of them.
        self._source: dict[int, _Source] = {}
        self._layouts: dict[int, list[tuple[torch.Tensor, RowLayout]]] = {}
        self._latest: dict[int, RowLayout] = {}
        # Per (layer, device): layouts of forwards awaiting their backward.
        self._pending: dict[tuple[str, int], list[RowLayout]] = {}
        self._multipliers: dict[tuple[int, torch.device], torch.Tensor] = {}

    # -- registration ------------------------------------------------------ #

    def covers(self, layer_name: str) -> bool:
        """Whether *layer_name* is at or under a declared module."""
        return any(
            layer_name == d or layer_name.startswith(d + ".") for d in self._declared
        )

    def bind(self, hooked_layers: Iterable[str]) -> None:
        """Mark the hooked layers whose captures are routed, and install the
        pre-forward hooks that track block inputs and 2-D row matrices.
        """
        self.layers = {n for n in hooked_layers if self.covers(n)}
        # Modules may have been added since construction (fused-expert
        # proxies); hook the tree as it stands now.
        self._modules = dict(self._root.named_modules())
        context: set[str] = set()
        subtree: set[str] = set()
        for d in self._declared:
            parts = d.split(".")
            context.update(".".join(parts[:i]) for i in range(len(parts)))
            subtree.update(n for n in self._modules if n == d or n.startswith(d + "."))
        # A module both above and below a declared name (a declared name
        # under another) is a subtree module: its input is matched, not
        # recorded as a source.
        context -= subtree
        for name in context:
            self._handles.append(
                self._modules[name].register_forward_pre_hook(
                    self._on_context_input,
                    with_kwargs=True,
                ),
            )
        for name in subtree:
            self._handles.append(
                self._modules[name].register_forward_pre_hook(
                    self._on_routed_input,
                    with_kwargs=True,
                ),
            )

    def remove(self) -> None:
        """Remove the tracker's hooks and drop all state."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self.reset()

    def reset(self) -> None:
        """Forget the step's sources, layouts and pending forwards."""
        with self._lock:
            self._source.clear()
            self._layouts.clear()
            self._latest.clear()
            self._pending.clear()

    # -- hooks --------------------------------------------------------------- #

    @staticmethod
    def _first_tensor(args: tuple, kwargs: dict) -> torch.Tensor | None:
        for a in args:
            if isinstance(a, torch.Tensor):
                return a
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    @staticmethod
    def _device_index(t: torch.Tensor) -> int:
        return t.device.index if t.is_cuda else 0

    def _on_context_input(self, _module: nn.Module, args: tuple, kwargs: dict) -> None:
        if not torch.is_grad_enabled():
            return
        x = self._first_tensor(args, kwargs)
        if x is None or x.ndim != 3 or not x.is_floating_point():
            return
        self._record_source(x)

    def _record_source(self, x: torch.Tensor) -> None:
        x = x.detach()
        batch, tokens, width = x.shape
        dev = self._device_index(x)
        with self._lock:
            # Ancestors fire before descendants, so the last 3-D input seen
            # before a routed module runs is its nearest 3-D ancestor's.
            self._source[dev] = _Source(x.reshape(batch * tokens, width), batch, tokens)
            self._layouts[dev] = []
            self._latest.pop(dev, None)

    def _on_routed_input(self, _module: nn.Module, args: tuple, kwargs: dict) -> None:
        if not torch.is_grad_enabled():
            return
        x = self._first_tensor(args, kwargs)
        if x is None or not x.is_floating_point():
            return
        if x.ndim == 3:
            # The declared module itself (or a sub-block of it) runs on the
            # (B, T, d) tokens: that is the input its routed rows come from.
            self._record_source(x)
            return
        if x.ndim != 2:
            return
        x = x.detach()
        dev = self._device_index(x)
        with self._lock:
            src = self._source.get(dev)
            if src is None or x.shape[1] != src.flat.shape[1]:
                return
            for seen, layout in self._layouts.get(dev, ()):
                if seen is x or (
                    seen.data_ptr() == x.data_ptr() and seen.shape == x.shape
                ):
                    self._latest[dev] = layout
                    return
            layout = self._match(x, src)
            if layout is None:
                # Not block rows (an expert's down projection whose width
                # happens to equal the hidden size): the layer keeps the
                # layout of the rows it was computed from.
                return
            self._layouts.setdefault(dev, []).append((x, layout))
            self._latest[dev] = layout

    # -- layer API ------------------------------------------------------------ #

    def push(self, layer_name: str, a: torch.Tensor, dev: int) -> RowLayout | None:
        """Layout for a routed layer's forward capture *a*; ``None`` when *a*
        already carries a batch axis (a 3-D input) and needs no regrouping.

        Raises:
            ValueError: If no layout matching *a*'s row count was established
                in this forward.
        """
        if a.ndim != 2:
            return None
        with self._lock:
            layout = self._latest.get(dev)
            if layout is None or layout.rows != a.shape[0]:
                have = None if layout is None else layout.rows
                raise ValueError(
                    f"{layer_name} is declared in moe_routed_layers and received "
                    f"{a.shape[0]} token rows, but no routed layout with that row "
                    f"count was established in this forward (latest: {have}). "
                    "The rows of the layer, or of a module above it under the "
                    "declared name, must be gathered from the (B, T, d) input of "
                    "an enclosing module, or be that input flattened to (B*T, d).",
                )
            self._pending.setdefault((layer_name, dev), []).append(layout)
        return layout

    def pop(self, layer_name: str, dev: int) -> RowLayout | None:
        """Layout of the matching forward for a routed layer's backward."""
        with self._lock:
            stack = self._pending.get((layer_name, dev))
            if not stack:
                return None
            return stack.pop()

    # -- matching -------------------------------------------------------------- #

    def _match(self, x: torch.Tensor, src: _Source) -> RowLayout | None:
        """Locate *x*'s rows in *src*; ``None`` when they are not its rows."""
        n = x.shape[0]
        total = src.batch * src.tokens
        if n == total and torch.equal(x, src.flat):
            return RowLayout(src.batch, src.tokens, n)
        if src.keys is None:
            keys = self._row_keys(src.flat)
            # Stable: tokens with identical rows stay in ascending order.
            src.order = torch.argsort(keys, stable=True)
            src.keys = keys[src.order]
        keys_x = self._row_keys(x)
        first = torch.searchsorted(src.keys, keys_x)
        last = torch.searchsorted(src.keys, keys_x, right=True)
        # Bit-identical rows (a repeated sample, a shared prefix under causal
        # attention) share a fingerprint.  Their candidate tokens are matched
        # to the rows in order of appearance: the k-th occurrence of a row
        # takes the k-th such token.  This is exact when the module lays out
        # identical tokens in ascending token order, which the gather
        # ``hidden[torch.where(mask)]`` and stable sort-by-expert kernels do.
        by_key = torch.argsort(keys_x, stable=True)
        sorted_keys = keys_x[by_key]
        run_start = torch.searchsorted(sorted_keys, sorted_keys)
        rank = torch.empty_like(keys_x)
        rank[by_key] = torch.arange(n, device=x.device) - run_start
        pos = first + rank
        if bool((pos >= last).any()):
            return None  # more copies of a row than the block holds
        token = src.order[pos]
        if not torch.equal(src.flat[token], x):
            return None
        sample = torch.div(token, src.tokens, rounding_mode="floor")
        counts = torch.bincount(sample, minlength=src.batch)
        by_sample = torch.argsort(sample, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        position = torch.empty_like(sample)
        position[by_sample] = (
            torch.arange(n, device=sample.device) - starts[sample[by_sample]]
        )
        return RowLayout(src.batch, int(counts.max()), n, sample, position)

    def _row_keys(self, t: torch.Tensor) -> torch.Tensor:
        """One int64 fingerprint per row of *t*, exact in the row's bits."""
        int_dtype = _INT_VIEW[t.element_size()]
        mult = self._multiplier(t.shape[1], t.device)
        rows = max(1, _FINGERPRINT_CHUNK_ELEMENTS // max(1, t.shape[1]))
        out = torch.empty(t.shape[0], dtype=torch.int64, device=t.device)
        for start in range(0, t.shape[0], rows):
            chunk = t[start : start + rows].contiguous().view(int_dtype)
            out[start : start + rows] = (chunk.to(torch.int64) * mult).sum(-1)
        return out

    def _multiplier(self, width: int, device: torch.device) -> torch.Tensor:
        key = (width, device)
        mult = self._multipliers.get(key)
        if mult is None:
            gen = torch.Generator().manual_seed(_FINGERPRINT_SEED)
            mult = torch.randint(
                -(2**62),
                2**62,
                (width,),
                generator=gen,
                dtype=torch.int64,
            )
            mult = (mult | 1).to(device)
            self._multipliers[key] = mult
        return mult
