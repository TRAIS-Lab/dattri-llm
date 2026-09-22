"""Optimizer state in the layout of a layer's materialized gradient.

Optimizer-aware capture combines a layer's per-sample gradient entries with
the optimizer's state at the *same* coordinates.  The per-sample gradient is
laid out by :func:`~dattri_llm.gradient.ops.materialize` -- for a linear or
conv layer with a bias, ``[W[o, :], b[o]]`` row after row; for a norm layer
``[gamma, beta]``; for an embedding the flattened weight -- while
``torch.optim`` keeps one state tensor per parameter.
:class:`OptimizerSnapshot` bridges the two: it reads a layer's state tensors
(``exp_avg``, ``momentum_buffer``, ...) in that layout, gathering just the
coordinates a caller asks for, and applies :func:`~dattri_llm.gradient.ops.precondition`
to a layer's captured entries.  :class:`GradientPreconditioner` is the
switchable form the :class:`~dattri_llm.gradient.hooks.HookManager` runs
inside its capture.

Single-process and DDP (replicated) optimizers are read directly.  Under the
``FullyShardedDataParallel`` wrapper (``use_orig_params=True``) every rank
holds a contiguous slice of each flattened state tensor; a read lays the
rank's slice out on zeros and sums across ranks (one ``all_reduce`` per
layer read), so it runs on every rank in step.  A layer's slice bounds are taken
from its flat parameter, which is reachable only while the layer's unit is
unsharded: :meth:`OptimizerSnapshot.bind` records them from inside the
layer's backward.  Under ``fully_shard`` a state tensor is a ``DTensor``
whose ``full_tensor()`` gathers it -- on every rank, so a read runs in step
there too -- and inside a backward the layer's unsharded weight is mapped
back to the sharded parameter the optimizer holds.  ``use_orig_params=False``
is not supported.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import base_layer_name
from dattri_llm.gradient.ops.types import (
    canonical_class_name,
    is_conv,
    is_embedding,
    is_linear,
    is_norm,
)
from dattri_llm.utils.distributed import dist_world_size, is_dist_initialized

HF_CONV1D = "transformers.pytorch_utils.Conv1D"

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn


_UNSUPPORTED_SHARDING = (
    "the optimizer holds flat parameters this reader cannot lay out: "
    "FullyShardedDataParallel needs use_orig_params=True."
)


class _Shard(NamedTuple):
    """One rank's slice of a parameter under the FSDP wrapper: the original
    parameter (the optimizer's key), the full shape, and the inclusive bounds
    of the slice in the flattened parameter (``None`` when the rank holds
    none of it).
    """

    param: torch.nn.Parameter
    shape: tuple[int, ...]
    start: int | None
    end: int | None


def optimizer_type(optimizer: torch.optim.Optimizer) -> str:
    """The :data:`~dattri_llm.gradient.ops.OPTIMIZER_STATE_KEYS` key for an optimizer.

    Resolved by class name along the MRO, so a subclass or a re-export
    (``transformers.AdamW``, a fused variant) maps to its torch family.
    """
    for cls in type(optimizer).__mro__:
        if cls.__name__ in ops.OPTIMIZER_STATE_KEYS:
            return cls.__name__
    raise NotImplementedError(
        f"{type(optimizer).__name__} is not a supported optimizer: its update is "
        "not coordinate-wise, or its state is not a per-parameter tensor. "
        f"Supported: {sorted(ops.OPTIMIZER_STATE_KEYS)}.",
    )


def hf_trainer_param_names(model: nn.Module) -> list[str]:
    """Parameter names in the order HF ``Trainer`` builds its optimizer.

    ``Trainer.create_optimizer`` puts the weight-decayed parameters (every
    parameter except biases and normalization weights) in the first group and
    the rest in the second, so a saved ``optimizer.pt`` indexes parameters
    in that order.  Use with :func:`optimizer_from_state_dict`.
    """
    from transformers.trainer_pt_utils import get_parameter_names

    norm_types = tuple(
        t
        for t in (torch.nn.LayerNorm, getattr(torch.nn, "RMSNorm", None))
        if t is not None
    )
    decay = [n for n in get_parameter_names(model, norm_types) if "bias" not in n]
    decay_set = set(decay)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    return [n for n in trainable if n in decay_set] + [
        n for n in trainable if n not in decay_set
    ]


def optimizer_from_state_dict(
    model: nn.Module,
    state_dict: dict,
    param_names: Iterable[str] | None = None,
    *,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
) -> torch.optim.Optimizer:
    """Rebuild a saved ``optimizer.state_dict()`` as a live optimizer over *model*.

    A saved state indexes parameters by their position in the saved groups;
    *param_names* gives the parameter name behind each global index, in
    order (default: :func:`hf_trainer_param_names`, the HF ``Trainer``
    layout).  The state is loaded into a fresh *optimizer_cls* over the named
    parameters, so a checkpoint's optimizer can be handed to a
    :class:`~dattri_llm.gradient.hooks.HookManager` like a live one.
    """
    params = dict(model.named_parameters())
    names = list(hf_trainer_param_names(model) if param_names is None else param_names)
    groups = []
    for saved in state_dict["param_groups"]:
        group = {k: v for k, v in saved.items() if k != "params"}
        group["params"] = [params[names[i]] for i in saved["params"]]
        groups.append(group)
    optimizer = optimizer_cls(groups)
    optimizer.load_state_dict(state_dict)
    return optimizer


class OptimizerSnapshot:
    """Read an optimizer's state per hooked layer, in materialized layout.

    A view, not a copy: every read goes to the optimizer's current tensors,
    so a snapshot built before training follows the state as it evolves.

    Args:
        model: The model the optimizer trains (the same module tree the
            :class:`~dattri_llm.gradient.hooks.HookManager` hooks).
        optimizer: A live ``torch.optim`` optimizer over *model*'s parameters
            (see :func:`optimizer_from_state_dict` for a saved one).
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self._model = getattr(model, "module", model)
        self._optimizer = optimizer
        self.optimizer_type = optimizer_type(optimizer)
        self._groups: dict[int, dict] = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                self._groups[id(p)] = group
                if getattr(p, "_is_flat_param", False):
                    raise NotImplementedError(_UNSUPPORTED_SHARDING)
        # (layer, role) -> the rank's slice, for layers under the FSDP wrapper
        # (filled by bind()).
        self._shards: dict[tuple[str, str], _Shard] = {}
        # (layer, role) -> the sharded (DTensor) parameter, for layers under
        # fully_shard (filled by bind()).
        self._dtensors: dict[tuple[str, str], torch.nn.Parameter] = {}

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """The optimizer being read."""
        return self._optimizer

    # ------------------------------------------------------------------ #
    # Layout                                                              #
    # ------------------------------------------------------------------ #
    def module(self, layer_name: str) -> nn.Module:
        """The hooked module behind a (possibly virtual) layer name."""
        return self._model.get_submodule(base_layer_name(layer_name))

    def parameters(
        self, layer_name: str, include_bias: bool = True
    ) -> list[tuple[str, torch.nn.Parameter]]:
        """``[(role, parameter), ...]`` making up the layer's flat gradient."""
        module = self.module(layer_name)
        lt = canonical_class_name(module)
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None) if include_bias else None
        if weight is None:
            raise ValueError(f"layer {layer_name!r} ({lt}) has no weight parameter.")
        if not (is_linear(lt) or is_conv(lt) or is_norm(lt) or is_embedding(lt)):
            raise NotImplementedError(
                f"optimizer-state layout is not defined for layer type {lt} "
                f"(layer {layer_name!r}); supported: linear, conv, norm, embedding.",
            )
        out = [("weight", weight)]
        if bias is not None:
            out.append(("bias", bias))
        self.bind(layer_name)
        return [
            (
                role,
                self._shards[layer_name, role].param
                if (layer_name, role) in self._shards
                else self._dtensors.get((layer_name, role), p),
            )
            for role, p in out
        ]

    def bind(self, layer_name: str) -> None:
        """Record where a layer's parameters sit in their FSDP flat parameter.

        A no-op for a layer that is not under the FSDP wrapper, or is bound
        already.  While a unit is unsharded its module attributes are views
        of the flat parameter, which names each original parameter and the
        slice of it this rank keeps; once the unit is resharded that link is
        gone.  So this must first run from inside the layer's forward or
        backward (the :class:`~dattri_llm.gradient.hooks.HookManager` reads a
        preconditioned layer there).
        """
        module = self.module(layer_name)
        for role in ("weight", "bias"):
            tensor = getattr(module, role, None)
            if tensor is None or (layer_name, role) in self._shards:
                continue
            if (layer_name, role) in self._dtensors:
                continue
            if _is_dtensor(tensor):  # fully_shard, idle: the parameter itself
                self._dtensors[layer_name, role] = tensor
                continue
            sharded = self._fully_shard_param(module, role)
            if sharded is not None:  # fully_shard, inside forward/backward
                self._dtensors[layer_name, role] = sharded
                continue
            flat = getattr(tensor, "_base", None)
            infos = getattr(flat, "_shard_param_infos", None)
            if infos is None:
                if getattr(tensor, "_fsdp_flattened", False):
                    raise RuntimeError(
                        f"layer {layer_name!r} is sharded by FSDP and its "
                        "optimizer state was read before the layer was bound: "
                        "call OptimizerSnapshot.bind(layer_name) from inside "
                        "the layer's backward first.",
                    )
                continue
            if flat._params is None:  # use_orig_params=False
                raise NotImplementedError(_UNSUPPORTED_SHARDING)
            for param, info, shape, where in zip(
                flat._params, infos, flat._shapes, flat._param_infos, strict=True
            ):
                if where.module is module and where.param_name == role:
                    self._shards[layer_name, role] = _Shard(
                        param,
                        tuple(shape),
                        info.intra_param_start_idx if info.in_shard else None,
                        info.intra_param_end_idx if info.in_shard else None,
                    )

    def _fully_shard_param(
        self, module: nn.Module, role: str
    ) -> torch.nn.Parameter | None:
        """The sharded parameter behind ``module.<role>`` while a
        ``fully_shard`` unit is unsharded, or ``None`` when *module* is not
        under one.
        """
        for unit in self._model.modules():
            state = getattr(unit, "_get_fsdp_state", None)
            group = getattr(state(), "_fsdp_param_group", None) if state else None
            for fsdp_param in getattr(group, "fsdp_params", ()):
                info = fsdp_param._module_info
                if info.module is module and info.param_name == role:
                    return fsdp_param.sharded_param
        return None

    def _full(
        self, layer_name: str, role: str, local: torch.Tensor | None
    ) -> torch.Tensor | None:
        """A state tensor in its parameter's full shape.  For a sharded layer
        that is the rank's slice laid out on zeros (``None`` reads as zeros),
        which sums across ranks to the full tensor.
        """
        shard = self._shards.get((layer_name, role))
        if shard is None:
            return local
        dtype = torch.float32 if local is None else local.dtype
        device = "cpu" if local is None else local.device
        full = torch.zeros(math.prod(shard.shape), dtype=dtype, device=device)
        if shard.start is not None and local is not None and local.numel():
            full[shard.start : shard.end + 1] = local.reshape(-1)
        return full.reshape(shard.shape)

    def _sharded(self, layer_name: str) -> bool:
        return (layer_name, "weight") in self._shards

    def width(self, layer_name: str, include_bias: bool = True) -> int:
        """Flattened width of the layer's materialized gradient."""
        return sum(
            math.prod(self._shards[layer_name, role].shape)
            if (layer_name, role) in self._shards
            else p.numel()
            for role, p in self.parameters(layer_name, include_bias)
        )

    def _row_layout(self, layer_name: str) -> bool:
        """Whether the layer lays its bias out as a column per output row."""
        lt = canonical_class_name(self.module(layer_name))
        return is_linear(lt) or is_conv(lt)

    def _weight_view(self, layer_name: str, w: torch.Tensor) -> torch.Tensor:
        """*w* with the output features leading, as the gradient is laid out.

        HF's ``Conv1D`` (GPT-2) stores its weight as ``(in, out)``, the
        transpose of ``nn.Linear``; its captured gradient is nonetheless laid
        out ``(out, in)`` like every linear layer.
        """
        if canonical_class_name(self.module(layer_name)) == HF_CONV1D:
            return w.transpose(0, 1)
        return w

    def flatten(
        self,
        layer_name: str,
        tensors: dict[str, torch.Tensor],
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lay ``{role: tensor}`` (parameter-shaped) out as the layer's flat
        gradient, gathered at *idx* when given.

        Roles are ``"weight"`` and optionally ``"bias"`` -- the flat layout is
        the one :func:`~dattri_llm.gradient.ops.materialize` produces with
        ``include_bias`` matching whether a bias tensor is present.  A gather
        reads only the ``len(idx)`` addressed elements.
        """
        w = self._weight_view(layer_name, tensors["weight"])
        b = tensors.get("bias")
        rows = w.shape[0] if self._row_layout(layer_name) else 1
        w_rows = w.reshape(rows, -1)
        if idx is None:
            if b is None:
                return w.reshape(-1)
            if rows > 1:
                return torch.cat([w_rows, b.reshape(-1, 1)], dim=1).reshape(-1)
            return torch.cat([w.reshape(-1), b.reshape(-1)])
        idx = idx.to(w.device)
        d_in = w_rows.shape[1]
        if b is None:
            return w.reshape(-1)[idx]
        if rows > 1:
            # Row layout: coordinate o * (d_in + 1) + i; i == d_in is the bias.
            o, i = idx // (d_in + 1), idx % (d_in + 1)
            from_bias = i == d_in
            return torch.where(
                from_bias, b.reshape(-1)[o], w_rows[o, i.clamp(max=d_in - 1)]
            )
        # Contiguous layout: the bias follows the whole weight.
        from_bias = idx >= d_in
        return torch.where(
            from_bias,
            b.reshape(-1)[(idx - d_in).clamp(min=0)],
            w.reshape(-1)[idx.clamp(max=d_in - 1)],
        )

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #
    def group(self, layer_name: str) -> dict:
        """The parameter group of the layer's weight (its hyperparameters)."""
        _, weight = self.parameters(layer_name)[0]
        try:
            return self._groups[id(weight)]
        except KeyError:
            raise ValueError(
                f"layer {layer_name!r}'s weight is not in the optimizer.",
            ) from None

    def hyperparameters(self, layer_name: str) -> dict:
        """The layer's group hyperparameters (``lr``, ``betas``, ``eps``, ...)."""
        group = self.group(layer_name)
        out = dict(self._optimizer.defaults)
        out.update({k: v for k, v in group.items() if k != "params"})
        return out

    def step_count(self, layer_name: str) -> int:
        """Updates applied so far to the layer's weight (0 before the first)."""
        _, weight = self.parameters(layer_name)[0]
        step = self._optimizer.state.get(weight, {}).get("step")
        if step is None and self._sharded(layer_name):
            # A rank that holds none of the weight has no state for it; the
            # count is the one its other parameters were stepped to.
            step = max(
                (st["step"] for st in self._optimizer.state.values() if "step" in st),
                default=0,
            )
        step = 0 if step is None else step
        return int(step.item() if isinstance(step, torch.Tensor) else step)

    def state(
        self,
        layer_name: str,
        key: str,
        idx: torch.Tensor | None = None,
        include_bias: bool = True,
    ) -> torch.Tensor | None:
        """A per-parameter state tensor (``"exp_avg"``, ``"momentum_buffer"``,
        ...) laid out as the layer's flat gradient, gathered at *idx*.

        ``None`` when no parameter of the layer has that state yet (before
        the first update); a per-parameter scalar (NAdam's ``mu_product``)
        is returned 0-d.  A parameter missing the state while another has it
        reads as zeros.
        """
        params = self.parameters(layer_name, include_bias)
        values = {role: self._optimizer.state.get(p, {}).get(key) for role, p in params}
        if self._sharded(layer_name):
            return self._sharded_state(layer_name, key, values, idx)
        # fully_shard: gather each DTensor state (a collective on every rank).
        values = {
            r: v.full_tensor() if _is_dtensor(v) else v for r, v in values.items()
        }
        if all(v is None for v in values.values()):
            return None
        first = next(v for v in values.values() if v is not None)
        if first.ndim == 0:
            return first
        tensors = {
            role: torch.zeros(tuple(p.shape), dtype=first.dtype, device=first.device)
            if values[role] is None
            else values[role]
            for role, p in params
        }
        return self.flatten(layer_name, tensors, idx)

    def _sharded_state(
        self,
        layer_name: str,
        key: str,
        values: dict[str, torch.Tensor | None],
        idx: torch.Tensor | None,
        reduce: bool = True,
    ) -> torch.Tensor | None:
        """:meth:`state` of a layer under the FSDP wrapper (``reduce=False``
        leaves the sum across ranks to the caller).

        Whether a rank has the state of a given parameter depends on its
        slice, so nothing here branches on that: every rank that has stepped
        the optimizer reduces.  The layout is linear in the state tensors, so
        the rank's slices are laid out on zeros, gathered at *idx*, and only
        that result is summed across ranks.
        """
        states = [st[key] for st in self._optimizer.state.values() if key in st]
        if not states:
            return None  # before the first update, on every rank
        if states[0].ndim == 0:
            return next((v for v in values.values() if v is not None), states[0])
        tensors = {role: self._full(layer_name, role, v) for role, v in values.items()}
        device, dtype = states[0].device, states[0].dtype
        tensors = {r: t.to(device=device, dtype=dtype) for r, t in tensors.items()}
        out = self.flatten(layer_name, tensors, idx)
        if reduce:
            _all_reduce_sum(out)
        return out

    def states(
        self,
        layer_name: str,
        idx: torch.Tensor | None = None,
        include_bias: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Every state tensor the optimizer type keeps, on *idx*."""
        keys = ops.OPTIMIZER_STATE_KEYS[self.optimizer_type]
        self.bind(layer_name)
        if not self._sharded(layer_name):
            return {key: self.state(layer_name, key, idx, include_bias) for key in keys}
        # Sharded: lay every key out locally, then sum them across ranks in
        # one reduction.
        params = self.parameters(layer_name, include_bias)
        out = {}
        for key in keys:
            values = {r: self._optimizer.state.get(p, {}).get(key) for r, p in params}
            out[key] = self._sharded_state(layer_name, key, values, idx, reduce=False)
        packed = [k for k, v in out.items() if v is not None and v.ndim > 0]
        if packed:
            buf = torch.cat([out[k].reshape(-1) for k in packed])
            _all_reduce_sum(buf)
            for k, part in zip(
                packed, buf.split([out[k].numel() for k in packed]), strict=True
            ):
                out[k] = part.reshape(out[k].shape)
        return out

    def precondition(
        self,
        layer_name: str,
        entries: torch.Tensor,
        idx: torch.Tensor | None = None,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """:func:`~dattri_llm.gradient.ops.precondition` of a layer's ``(B, k)``
        entries with the optimizer's current (pre-step) state on *idx*.
        """
        state = {
            k: None if v is None else v.to(entries.device)
            for k, v in self.states(layer_name, idx, include_bias).items()
        }
        return ops.precondition(
            entries,
            state,
            optimizer_type=self.optimizer_type,
            step=self.step_count(layer_name) + 1,
            **self.hyperparameters(layer_name),
        )


def _all_reduce_sum(t: torch.Tensor) -> None:
    """Sum *t* across ranks in place (a no-op outside a process group)."""
    if is_dist_initialized() and dist_world_size() > 1:
        import torch.distributed as dist

        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def _is_dtensor(t: object) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - torch without DTensor
        return False
    return isinstance(t, DTensor)


class GradientPreconditioner:
    """The optimizer map a :class:`~dattri_llm.gradient.hooks.HookManager`
    applies to each layer's captured entries, switchable per pass.

    Args:
        snapshot: The :class:`OptimizerSnapshot` to read, or a zero-argument
            callable building it on first use (a streamer's optimizer exists
            only once its pass starts).
        projector: The manager's :class:`~dattri_llm.gradient.ops.DattriProjector`,
            needed to regenerate a ``"mask"`` layer's coordinates.
    """

    def __init__(
        self,
        snapshot: OptimizerSnapshot | Callable[[], OptimizerSnapshot],
        projector: ops.DattriProjector | None = None,
    ) -> None:
        self._snapshot_src = snapshot
        self._snapshot: OptimizerSnapshot | None = (
            snapshot if isinstance(snapshot, OptimizerSnapshot) else None
        )
        self.projector = projector
        #: Whether captures are preconditioned; a pass that wants raw
        #: gradients through the same hooks switches this off.
        self.enabled = True

    @property
    def snapshot(self) -> OptimizerSnapshot:
        """The optimizer snapshot (resolved on first access)."""
        if self._snapshot is None:
            self._snapshot = self._snapshot_src()  # type: ignore[operator]
        return self._snapshot

    def __call__(
        self,
        layer_name: str,
        entries: torch.Tensor,
        proj_kw: dict | None = None,
    ) -> torch.Tensor:
        """Precondition a layer's ``(B, k)`` entries captured under *proj_kw*.

        *proj_kw* is the layer's projection config: ``None`` or a ``"dense"``
        config means *entries* is the whole flat gradient; ``"mask"`` means the
        coordinates that config draws.
        """
        include_bias = (
            True if proj_kw is None else bool(proj_kw.get("include_bias", True))
        )
        idx = None
        if proj_kw is not None and proj_kw.get("style", "logra") == "mask":
            idx = ops.mask_coordinates(
                {layer_name: proj_kw},
                layer_name,
                self.snapshot.width(layer_name, include_bias),
                self.projector,
                device=entries.device,
            )
        return self.snapshot.precondition(layer_name, entries, idx, include_bias)
