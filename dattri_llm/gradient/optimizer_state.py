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

Single-process and DDP (replicated) optimizers are supported.  FSDP shards
the state, and mapping coordinates onto shards is not implemented here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

HF_CONV1D = "transformers.pytorch_utils.Conv1D"

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn


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
        return out

    def width(self, layer_name: str, include_bias: bool = True) -> int:
        """Flattened width of the layer's materialized gradient."""
        return sum(p.numel() for _, p in self.parameters(layer_name, include_bias))

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
        step = self._optimizer.state.get(weight, {}).get("step", 0)
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
        if all(v is None for v in values.values()):
            return None
        first = next(v for v in values.values() if v is not None)
        if first.ndim == 0:
            return first
        tensors = {
            role: torch.zeros_like(p) if values[role] is None else values[role]
            for role, p in params
        }
        return self.flatten(layer_name, tensors, idx)

    def states(
        self,
        layer_name: str,
        idx: torch.Tensor | None = None,
        include_bias: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Every state tensor the optimizer type keeps, on *idx*."""
        return {
            key: self.state(layer_name, key, idx, include_bias)
            for key in ops.OPTIMIZER_STATE_KEYS[self.optimizer_type]
        }

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


class GradientPreconditioner:
    """The optimizer map a :class:`~dattri_llm.gradient.hooks.HookManager`
    applies to each layer's captured entries, switchable per pass.

    Args:
        snapshot: The :class:`OptimizerSnapshot` to read, or a zero-argument
            callable building it on first use (a streamer's optimizer exists
            only once its pass starts).
        projector: The manager's :class:`~dattri_llm.gradient.ops.DattriProjector`,
            needed to regenerate a ``"subset_materialized"`` layer's coordinates.
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

        *proj_kw* is the layer's projection config: ``None`` or a
        ``"materialized"`` config means *entries* is the whole flat gradient;
        ``"subset_materialized"`` means the coordinates that config draws.
        """
        include_bias = (
            True if proj_kw is None else bool(proj_kw.get("include_bias", True))
        )
        idx = None
        if proj_kw is not None and proj_kw.get("style") == "subset_materialized":
            idx = ops.subset_coordinates(
                {layer_name: proj_kw},
                layer_name,
                self.snapshot.width(layer_name, include_bias),
                self.projector,
                device=entries.device,
            )
        return self.snapshot.precondition(layer_name, entries, idx, include_bias)
