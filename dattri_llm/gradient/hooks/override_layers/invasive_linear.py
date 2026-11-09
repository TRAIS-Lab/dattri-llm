"""Forward override skipping the weight-gradient matmul of ``nn.Linear``.

The ``invasive_linear_io`` hook family captures exactly what ``linear_io``
captures (the input activation and the output gradient, through the ordinary
module hooks) but routes the layer's forward through
:class:`InvasiveLinearFunction`, whose backward returns no weight or bias
gradient, so autograd skips the ``d_out x d_in`` weight-gradient matmul the
factorized capture never reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from dattri_llm.gradient.ops import canonical_class_name

if TYPE_CHECKING:
    from collections.abc import Callable


def is_invasive_capable(module: nn.Module) -> bool:
    """Return ``True`` if *module* supports ``invasive_linear_io`` hooks.

    The invasive variant replaces the layer's forward with a custom autograd
    ``Function`` whose forward is exactly ``F.linear``, so it is restricted to
    plain :class:`torch.nn.Linear`.  (Unlike :func:`_is_linear_io_capable`,
    which also admits conv/embedding/norm layers whose forwards differ.)
    """
    return isinstance(module, nn.Linear)


class InvasiveLinearFunction(torch.autograd.Function):
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
        """``F.linear(tensor_input, weight, bias)``."""
        ctx.save_for_backward(weight)
        return nn.functional.linear(tensor_input, weight, bias)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        """The input gradient only; ``None`` for the weight and bias."""
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

    Routes the layer's forward through :class:`InvasiveLinearFunction` so the
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
        if not is_invasive_capable(module):
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
                    return InvasiveLinearFunction.apply(
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
