"""Callback that fits the K-FAC covariance factors during collection.

Standard K-FAC estimates the per-layer Kronecker factors ``(A, G)`` -- the
input-activation and output-gradient covariances -- from the training gradients.
The store-then-attribute path fits them in a separate re-pass over the on-disk
gradients (see :meth:`KFACAttributor.fit`); this callback fits them **inline
during the capture pass instead**, from the raw factors the
:class:`~dattri_llm.gradient.hooks.HookManager` already emits to
``on_layer_forward`` / ``on_layer_backward``.

Two use sites share it:

* **Manual collection** -- attach it beside an
  :class:`~dattri_llm.gradient.callbacks.OffloadCallback` when wrapping a
  training loop, so one pass both stores gradients and fits the covariances (no
  re-iterable source needed -- the training data streams by once).
* **On-the-fly** -- an attributor's ``cache``/``collect_to_disk`` attaches it to
  the streamer's manager so the Fisher is fit in the collection pass.

The factors are accumulated with the same
:class:`~dattri_llm.gradient.ops.LayerKroneckerAccumulator` the fit pass uses,
so :meth:`result` reproduces :meth:`KFACAttributor.fit`'s covariances.  They are
built from the **raw** (un-projected) factors, i.e. in full weight space -- the
covariances a factorized/full store is scored with.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks.base import HookManagerCallback

if TYPE_CHECKING:
    import torch


def _is_kfac_eligible(layer_type: str) -> bool:
    """Whether a layer type carries K-FAC (Kronecker) covariances.

    Matches ``KFACAttributor._kfac_layers``: linear and convolution layers.
    Norm / embedding / ``param_grad`` layers are skipped.
    """
    return (
        ops.is_linear(layer_type)
        or ops.is_conv(layer_type)
        or ops.is_conv_transpose(layer_type)
    )


class KroneckerCovarianceCallback(HookManagerCallback):
    """Accumulate per-layer K-FAC covariances ``(A, G)`` during collection.

    Pairs each layer's forward activation with its backward output-gradient and
    folds the pair into a per-layer accumulator.  Only the fixed-size
    covariances are held (``d_in x d_in`` and ``d_out x d_out``), never the
    per-token factors, so memory does not grow with the dataset.

    Args:
        include_bias: Append the bias row/column to the activation covariance
            (as K-FAC does for a layer with a bias term).  Matches
            :meth:`KFACAttributor.fit`'s default.
    """

    def __init__(self, *, include_bias: bool = True) -> None:
        self._include_bias = include_bias
        self._accumulators: dict[str, ops.LayerKroneckerAccumulator] = {}
        # Per-layer stack of pending activations awaiting their backward, matched
        # LIFO -- the order the capture hooks themselves pair a layer invoked
        # more than once per step (weight tying / RNN unroll).
        self._pending: dict[str, list[torch.Tensor]] = {}

    def on_layer_forward(
        self,
        layer_name: str,
        activation: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,  # noqa: ARG002 - used at backward pairing
    ) -> None:
        """Buffer a K-FAC-eligible layer's activation for its backward."""
        if not _is_kfac_eligible(layer_type):
            return
        self._pending.setdefault(layer_name, []).append(activation)

    def on_layer_backward(
        self,
        layer_name: str,
        grad_output: torch.Tensor,
        layer_type: str,
        module_kwargs: dict | None,
    ) -> None:
        """Pair with the buffered activation and fold ``(a, g)`` into ``(A, G)``."""
        if not _is_kfac_eligible(layer_type):
            return
        pending = self._pending.get(layer_name)
        if not pending:
            # No unconsumed forward for this backward -- the manager warns about
            # the orphan; skip it here rather than pairing the wrong activation.
            return
        activation = pending.pop()
        # module_kwargs (has_bias, conv stride/padding, ...) is what makes the
        # covariance match KFACAttributor.fit -- e.g. a biased layer appends the
        # bias row so A is (d_in + 1) x (d_in + 1).
        self._accumulators.setdefault(
            layer_name,
            ops.LayerKroneckerAccumulator(),
        ).update(
            activation,
            grad_output,
            layer_type,
            module_kwargs,
            include_bias=self._include_bias,
        )

    def result(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return the fitted ``{layer: (A, G)}`` covariances (normalized).

        These are the raw, damping-free factors :meth:`KFACAttributor.fit`
        persists; a layer that saw no gradient-carrying data is omitted.
        """
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, acc in self._accumulators.items():
            # suppress (not try/except in-loop): a layer whose every row was
            # gradient-free accumulated nothing and is simply omitted.
            with contextlib.suppress(RuntimeError):
                out[layer_name] = acc.result()
        return out

    def reset(self) -> None:
        """Drop all accumulated covariances and pending activations."""
        self._accumulators.clear()
        self._pending.clear()
