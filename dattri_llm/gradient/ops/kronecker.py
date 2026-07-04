"""K-FAC / EK-FAC and empirical-Fisher kernels, plus streaming accumulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import torch

from dattri_llm.gradient.ops.dot import _cross_gram
from dattri_llm.gradient.ops.materialize import _materialize, materialize
from dattri_llm.gradient.ops.preprocess import _preprocess_factorized
from dattri_llm.gradient.ops.types import (
    is_conv,
    is_conv_transpose,
    is_embedding,
    is_norm,
)

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Factorized, Gradient


# ---------------------------------------------------------------------------
# kfac
# ---------------------------------------------------------------------------

def _flatten_for_kfac(
    a: torch.Tensor, g: torch.Tensor, layer_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten spatial/token dims for K-FAC; raise for norm/embedding types."""
    if is_norm(layer_type):
        raise NotImplementedError("K-FAC is not defined for normalization layers")
    if is_embedding(layer_type):
        raise NotImplementedError("K-FAC is not defined for embedding layers")
    a_f = a.float().reshape(-1, a.shape[-1])
    g_f = g.float().reshape(-1, g.shape[-1])
    return a_f, g_f


def _drop_gradient_free_rows(
    a_f: torch.Tensor, g_f: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop flattened token rows whose output-gradient is exactly zero.

    Such rows contribute nothing to the true Fisher: their ``g gᵀ`` term
    vanishes and their activation carries no learning signal.  Excluding them
    from both factors *and* the row count keeps ``(A, G)`` proper means over
    the gradient-carrying token distribution.  Detection is on the gradient
    side (exact zeros), so no mask metadata is needed.  Whether to drop at all
    is the caller's policy decision.
    """
    keep = ~(g_f == 0).all(dim=1)
    if bool(keep.all()):
        return a_f, g_f
    return a_f[keep], g_f[keep]


def _kfac(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (A, G) K-FAC covariance factor matrices.

    *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
    For sequence (linear) layers, gradient-free (padded / fully masked) token
    rows are excluded from both factors — see :func:`_drop_gradient_free_rows`.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    a_f, g_f = _flatten_for_kfac(a, g, layer_type)
    # Sequence layers only: conv zero-gradient spatial rows are architectural
    # (max-pooling losers, dead ReLU paths), not padding — the KFC convention
    # estimates A over all spatial positions.
    if not (is_conv(layer_type) or is_conv_transpose(layer_type)):
        a_f, g_f = _drop_gradient_free_rows(a_f, g_f)
    N = a_f.shape[0]
    if N == 0:
        raise ValueError(
            "K-FAC factors are undefined: every token row carries a zero "
            "gradient (fully padded / masked input)."
        )
    A = a_f.T @ a_f / N
    G = g_f.T @ g_f / N
    return A, G


# ---------------------------------------------------------------------------
# fim
# ---------------------------------------------------------------------------

def _fim(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Return (d, d) empirical Fisher information matrix.

    Built from the per-sample :func:`_materialize` (token-summed for norm layers,
    so the Fisher is over the layer's actual parameters).  *module_kwargs* is
    passed to :func:`_preprocess_factorized` when provided.
    """
    grad = _materialize(a, g, layer_type, module_kwargs, include_bias)   # (B, d)
    B = grad.shape[0]
    return grad.T @ grad / B


# ---------------------------------------------------------------------------
# K-FAC / EK-FAC attribution kernels
# ---------------------------------------------------------------------------

def sym_inverse(matrix: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    """Damped symmetric inverse ``(matrix + damping·I)^{-1}``.

    Computed via eigendecomposition so the result stays symmetric and a small
    *damping* keeps a rank-deficient K-FAC covariance factor invertible.
    """
    evals, evecs = torch.linalg.eigh(matrix.float())
    return (evecs / (evals + damping)) @ evecs.T


def _kfac_cross(
    a1: torch.Tensor, g1: torch.Tensor,
    a2: torch.Tensor, g2: torch.Tensor,
    layer_type: str,
    A_inv: torch.Tensor,
    G_inv: torch.Tensor,
    module_kwargs1: Optional[dict] = None,
    module_kwargs2: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """K-FAC preconditioned cross-gram between two factorized gradient sets.

    Returns ``K[i, j] = vec(∇W1_i)ᵀ (A⁻¹ ⊗ G⁻¹) vec(∇W2_j)`` — i.e.
    :func:`_cross_dot` with the side-1 factors whitened by the inverse K-FAC
    covariances (``A_inv`` over the input dim, ``G_inv`` over the output dim).
    Both inverses are symmetric, so whitening either side gives the same value.
    Defined for linear and convolution layers.
    """
    a1, g1 = _preprocess_factorized(a1, g1, layer_type, module_kwargs1, include_bias)
    a2, g2 = _preprocess_factorized(a2, g2, layer_type, module_kwargs2, include_bias)
    a1 = a1.float() @ A_inv.float()
    g1 = g1.float() @ G_inv.float()
    return _cross_gram(a1, g1, a2, g2, layer_type)


def kfac_eigh(
    A: torch.Tensor, G: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigendecompose both K-FAC factors: returns ``(s_A, U_A, s_G, U_G)``."""
    s_A, U_A = torch.linalg.eigh(A.float())
    s_G, U_G = torch.linalg.eigh(G.float())
    return s_A, U_A, s_G, U_G


def _ekfac_materialize(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    U_A: torch.Tensor,
    U_G: torch.Tensor,
    module_kwargs: Optional[dict] = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Per-sample weight gradient rotated into the K-FAC eigenbasis.

    Returns ``(B, d_out·d_in)`` flattened ``U_Gᵀ ∇W U_A`` — the eigenbasis
    coordinates whose empirical second moments are the EK-FAC corrected
    eigenvalues, and against which test/train gradients are scored.
    """
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    a = a.float() @ U_A.float()
    g = g.float() @ U_G.float()
    return _materialize(a, g, layer_type)


def kfac(
    f: "Factorized", layer_type: str, include_bias: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`_kfac` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _kfac(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs, include_bias
    )


def fim(
    f: "Factorized", layer_type: str, include_bias: bool = True
) -> torch.Tensor:
    """:func:`_fim` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _fim(
        bf.activation, bf.pre_activation_grad, layer_type, bf.module_kwargs, include_bias
    )


def kfac_cross(
    f1: "Factorized", f2: "Factorized", layer_type: str,
    A_inv: torch.Tensor, G_inv: torch.Tensor, include_bias: bool = True,
) -> torch.Tensor:
    """:func:`_kfac_cross` on two :class:`Factorized` (batch-first-safe)."""
    b1, b2 = f1.as_batch_first(), f2.as_batch_first()
    return _kfac_cross(
        b1.activation, b1.pre_activation_grad, b2.activation, b2.pre_activation_grad,
        layer_type, A_inv, G_inv, b1.module_kwargs, b2.module_kwargs, include_bias,
    )


def ekfac_materialize(
    f: "Factorized", layer_type: str,
    U_A: torch.Tensor, U_G: torch.Tensor, include_bias: bool = True,
) -> torch.Tensor:
    """:func:`_ekfac_materialize` on a :class:`Factorized` (batch-first-safe)."""
    bf = f.as_batch_first()
    return _ekfac_materialize(
        bf.activation, bf.pre_activation_grad, layer_type, U_A, U_G,
        bf.module_kwargs, include_bias,
    )


# ---------------------------------------------------------------------------
# Per-layer streaming accumulators
# ---------------------------------------------------------------------------

@dataclass
class LayerKroneckerAccumulator:
    """Streaming K-FAC covariance accumulator for a *single* layer.

    Accumulates the Kronecker factors ``(A, G)`` (input-activation and
    output-gradient covariances).  For the across-layers version see
    :class:`KroneckerAccumulator`.
    """

    _A: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _G: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def update(
        self,
        a: torch.Tensor,
        g: torch.Tensor,
        layer_type: str,
        module_kwargs: Optional[dict] = None,
        include_bias: bool = True,
    ) -> None:
        """Accumulate one batch of factorized gradient data.

        *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
        """
        a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
        a_f, g_f = _flatten_for_kfac(a, g, layer_type)
        # Sequence layers only: padded / fully masked positions carry
        # exactly-zero gradients and no learning signal; keep (A, G) means over
        # gradient-carrying rows.  Conv spatial zeros are architectural — kept.
        if not (is_conv(layer_type) or is_conv_transpose(layer_type)):
            a_f, g_f = _drop_gradient_free_rows(a_f, g_f)
        N = a_f.shape[0]
        if self._A is None:
            self._A = torch.zeros(
                a_f.shape[-1], a_f.shape[-1], dtype=a_f.dtype, device=a_f.device
            )
            self._G = torch.zeros(
                g_f.shape[-1], g_f.shape[-1], dtype=g_f.dtype, device=g_f.device
            )
        self._A += a_f.T @ a_f
        self._G += g_f.T @ g_f
        self._n += N

    def result(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A, G) normalized covariance matrices."""
        if self._A is None or self._n == 0:
            raise RuntimeError("No gradient-carrying data has been accumulated")
        return self._A / self._n, self._G / self._n

    def reset(self) -> None:
        """Reset accumulator state."""
        self._A = None
        self._G = None
        self._n = 0


@dataclass
class LayerFisherAccumulator:
    """Streaming empirical Fisher accumulator for a *single* layer.

    Accumulates ``Σ_i g_i g_iᵀ`` over the per-sample :func:`_materialize` ``g_i``.
    For the across-layers version see :class:`FisherAccumulator`.
    """

    _F: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def update(
        self,
        a: torch.Tensor,
        g: torch.Tensor,
        layer_type: str,
        module_kwargs: Optional[dict] = None,
        include_bias: bool = True,
    ) -> None:
        """Accumulate one batch from its factorized factors.

        *module_kwargs* is passed to :func:`_preprocess_factorized` when provided.
        """
        self.update_from_grad(_materialize(a, g, layer_type, module_kwargs, include_bias))

    def update_from_grad(self, grad: torch.Tensor) -> None:
        """Accumulate one batch from already-materialized per-sample gradients.

        *grad* is ``(B, d)``.  Use this when the caller has materialized the
        per-sample weight gradient itself (e.g. with a non-default token
        reduction) and only needs the streaming outer-product accumulation.
        """
        B = grad.shape[0]
        if self._F is None:
            self._F = torch.zeros(
                grad.shape[-1], grad.shape[-1], dtype=grad.dtype, device=grad.device
            )
        self._F += grad.T @ grad
        self._n += B

    def result(self) -> torch.Tensor:
        """Return normalized empirical Fisher matrix."""
        if self._F is None:
            raise RuntimeError("No data has been accumulated")
        return self._F / self._n

    def reset(self) -> None:
        """Reset accumulator state."""
        self._F = None
        self._n = 0


# ---------------------------------------------------------------------------
# Multi-layer streaming accumulators
# ---------------------------------------------------------------------------

class KroneckerAccumulator:
    """Streaming K-FAC covariance accumulator across a model's layers.

    Holds one :class:`LayerKroneckerAccumulator` per layer and fans each streamed
    :class:`~dattri_llm.gradient.gradient.Gradient` block out to the requested
    layers, so a fit loop is a single ``update`` call per block::

        acc = KroneckerAccumulator()
        for block in blocks:
            acc.update(block, kfac_eligible_layers)
        factors = acc.result()          # {layer: (A, G)}
    """

    def __init__(self) -> None:
        self._layers: Dict[str, LayerKroneckerAccumulator] = {}

    def update(self, gradient: "Gradient", layers: Iterable[str]) -> None:
        """Accumulate the factors for *layers* from one gradient block."""
        for name in layers:
            bf = gradient.data[name].as_batch_first()
            self._layers.setdefault(name, LayerKroneckerAccumulator()).update(
                bf.activation, bf.pre_activation_grad,
                gradient.layer_types[name], bf.module_kwargs,
            )

    def result(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return ``{layer: (A, G)}`` for every accumulated layer."""
        return {name: acc.result() for name, acc in self._layers.items()}


class FisherAccumulator:
    """Streaming empirical-Fisher accumulator across a model's layers.

    Holds one :class:`LayerFisherAccumulator` per layer, accumulating the dense
    Fisher from each layer's per-sample :func:`_materialize`.  When *max_params*
    is given, layers whose parameter count exceeds it are skipped (and recorded
    in :attr:`skipped`) to bound the dense ``O(d²)`` Fisher::

        acc = FisherAccumulator(max_params=4096)
        for block in blocks:
            acc.update(block, fisher_eligible_layers)
        fishers = acc.result()          # {layer: F}
        dropped = acc.skipped           # {layer: param_count}
    """

    def __init__(self, max_params: Optional[int] = None) -> None:
        self._max_params = max_params
        self._layers: Dict[str, LayerFisherAccumulator] = {}
        self._skipped: Dict[str, int] = {}

    def update(self, gradient: "Gradient", layers: Iterable[str]) -> None:
        """Accumulate the Fisher for *layers* from one gradient block."""
        for name in layers:
            if name in self._skipped:
                continue
            g = materialize(gradient.data[name], gradient.layer_types[name])  # (B, P)
            if self._max_params is not None and g.shape[-1] > self._max_params:
                self._skipped[name] = g.shape[-1]
                self._layers.pop(name, None)
                continue
            self._layers.setdefault(name, LayerFisherAccumulator()).update_from_grad(g)

    def result(self) -> Dict[str, torch.Tensor]:
        """Return ``{layer: F}`` for every accumulated (non-skipped) layer."""
        return {name: acc.result() for name, acc in self._layers.items()}

    @property
    def skipped(self) -> Dict[str, int]:
        """``{layer: param_count}`` for layers dropped by the ``max_params`` cap."""
        return dict(self._skipped)
