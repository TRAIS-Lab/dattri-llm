"""Random projection -- TRAK-style (materialized) and LoGRA-style (factorized)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.ops import dtypes
from dattri_llm.gradient.ops.materialize import _materialize
from dattri_llm.gradient.ops.preprocess import _preprocess_factorized, _to_3d
from dattri_llm.gradient.ops.types import is_embedding, is_linear, is_norm

if TYPE_CHECKING:
    from collections.abc import Callable

    from dattri_llm.gradient.gradient import Factorized


# The three capture/projection styles (the ``style`` key of a projection
# config).  ``logra_*`` are the double-sided (Kronecker) factor projection --
# keeping the factors, or materializing them into a compact per-sample block;
# ``materialized`` is the single-sided (TRAK) materialize-then-project.
PROJECTION_STYLES = (
    "auto",
    "logra_factorized",
    "logra_materialized",
    "materialized",
)


# Materialized projection matrices, one per (projector, D, proj_dim, seed,
# device, dtype, projector kwargs).  A seeded random projection is a *fixed*
# linear map, so the matrix is built once and every later call is a single
# matmul.  Without the cache each call rebuilt it through the projector factory
# (allocate, seed a generator, draw the Rademacher/Gaussian entries, scale):
# ~12 dispatched ops per call, of which only the matmul does useful work, and
# two calls per hooked layer per step -- a third of the per-layer capture cost
# at batch 1, where capture is launch-bound.  A rank-64 matrix is D x 64, a few
# MB even for the widest LLM layers, so the cache stays small.
_PROJECTION_MATRICES: dict[tuple, torch.Tensor] = {}
# Rows of the identity materialized per chunk while building a matrix, so the
# largest layers (D ~ 50k) never allocate a D x D identity.
_IDENTITY_CHUNK_ROWS = 8192


def clear_projection_cache() -> None:
    """Drop every cached projection matrix (e.g. to free device memory)."""
    _PROJECTION_MATRICES.clear()


def _projector_key(projector: Callable) -> str:
    module = getattr(projector, "__module__", "")
    name = getattr(projector, "__qualname__", repr(projector))
    return f"{module}.{name}"


def _projection_matrix(
    projector: Callable,
    d_in: int,
    *,
    proj_dim: int,
    proj_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    **proj_kwargs,
) -> torch.Tensor:
    """The ``(d_in, proj_dim)`` matrix *projector* applies, built once and cached.

    The matrix is recovered exactly by projecting the identity through the
    projector itself, chunked over rows, so it is the same map the projector
    would apply directly (same seed, same device-specific generator).  It is
    generated in float32 and stored in *dtype*: the entries of a scaled
    Rademacher/Gaussian map are what the projector would hold in that dtype,
    so applying it in the feature's dtype reproduces the projector's own
    dtype behaviour (dattri projects in the feature's dtype).
    """
    key = (
        _projector_key(projector),
        d_in,
        proj_dim,
        proj_seed,
        str(device),
        dtype,
        tuple(sorted((k, repr(v)) for k, v in proj_kwargs.items())),
    )
    matrix = _PROJECTION_MATRICES.get(key)
    if matrix is None:
        rows = []
        for start in range(0, d_in, _IDENTITY_CHUNK_ROWS):
            stop = min(start + _IDENTITY_CHUNK_ROWS, d_in)
            block = torch.zeros(stop - start, d_in, device=device, dtype=torch.float32)
            rows_idx = torch.arange(stop - start, device=device)
            block[rows_idx, rows_idx + start] = 1.0
            rows.append(
                projector(
                    block,
                    block.shape[0],
                    proj_dim=proj_dim,
                    proj_seed=proj_seed,
                    device=device,
                    **proj_kwargs,
                )(block),
            )
        matrix = torch.cat(rows, dim=0).to(dtype)
        _PROJECTION_MATRICES[key] = matrix
    return matrix


def _apply_projector(
    projector: Callable,
    x: torch.Tensor,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Random-project the last axis of *x* from ``D`` to ``proj_dim``.

    *projector* follows dattri's ``random_project`` protocol:
    ``projector(feature, batch_size, proj_dim=..., proj_seed=..., **kw)`` returns
    a callable mapping a ``(N, D)`` feature to ``(N, proj_dim)``.  Any leading
    axes of *x* (the batch, plus the token axis when projecting a factor) are
    folded into ``N`` and restored afterward.

    The projector is only ever asked for its matrix (see
    :func:`_projection_matrix`), once per ``(D, proj_dim, seed, device, dtype)``;
    the projection itself is one matmul against the cached matrix, in *x*'s
    floating dtype.

    ``device`` selects where the projection runs (dattri builds a
    device-specific projector for it) and defaults to *x*'s own device.  The
    feature is moved there before projecting and the result is returned on
    that same (projection) device.  Note that dattri's CPU and CUDA projectors
    do **not** produce the same projection for the same seed -- use one device
    consistently across every gradient that will be compared.
    """
    lead = x.shape[:-1]
    # as_float, not align: dattri's projectors multiply by a random matrix, so
    # an embedding's integer one-hot has to become floating point here.
    (x,) = dtypes.as_float(x)
    flat = x.reshape(-1, x.shape[-1])  # (N, D)
    device = torch.device(proj_kwargs.pop("device", flat.device))
    flat = flat.to(device)
    matrix = _projection_matrix(
        projector,
        flat.shape[-1],
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        device=device,
        dtype=flat.dtype,
        **proj_kwargs,
    )
    # (..., proj_dim), on the projection device
    return (flat @ matrix).reshape(*lead, proj_dim)


def _project_materialized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """TRAK-style: materialize the per-sample weight gradient, then project it.

    Returns a dense ``(B, proj_dim)`` tensor -- the full gradient is reduced to a
    single random-projected vector per sample.
    """
    mat = _materialize(a, g, layer_type, module_kwargs, include_bias)  # (B, D)
    return _apply_projector(
        projector,
        mat,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def _project_factorized(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LoGRA-style: project the two factorized factors, keeping the structure.

    Each factor is independently projected to width ``proj_dim`` -- the output
    factor with ``proj_seed`` and the input factor with ``proj_seed + 1`` (dattri's
    LoGRA convention) -- so the per-sample gradient stays the outer product of two
    ``(B, T, proj_dim)`` factors.  Returns ``(a_p, g_p)``.

    Supported for every layer type whose gradient *is* an outer product of the
    factors: linear, conv, transposed conv, and the embedding family --
    an embedding is a linear layer over one-hot inputs
    (``dW = sum_t onehot(id_t) x g_t``), so its integer ids are expanded to
    one-hot vectors of width ``num_embeddings`` before the input-side
    projection.  (The transient one-hot is ``(B, T, num_embeddings)`` floats;
    a cached identity-projection lookup table would avoid it -- acceptable
    until vocab sizes make it hurt.)  Norm layers (diagonal gradient, not an
    outer product) must use materialized projection instead.
    """
    if is_norm(layer_type):
        raise ValueError(
            f"factorized projection is undefined for {layer_type!r}: its gradient "
            "is not an outer product of the factors -- use materialized projection",
        )
    a, g = _preprocess_factorized(a, g, layer_type, module_kwargs, include_bias)
    if is_embedding(layer_type):
        # Embedding == linear over one-hot inputs; padding/bag handling already
        # happened in preprocessing (pad positions carry zero g).
        if module_kwargs is None:
            raise ValueError(
                "Factorized projection of an embedding requires module_kwargs "
                "with 'num_embeddings' (the one-hot width cannot be inferred "
                "from the captured factors).",
            )
        a = torch.nn.functional.one_hot(
            a.long(),
            num_classes=module_kwargs["num_embeddings"],
        )
    a, g = dtypes.align(a, g)
    a_f = _to_3d(a)  # (B, T, d_in)
    g_f = _to_3d(g)  # (B, T, d_out)
    g_p = _apply_projector(
        projector,
        g_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )
    a_p = _apply_projector(
        projector,
        a_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed + 1,
        **proj_kwargs,
    )
    return a_p, g_p


def project_activation(
    a: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: dict | None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Project **only** a linear layer's activation factor (the a-side of
    :func:`_project_factorized`).

    For a linear layer the a-side prep is just the bias ones-column and does not
    depend on the gradient, so this runs in the *forward* hook -- the capture
    buffer then holds the small ``(B, T, proj_dim)`` factor instead of the full
    ``(B, T, d_in)`` activation.  Uses ``proj_seed + 1`` (dattri's LoGRA input-side
    convention), so composing it with :func:`project_gradient` reproduces
    :func:`_project_factorized` exactly.
    """
    if not is_linear(layer_type):
        raise ValueError(
            f"project_activation is for linear layers only, got {layer_type!r}.",
        )
    if module_kwargs is not None and module_kwargs["has_bias"] and include_bias:
        a = torch.cat([a, torch.ones_like(a[..., :1])], dim=-1)
    a_f = _to_3d(dtypes.align(a)[0])
    return _apply_projector(
        projector,
        a_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed + 1,
        **proj_kwargs,
    )


def project_gradient(
    g: torch.Tensor,
    layer_type: str,
    projector: Callable,
    module_kwargs: dict | None,  # noqa: ARG001 - parity with the a-side signature
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Project **only** a linear layer's gradient factor (the g-side of
    :func:`_project_factorized`).

    A linear layer's gradient needs no per-layer prep, so this is a plain
    projection with ``proj_seed`` -- run in the *backward* hook and paired with
    the forward-projected activation.
    """
    if not is_linear(layer_type):
        raise ValueError(
            f"project_gradient is for linear layers only, got {layer_type!r}.",
        )
    g_f = _to_3d(dtypes.align(g)[0])
    return _apply_projector(
        projector,
        g_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def project_materialized(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable,
    *,
    proj_dim: int,
    include_bias: bool = True,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """:func:`_project_materialized` on a :class:`Factorized` (batch-first-safe).

    Also accepts an already-dense ``(B, D)`` tensor, which is projected directly.
    """
    if isinstance(f, torch.Tensor):
        return _apply_projector(
            projector,
            f,
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            **proj_kwargs,
        )
    bf = f.as_batch_first()
    return _project_materialized(
        bf.activation,
        bf.pre_activation_grad,
        layer_type,
        projector,
        bf.module_kwargs,
        include_bias,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def project_factorized(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable,
    *,
    proj_dim: int,
    include_bias: bool = True,
    proj_seed: int = 0,
    **proj_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_project_factorized` on a :class:`Factorized` (batch-first-safe).

    Returns the projected ``(a_p, g_p)`` factor tuple; the caller rewraps it into
    a :class:`Factorized` with ``module_kwargs=None`` (the factors are final).
    """
    bf = f.as_batch_first()
    return _project_factorized(
        bf.activation,
        bf.pre_activation_grad,
        layer_type,
        projector,
        bf.module_kwargs,
        include_bias,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def maybe_materialize_projected(
    seq_len: int,
    k_a: int,
    k_g: int,
    kappa: float = 1.0,
) -> bool:
    """``True`` when the projected factors should be materialized at capture.

    The scoring path already routes between the factorized and materialized
    cross-gram by :func:`~dattri_llm.gradient.ops.dot.maybe_use_materialized_gram`;
    capture had no equivalent, so a projected capture kept the token axis however
    unfavourable that was.  The same rule applies here, on the *projected* dims:
    per sample and layer the factors cost ``S*(k_a+k_g)`` while their outer
    product costs ``k_a*k_g``, so materializing wins once

        S >= H = k_a*k_g / (k_a + k_g).

    Projection makes this bite: at ``k_a=k_g=64`` the crossover is ``H=32``, so a
    512-token sequence stores 16x more as factors than as the outer product --
    the opposite of the unprojected case, where ``H`` is in the hundreds and
    keeping the factors is what saves memory.

    ``kappa > 1`` biases toward keeping the factors (they are what per-token
    attribution needs; the outer product has summed the token axis away).
    """
    denom = k_a + k_g
    if denom <= 0:
        return False
    return seq_len >= kappa * (k_a * k_g) / denom


def project_layer(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable,
    *,
    style: str = "logra_factorized",
    **proj_kwargs,
) -> tuple[object, bool]:
    """Route one layer to one of the three projection styles.

    Returns ``(payload, is_factorized)``:

    * ``"auto"`` -- project the factors, then keep or materialize them according
      to :func:`maybe_materialize_projected`, the capture-time counterpart of the
      scoring cost model.

    * ``"logra_factorized"`` -- double-sided (LoGRA) projection, **keeping the
      factors**: payload is the ``(a_p, g_p)`` tuple (the caller rewraps it into
      a :class:`Factorized` with ``module_kwargs=None``), ``is_factorized`` True.
    * ``"logra_materialized"`` -- double-sided (LoGRA) projection, then
      **materialize** the projected factors into one ``(B, k_g*k_a)`` per-sample
      block (token-summed outer product, cheap because it happens in the small
      projected space).  payload is a dense tensor, ``is_factorized`` False.
    * ``"materialized"`` -- single-sided (TRAK) projection: materialize the full
      per-sample weight gradient **first**, then project it to ``(B, proj_dim)``.
      payload is a dense tensor, ``is_factorized`` False.

    A materialized input tensor can only take the ``"materialized"`` path (there
    are no factors to project), whatever the requested style.
    """
    if not isinstance(f, torch.Tensor):
        if style == "auto":
            a_p, g_p = project_factorized(f, layer_type, projector, **proj_kwargs)
            # Decide on the ACTUAL projected shapes rather than the requested
            # proj_dim: a layer may project asymmetrically, and the bias column
            # (include_bias) widens one side.
            seq_len = a_p.shape[1] if a_p.ndim == 3 else 1
            if maybe_materialize_projected(seq_len, a_p.shape[-1], g_p.shape[-1]):
                return _materialize(a_p, g_p, "nn.Linear"), False
            return (a_p, g_p), True
        if style == "logra_factorized":
            return project_factorized(f, layer_type, projector, **proj_kwargs), True
        if style == "logra_materialized":
            a_p, g_p = project_factorized(f, layer_type, projector, **proj_kwargs)
            # Projected outer-product factors behave as a plain linear layer;
            # module_kwargs=None so they are not re-preprocessed.
            return _materialize(a_p, g_p, "nn.Linear"), False
    return project_materialized(f, layer_type, projector, **proj_kwargs), False
