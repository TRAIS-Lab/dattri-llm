"""Random projection -- TRAK-style (materialized) and LoGRA-style (factorized)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
PROJECTION_STYLES = ("logra_factorized", "logra_materialized", "materialized")


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

    ``device`` selects where the projection runs (dattri builds a
    device-specific projector for it) and defaults to *x*'s own device.  The
    feature is moved there before projecting and the result is returned on
    that same (projection) device.  Note that dattri's CPU and CUDA projectors
    do **not** produce the same projection for the same seed -- use one device
    consistently across every gradient that will be compared.
    """
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).float()  # (N, D)
    device = proj_kwargs.setdefault("device", flat.device)
    flat = flat.to(device)
    out = projector(
        flat,
        flat.shape[0],
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )(flat)  # (N, proj_dim), on the projection device
    return out.reshape(*lead, proj_dim)


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
    a_f = _to_3d(a.float())  # (B, T, d_in)
    g_f = _to_3d(g.float())  # (B, T, d_out)
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
    a_f = _to_3d(a.float())
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
    g_f = _to_3d(g.float())
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
        if style == "logra_factorized":
            return project_factorized(f, layer_type, projector, **proj_kwargs), True
        if style == "logra_materialized":
            a_p, g_p = project_factorized(f, layer_type, projector, **proj_kwargs)
            # Projected outer-product factors behave as a plain linear layer;
            # module_kwargs=None so they are not re-preprocessed.
            return _materialize(a_p, g_p, "nn.Linear"), False
    return project_materialized(f, layer_type, projector, **proj_kwargs), False
