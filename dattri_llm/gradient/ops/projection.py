"""Dimension reduction of per-sample gradients.

Three projection *styles* (the ``style`` key of a projection config) say how a
layer's per-sample gradient is reduced -- in what order projection and
materialization happen:

* ``"logra"`` -- double-sided: random matrices on the two factors (a Kronecker
  projection of the gradient), so the layer can stay factorized at the
  projected width.
* ``"dense"`` -- single-sided: materialize the per-sample gradient first, then
  one random matrix on the flat vector; always dense.
* ``"mask"`` -- keep a fixed random subset of the gradient's coordinates,
  gathered from the factors without materializing; always dense, every kept
  entry exact.

Orthogonal to the style, the *capture style* (``"factorized"``,
``"materialized"`` or ``"auto"``) picks the representation wherever there is a
choice: an unprojected layer, or a ``"logra"``-projected one.  ``"auto"`` is
the capture-time counterpart of the scoring cost model
(:func:`should_materialize`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.ops import dtypes
from dattri_llm.gradient.ops.materialize import materialize_factors
from dattri_llm.gradient.ops.preprocess import preprocess_factors, to_3d
from dattri_llm.gradient.ops.types import (
    is_conv_transpose,
    is_embedding,
    is_linear,
    is_norm,
)
from dattri_llm.utils.cache import TensorCache

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from dattri_llm.gradient.gradient import Factorized


# The projection styles (the ``style`` key of a projection config): see the
# module docstring.
PROJECTION_STYLES = ("logra", "dense", "mask")

# The capture styles: the representation a layer is buffered in wherever there
# is a choice (see :func:`should_materialize`).
CAPTURE_STYLES = ("factorized", "materialized", "auto")

# Keys a ``"mask"`` projection config may carry: there is no projection
# matrix, so the projector kwargs (``proj_type``, ...) have no meaning and are
# rejected rather than ignored.
MASK_KEYS = frozenset({"style", "proj_dim", "proj_seed", "include_bias", "device"})

# Coordinates gathered per chunk in :func:`mask_factors`, bounding the
# ``(B, T, chunk)`` temporaries the coordinate trick forms.
_MASK_CHUNK_ELEMS = 1 << 24


# Rows of the identity materialized per chunk while building a projection
# matrix, so the largest layers (D ~ 50k) never allocate a D x D identity.
_IDENTITY_CHUNK_ROWS = 8192


def _projector_key(projector: Callable) -> str:
    module = getattr(projector, "__module__", "")
    name = getattr(projector, "__qualname__", repr(projector))
    return f"{module}.{name}"


def _mask_seed(proj_seed: int, d: int, proj_dim: int) -> int:
    """Generator seed of a coordinate subset: one per ``(seed, width, size)``."""
    return ((proj_seed * 1_000_003 + d) * 1_000_003 + proj_dim) & ((1 << 63) - 1)


def _check_mask_kwargs(proj_kwargs: dict) -> None:
    extra = set(proj_kwargs) - MASK_KEYS
    if extra:
        raise ValueError(
            "style 'mask' keeps coordinates rather than projecting, so it "
            f"takes only {sorted(MASK_KEYS - {'style'})}; got unexpected "
            f"{sorted(extra)}.",
        )


class DattriProjector:
    """A random-projection factory plus the cache of the matrices it builds.

    *projector* follows dattri's ``random_project`` protocol:
    ``projector(feature, batch_size, proj_dim=..., proj_seed=..., **kw)``
    returns a callable mapping an ``(N, D)`` feature to ``(N, proj_dim)``.
    ``None`` lazily resolves to dattri's own ``random_project`` (so importing
    dattri is only required when projection is actually used).

    A seeded random projection is a *fixed* linear map, so its ``(D, proj_dim)``
    matrix is built once -- by projecting the identity through the factory --
    and every later call is a single matmul.  Without the cache each call would
    rebuild it (allocate, seed a generator, draw the entries, scale): ~12
    dispatched ops per call, two calls per hooked layer per step, a third of
    the per-layer capture cost at batch 1.  A rank-64 matrix is ``D x 64``, a
    few MB even for the widest LLM layers, so the cache stays small.

    The cache is a :class:`~dattri_llm.utils.cache.TensorCache` owned by this
    object: the projector's lifetime *is* the cache's lifetime.  A
    :class:`~dattri_llm.gradient.hooks.HookManager` holds one for as long as
    its hooks are registered; :meth:`Gradient.project` builds one per call.
    Use it as a context manager, or call :meth:`close`, to release the
    matrices explicitly.

    Args:
        projector: The projection factory (``None`` = dattri's).
        cache: A cache to store the matrices in; by default a private
            in-memory cache.
    """

    def __init__(
        self,
        projector: Callable | None = None,
        *,
        cache: TensorCache | None = None,
    ) -> None:
        self._factory = projector
        self._cache = cache if cache is not None else TensorCache("memory")

    @classmethod
    def coerce(cls, projector: Callable | DattriProjector | None) -> DattriProjector:
        """*projector* itself if it already is one, else a fresh wrapper."""
        if isinstance(projector, DattriProjector):
            return projector
        return cls(projector)

    @property
    def factory(self) -> Callable:
        """The underlying projection factory (dattri's when none was given)."""
        if self._factory is None:
            from dattri.func.projection import random_project

            self._factory = random_project
        return self._factory

    @property
    def cache(self) -> TensorCache:
        """The cache of materialized projection matrices."""
        return self._cache

    def matrix(
        self,
        d_in: int,
        *,
        proj_dim: int,
        proj_seed: int,
        device: torch.device,
        dtype: torch.dtype,
        **proj_kwargs,
    ) -> torch.Tensor:
        """The ``(d_in, proj_dim)`` matrix the factory applies, built once.

        The matrix is recovered exactly by projecting the identity through the
        factory itself, chunked over rows, so it is the same map the factory
        would apply directly (same seed, same device-specific generator).  It
        is generated in float32 and stored in *dtype*: the entries of a scaled
        Rademacher/Gaussian map are what the factory would hold in that dtype,
        so applying it in the feature's dtype reproduces the factory's own
        dtype behaviour (dattri projects in the feature's dtype).
        """
        key = (
            _projector_key(self.factory),
            d_in,
            proj_dim,
            proj_seed,
            str(device),
            dtype,
            tuple(sorted((k, repr(v)) for k, v in proj_kwargs.items())),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rows = []
        for start in range(0, d_in, _IDENTITY_CHUNK_ROWS):
            stop = min(start + _IDENTITY_CHUNK_ROWS, d_in)
            block = torch.zeros(stop - start, d_in, device=device, dtype=torch.float32)
            rows_idx = torch.arange(stop - start, device=device)
            block[rows_idx, rows_idx + start] = 1.0
            rows.append(
                self.factory(
                    block,
                    block.shape[0],
                    proj_dim=proj_dim,
                    proj_seed=proj_seed,
                    device=device,
                    **proj_kwargs,
                )(block),
            )
        matrix = torch.cat(rows, dim=0).to(dtype)
        self._cache.put(key, matrix)
        return matrix

    def mask_indices(
        self,
        d: int,
        *,
        proj_dim: int,
        proj_seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """The ``proj_dim`` coordinates a ``"mask"``-projected layer keeps.

        A sorted ``(proj_dim,)`` index tensor into a flattened width-``d``
        gradient, drawn once without replacement from a CPU generator seeded
        by ``(proj_seed, d, proj_dim)`` -- so, like a projection matrix, the
        same seed and width give the same subset on every device and in every
        process, and two sides of a score can only be compared when they were
        reduced with the same subset.  Cached like the matrices.
        """
        if not 0 < proj_dim <= d:
            raise ValueError(
                f"a mask needs 0 < proj_dim <= d, got proj_dim="
                f"{proj_dim} for a width-{d} layer.",
            )
        key = ("subset", d, proj_dim, proj_seed, str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        gen = torch.Generator(device="cpu").manual_seed(
            _mask_seed(proj_seed, d, proj_dim)
        )
        idx = torch.randperm(d, generator=gen)[:proj_dim].sort().values.to(device)
        self._cache.put(key, idx)
        return idx

    def apply(
        self,
        x: torch.Tensor,
        *,
        proj_dim: int,
        proj_seed: int = 0,
        include_bias: bool = False,
        **proj_kwargs,
    ) -> torch.Tensor:
        """Random-project the last axis of *x* from ``D`` to ``proj_dim``.

        Any leading axes of *x* (the batch, plus the token axis when projecting
        a factor) are folded into ``N`` and restored afterward.  ``device``
        (in *proj_kwargs*) selects where the projection runs and defaults to
        *x*'s own device; the feature is moved there and the result stays on
        that device.  Note that dattri's CPU and CUDA projectors do **not**
        produce the same projection for the same seed -- use one device
        consistently across every gradient that will be compared.

        ``include_bias=True`` projects ``x`` as the activation of a linear
        layer with a bias, i.e. *as if* the bias ones column had been appended
        (the convention of :func:`preprocess_factors`): the ``(D + 1,
        proj_dim)`` matrix is used, and the ones column contributes its row
        as a broadcast add, so the augmented activation -- a full copy of
        *x* per layer and step -- is never formed.
        """
        lead = x.shape[:-1]
        # as_float, not align: the factory multiplies by a random matrix, so
        # an embedding's integer one-hot has to become floating point here.
        (x,) = dtypes.as_float(x)
        flat = x.reshape(-1, x.shape[-1])  # (N, D)
        device = torch.device(proj_kwargs.pop("device", flat.device))
        flat = flat.to(device)
        matrix = self.matrix(
            flat.shape[-1] + (1 if include_bias else 0),
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            device=device,
            dtype=flat.dtype,
            **proj_kwargs,
        )
        if include_bias:
            out = torch.addmm(matrix[-1], flat, matrix[:-1])
        else:
            out = flat @ matrix
        return out.reshape(*lead, proj_dim)

    def resolve(
        self,
        d_in: int,
        *,
        proj_dim: int,
        proj_seed: int = 0,
        include_bias: bool = False,
        device: torch.device | str,
        dtype: torch.dtype,
        **proj_kwargs,
    ) -> ProjectionMatrix:
        """The projection :meth:`apply` would perform on a ``(..., d_in)``
        feature of this *dtype* on this *device*, resolved once.

        :meth:`apply` re-derives the matrix key from its arguments and looks
        the matrix up on every call -- fine for a one-off, but a capture hook
        projects the same layer twice per step, thousands of times per
        attribution, and that dispatch costs more than the kernel.  A
        :class:`ProjectionMatrix` holds the matrix (and, with
        *include_bias*, its bias row) and applies it with one ``addmm``.
        """
        device = torch.device(proj_kwargs.pop("device", device))
        matrix = self.matrix(
            d_in + (1 if include_bias else 0),
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            device=device,
            dtype=dtype,
            **proj_kwargs,
        )
        return ProjectionMatrix(matrix, include_bias, d_in, dtype, device)

    def clear(self) -> None:
        """Drop every cached projection matrix (e.g. to free device memory)."""
        self._cache.clear()

    def close(self) -> None:
        """Release the matrix cache; the projector stays usable (it will
        rebuild matrices on demand).
        """
        self._cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


class ProjectionMatrix:
    """One layer's random projection with everything but the kernel done.

    Built by :meth:`DattriProjector.resolve`; :meth:`matches` says whether a
    feature is the one it was resolved for (width, dtype, device), and
    :meth:`__call__` projects it exactly as :meth:`DattriProjector.apply`
    would (same matrix, same ``addmm``), without re-deriving anything.
    """

    __slots__ = ("bias_row", "d_in", "device", "dtype", "matrix", "proj_dim")

    def __init__(
        self,
        matrix: torch.Tensor,
        include_bias: bool,
        d_in: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.matrix = matrix[:-1] if include_bias else matrix
        self.bias_row = matrix[-1] if include_bias else None
        self.d_in, self.dtype, self.device = d_in, dtype, device
        self.proj_dim = matrix.shape[1]

    def matches(self, x: torch.Tensor) -> bool:
        """Whether *x* is a feature this projection was resolved for."""
        return (
            x.shape[-1] == self.d_in
            and x.dtype == self.dtype
            and x.device == self.device
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last axis of *x* (``(..., d_in)`` to ``(..., proj_dim)``)."""
        flat = x.reshape(-1, self.d_in)
        if self.bias_row is None:
            out = flat @ self.matrix
        else:
            out = torch.addmm(self.bias_row, flat, self.matrix)
        return out.reshape(*x.shape[:-1], self.proj_dim)


def apply_projection(
    projector: Callable | DattriProjector | None,
    x: torch.Tensor,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    include_bias: bool = False,
    **proj_kwargs,
) -> torch.Tensor:
    """:meth:`DattriProjector.apply` for a factory **or** a projector.

    A bare factory is wrapped on the fly, so the matrix is not retained past
    this call -- pass a :class:`DattriProjector` to share matrices across calls.
    """
    return DattriProjector.coerce(projector).apply(
        x,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        include_bias=include_bias,
        **proj_kwargs,
    )


def project_materialized_factors(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
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
    mat = materialize_factors(a, g, layer_type, module_kwargs, include_bias)  # (B, D)
    return apply_projection(
        projector,
        mat,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def project_factors(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
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
    a, g = preprocess_factors(a, g, layer_type, module_kwargs, include_bias)
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
    a_f = to_3d(a)  # (B, T, d_in)
    g_f = to_3d(g)  # (B, T, d_out)
    g_p = apply_projection(
        projector,
        g_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )
    a_p = apply_projection(
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
    projector: Callable | DattriProjector | None,
    module_kwargs: dict | None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Project **only** a linear layer's activation factor (the a-side of
    :func:`project_factors`).

    For a linear layer the a-side prep is just the bias ones-column and does not
    depend on the gradient, so this runs in the *forward* hook -- the capture
    buffer then holds the small ``(B, T, proj_dim)`` factor instead of the full
    ``(B, T, d_in)`` activation.  Uses ``proj_seed + 1`` (dattri's LoGRA input-side
    convention), so composing it with :func:`project_gradient` reproduces
    :func:`project_factors` exactly.
    """
    if not is_linear(layer_type):
        raise ValueError(
            f"project_activation is for linear layers only, got {layer_type!r}.",
        )
    # The bias ones-column is folded into the projection (a broadcast add of
    # the matrix's last row) rather than appended to the activation, which
    # would copy the whole ``(B, T, d_in)`` tensor once per layer and step.
    with_bias = module_kwargs is not None and module_kwargs["has_bias"] and include_bias
    a_f = to_3d(dtypes.align(a)[0])
    return apply_projection(
        projector,
        a_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed + 1,
        include_bias=with_bias,
        **proj_kwargs,
    )


def project_gradient(
    g: torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
    module_kwargs: dict | None,  # noqa: ARG001 - parity with the a-side signature
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Project **only** a linear layer's gradient factor (the g-side of
    :func:`project_factors`).

    A linear layer's gradient needs no per-layer prep, so this is a plain
    projection with ``proj_seed`` -- run in the *backward* hook and paired with
    the forward-projected activation.
    """
    if not is_linear(layer_type):
        raise ValueError(
            f"project_gradient is for linear layers only, got {layer_type!r}.",
        )
    g_f = to_3d(dtypes.align(g)[0])
    return apply_projection(
        projector,
        g_f,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        **proj_kwargs,
    )


def project_materialized(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
    *,
    proj_dim: int,
    include_bias: bool = True,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """:func:`project_materialized_factors` on a :class:`Factorized` (batch-first-safe).

    Also accepts an already-dense ``(B, D)`` tensor, which is projected directly.
    """
    if isinstance(f, torch.Tensor):
        return apply_projection(
            projector,
            f,
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            **proj_kwargs,
        )
    bf = f.as_batch_first()
    return project_materialized_factors(
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
    projector: Callable | DattriProjector | None,
    *,
    proj_dim: int,
    include_bias: bool = True,
    proj_seed: int = 0,
    **proj_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`project_factors` on a :class:`Factorized` (batch-first-safe).

    Returns the projected ``(a_p, g_p)`` factor tuple; the caller rewraps it into
    a :class:`Factorized` with ``module_kwargs=None`` (the factors are final).
    """
    bf = f.as_batch_first()
    return project_factors(
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


def mask_coordinates(
    projection: dict[str, dict] | None,
    layer_name: str,
    width: int,
    projector: Callable | DattriProjector | None,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    """The coordinates a ``"mask"`` capture keeps for a layer.

    Regenerated from the projection config the gradients were captured with
    (``proj_dim``, ``proj_seed``) and the layer's flat *width*, exactly as
    the capture drew them.  ``None`` when the layer is not masked -- its
    entries are then the whole flat gradient.  Any other projection style
    raises: the entries of a projected gradient are not coordinates.
    """
    if projection is None:
        return None
    kw = projection.get(layer_name, projection.get("__default__"))
    if kw is None:
        return None
    if kw.get("style", "logra") != "mask":
        raise ValueError(
            f"layer {layer_name!r} is captured with style {kw.get('style')!r}; "
            "coordinate-wise maps need exact gradient entries, i.e. no "
            "projection or style 'mask'.",
        )
    return DattriProjector.coerce(projector).mask_indices(
        width,
        proj_dim=kw["proj_dim"],
        proj_seed=kw.get("proj_seed", 0),
        device=torch.device(device),
    )


def mask_materialized(
    x: torch.Tensor,
    projector: Callable | DattriProjector | None,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """Keep ``proj_dim`` fixed random coordinates of a dense ``(..., D)`` tensor.

    The mask is :meth:`DattriProjector.mask_indices` for ``(D, proj_dim,
    proj_seed)``; ``device`` selects where the result lives (default: *x*'s).
    """
    _check_mask_kwargs(proj_kwargs)
    device = torch.device(proj_kwargs.get("device", x.device))
    x = x.to(device)
    idx = DattriProjector.coerce(projector).mask_indices(
        x.shape[-1],
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        device=device,
    )
    return x.index_select(-1, idx)


def mask_factors(
    a: torch.Tensor,
    g: torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
    module_kwargs: dict | None = None,
    include_bias: bool = True,
    *,
    proj_dim: int,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """``materialize_factors(...)[:, mask]`` without materializing.

    An entry of the per-sample weight gradient of an outer-product layer is a
    contraction of one output-gradient column with one activation column over
    the token axis -- ``dW[b, o, i] = sum_t g[b, t, o] * a[b, t, i]`` -- so the
    kept coordinates are gathered straight from the factors at ``O(B*T*k)``,
    never forming the ``(B, d_out*d_in)`` gradient.  The subset indexes the
    **flattened** gradient exactly as :func:`materialize_factors` lays it out
    (``o * d_in + i`` for linear/conv, ``c * P + p`` for transposed conv,
    ``v * E + e`` for embeddings), so the result equals a gather on the
    materialized gradient.  Norm layers are tiny and are materialized and
    gathered directly.

    Returns a dense ``(B, proj_dim)`` tensor.
    """
    _check_mask_kwargs(proj_kwargs)
    a, g = preprocess_factors(a, g, layer_type, module_kwargs, include_bias)
    projector = DattriProjector.coerce(projector)
    device = torch.device(proj_kwargs.get("device", g.device))

    if is_embedding(layer_type):
        if module_kwargs is None:
            raise ValueError(
                "Subsetting an embedding gradient requires module_kwargs with "
                "'num_embeddings' (the flattened width is vocab x embed_dim).",
            )
        ids = a.to(device)  # (B, T) int
        (grad,) = dtypes.align(g)
        grad = grad.to(device)  # (B, T, E)
        vocab, embed = module_kwargs["num_embeddings"], grad.shape[-1]
        idx = projector.mask_indices(
            vocab * embed, proj_dim=proj_dim, proj_seed=proj_seed, device=device
        )
        rows, cols = idx // embed, idx % embed
        # Entry (v, e) is the sum over tokens equal to v of the gradient column e.
        return _gather_pairs(
            (ids.unsqueeze(-1) == rows).to(grad.dtype), grad, rows, cols, one_hot=True
        )

    a, g = dtypes.align(a, g)
    a_f, g_f = to_3d(a).to(device), to_3d(g).to(device)

    if is_norm(layer_type):
        dense = (a_f * g_f).sum(1)  # (B, d): elementwise, cheap
        idx = projector.mask_indices(
            dense.shape[-1], proj_dim=proj_dim, proj_seed=proj_seed, device=device
        )
        return dense.index_select(-1, idx)

    if is_conv_transpose(layer_type):
        # materialize: einsum("blc,blp->bcp") -> flat index c * P + p
        n_cols = g_f.shape[-1]
        idx = projector.mask_indices(
            a_f.shape[-1] * n_cols,
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            device=device,
        )
        return _gather_pairs(a_f, g_f, idx // n_cols, idx % n_cols)

    # Linear and Conv: einsum("bto,bti->boi") -> flat index o * d_in + i
    n_cols = a_f.shape[-1]
    idx = projector.mask_indices(
        g_f.shape[-1] * n_cols, proj_dim=proj_dim, proj_seed=proj_seed, device=device
    )
    return _gather_pairs(g_f, a_f, idx // n_cols, idx % n_cols)


def _gather_pairs(
    rows_src: torch.Tensor,
    cols_src: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    one_hot: bool = False,
) -> torch.Tensor:
    """``sum_t rows_src[b, t, rows[k]] * cols_src[b, t, cols[k]]`` -> ``(B, k)``.

    Chunked over ``k`` so the ``(B, T, chunk)`` products stay bounded.  With
    ``one_hot`` the row source is already the gathered ``(B, T, k)`` indicator
    (embeddings), so only the column source is gathered.
    """
    B, T = cols_src.shape[:2]
    chunk = max(1, _MASK_CHUNK_ELEMS // max(1, B * T))
    out = []
    for start in range(0, rows.numel(), chunk):
        r, c = rows[start : start + chunk], cols[start : start + chunk]
        left = (
            rows_src[..., start : start + chunk]
            if one_hot
            else rows_src.index_select(-1, r)
        )
        out.append((left * cols_src.index_select(-1, c)).sum(1))
    return torch.cat(out, dim=-1)


def mask_factorized(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
    *,
    proj_dim: int,
    include_bias: bool = True,
    proj_seed: int = 0,
    **proj_kwargs,
) -> torch.Tensor:
    """:func:`mask_factors` on a :class:`Factorized` (batch-first-safe).

    Also accepts an already-dense ``(B, D)`` tensor, which is masked directly.
    """
    if isinstance(f, torch.Tensor):
        return mask_materialized(
            f, projector, proj_dim=proj_dim, proj_seed=proj_seed, **proj_kwargs
        )
    bf = f.as_batch_first()
    return mask_factors(
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


def should_materialize(
    capture_style: str,
    seq_len: int,
    k_a: int,
    k_g: int,
) -> bool:
    """Whether a layer's factors are materialized at capture under *capture_style*.

    ``"factorized"`` keeps the factors, ``"materialized"`` contracts them into
    the per-sample gradient, and ``"auto"`` applies the cost rule of
    :func:`maybe_materialize_projected` to the actual widths -- the projected
    widths of a ``"logra"`` layer, or the raw input (plus bias) and output
    widths of an unprojected one.
    """
    if capture_style == "factorized":
        return False
    if capture_style == "materialized":
        return True
    if capture_style != "auto":
        raise ValueError(
            f"capture_style must be one of {CAPTURE_STYLES}, got {capture_style!r}."
        )
    return maybe_materialize_projected(seq_len, k_a, k_g)


def project_layer(
    f: Factorized | torch.Tensor,
    layer_type: str,
    projector: Callable | DattriProjector | None,
    *,
    style: str = "logra",
    capture_style: str = "factorized",
    **proj_kwargs,
) -> tuple[object, bool]:
    """Route one layer through a projection style.

    Returns ``(payload, is_factorized)``:

    * ``"logra"`` -- project the two factors (Kronecker / double-sided), then
      keep them (payload ``(a_p, g_p)``, ``is_factorized`` True; the caller
      rewraps it into a :class:`Factorized` with ``module_kwargs=None``) or,
      when :func:`should_materialize` says so under *capture_style*, contract
      them into one ``(B, k_g*k_a)`` per-sample block (cheap, in the small
      projected space).
    * ``"dense"`` -- materialize the full per-sample gradient **first**, then
      project it with one matrix to ``(B, proj_dim)``.
    * ``"mask"`` -- keep ``proj_dim`` fixed random coordinates of the
      per-sample gradient, gathered from the factors without materializing
      (:func:`mask_factors`).

    ``"dense"`` and ``"mask"`` are dense by construction, so *capture_style*
    does not apply to them.  A materialized input tensor can only take the
    ``"dense"`` or ``"mask"`` path (there are no factors to project).
    """
    if style not in PROJECTION_STYLES:
        raise ValueError(f"style must be one of {PROJECTION_STYLES}, got {style!r}.")
    if style == "mask":
        return mask_factorized(f, layer_type, projector, **proj_kwargs), False
    if style == "logra" and not isinstance(f, torch.Tensor):
        a_p, g_p = project_factorized(f, layer_type, projector, **proj_kwargs)
        # Decide on the ACTUAL projected shapes rather than the requested
        # proj_dim: a layer may project asymmetrically, and the bias column
        # (include_bias) widens one side.
        seq_len = a_p.shape[1] if a_p.ndim == 3 else 1
        if should_materialize(capture_style, seq_len, a_p.shape[-1], g_p.shape[-1]):
            return materialize_factors(a_p, g_p, "nn.Linear"), False
        return (a_p, g_p), True
    return project_materialized(f, layer_type, projector, **proj_kwargs), False
