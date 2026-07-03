from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Callable, Dict, Iterable, Literal, Optional, Union

import torch

from dattri_llm.gradient import ops


GradientRepresentation = Literal["materialized", "factorized"]
Indexing = Literal["batch", "batch_token"]


@dataclass(frozen=True)
class Factorized:
    activation: torch.Tensor
    pre_activation_grad: torch.Tensor
    # Minimal serialisable hyperparameters extracted at hook-registration time
    # via ops.extract_module_kwargs.  Excluded from equality and hashing so
    # two Factorized tensors with identical data compare equal regardless of
    # which layer produced them.
    module_kwargs: Optional[dict] = field(default=None, compare=False, repr=False, hash=False)
    # Tensor layout flag.  ``True`` (default) means the batch axis is dim 0 and
    # the token/sequence axis is dim 1 — the layout every ``ops`` kernel assumes.
    # ``False`` marks a *sequence-first* capture ``(T, B, ...)`` (e.g. a layer
    # internal to a sequence-first model such as MusicTransformer); call
    # :meth:`as_batch_first` to get the canonical ``(B, T, ...)`` view.
    batch_first: bool = field(default=True, compare=False)

    def to(self, device=None, dtype=None) -> "Factorized":
        return Factorized(
            activation=self.activation.to(device=device, dtype=dtype),
            pre_activation_grad=self.pre_activation_grad.to(device=device, dtype=dtype),
            module_kwargs=self.module_kwargs,
            batch_first=self.batch_first,
        )

    def as_batch_first(self) -> "Factorized":
        """Return a batch-first view of these factors.

        For a sequence-first capture this swaps the leading two axes
        (``(T, B, ...) -> (B, T, ...)``) of both factors; for an already
        batch-first capture it returns ``self`` unchanged.  Every ``ops`` kernel
        expects batch-first input, so call this before handing the raw factors to
        :mod:`dattri_llm.gradient.ops`.
        """
        if self.batch_first:
            return self
        return Factorized(
            activation=self.activation.transpose(0, 1),
            pre_activation_grad=self.pre_activation_grad.transpose(0, 1),
            module_kwargs=self.module_kwargs,
            batch_first=True,
        )

GradientData = Union[torch.Tensor, Factorized]


@dataclass(frozen=True)
class Gradient:
    representation: Dict[str, GradientRepresentation]
    data: Dict[str, GradientData]
    layer_types: Dict[str, str]
    indexing: Dict[str, Indexing] = field(default_factory=dict)
    validate_on_init: InitVar[bool] = True

    def __post_init__(self, validate_on_init: bool) -> None:
        full_indexing = {name: self.indexing.get(name, "batch") for name in self.data}
        object.__setattr__(self, "indexing", full_indexing)
        if validate_on_init:
            self.validate()

    def _layer_indexing(self, name: str) -> Indexing:
        """Return the indexing mode for *name*; raises ``KeyError`` if absent."""
        return self.indexing[name]

    @property
    def layer_names(self) -> set[str]:
        return set(self.data.keys())

    @property
    def batch_size(self) -> int:
        # The real batch size is the largest per-layer batch dim: a broadcast
        # layer (e.g. a positional embedding) carries batch 1 and must not be
        # mistaken for the whole batch when it happens to come first.
        # ``param_grad`` tensors have no sample axis, so they are excluded.
        best = 0
        for name, x in self.data.items():
            if self.layer_types.get(name) == ops.PARAM_GRAD_TYPES:
                continue
            if isinstance(x, Factorized):
                # Batch axis is dim 0 when batch-first, dim 1 when sequence-first.
                best = max(best, x.activation.shape[0 if x.batch_first else 1])
            else:
                best = max(best, x.shape[0])
        if best:
            return best
        # Only ``param_grad`` layers present — fall back to the first tensor.
        x = next(iter(self.data.values()))
        return x.activation.shape[0] if isinstance(x, Factorized) else x.shape[0]

    @property
    def token_dim(self) -> Dict[str, "int | None"]:
        """Per-layer token dimension; ``None`` for layers with ``"batch"`` indexing."""
        result: Dict[str, "int | None"] = {}
        for name, value in self.data.items():
            if self._layer_indexing(name) == "batch_token":
                if isinstance(value, Factorized):
                    # Token axis is dim 1 when batch-first, dim 0 when seq-first.
                    result[name] = value.activation.shape[1 if value.batch_first else 0]
                else:
                    result[name] = value.shape[1]
            else:
                result[name] = None
        return result

    @property
    def device(self) -> torch.device:
        x = next(iter(self.data.values()))
        return x.activation.device if isinstance(x, Factorized) else x.device

    @property
    def dtype(self) -> torch.dtype:
        x = next(iter(self.data.values()))
        return x.activation.dtype if isinstance(x, Factorized) else x.dtype

    def validate(self) -> None:
        if not self.data:
            raise ValueError("data cannot be empty")

        missing = set(self.data.keys()) - set(self.representation.keys())
        if missing:
            raise ValueError(f"Missing representation for layers: {sorted(missing)}")

        batch_size = None

        for name, value in self.data.items():
            layer_repr = self.representation[name]
            layer_type = self.layer_types[name]

            if layer_repr == "factorized":
                if not isinstance(value, Factorized):
                    raise TypeError(f"{name} must be Factorized")

                # Validate against the canonical batch-first layout, so a
                # sequence-first capture is checked exactly like a batch-first one.
                bf = value.as_batch_first()
                act = bf.activation
                gout = bf.pre_activation_grad

                if layer_type == "nn.EmbeddingBag":
                    # EmbeddingBag: activation is (B, T) int token indices,
                    # grad is the per-bag output gradient (B, embed_dim).
                    if act.ndim != 2 or gout.ndim != 2:
                        raise ValueError(
                            f"{name} has invalid embedding-bag factor dimensions"
                        )
                    if act.shape[0] != gout.shape[0]:
                        raise ValueError(
                            f"{name} embedding-bag batch dimension mismatch"
                        )
                elif ops.is_embedding(layer_type):
                    # Embedding: activation is (B, T) int, grad is (B, T, embed_dim).
                    if act.ndim != 2 or gout.ndim != 3:
                        raise ValueError(
                            f"{name} has invalid embedding factor dimensions"
                        )
                    if act.shape != gout.shape[:2]:
                        raise ValueError(
                            f"{name} embedding batch/token dimensions mismatch"
                        )
                elif ops.is_conv(layer_type) or ops.is_conv_transpose(layer_type):
                    # Raw conv data: a=(N, C_in, *spatial_in), g=(N, C_out, *spatial_out).
                    # Spatial and channel dims differ; only batch size must match.
                    if act.shape[0] != gout.shape[0]:
                        raise ValueError(
                            f"{name} conv factor batch size mismatch: "
                            f"{act.shape[0]} != {gout.shape[0]}"
                        )
                else:
                    if act.shape[:-1] != gout.shape[:-1]:
                        raise ValueError(
                            f"{name} factor batch/token dimensions mismatch"
                        )

                cur_batch = act.shape[0]

            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} must be Tensor")

                if value.ndim < 1:
                    raise ValueError(f"{name} must be at least 1D")
                if (
                    layer_type != ops.PARAM_GRAD_TYPES
                    and self._layer_indexing(name) == "batch_token"
                    and value.ndim != 3
                ):
                    raise ValueError(
                        f"{name} declared as batch_token but tensor is {value.ndim}D, expected 3D"
                    )
                cur_batch = value.shape[0]

            # A layer with batch size 1 is a *broadcast* / shared gradient — e.g.
            # a positional embedding added to every sample, whose gradient is
            # summed over the batch.  Like ``param_grad`` it has no per-sample
            # axis, so it is compatible with any real batch size and does not
            # constrain the check.
            is_param_grad = layer_type == ops.PARAM_GRAD_TYPES
            if not is_param_grad and cur_batch != 1:
                if batch_size is None:
                    batch_size = cur_batch
                elif batch_size != cur_batch:
                    raise ValueError("All layers must have the same batch size")

    def clone(self) -> "Gradient":
        new_data = {}

        for name, value in self.data.items():
            if isinstance(value, Factorized):
                new_data[name] = Factorized(
                    value.activation.clone(),
                    value.pre_activation_grad.clone(),
                    module_kwargs=value.module_kwargs,
                    batch_first=value.batch_first,
                )
            else:
                new_data[name] = value.clone()

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=dict(self.layer_types),
            indexing=dict(self.indexing),
        )

    def to(self, device=None, dtype=None) -> "Gradient":
        new_data = {}

        for name, value in self.data.items():
            new_data[name] = value.to(device=device, dtype=dtype)

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=dict(self.indexing),
        )

    def materialize(self) -> "Gradient":
        if all(r == "materialized" for r in self.representation.values()):
            return self

        new_data = {}
        new_repr = {}
        new_indexing = dict(self.indexing)

        for name, value in self.data.items():
            layer_repr = self.representation[name]
            if layer_repr == "factorized" and isinstance(value, Factorized):
                layer_type = self.layer_types[name]
                new_data[name] = ops.materialize(value, layer_type)
                new_repr[name] = "materialized"
                # ops.materialize always returns (B, d) — collapse to "batch".
                new_indexing[name] = "batch"
            else:
                new_data[name] = value
                new_repr[name] = layer_repr

        return Gradient(
            representation=new_repr,
            data=new_data,
            layer_types=self.layer_types,
            indexing=new_indexing,
        )

    def project(self, projector: Callable, proj_kwargs: Dict[str, dict]) -> "Gradient":
        """Random-project each layer's per-sample gradient to a smaller dimension.

        Two styles, chosen per layer by ``proj_kwargs[name]["factorize"]``:

        * ``factorize=True`` (LoGRA) — project the factorized factors, keeping the
          Kronecker structure at width ``proj_dim``; the layer stays *factorized*.
          Defined only for outer-product gradients (linear / conv); norm and
          embedding layers must use ``factorize=False``.
        * ``factorize=False`` (TRAK) — materialize the per-sample weight gradient,
          then project it, collapsing the layer to a dense ``(B, proj_dim)`` block.

        Args:
            projector: a projection factory following dattri's ``random_project``
                protocol — ``projector(feature, batch_size, proj_dim=..., **kw)``
                returns a callable mapping ``(N, D) -> (N, proj_dim)``.
            proj_kwargs: ``{layer_name: dict}`` per-layer config.  Each dict carries
                ``proj_dim`` and optionally ``factorize`` (default ``True``),
                ``proj_seed`` and any projector kwargs (``proj_type``,
                ``proj_max_batch_size``, ``device``, ...).  A ``"__default__"``
                entry supplies the config for layers without their own; layers
                with **neither** an entry nor ``"__default__"`` are left unchanged.

        Returns:
            A new :class:`Gradient` holding each layer's projection.
        """
        new_data: Dict[str, GradientData] = {}
        new_repr: Dict[str, GradientRepresentation] = {}
        new_types: Dict[str, str] = {}
        new_indexing: Dict[str, Indexing] = {}

        for name, value in self.data.items():
            kw = proj_kwargs.get(name, proj_kwargs.get("__default__"))
            if kw is None:
                # No config for this layer — pass it through untouched.
                new_data[name] = value
                new_repr[name] = self.representation[name]
                new_types[name] = self.layer_types[name]
                new_indexing[name] = self.indexing[name]
                continue
            kw = dict(kw)
            factorize = kw.pop("factorize", True)
            payload, is_factorized = ops.project_layer(
                value, self.layer_types[name], projector, factorize=factorize, **kw
            )
            if is_factorized:
                # module_kwargs=None: the projected factors are final; projected
                # outer-product factors behave as a plain linear layer.
                new_data[name] = Factorized(*payload)
                new_repr[name] = "factorized"
                new_types[name] = "nn.Linear"
                new_indexing[name] = self.indexing[name]
            else:
                new_data[name] = payload
                new_repr[name] = "materialized"
                new_types[name] = self.layer_types[name]
                new_indexing[name] = "batch"

        return Gradient(
            representation=new_repr,
            data=new_data,
            layer_types=new_types,
            indexing=new_indexing,
        )

    def select_layers(self, layer_names: Iterable[str]) -> "Gradient":
        names = set(layer_names)
        missing = names - self.layer_names
        if missing:
            raise KeyError(f"Unknown layers: {sorted(missing)}")

        return Gradient(
            representation={name: self.representation[name] for name in names},
            data={name: self.data[name] for name in names},
            layer_types={k: v for k, v in self.layer_types.items() if k in names},
            indexing={name: idx for name, idx in self.indexing.items() if name in names},
        )

    def concatenate(
        self,
        other: "Gradient",
        dim: Literal["batch", "token"] = "batch",
    ) -> "Gradient":
        self._check_compatible(other, require_same_batch=False)

        if dim == "token":
            non_bt = [
                n for n in self.layer_names
                if self._layer_indexing(n) != "batch_token"
            ]
            if non_bt:
                raise ValueError(
                    f"Token concatenation requires all layers to have "
                    f"indexing='batch_token'. Non-conforming: {sorted(non_bt)}"
                )
            if self.batch_size != other.batch_size:
                raise ValueError("Token concatenation requires the same batch size")

        new_data = {}

        def cat_axis(batch_first: bool) -> int:
            # Batch axis is 0 (1) and token axis 1 (0) for batch-first (seq-first).
            if dim == "batch":
                return 0 if batch_first else 1
            return 1 if batch_first else 0

        for name in self.layer_names:
            a = self.data[name]
            b = other.data[name]

            if isinstance(a, Factorized) and isinstance(b, Factorized):
                if a.batch_first != b.batch_first:
                    raise ValueError(
                        f"{name} has mismatched batch_first layout between gradients"
                    )
                cat_dim = cat_axis(a.batch_first)
                new_data[name] = Factorized(
                    activation=torch.cat([a.activation, b.activation], dim=cat_dim),
                    pre_activation_grad=torch.cat(
                        [a.pre_activation_grad, b.pre_activation_grad],
                        dim=cat_dim,
                    ),
                    module_kwargs=a.module_kwargs,
                    batch_first=a.batch_first,
                )
            elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                new_data[name] = torch.cat([a, b], dim=cat_axis(True))
            else:
                raise TypeError(f"{name} has mismatched data types")

        out = Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=dict(self.indexing),
        )
        out.validate()
        return out

    def aggregate(
        self,
        dim: Literal["token"] = "token",
        mode: Literal["sum", "mean"] = "sum",
    ) -> "Gradient":
        """Reduce the token dimension of ``"batch_token"`` layers.

        Layers with ``"batch"`` indexing are passed through unchanged.
        This allows a mixed-indexing :class:`Gradient` to be unified into
        a purely ``"batch"``-indexed one.

        Args:
            dim: Must be ``"token"`` (only token aggregation is supported).
            mode: ``"sum"`` (default) or ``"mean"``.

        Returns:
            A new :class:`Gradient` with all layers at ``"batch"`` indexing.
        """
        if dim != "token":
            raise NotImplementedError("Only token aggregation is supported")
        if mode not in {"sum", "mean"}:
            raise ValueError("mode must be 'sum' or 'mean'")

        reduce_fn = torch.sum if mode == "sum" else torch.mean
        new_data = {}
        new_repr = {}

        for name, value in self.data.items():
            layer_type = self.layer_types[name]
            if self._layer_indexing(name) == "batch_token":
                if isinstance(value, Factorized):
                    mat = ops.materialize(value, layer_type)  # (B, d), summed over T
                    if mode == "mean":
                        # T is the token count after preprocessing: for raw Conv
                        # this is L_out (spatial positions), not C_in.
                        if value.module_kwargs is not None and (
                            ops.is_conv(layer_type) or ops.is_conv_transpose(layer_type)
                        ):
                            # L_out is encoded in mat's flattened dim; recover it
                            # from the preprocessed activation shape.
                            a_p, _ = ops.preprocess_factorized(value, layer_type)
                            T = a_p.shape[1]
                        else:
                            T = value.as_batch_first().activation.shape[1]
                        mat = mat / T
                    new_data[name] = mat
                    new_repr[name] = "materialized"
                else:
                    new_data[name] = reduce_fn(value, dim=1)
                    new_repr[name] = self.representation[name]
            else:
                # "batch" layer — pass through unchanged.
                new_data[name] = value
                new_repr[name] = self.representation[name]

        return Gradient(
            representation=new_repr,
            data=new_data,
            layer_types=self.layer_types,
            indexing={},  # all layers now "batch" (default)
        )

    def slice(
        self,
        dim: Literal["batch", "token"],
        index,
    ) -> "Gradient":
        if dim == "token":
            non_bt = [
                n for n in self.layer_names
                if self._layer_indexing(n) != "batch_token"
            ]
            if non_bt:
                raise ValueError(
                    f"Token slicing requires all layers to have "
                    f"indexing='batch_token'. Non-conforming: {sorted(non_bt)}"
                )

        new_data = {}

        def slice_tensor(x: torch.Tensor, slice_dim: int) -> torch.Tensor:
            idx = [slice(None)] * x.ndim
            idx[slice_dim] = index
            y = x[tuple(idx)]

            if isinstance(index, int):
                y = y.unsqueeze(slice_dim)

            return y

        def axis(batch_first: bool) -> int:
            # Batch axis is 0 (1) and token axis 1 (0) for batch-first (seq-first).
            if dim == "batch":
                return 0 if batch_first else 1
            return 1 if batch_first else 0

        for name, value in self.data.items():
            if self.layer_types[name] == ops.PARAM_GRAD_TYPES:
                # Batch-level parameter gradients have no sample axis. Preserve
                # them until per-sample attribution loading warns and skips them.
                new_data[name] = value
            elif isinstance(value, Factorized):
                sdim = axis(value.batch_first)
                new_data[name] = Factorized(
                    activation=slice_tensor(value.activation, sdim),
                    pre_activation_grad=slice_tensor(value.pre_activation_grad, sdim),
                    module_kwargs=value.module_kwargs,
                    batch_first=value.batch_first,
                )
            else:
                new_data[name] = slice_tensor(value, axis(True))

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=dict(self.indexing),
        )

    def similarity(
        self,
        other: "Gradient",
        metric: Literal["dot", "cosine"] = "dot",
        reduce: Literal["none", "all"] = "none",
        mode: Literal["factorized", "materialized", "auto"] = "auto",
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Per-sample gradient similarity between this gradient and ``other``.

        For each shared layer this forms the full ``(B_self, B_other)`` cross-gram
        ``K[i, j] = ⟨∇W_i (self), ∇W_j (other)⟩`` — the fundamental object for
        gradient-based data attribution.  The aligned ``i == j`` case is its
        diagonal; per-sample influence scoring against a target gradient is the
        row-sum over ``other``'s batch (``⟨∇W_i, Σ_j ∇W_target_j⟩``).

        Layers absent from *other* are skipped (so a target gradient need only
        overlap on some layers).  A layer stored factorized on one side and
        materialized on the other is compared by materializing the factorized
        side.

        Args:
            other: Gradient to compare against (e.g. a target/reference batch).
            metric: ``"dot"`` for the raw inner product, or ``"cosine"`` for the
                cosine similarity (each entry divided by the two gradients'
                norms).
            reduce: ``"none"`` (default) keeps the result broken down per layer,
                returning ``{layer: (B_self, B_other)}``.  ``"all"`` returns a
                single ``(B_self, B_other)`` matrix — the full-model gradient
                cross-gram, i.e. the per-layer matrices summed over layers
                (``⟨g_i, g_j⟩ = Σ_layer ⟨g_i^layer, g_j^layer⟩``).  Either way
                the per-sample pair structure is preserved.
            mode: ``"factorized"`` (ghost, no materialisation, via
                :func:`ops.cross_dot`), ``"materialized"`` (materialize each side
                then matrix-multiply), or ``"auto"`` (choose per layer by the cost
                heuristic in :func:`ops.cross_gram_auto` — factorized for small
                token/patch counts, materialized once the $S^2$ factor dominates).
                All three are numerically equivalent.
            eps: Numerical floor added to the cosine denominator.

        Returns:
            ``{layer: (B_self, B_other) tensor}`` for ``reduce="none"``, else a
            single ``(B_self, B_other)`` tensor.

        Note:
            ``reduce="all"`` requires every shared layer to have the same
            ``B_self`` (and ``B_other``); summing is undefined when a layer's
            gradient was collapsed over the batch during a forward broadcast.
        """
        if metric not in {"dot", "cosine"}:
            raise ValueError("metric must be 'dot' or 'cosine'")
        if reduce not in {"none", "all"}:
            raise ValueError("reduce must be 'none' or 'all'")
        if mode not in {"factorized", "materialized", "auto"}:
            raise ValueError("mode must be 'factorized', 'materialized', or 'auto'")

        per_layer: Dict[str, torch.Tensor] = {}
        for name in self.layer_names:
            if name not in other.data:
                continue
            matrix = self._layer_cross_matrix(other, name, mode)
            if matrix is None:
                continue
            if metric == "cosine" and reduce == "none":
                # Per-layer cosine: normalise each layer independently.
                n_s = self._layer_norm_sq(name, mode).clamp_min(0).sqrt()   # (B_self,)
                n_o = other._layer_norm_sq(name, mode).clamp_min(0).sqrt()  # (B_other,)
                matrix = matrix / (n_s[:, None] * n_o[None, :] + eps)

            per_layer[name] = matrix

        if reduce == "none":
            return per_layer
        # reduce == "all": full-model gradient cross-gram (sum the per-layer
        # matrices, since the whole-model gradient is the concatenation of layers).
        if not per_layer:
            raise ValueError("No shared layers to compute an overall similarity")
        total = torch.stack(list(per_layer.values())).sum(0)
        if metric == "cosine":
            # Full-model cosine: normalise by the concatenated-gradient norms,
            # i.e. sqrt(sum of per-layer squared norms).
            norm_sq_s = torch.stack(
                [self._layer_norm_sq(n, mode) for n in per_layer]
            ).sum(0).clamp_min(0)                                         # (B_self,)
            norm_sq_o = torch.stack(
                [other._layer_norm_sq(n, mode) for n in per_layer]
            ).sum(0).clamp_min(0)                                         # (B_other,)
            total = total / (norm_sq_s.sqrt()[:, None] * norm_sq_o.sqrt()[None, :] + eps)
        return total

    def _layer_cross_matrix(
        self,
        other: "Gradient",
        name: str,
        mode: Literal["factorized", "materialized", "auto"],
    ):
        """Return the ``(B_self, B_other)`` cross-gram for one layer, or ``None``
        when the representations are incompatible.

        The factorized↔materialized routing lives in :func:`ops.cross_dot` /
        :func:`ops._cross_gram` (the shared kernel, also used by K-FAC), so this
        method just forwards ``mode``; ``"auto"`` picks the cheaper path per layer
        from :func:`ops.maybe_use_materialized_gram`.
        """
        sv = self.data[name]
        ov = other.data[name]
        layer_type = self.layer_types[name]

        if isinstance(sv, Factorized) and isinstance(ov, Factorized):
            return ops.cross_dot(sv, ov, layer_type, mode=mode)

        if isinstance(sv, Factorized) or isinstance(ov, Factorized):
            if isinstance(sv, Factorized):
                sv = ops.materialize(sv, layer_type)
            else:
                ov = ops.materialize(ov, other.layer_types[name])

        # Both plain materialized tensors: flatten non-batch dims and dot.
        xf = sv.reshape(sv.shape[0], -1).float()
        yf = ov.reshape(ov.shape[0], -1).float()
        return xf @ yf.T

    def _layer_norm_sq(
        self,
        name: str,
        mode: Literal["factorized", "materialized", "auto"],
    ) -> torch.Tensor:
        """Per-sample squared gradient norms ``(B,)`` for one layer.

        Used by :meth:`similarity` for the cosine denominator.  The factorized↔
        materialized routing lives in :func:`ops.grad_norm_sq` /
        :func:`ops._grad_norm_sq`; this method just forwards ``mode`` (``"auto"``
        picks the cheaper path via :func:`ops.maybe_use_materialized_norm`).  The
        value is identical across modes.
        """
        value = self.data[name]
        if not isinstance(value, Factorized):
            flat = value.reshape(value.shape[0], -1).float()
            return (flat * flat).sum(-1)
        return ops.grad_norm_sq(value, self.layer_types[name], mode=mode)

    @staticmethod
    def _align_embedding_width(
        mat_s: torch.Tensor, mat_t: torch.Tensor, embed_dim: int
    ):
        """Zero-pad two flattened embedding gradients to a common vocab width.

        Embedding ``materialize`` uses ``vocab = max(token) + 1`` per batch, so
        the two sides may differ; padding makes the dot product well-defined.
        """
        v_s = mat_s.shape[1] // embed_dim
        v_t = mat_t.shape[1] // embed_dim
        v = max(v_s, v_t)
        if v_s < v:
            mat_s = torch.cat(
                [mat_s, mat_s.new_zeros(mat_s.shape[0], (v - v_s) * embed_dim)], dim=1
            )
        if v_t < v:
            mat_t = torch.cat(
                [mat_t, mat_t.new_zeros(mat_t.shape[0], (v - v_t) * embed_dim)], dim=1
            )
        return mat_s, mat_t

    def _check_compatible(
        self,
        other: "Gradient",
        require_same_batch: bool,
    ) -> None:
        if self.layer_names != other.layer_names:
            raise ValueError("Layer sets differ")

        mismatched = {
            name
            for name in self.layer_names
            if self.representation.get(name) != other.representation.get(name)
        }
        if mismatched:
            raise ValueError(
                f"Layers have differing representations: {sorted(mismatched)}"
            )

        indexing_mismatch = {
            name
            for name in self.layer_names
            if self._layer_indexing(name) != other._layer_indexing(name)
        }
        if indexing_mismatch:
            raise ValueError(
                f"Layers have differing indexing: {sorted(indexing_mismatch)}"
            )

        if self.layer_types != other.layer_types:
            raise ValueError(
                f"Layer types differ: {self.layer_types} vs {other.layer_types}"
            )

        if require_same_batch and self.batch_size != other.batch_size:
            raise ValueError("Batch sizes differ")

        if require_same_batch:
            self_td = self.token_dim
            other_td = other.token_dim
            mismatched_td = {
                name
                for name in self.layer_names
                if self_td[name] != other_td[name]
            }
            if mismatched_td:
                raise ValueError(
                    f"Token dimensions differ for layers: {sorted(mismatched_td)}"
                )


# --------------------------------------------------------------------------- #
# Gradient record                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class GradientRecord:
    """A gradient snapshot with its identity baked in.

    Attributes:
        step: Global batch-step counter, incremented once per completed
            backward pass.
        input_hash: Content-based identifier for the sample(s) in this record.

            * **Per-sample** (``recording_type="per_sample"``): a single
              64-character SHA-256 hex string identifying the one sample whose
              gradient is stored.
            * **Per-batch** (``recording_type="per_batch"``): a list of
              64-character SHA-256 hex strings, one per sample in the batch,
              in batch order.  The :class:`GradientFileManager` indexes every
              hash in the list so per-sample lookup works even when many
              records are bundled in one file.

        gradient: The collected :class:`Gradient`.
    """

    step: int
    input_hash: str | list[str]
    gradient: Gradient

    def __repr__(self) -> str:
        if isinstance(self.input_hash, list):
            h_repr = f"[{self.input_hash[0][:16]}…+{len(self.input_hash)-1}]"
        else:
            h_repr = f"{self.input_hash[:16]}…"
        return f"GradientRecord(step={self.step}, input_hash={h_repr})"
