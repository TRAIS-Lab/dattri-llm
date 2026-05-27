from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Union

import torch


GradientRepresentation = Literal["materialized", "factorized"]
Indexing = Literal["batch", "batch_token"]


@dataclass(frozen=True)
class Factorized:
    activation: torch.Tensor
    pre_activation_grad: torch.Tensor

    def to(self, device=None, dtype=None) -> "Factorized":
        return Factorized(
            activation=self.activation.to(device=device, dtype=dtype),
            pre_activation_grad=self.pre_activation_grad.to(device=device, dtype=dtype),
        )

    @property
    def is_embedding(self) -> bool:
        """True when activation stores integer token IDs (``nn.Embedding`` hook)."""
        return not self.activation.is_floating_point()

    def reduce_activation(self, reduction: str = "mean") -> torch.Tensor:
        """Return the activation reduced to a float ``(B, F)`` tensor.

        This is the counterpart of reducing ``pre_activation_grad`` — both are
        needed when computing gram-matrix-based scores (ghost inner product).

        For standard layers (``nn.Linear``, ``nn.LayerNorm``, …) the stored
        activation is already a float tensor:

        * ``(B, F)``      — returned as-is (cast to float).
        * ``(B, T, F)``   — reduced over the token dimension via *reduction*
          (``"mean"`` or ``"sum"``), giving ``(B, F)``.

        For ``nn.Embedding`` the stored activation is an integer token-ID
        tensor of shape ``(B, T)``.  Its MLP-equivalent activation is
        ``one_hot(token_ids)`` of shape ``(B, T, vocab_size)``, which would be
        prohibitively large to materialise.  Instead we compute the reduced
        form directly via scatter-add:

        * ``"mean"`` → token-frequency vectors ``(B, effective_vocab)``,
          i.e. ``one_hot(token_ids).float().mean(dim=1)``.
        * ``"sum"``  → token-count vectors ``(B, effective_vocab)``,
          i.e. ``one_hot(token_ids).float().sum(dim=1)``.

        ``effective_vocab = max(token_ids) + 1`` for this batch; rows beyond
        that are zero and do not affect dot products with other batches.

        Args:
            reduction: ``"mean"`` (default) or ``"sum"``.

        Returns:
            Float tensor of shape ``(B, F)``.
        """
        if self.is_embedding:
            token_ids = self.activation          # (B, T) int
            B, T = token_ids.shape
            effective_vocab = int(token_ids.max().item()) + 1
            freq = torch.zeros(B, effective_vocab, dtype=torch.float,
                               device=token_ids.device)
            freq.scatter_add_(1, token_ids,
                              torch.ones(B, T, dtype=torch.float,
                                         device=token_ids.device))
            if reduction == "mean":
                freq = freq / T
            return freq                          # (B, effective_vocab)

        a = self.activation.float()
        if a.ndim == 2:
            return a                             # already (B, F)
        return a.mean(1) if reduction == "mean" else a.sum(1)  # (B, F)

    def materialize(self) -> torch.Tensor:
        """Convert factorized layer data into materialized per-sample gradients.

        Standard linear layers (``nn.Linear``, ``Conv1D``, ``nn.LayerNorm``):

        * ``[B, I], [B, O]       → [B, O * I]``
        * ``[B, T, I], [B, T, O] → [B, T, O * I]``

        Embedding layers (``nn.Embedding``), where activation stores integer
        token IDs of shape ``(B, T)`` and pre_activation_grad has shape
        ``(B, T, embed_dim)``:

        * Per-sample weight gradients are computed via scatter-add over the
          token dimension, producing ``[B, vocab_size * embed_dim]`` where
          ``vocab_size`` is inferred as ``max(token_ids) + 1`` in this batch.

        Note:
            Embedding materialization yields a *dense* gradient over the
            observed vocabulary — rows not present in the batch are zero.  The
            effective ``vocab_size`` may differ across batches; do not
            concatenate materialized Embedding gradients from different steps.
        """
        if self.is_embedding:
            return self._materialize_embedding()

        if self.activation.ndim == 2:
            grad = torch.einsum(
                "bo,bi->boi",
                self.pre_activation_grad,
                self.activation,
            )
            return grad.flatten(start_dim=1)

        if self.activation.ndim == 3:
            grad = torch.einsum(
                "bto,bti->btoi",
                self.pre_activation_grad,
                self.activation,
            )
            return grad.flatten(start_dim=2)

        raise ValueError("Expected activation and pre_activation_grad to be 2D or 3D.")

    def _materialize_embedding(self) -> torch.Tensor:
        """Scatter-add materialization for ``nn.Embedding`` weight gradients.

        The embedding weight gradient for sample *i* is:

        .. math::
            \\frac{\\partial L}{\\partial W}[k, :] =
            \\sum_{t:\\ \\text{token\\_ids}[i,t] = k}
            \\frac{\\partial L}{\\partial \\text{output}}[i, t, :]

        Args:
            (no args — uses ``self.activation`` and ``self.pre_activation_grad``)

        Returns:
            Float tensor of shape ``(B, vocab_size * embed_dim)`` where
            ``vocab_size = max(token_ids) + 1``.
        """
        token_ids = self.activation                 # (B, T) int
        grad = self.pre_activation_grad.float()     # (B, T, embed_dim)
        B, T = token_ids.shape
        embed_dim = grad.shape[-1]
        vocab_size = int(token_ids.max().item()) + 1

        # idx: (B, T, embed_dim) — broadcast token IDs along the embed dim
        idx = token_ids.unsqueeze(-1).expand(B, T, embed_dim)  # (B, T, embed_dim)
        result = torch.zeros(B, vocab_size, embed_dim, dtype=grad.dtype)
        result.scatter_add_(1, idx, grad)           # (B, vocab_size, embed_dim)
        return result.reshape(B, -1)                # (B, vocab_size * embed_dim)


GradientData = Union[torch.Tensor, Factorized]


@dataclass(frozen=True)
class Gradient:
    representation: Dict[str, GradientRepresentation]
    data: Dict[str, GradientData]
    layer_types: Optional[Dict[str, str]] = None
    indexing: Indexing = "batch"

    @property
    def layer_names(self) -> set[str]:
        return set(self.data.keys())

    @property
    def batch_size(self) -> int:
        x = next(iter(self.data.values()))
        return x.activation.shape[0] if isinstance(x, Factorized) else x.shape[0]

    @property
    def token_dim(self) -> Optional[int]:
        if self.indexing != "batch_token":
            return None
        x = next(iter(self.data.values()))
        return x.activation.shape[1] if isinstance(x, Factorized) else x.shape[1]

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

        expected_ndim = 3 if self.indexing == "batch_token" else 2
        batch_size = None
        token_dim = None

        for name, value in self.data.items():
            layer_repr = self.representation[name]

            if layer_repr == "factorized":
                if not isinstance(value, Factorized):
                    raise TypeError(f"{name} must be Factorized")

                act = value.activation
                gout = value.pre_activation_grad

                if act.ndim != expected_ndim or gout.ndim != expected_ndim:
                    raise ValueError(f"{name} has invalid factor dimensions")

                if act.shape[:-1] != gout.shape[:-1]:
                    raise ValueError(f"{name} factor batch/token dimensions mismatch")

                cur_batch = act.shape[0]
                cur_token = act.shape[1] if self.indexing == "batch_token" else None

            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} must be Tensor")

                if value.ndim != expected_ndim:
                    raise ValueError(f"{name} expected {expected_ndim}D tensor")

                cur_batch = value.shape[0]
                cur_token = value.shape[1] if self.indexing == "batch_token" else None

            if batch_size is None:
                batch_size = cur_batch
            elif batch_size != cur_batch:
                raise ValueError("All layers must have the same batch size")

            if token_dim is None:
                token_dim = cur_token
            elif token_dim != cur_token:
                raise ValueError("All layers must have the same token dimension")

    def clone(self) -> "Gradient":
        new_data = {}

        for name, value in self.data.items():
            if isinstance(value, Factorized):
                new_data[name] = Factorized(
                    value.activation.clone(),
                    value.pre_activation_grad.clone(),
                )
            else:
                new_data[name] = value.clone()

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=dict(self.layer_types) if self.layer_types else None,
            indexing=self.indexing,
        )

    def to(self, device=None, dtype=None) -> "Gradient":
        new_data = {}

        for name, value in self.data.items():
            new_data[name] = value.to(device=device, dtype=dtype)

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=self.indexing,
        )

    def materialize(self) -> "Gradient":
        if all(r == "materialized" for r in self.representation.values()):
            return self

        new_data = {}
        new_repr = {}

        for name, value in self.data.items():
            layer_repr = self.representation[name]
            if layer_repr == "factorized" and isinstance(value, Factorized):
                new_data[name] = value.materialize()
                new_repr[name] = "materialized"
            else:
                new_data[name] = value
                new_repr[name] = layer_repr

        return Gradient(
            representation=new_repr,
            data=new_data,
            layer_types=self.layer_types,
            indexing=self.indexing,
        )

    def select_layers(self, layer_names: Iterable[str]) -> "Gradient":
        names = set(layer_names)
        missing = names - self.layer_names
        if missing:
            raise KeyError(f"Unknown layers: {sorted(missing)}")

        return Gradient(
            representation={name: self.representation[name] for name in names},
            data={name: self.data[name] for name in names},
            layer_types=(
                {name: self.layer_types[name] for name in names}
                if self.layer_types
                else None
            ),
            indexing=self.indexing,
        )

    def concatenate(
        self,
        other: "Gradient",
        dim: Literal["batch", "token"] = "batch",
    ) -> "Gradient":
        self._check_compatible(other, require_same_batch=False)

        if dim == "token":
            if self.indexing != "batch_token":
                raise ValueError("Token concatenation requires indexing='batch_token'")
            if self.batch_size != other.batch_size:
                raise ValueError("Token concatenation requires the same batch size")

        cat_dim = 0 if dim == "batch" else 1
        new_data = {}

        for name in self.layer_names:
            a = self.data[name]
            b = other.data[name]

            if isinstance(a, Factorized) and isinstance(b, Factorized):
                new_data[name] = Factorized(
                    activation=torch.cat([a.activation, b.activation], dim=cat_dim),
                    pre_activation_grad=torch.cat(
                        [a.pre_activation_grad, b.pre_activation_grad],
                        dim=cat_dim,
                    ),
                )
            elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                new_data[name] = torch.cat([a, b], dim=cat_dim)
            else:
                raise TypeError(f"{name} has mismatched data types")

        out = Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=self.indexing,
        )
        out.validate()
        return out

    def aggregate(
        self,
        dim: Literal["token"] = "token",
        mode: Literal["sum", "mean"] = "sum",
    ) -> "Gradient":
        if dim != "token":
            raise NotImplementedError("Only token aggregation is supported")
        if self.indexing != "batch_token":
            raise ValueError("Token aggregation requires indexing='batch_token'")
        if mode not in {"sum", "mean"}:
            raise ValueError("mode must be 'sum' or 'mean'")

        reduce_fn = torch.sum if mode == "sum" else torch.mean
        new_data = {}

        for name, value in self.data.items():
            if isinstance(value, Factorized):
                # Sum of outer products is not generally representable
                # as a single outer product, so we materialize first.
                new_data[name] = reduce_fn(value.materialize(), dim=1)
            else:
                new_data[name] = reduce_fn(value, dim=1)

        new_repr = {
            name: "materialized" if self.representation[name] == "factorized" else self.representation[name]
            for name in self.data
        }

        return Gradient(
            representation=new_repr,
            data=new_data,
            layer_types=self.layer_types,
            indexing="batch",
        )

    def slice(
        self,
        dim: Literal["batch", "token"],
        index,
    ) -> "Gradient":
        if dim == "token" and self.indexing != "batch_token":
            raise ValueError("Token slicing requires indexing='batch_token'")

        slice_dim = 0 if dim == "batch" else 1
        new_data = {}

        def slice_tensor(x: torch.Tensor) -> torch.Tensor:
            idx = [slice(None)] * x.ndim
            idx[slice_dim] = index
            y = x[tuple(idx)]

            if isinstance(index, int):
                y = y.unsqueeze(slice_dim)

            return y

        for name, value in self.data.items():
            if isinstance(value, Factorized):
                new_data[name] = Factorized(
                    activation=slice_tensor(value.activation),
                    pre_activation_grad=slice_tensor(value.pre_activation_grad),
                )
            else:
                new_data[name] = slice_tensor(value)

        return Gradient(
            representation=dict(self.representation),
            data=new_data,
            layer_types=self.layer_types,
            indexing=self.indexing,
        )

    def similarity(
        self,
        other: "Gradient",
        metric: Literal["dot", "cosine"] = "dot",
        reduce: Literal["none", "layer", "all"] = "all",
        eps: float = 1e-8,
    ):
        self._check_compatible(other, require_same_batch=True)

        per_layer = {}

        for name in self.layer_names:
            x = self._layer_tensor(name)
            y = other._layer_tensor(name)

            dot = (x * y).sum(dim=-1)

            if metric == "dot":
                sim = dot
            elif metric == "cosine":
                sim = dot / (x.norm(dim=-1) * y.norm(dim=-1) + eps)
            else:
                raise ValueError("metric must be 'dot' or 'cosine'")

            per_layer[name] = sim

        if reduce == "none":
            return per_layer

        if reduce == "layer":
            return {name: value.sum() for name, value in per_layer.items()}

        if reduce == "all":
            return torch.stack([value.sum() for value in per_layer.values()]).sum()

        raise ValueError("reduce must be 'none', 'layer', or 'all'")

    def _layer_tensor(self, name: str) -> torch.Tensor:
        value = self.data[name]
        return value.materialize() if isinstance(value, Factorized) else value

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

        if self.indexing != other.indexing:
            raise ValueError(f"Different indexing: {self.indexing} vs {other.indexing}")

        if self.layer_types != other.layer_types:
            raise ValueError(
                f"Layer types differ: {self.layer_types} vs {other.layer_types}"
            )

        if require_same_batch and self.batch_size != other.batch_size:
            raise ValueError("Batch sizes differ")

        if (
            require_same_batch
            and self.indexing == "batch_token"
            and self.token_dim != other.token_dim
        ):
            raise ValueError("Token dimensions differ")


# --------------------------------------------------------------------------- #
# Sample hashing                                                               #
# --------------------------------------------------------------------------- #


def hash_sample(inputs: dict[str, torch.Tensor], sample_idx: int) -> str:
    """Compute a SHA-256 hash that uniquely identifies one sample in a batch.

    All tensor fields in *inputs* are included, sorted by key for
    determinism.  Non-tensor values are skipped.  The hash is content-based
    and stable: the same sample always produces the same hash regardless of
    batch position, epoch, or whether the trainer has stripped metadata fields
    like dataset indices.

    Args:
        inputs: The full input dict passed to the model (e.g.
            ``{"input_ids": ..., "attention_mask": ..., "labels": ...}``).
        sample_idx: Index of the sample within the batch dimension.

    Returns:
        A 64-character hex string (SHA-256 digest).
    """
    h = hashlib.sha256()
    for key in sorted(inputs):
        val = inputs[key]
        if isinstance(val, torch.Tensor) and val.ndim > 0:
            h.update(key.encode())
            h.update(val[sample_idx].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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
