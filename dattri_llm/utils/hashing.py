"""Content hashing of model inputs (the sample identity for capture/retrieval)."""

from __future__ import annotations

import hashlib

import torch


def _digest_field(h: hashlib._hashlib.HASH, key: str, value: object) -> bool:
    """Digest one input field into *h*; return whether it was digestible.

    Tensors are digested directly.  Lists, tuples, and numeric scalars are
    converted with ``torch.tensor`` first, so ``dataset[i]`` yielding plain
    Python lists (the Hugging Face default) hashes identically to the
    captured tensor row -- provided the dtypes agree under torch's default
    conversion (ints -> int64, floats -> float32); the digest is over raw
    bytes and therefore dtype-sensitive.  Everything else (strings, ``None``,
    nested dicts, ragged lists) is not digestible and is skipped.
    """
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (list, tuple, int, float, bool)):
        try:
            tensor = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    else:
        return False
    h.update(key.encode())
    h.update(tensor.cpu().contiguous().numpy().tobytes())
    return True


def hash_sample(inputs: dict[str, object]) -> str:
    """SHA-256 content hash identifying **one sample** by its model inputs.

    *inputs* holds a single sample's fields -- e.g. what ``dataset[i]`` yields,
    or one row sliced out of a batch.  Tensor fields, numeric list/tuple
    fields, and numeric scalars are digested (sorted by key for determinism);
    a plain Python list hashes identically to the equivalent tensor, so a raw
    ``dataset[i]`` looks up gradients captured from the collated tensor batch.
    Non-digestible fields (strings, ``None``, ...) are skipped.  The digest is
    over raw bytes, so the same sample hashes identically wherever shuffling
    put it: a ``(T,)`` row, its ``(1, T)`` batched form, and ``batch[i]`` all
    produce the same hash.

    Args:
        inputs: One sample's model-input dict (e.g.
            ``{"input_ids": ..., "attention_mask": ..., "labels": ...}``).

    Returns:
        A 64-character hex string (SHA-256 digest).

    Raises:
        ValueError: If no field is digestible -- an empty digest would give
            every sample the same identity and silently corrupt lookups.
    """
    h = hashlib.sha256()
    digested = False
    for key in sorted(inputs):
        digested |= _digest_field(h, key, inputs[key])
    if not digested:
        raise ValueError(
            "hash_sample found no digestible field (tensor, numeric "
            f"list/tuple, or numeric scalar) among keys {sorted(inputs)}; "
            "an empty digest would give every sample the same identity.",
        )
    return h.hexdigest()


def hash_batch(
    inputs: dict[str, object],
    batch_size: int | None = None,
) -> list[str]:
    """Per-sample content hashes for a **batched** input dict, in batch order.

    Fields whose leading dimension is the batch dimension -- tensors of shape
    ``(batch_size, ...)`` and lists/tuples of length ``batch_size`` -- are
    sliced per sample and hashed with :func:`hash_sample`, so
    ``hash_batch(batch)[i] == hash_sample({k: v[i] for k, v in batch.items()})``
    over those fields.

    **Broadcast fields are skipped**: a tensor whose leading dimension is not
    ``batch_size`` (e.g. ``position_ids`` of shape ``(1, T)``, shared across
    the batch) carries no per-sample identity -- and may even be
    batch-dependent -- so it does not enter the hash.  Lookups must likewise
    hash only per-sample fields (a raw ``dataset[i]`` naturally does).

    Args:
        inputs: The batched model-input dict.
        batch_size: The expected batch size.  ``None`` infers it as the
            **largest** leading dimension over the tensor fields (robust to a
            broadcast field appearing first).

    Returns:
        List of ``batch_size`` hashes, one per sample, in batch order.

    Raises:
        ValueError: If the batch size cannot be inferred, or no batch-first
            field remains after skipping broadcast fields (there would be
            nothing identifying the samples).
    """
    tensors = {
        k: v for k, v in inputs.items() if isinstance(v, torch.Tensor) and v.ndim > 0
    }
    if batch_size is None:
        batch_size = max((v.shape[0] for v in tensors.values()), default=0)
        if batch_size == 0:
            raise ValueError(
                "hash_batch could not infer a batch size: no non-scalar "
                f"tensor field among keys {sorted(inputs)}.",
            )
    sliceable: dict[str, object] = {
        k: v for k, v in tensors.items() if v.shape[0] == batch_size
    }
    sliceable.update(
        {
            k: v
            for k, v in inputs.items()
            if isinstance(v, (list, tuple)) and len(v) == batch_size
        },
    )
    if not sliceable:
        raise ValueError(
            f"hash_batch found no batch-first field (batch size {batch_size}) "
            f"among keys {sorted(inputs)}; every field is broadcast or scalar, "
            "so there is nothing identifying the individual samples.",
        )
    return [
        hash_sample({k: v[i] for k, v in sliceable.items()}) for i in range(batch_size)
    ]
