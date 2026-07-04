from __future__ import annotations

import hashlib

import torch


def hash_sample(inputs: dict[str, torch.Tensor]) -> str:
    """SHA-256 content hash identifying **one sample** by its model inputs.

    *inputs* holds a single sample's tensors -- e.g. what ``dataset[i]`` yields,
    or one row sliced out of a batch.  All tensor fields are digested, sorted
    by key for determinism; non-tensor values are skipped.  The digest is over
    raw bytes, so the same sample hashes identically wherever shuffling put
    it: a ``(T,)`` row, its ``(1, T)`` batched form, and ``batch[i]`` all
    produce the same hash.

    Args:
        inputs: One sample's model-input dict (e.g.
            ``{"input_ids": ..., "attention_mask": ..., "labels": ...}``).

    Returns:
        A 64-character hex string (SHA-256 digest).
    """
    h = hashlib.sha256()
    for key in sorted(inputs):
        val = inputs[key]
        if isinstance(val, torch.Tensor):
            h.update(key.encode())
            h.update(val.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hash_batch(
    inputs: dict[str, torch.Tensor],
    batch_size: int | None = None,
) -> list[str]:
    """Per-sample content hashes for a **batched** input dict, in batch order.

    The first dimension of every tensor field is taken as the batch dimension;
    each sample's slice is hashed with :func:`hash_sample`.  This is what the
    capture side records at every step, so ``hash_batch(batch)[i] ==
    hash_sample({k: v[i] for k, v in batch.items()})``.

    Args:
        inputs: The batched model-input dict.
        batch_size: The expected batch size.  ``None`` infers it from the first
            tensor field's leading dimension.

    Returns:
        List of ``batch_size`` hashes, one per sample, in batch order.

    Raises:
        NotImplementedError: If a tensor field is not batch-first (its leading
            dimension disagrees with the batch size), e.g. sequence-first
            ``(T, B)`` inputs.
    """
    tensors = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor) and v.ndim > 0
    }
    if batch_size is None:
        batch_size = next((v.shape[0] for v in tensors.values()), 1)
    mismatched = {k: tuple(v.shape) for k, v in tensors.items()
                  if v.shape[0] != batch_size}
    if mismatched:
        raise NotImplementedError(
            f"hash_batch assumes every tensor field is batch-first with batch "
            f"size {batch_size}; non-conforming fields: {mismatched}. "
            "Sequence-first or broadcast model inputs are not supported yet."
        )
    return [
        hash_sample({k: v[i] for k, v in tensors.items()})
        for i in range(batch_size)
    ]
