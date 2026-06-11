from __future__ import annotations

import hashlib

import torch


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
