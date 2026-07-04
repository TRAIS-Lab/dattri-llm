"""Common utilities shared across the library.

* :mod:`~dattri_llm.utils.hashing` — content hashing of model inputs
  (``hash_sample`` / ``hash_batch``), the sample identity used by capture
  and retrieval.
* :mod:`~dattri_llm.utils.file_manager` — on-disk storage and retrieval of
  gradient records (:class:`GradientFileManager`).
* :mod:`~dattri_llm.utils.distributed` — guarded ``torch.distributed``
  helpers, safe to call outside a distributed context.
"""

from dattri_llm.utils.distributed import (
    dist_rank,
    dist_world_size,
    is_dist_initialized,
)
from dattri_llm.utils.file_manager import GradientFileManager
from dattri_llm.utils.hashing import hash_batch, hash_sample

__all__ = [
    "GradientFileManager",
    "dist_rank",
    "dist_world_size",
    "hash_batch",
    "hash_sample",
    "is_dist_initialized",
]
