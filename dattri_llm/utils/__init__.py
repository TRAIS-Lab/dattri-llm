"""Common utilities shared across the library.

* :mod:`~dattri_llm.utils.hashing` -- content hashing of model inputs
  (``hash_sample`` / ``hash_batch``), the sample identity used by capture
  and retrieval.
* :mod:`~dattri_llm.utils.distributed` -- guarded ``torch.distributed``
  helpers, safe to call outside a distributed context.
* :mod:`~dattri_llm.utils.autograd` -- guarded autograd-engine helpers for
  scheduling work at the end of the in-flight backward pass.
"""

from dattri_llm.utils.autograd import (
    queue_after_backward_finalization,
    queue_backward_end_callback,
)
from dattri_llm.utils.distributed import (
    dist_rank,
    dist_world_size,
    is_dist_initialized,
)
from dattri_llm.utils.hashing import hash_batch, hash_sample

__all__ = [
    "dist_rank",
    "dist_world_size",
    "hash_batch",
    "hash_sample",
    "is_dist_initialized",
    "queue_after_backward_finalization",
    "queue_backward_end_callback",
]
