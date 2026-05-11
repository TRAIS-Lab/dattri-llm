"""Utility constants for OLMo gradient collection.

OLMo's feed-forward projections (``ff_proj`` / ``ff_out``) do not match the
default MLP-keyword heuristic used by
:class:`~dattri_llm.gradient.hooks.HookManager`.  Pass
:data:`OLMO_MLP_PATTERNS` to :class:`~dattri_llm.gradient.hooks.HookManagerConfig`
to target the correct layers.

Usage::

    from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, OffloadCallback
    from dattri_llm.gradient.file_manager import GradientFileManager
    from dattri_llm.trainers.olmo import OLMO_MLP_PATTERNS

    hook_cfg  = HookManagerConfig(mlp_name_patterns=OLMO_MLP_PATTERNS)
    manager   = GradientFileManager(grad_dir)
    offload   = OffloadCallback(every_n_steps=1, file_manager=manager)
    collector = HookManager(olmo_model, config=hook_cfg, callbacks=[offload])

    with collector.collect():
        for batch in dataloader:
            loss = olmo_model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
"""

from __future__ import annotations

try:
    from olmo.model import OLMo as _OLMo  # noqa: F401
    _OLMO_AVAILABLE = True
except ImportError:
    _OLMO_AVAILABLE = False

OLMO_MLP_PATTERNS: list[str] = [
    r"transformer\.blocks\.\d+\.ff_proj",
    r"transformer\.blocks\.\d+\.ff_out",
]
"""Regex patterns that match OLMo's MLP linear projections.

Pass as ``mlp_name_patterns`` in :class:`~dattri_llm.gradient.hooks.HookManagerConfig`.
"""
