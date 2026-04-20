"""OLMo-specific helpers for GradientCollector.

OLMo's block-level feed-forward linears are named ``ff_proj`` and ``ff_out``
under ``transformer.blocks.<i>``. Those names do not match the default MLP
heuristics in :mod:`dattri_llm.gradient.hooks`, so OLMo models must use
explicit regex patterns when registering hooks.
"""

from __future__ import annotations

import torch.nn as nn

try:
    import olmo  # noqa: F401 - confirms ai2-olmo is installed
except ImportError as exc:
    raise ImportError(
        "OLMo is required for this module. Install with: pip install ai2-olmo"
    ) from exc

from dattri_llm.gradient.collector import GradientCollector

# Regex patterns targeting OLMo's block-level MLP linears.
OLMO_MLP_PATTERNS: list[str] = [
    r"transformer\.blocks\.\d+\.ff_proj",
    r"transformer\.blocks\.\d+\.ff_out",
]


def make_olmo_collector(model: nn.Module) -> GradientCollector:
    """Return a GradientCollector pre-configured for OLMo MLP layers."""
    return GradientCollector(model, name_patterns=OLMO_MLP_PATTERNS)
