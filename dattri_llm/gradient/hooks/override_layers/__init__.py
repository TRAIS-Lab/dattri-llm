"""Forward overrides installed on a model while a ``HookManager`` collects.

Some layers cannot be captured by module hooks alone, or do work the capture
does not need.  Each module here replaces a layer's ``forward`` for the
duration of collection and restores it afterwards, through an
``install_*`` function that returns revert closures and takes a
``should_intervene`` predicate (the manager's collecting flag), so the
model trains unchanged outside ``collect()``:

* :mod:`~dattri_llm.gradient.hooks.override_layers.invasive_linear` -- an
  ``nn.Linear`` forward whose backward skips the weight-gradient matmul.
* :mod:`~dattri_llm.gradient.hooks.override_layers.fused_experts` -- per-expert
  proxies for Mixture-of-Experts modules that store their experts as 3-D
  parameters and have no ``nn.Linear`` to hook.

New layer kinds that need an override go here as further modules.
"""

from dattri_llm.gradient.hooks.override_layers.fused_experts import (
    RouterLinear,
    SliceLinear,
    find_fused_experts,
    install_fused_experts,
    is_fused_experts,
)
from dattri_llm.gradient.hooks.override_layers.invasive_linear import (
    InvasiveLinearFunction,
    install_invasive_forward,
    is_invasive_capable,
)

__all__ = [
    "InvasiveLinearFunction",
    "RouterLinear",
    "SliceLinear",
    "find_fused_experts",
    "install_fused_experts",
    "install_invasive_forward",
    "is_fused_experts",
    "is_invasive_capable",
]
