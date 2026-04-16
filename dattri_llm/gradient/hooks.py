"""Core PyTorch hook utilities for gradient collection.

This module provides functions to register forward and backward hooks
on MLP (feed-forward) linear layers of transformer-based LLMs. The hooks
capture input activations and output gradients needed to reconstruct
per-sample gradients via the outer-product identity:

    dL/dW ≈ g^T ⊗ a    (per sample)

where `a` is the layer input activation and `g` is the gradient of the
loss w.r.t. the layer output.

DataParallel support
--------------------
When a model is wrapped in ``nn.DataParallel``, PyTorch runs each replica
in its own thread (one per device).  Each thread fires the forward and
backward hooks independently, so a naïve single-slot buffer would be
overwritten by whichever replica finishes last.

To handle this correctly the hooks here *accumulate* all replica calls into
lists (``_act_parts`` / ``_grad_parts``), tagging each entry with the source
device index so the parts can be reassembled in scatter order before the
outer-product is computed.

Single-GPU and DDP usage is unaffected: there is exactly one forward and one
backward call per step, so the lists always contain a single element.
"""

from __future__ import annotations

import re
import threading

import torch
import torch.nn as nn

# HuggingFace's Conv1D behaves like a transposed nn.Linear — support it too.
try:
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
except ImportError:
    HF_Conv1D = None  # type: ignore[assignment,misc]

# Module types that behave as a linear projection (weight matrix + optional bias).
_LINEAR_TYPES: tuple[type, ...] = (nn.Linear,) + (
    (HF_Conv1D,) if HF_Conv1D is not None else ()
)

# Names that indicate a module is part of an MLP / feed-forward block.
_MLP_PARENT_PATTERNS = re.compile(
    r"(mlp|ffn|feed_forward|feedforward|fc|dense)",
    re.IGNORECASE,
)


def _device_sort_key(item: tuple[int, torch.Tensor]) -> int:
    return item[0]


def _is_mlp_linear(name: str, module: nn.Module) -> bool:
    """Return True if *module* is a linear projection inside an MLP block.

    Recognises both ``nn.Linear`` and HuggingFace's ``Conv1D`` (used by GPT-2
    and related models), which behaves as a transposed linear layer.

    Args:
        name: Fully-qualified module name (dot-separated path from the root).
        module: The module to test.

    Returns:
        True if the module is a linear projection whose parent path contains an
        MLP keyword, False otherwise.
    """
    if not isinstance(module, _LINEAR_TYPES):
        return False
    # Check whether any segment of the dotted path matches an MLP keyword.
    parts = name.split(".")
    # The *parent* segments are everything except the last token.
    parent_path = ".".join(parts[:-1])
    return bool(_MLP_PARENT_PATTERNS.search(parent_path))


# Buffer type alias for documentation purposes.
#
# Each layer entry:
#   "activation"  — the last-written activation tensor (single-GPU / DDP view)
#   "grad_output" — the last-written grad tensor (single-GPU / DDP view)
#   "_act_parts"  — list of (device_idx, cpu_tensor) from all replicas (DP)
#   "_grad_parts" — list of (device_idx, cpu_tensor) from all replicas (DP)
#   "_lock"       — threading.Lock guarding the lists under DataParallel
LayerBuffer = dict  # dict[str, Any]


def _make_layer_buffer() -> LayerBuffer:
    return {
        "activation": None,
        "grad_output": None,
        "_act_parts": [],
        "_grad_parts": [],
        "_lock": threading.Lock(),
    }


def register_mlp_hooks(
    model: nn.Module,
    name_patterns: list[str] | None = None,
) -> tuple[dict[str, LayerBuffer], list[torch.utils.hooks.RemovableHook]]:
    """Register forward and backward hooks on MLP linear layers.

    For each qualifying linear layer the function registers:

    * A **forward hook** that appends ``input[0]`` (moved to CPU) to
      ``buffers[layer_name]["_act_parts"]`` and also stores it in
      ``buffers[layer_name]["activation"]`` (last-write, backward-compat).
    * A **backward hook** that appends ``grad_output[0]`` (moved to CPU) to
      ``buffers[layer_name]["_grad_parts"]`` and also stores it in
      ``buffers[layer_name]["grad_output"]``.

    Both lists accumulate entries from all DataParallel replicas in a
    thread-safe manner; each entry is tagged with the source device index so
    they can be reassembled in scatter order.

    Supports ``DataParallel`` and ``DistributedDataParallel`` by traversing
    the underlying ``module`` attribute.

    Args:
        model: The PyTorch model (plain, DP-wrapped, or DDP-wrapped).
        name_patterns: Optional list of regex patterns.  When provided,
            *only* layers whose fully-qualified name matches at least one
            pattern are hooked.  When ``None``, the default MLP-keyword
            heuristic is used.

    Returns:
        A 2-tuple ``(buffers, handles)`` where:

        * ``buffers`` is a dict mapping layer name to a :data:`LayerBuffer`.
        * ``handles`` is a list of ``RemovableHook`` objects; pass them to
          :func:`remove_hooks` for clean teardown.
    """
    # Unwrap DataParallel / DistributedDataParallel transparently.
    root: nn.Module = getattr(model, "module", model)

    # Build compiled patterns if caller supplied custom regexes.
    compiled: list[re.Pattern[str]] | None = None
    if name_patterns is not None:
        compiled = [re.compile(p) for p in name_patterns]

    buffers: dict[str, LayerBuffer] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for name, module in root.named_modules():
        if compiled is not None:
            # Custom pattern mode: any linear whose name matches.
            if not isinstance(module, _LINEAR_TYPES):
                continue
            if not any(p.search(name) for p in compiled):
                continue
        else:
            if not _is_mlp_linear(name, module):
                continue

        buffers[name] = _make_layer_buffer()

        # ------------------------------------------------------------------ #
        # Closures capture `name` by value via default argument.              #
        # ------------------------------------------------------------------ #

        def _make_forward_hook(layer_name: str):
            def _forward_hook(
                module: nn.Module,
                inp: tuple[torch.Tensor, ...],
                out: torch.Tensor,
            ) -> None:
                t = inp[0].detach().cpu()
                dev_idx = inp[0].device.index if inp[0].is_cuda else 0
                buf = buffers[layer_name]
                with buf["_lock"]:
                    buf["_act_parts"].append((dev_idx, t))
                buf["activation"] = inp[0].detach()  # last-write (single-GPU view)

            return _forward_hook

        def _make_backward_hook(layer_name: str):
            def _backward_hook(
                module: nn.Module,
                grad_input: tuple[torch.Tensor | None, ...],
                grad_output: tuple[torch.Tensor, ...],
            ) -> None:
                t = grad_output[0].detach().cpu()
                dev_idx = grad_output[0].device.index if grad_output[0].is_cuda else 0
                buf = buffers[layer_name]
                with buf["_lock"]:
                    buf["_grad_parts"].append((dev_idx, t))
                buf["grad_output"] = grad_output[0].detach()  # last-write

            return _backward_hook

        handles.append(module.register_forward_hook(_make_forward_hook(name)))
        handles.append(module.register_full_backward_hook(_make_backward_hook(name)))

    return buffers, handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]) -> None:
    """Remove all registered hooks.

    Args:
        handles: List of hook handles returned by :func:`register_mlp_hooks`.
    """
    for h in handles:
        h.remove()
    handles.clear()
