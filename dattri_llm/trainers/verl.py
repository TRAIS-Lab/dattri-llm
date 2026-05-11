"""Utility functions and constants for verl actor gradient collection.

These are trainer-agnostic helpers; gradient collection itself is done via
:class:`~dattri_llm.gradient.hooks.HookManager`.

Usage::

    from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, OffloadCallback
    from dattri_llm.gradient.file_manager import GradientFileManager
    from dattri_llm.trainers.verl import VERL_GPT2_MLP_PATTERNS, compute_response_nll_loss

    manager   = GradientFileManager(grad_dir)
    offload   = OffloadCallback(every_n_steps=32, file_manager=manager)
    collector = HookManager(
        actor_model,
        config=HookManagerConfig(mlp_name_patterns=VERL_GPT2_MLP_PATTERNS),
        callbacks=[offload],
    )

    with collector.collect():
        for batch in rollout_batches:
            outputs = actor_model(input_ids=batch.input_ids)
            loss = compute_response_nll_loss(
                outputs.logits, batch.input_ids, batch.response_mask.float()
            )
            loss.backward()
            actor_model.zero_grad()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MLP layer name patterns
# ---------------------------------------------------------------------------

# GPT-2's MLP uses Conv1D (not nn.Linear) inside layers named "mlp.c_fc" /
# "mlp.c_proj" — these match the default heuristic — but when loaded via
# HuggingFace the parent is confirmed as "mlp", so the default works.
# These explicit patterns are provided for documentation and for users who
# prefer not to rely on the heuristic.
VERL_GPT2_MLP_PATTERNS: list[str] = [
    r"transformer\.h\.\d+\.mlp\.c_fc",
    r"transformer\.h\.\d+\.mlp\.c_proj",
]

# LLaMA / Mistral MLP layers.
VERL_LLAMA_MLP_PATTERNS: list[str] = [
    r"model\.layers\.\d+\.mlp\.gate_proj",
    r"model\.layers\.\d+\.mlp\.up_proj",
    r"model\.layers\.\d+\.mlp\.down_proj",
]


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def compute_response_nll_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute negative log-likelihood loss restricted to response tokens.

    verl batches concatenate prompt and response tokens.  This function masks
    out the prompt portion so that only response tokens contribute to the loss
    — and therefore to the gradients recorded by HookManager.

    Args:
        logits: Model output logits of shape ``(B, T, vocab_size)``.
        input_ids: Token IDs of shape ``(B, T)``.
        response_mask: Float mask of shape ``(B, T)`` — 1 for response tokens,
            0 for prompt tokens.

    Returns:
        Scalar mean NLL loss over response tokens.
    """
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask   = response_mask[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    masked = (loss_per_token * shift_mask).sum()
    denom  = shift_mask.sum().clamp(min=1.0)
    return masked / denom
