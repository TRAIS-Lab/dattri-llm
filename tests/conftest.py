"""Shared pytest fixtures for dattri-llm tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn


class TinyMLP(nn.Module):
    """A minimal 2-layer transformer-style model with explicit MLP blocks.

    Used as a lightweight stand-in for a real LLM during tests.  Structure::

        embedding -> transformer_block (attn_proj, mlp.fc1, mlp.fc2) -> lm_head
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 16,
        seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Simulated attention projection (NOT in mlp -> should NOT be hooked)
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        # MLP block: named so that _is_mlp_linear matches it
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),  # mlp.0
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False),  # mlp.2
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (B, T, d_model)
        x = self.attn_proj(x)
        x = self.mlp(x)
        return self.lm_head(x)  # (B, T, vocab_size)


@pytest.fixture
def tiny_model() -> TinyMLP:
    """Return a fresh TinyMLP on CPU with fixed seed."""
    torch.manual_seed(42)
    return TinyMLP()


@pytest.fixture
def tiny_batch(tiny_model: TinyMLP) -> dict[str, torch.Tensor]:
    """Return a small random input batch (B=2, T=8)."""
    B, T = 2, tiny_model.seq_len
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_model.vocab_size, (B, T))
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}
