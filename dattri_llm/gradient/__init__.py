"""Gradient collection utilities for training data attribution."""

from dattri_llm.gradient.collector import GradientCollector
from dattri_llm.gradient.hooks import register_mlp_hooks, remove_hooks

__all__ = ["GradientCollector", "register_mlp_hooks", "remove_hooks"]
