"""Trainer-specific integrations for dattri-llm gradient collection."""

from dattri_llm.gradient.trainers.transformers import AttributionTrainerCallback

__all__ = ["AttributionTrainerCallback"]
