"""Shared task definition for the ours-vs-kronfluence pair (their example's
``LanguageModelingTask``, verbatim), plus the pair's size constants."""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

N_QUERY = 32       # first 32 validation blocks = one documented query batch
QUERY_BATCH = 32   # their README setting
# Their README uses 64 on an A100-80GB; 32 fits the A40.  Overridable for
# batch-matched A/B runs (e.g. PAIR_TRAIN_BATCH=16 to match ours).
TRAIN_BATCH = int(os.environ.get("PAIR_TRAIN_BATCH", "32"))


class LanguageModelingTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")
        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]
