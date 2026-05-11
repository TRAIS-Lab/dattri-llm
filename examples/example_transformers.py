"""Gradient collection via HookManager with HuggingFace Transformers Trainer.

Two integration patterns are shown:

VERSION 1 — TrainerCallback
    Open and close collector.collect() via on_train_begin / on_train_end hooks.
    Useful when you cannot modify the call to trainer.train().

VERSION 2 — collect() context (simpler)
    Wrap trainer.train() directly inside a with collector.collect(): block.
    Trainer.training_step() calls forward + backward inside the same Python
    call stack, so HookManager sees every backward pass automatically.

Both patterns produce identical records.  Version 2 is preferred for new code;
Version 1 is useful in pipelines where trainer.train() is called deep inside a
library you do not control.

Run:
    python examples/example_transformers.py
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import hash_sample
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, OffloadCallback

# ============================================================================
# 1. Model + tokenizer
# ============================================================================

MODEL_ID = "sshleifer/tiny-gpt2"   # 2-layer GPT-2, runs on CPU

print(f"Loading {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# ============================================================================
# 2. Tiny dataset
# ============================================================================

SENTENCES = [
    "The cat sat on the mat.",
    "A quick brown fox jumps over the lazy dog.",
    "Training data attribution identifies influential samples.",
    "Gradient hooks capture per-sample signals efficiently.",
]
MAX_LENGTH = 32


class TextDataset(TorchDataset):
    def __init__(self, texts: list[str]) -> None:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.input_ids[idx].clone(),
        }


dataset = TextDataset(SENTENCES)

args = TrainingArguments(
    output_dir="__placeholder__",   # overridden inside the tempdir below
    num_train_epochs=2,             # two epochs so each sample appears twice
    per_device_train_batch_size=4,
    use_cpu=True,
    logging_steps=1,
    report_to="none",
)

# The Trainer passes {input_ids, attention_mask, labels} to the model, so all
# three are included in the recorded hash.  The query must include the same keys.
query_inputs = {
    "input_ids":      dataset.input_ids,
    "attention_mask": dataset.attention_mask,
    "labels":         dataset.input_ids.clone(),
}

hook_cfg = HookManagerConfig(recording_type="per_sample", hook_types=["mlp_io"])

# ============================================================================
# VERSION 1 — TrainerCallback
# ============================================================================

print("\n" + "=" * 60)
print("VERSION 1 — TrainerCallback")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    args.output_dir = tmpdir
    grad_dir = os.path.join(tmpdir, "gradients")

    manager1  = GradientFileManager(grad_dir)
    offload1  = OffloadCallback(every_n_steps=1, file_manager=manager1)
    collector1 = HookManager(model, config=hook_cfg, callbacks=[offload1])

    print(f"Hooked {len(collector1.layer_names)} MLP layer(s):")
    for name in collector1.layer_names:
        print(f"  {name}")

    class _CollectCtx(TrainerCallback):
        """Opens collect() before the first step, closes it after the last."""
        def on_train_begin(self, args, state, control, **kwargs):
            self._ctx = collector1.collect()
            self._ctx.__enter__()

        def on_train_end(self, args, state, control, **kwargs):
            self._ctx.__exit__(None, None, None)

    trainer1 = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        callbacks=[_CollectCtx()],
    )
    print("\nRunning trainer.train() (Version 1 — callback) …")
    trainer1.train()
    print(f"Steps collected : {collector1.steps_collected}")
    collector1.remove()

    records1 = manager1.load_all(query_inputs)
    print(f"Records loaded  : {len(records1)}")

    h0 = hash_sample(query_inputs, sample_idx=0)
    recs_s0 = manager1.load_all_by_hash(h0)
    print(f"Sample 0 across epochs : {len(recs_s0)} record(s)")
    if len(recs_s0) >= 2:
        g0 = recs_s0[0].gradient.materialize().aggregate(dim="token", mode="sum")
        g1 = recs_s0[1].gradient.materialize().aggregate(dim="token", mode="sum")
        sim = g0.similarity(g1, metric="cosine", reduce="layer")
        print("Cosine similarity sample 0 — epoch 1 vs epoch 2:")
        for layer, val in sim.items():
            print(f"  {layer}  cos_sim={val.item():.6f}")

# ============================================================================
# VERSION 2 — collect() context (simpler)
#
# collector.collect() can wrap trainer.train() directly.  The Trainer's
# training_step() runs forward+backward in the same thread, so every backward
# pass fires inside the context and is captured automatically.
# ============================================================================

print("\n" + "=" * 60)
print("VERSION 2 — collect() context")
print("=" * 60)

# Re-create the model so both versions start from the same weights.
model2 = AutoModelForCausalLM.from_pretrained(MODEL_ID)

with tempfile.TemporaryDirectory() as tmpdir:
    args.output_dir = tmpdir
    grad_dir = os.path.join(tmpdir, "gradients")

    manager2  = GradientFileManager(grad_dir)
    offload2  = OffloadCallback(every_n_steps=1, file_manager=manager2)
    collector2 = HookManager(model2, config=hook_cfg, callbacks=[offload2])

    trainer2 = Trainer(
        model=model2,
        args=args,
        train_dataset=dataset,
    )

    print("\nRunning trainer.train() (Version 2 — collect() context) …")
    with collector2.collect():
        trainer2.train()
    print(f"Steps collected : {collector2.steps_collected}")
    collector2.remove()

    records2 = manager2.load_all(query_inputs)
    print(f"Records loaded  : {len(records2)}")

    h0 = hash_sample(query_inputs, sample_idx=0)
    recs_s0 = manager2.load_all_by_hash(h0)
    print(f"Sample 0 across epochs : {len(recs_s0)} record(s)")
    if len(recs_s0) >= 2:
        g0 = recs_s0[0].gradient.materialize().aggregate(dim="token", mode="sum")
        g1 = recs_s0[1].gradient.materialize().aggregate(dim="token", mode="sum")
        sim = g0.similarity(g1, metric="cosine", reduce="layer")
        print("Cosine similarity sample 0 — epoch 1 vs epoch 2:")
        for layer, val in sim.items():
            print(f"  {layer}  cos_sim={val.item():.6f}")
