"""This example shows gradient collection around the HuggingFace Trainer.

The training loop is NOT modified — TDA is added by wrapping trainer.train()
in a HookManager collection context.  Two equivalent integration patterns:

  Pattern A — with-context: wrap trainer.train() directly (preferred).
  Pattern B — TrainerCallback: open/close the context from on_train_begin /
      on_train_end, for pipelines where trainer.train() is called deep inside
      a library you do not control.

Note: this file must not be named ``transformers.py`` — it would shadow the
HuggingFace package on import.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.utils.hashing import hash_sample

MODEL_ID = "sshleifer/tiny-gpt2"   # 2-layer GPT-2, runs on CPU
MAX_LENGTH = 32
SENTENCES = [
    "The cat sat on the mat.",
    "A quick brown fox jumps over the lazy dog.",
    "Training data attribution identifies influential samples.",
    "Gradient hooks capture per-sample signals efficiently.",
]


class TextDataset(TorchDataset):
    def __init__(self, tokenizer, texts: list[str]) -> None:
        enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                        padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx].clone(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2,
                        help="Two epochs so each sample is recorded twice.")
    args_cli = parser.parse_args()

    # load the tiny model and build the dataset
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(tokenizer, SENTENCES)

    # Hook the transformer blocks.  GPT-2's positional embedding (wpe) is fed an
    # unbatched position tensor, so its gradient is broadcast over the batch and
    # is NOT per-sample — exclude it from per-sample collection.
    hook_cfg = HookManagerConfig(linear_io=[r"transformer\.h\.", r"wte", r"lm_head"])

    results = {}
    for pattern in ("A (with-context)", "B (TrainerCallback)"):
        # a fresh model per pattern so both start from the same weights
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=args_cli.epochs,
                per_device_train_batch_size=4,
                use_cpu=True,
                logging_steps=100,
                report_to="none",
            )
            fm = GradientFileManager(os.path.join(tmpdir, "gradients"))
            collector = HookManager(
                model, config=hook_cfg,
                callbacks=[OffloadCallback(offload_interval=1, file_manager=fm,
                                           recording_type="per_batch")],
            )

            print(f"\nRunning trainer.train() — pattern {pattern} ...")
            if pattern.startswith("A"):
                # Pattern A — wrap trainer.train() in the collection context.
                trainer = Trainer(model=model, args=training_args,
                                  train_dataset=dataset)
                with collector.collect():
                    trainer.train()
            else:
                # Pattern B — open/close the context via Trainer callbacks.
                class CollectCallback(TrainerCallback):
                    def on_train_begin(self, args, state, control, **kwargs):
                        self._ctx = collector.collect()
                        self._ctx.__enter__()

                    def on_train_end(self, args, state, control, **kwargs):
                        self._ctx.__exit__(None, None, None)

                trainer = Trainer(model=model, args=training_args,
                                  train_dataset=dataset,
                                  callbacks=[CollectCallback()])
                trainer.train()
            collector.remove()

            # Retrieval is a two-step scheme.  The content hash of ONE sample's
            # model inputs — here simply dataset[0], since the Trainer passes
            # {input_ids, attention_mask, labels} through unchanged — identifies
            # WHAT the sample is, independent of shuffling.  lookup() then
            # reveals WHERE it was recorded: every (step, sample_idx)
            # pair, and load_sample() retrieves each pair's gradient by a
            # direct slice of the stored record (no scan over the batch).
            h0 = hash_sample(dataset[0])
            pairs = fm.lookup_by_hash(h0)                      # [(step, sample_idx), ...]
            g_first = fm.load_sample_by_hash(h0, *pairs[0])
            g_last = fm.load_sample_by_hash(h0, *pairs[-1])
            drift = g_first.similarity(g_last, metric="cosine", reduce="all").item()
            results[pattern] = dict(
                steps=collector.steps_collected,
                records=len(fm.index),
                pairs=pairs,
                drift=drift,
            )

    # both patterns must produce identical collection statistics; the Trainer
    # shuffles each epoch, so sample 0's (step, sample_idx) pairs show it landing
    # at different batch positions
    print(f"\n{'Pattern':<22}{'Steps':>7}{'Records':>9}"
          f"{'Sample-0 (step, sample_idx)':>30}{'cos(first, last)':>18}")
    print("-" * 86)
    for pattern, r in results.items():
        print(f"{pattern:<22}{r['steps']:>7}{r['records']:>9}"
              f"{str(r['pairs']):>30}{r['drift']:>18.6f}")
    print("-" * 86)
