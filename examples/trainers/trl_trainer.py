"""This example shows gradient collection around TRL's SFTTrainer.

The training loop is NOT modified -- TDA is added by wrapping trainer.train()
in a HookManager collection context, exactly as with the plain HF Trainer.
TRL is handed a raw-text dataset and does its own tokenization, collation, and
label masking internally; the hooks sit below all of that, so no cooperation
from the SFT pipeline is required.

Because TRL transforms the raw texts before the model sees them, samples are
identified by the content hash of the model inputs TRL actually produced --
the on-disk index maps each hash to every (step, sample_idx) it was recorded
at, so no re-implementation of TRL's preprocessing is needed for retrieval.

Run (with trl installed):
    python examples/trainers/trl_trainer.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    raise SystemExit(
        "trl (and datasets) are required for this example.\n"
        "Install with:  pip install trl",
    ) from exc

from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.storage_manager import GradientStorageManager

MODEL_ID = "sshleifer/tiny-gpt2"  # 2-layer GPT-2, runs on CPU
SENTENCES = [
    "The cat sat on the mat.",
    "A quick brown fox jumps over the lazy dog.",
    "Training data attribution identifies influential samples.",
    "Gradient hooks capture per-sample signals efficiently.",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Two epochs so each sample is recorded twice.",
    )
    args_cli = parser.parse_args()

    # load the tiny model; TRL receives the raw texts and tokenizes internally
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    dataset = Dataset.from_dict({"text": SENTENCES})

    # Hook the transformer blocks.  Two GPT-2 layers are excluded on purpose:
    # wpe (positional embedding) receives an unbatched position tensor, so its
    # gradient is broadcast over the batch and is NOT per-sample; lm_head is
    # never invoked as a module by TRL's SFT loss (it computes the LM loss
    # through a fused/functional linear on the tied weight), so its hooks
    # would never fire and step completion would stall waiting for them.
    hook_cfg = HookManagerConfig(linear_io=[r"transformer\.h\.", r"wte"])

    with tempfile.TemporaryDirectory() as tmpdir:
        # A sample's content hash covers the model inputs TRL feeds the
        # forward pass, and two of those inputs depend on which samples share
        # the batch: the pad-to-longest sequence length and the
        # num_items_in_batch loss-normalization count.  Train the full set as
        # ONE fixed-length batch so both are constant and each sample's hash
        # is identical across epochs.
        sft_config = SFTConfig(
            output_dir=tmpdir,
            num_train_epochs=args_cli.epochs,
            per_device_train_batch_size=len(SENTENCES),
            max_length=32,
            pad_to_multiple_of=32,
            # TRL (unlike plain TrainingArguments) turns gradient checkpointing
            # ON by default; its non-reentrant mode runs every checkpointed
            # block forward twice with grad enabled (build + recompute), which
            # would double-capture activations.  Disable it for collection.
            gradient_checkpointing=False,
            use_cpu=True,
            logging_steps=100,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        fm = GradientStorageManager(str(pathlib.Path(tmpdir) / "gradients"))
        collector = HookManager(
            model,
            config=hook_cfg,
            callbacks=[
                OffloadCallback(
                    offload_interval=1,
                    file_manager=fm,
                    recording_type="per_batch",
                ),
            ],
        )

        # the SFT loop itself is untouched: wrap trainer.train() and every
        # backward pass inside is captured automatically
        print("\nRunning SFTTrainer.train() with gradient collection ...")
        with collector.collect():
            trainer.train()
        collector.remove()

        # Retrieval: TRL shuffles each epoch, so the same sample lands at
        # different (step, sample_idx) positions; its content hash ties the
        # occurrences together.  Pick any sample recorded at >= 2 steps and
        # measure how far its gradient drifted over training.
        h0 = next(h for h in fm.index if len(fm.lookup_by_hash(h)) >= 2)
        pairs = fm.lookup_by_hash(h0)  # [(step, sample_idx), ...]
        g_first = fm.load_sample_by_hash(h0, *pairs[0])
        g_last = fm.load_sample_by_hash(h0, *pairs[-1])
        drift = g_first.similarity(g_last, metric="cosine", reduce="all").item()

        print(f"\n{'Steps collected':<30}{collector.steps_collected}")
        print(f"{'Sample records':<30}{len(fm.index)}")
        print("-" * 50)
        print(f"{'Sample (step, sample_idx)':<30}{pairs}")
        print(f"{'cos(first, last)':<30}{drift:.6f}")
        print("-" * 50)
