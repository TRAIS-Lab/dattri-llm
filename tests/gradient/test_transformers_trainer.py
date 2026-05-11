"""Unit tests for GradientCollectingTrainer (HuggingFace Transformers integration).

These tests use a tiny GPT-2-config model so they run entirely on CPU in CI.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

try:
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    from transformers import set_seed
    from dattri_llm.trainers.transformers import GradientCollectingTrainer
    # Verify that accelerate is available by exercising TrainingArguments device setup
    import accelerate  # noqa: F401
    _HAS_TRANSFORMERS_STACK = True
except (ImportError, Exception):
    _HAS_TRANSFORMERS_STACK = False

pytestmark = pytest.mark.skipif(
    not _HAS_TRANSFORMERS_STACK,
    reason="transformers[torch] / accelerate>=1.1.0 not installed",
)


# --------------------------------------------------------------------------- #
# Shared fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def tiny_gpt2():
    """A minimal 2-layer GPT-2 that fits on CPU in seconds."""
    set_seed(42)
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    return GPT2LMHeadModel(cfg)


@pytest.fixture()
def tiny_dataset():
    """An iterable of 4 tiny token sequences (B=2 steps × 2 samples)."""
    from torch.utils.data import Dataset

    class TinyTokenDataset(Dataset):
        def __init__(self, n: int = 4, seq_len: int = 8, vocab_size: int = 64):
            torch.manual_seed(0)
            self.data = torch.randint(0, vocab_size, (n, seq_len))

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            ids = self.data[idx]
            return {"input_ids": ids, "labels": ids.clone()}

    return TinyTokenDataset()


@pytest.fixture()
def training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path / "output"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=2,
        use_cpu=True,
        logging_steps=1,
        save_steps=100,  # no checkpoint saves
        report_to="none",
    )


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


class TestGradientCollectingTrainerMlpIo:
    def test_gradient_files_created(self, tiny_gpt2, tiny_dataset, training_args):
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="mlp_io",
        )
        trainer.train()

        grad_dir = os.path.join(training_args.output_dir, "gradients")
        assert os.path.isdir(grad_dir), "gradients/ directory was not created"
        pt_files = [f for f in os.listdir(grad_dir) if f.endswith(".pt")]
        assert len(pt_files) > 0, "No .pt gradient files written"

    def test_gradient_file_contains_tensors(self, tiny_gpt2, tiny_dataset, training_args):
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="mlp_io",
        )
        trainer.train()

        grad_dir = os.path.join(training_args.output_dir, "gradients")
        pt_file = sorted(os.listdir(grad_dir))[0]
        data = torch.load(os.path.join(grad_dir, pt_file), weights_only=True)

        assert isinstance(data, dict)
        assert len(data) > 0
        for name, tensor in data.items():
            assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
            assert torch.isfinite(tensor).all(), f"{name} has non-finite values"

    def test_per_sample_gradient_batch_dim(self, tiny_gpt2, tiny_dataset, training_args):
        """Verify the first dim of each tensor equals per_device_train_batch_size."""
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="mlp_io",
        )
        trainer.train()

        grad_dir = os.path.join(training_args.output_dir, "gradients")
        pt_file = sorted(os.listdir(grad_dir))[0]
        data = torch.load(os.path.join(grad_dir, pt_file), weights_only=True)

        B = training_args.per_device_train_batch_size
        for name, tensor in data.items():
            assert tensor.shape[0] == B, (
                f"{name}: expected batch dim {B}, got {tensor.shape[0]}"
            )

    def test_custom_save_dir(self, tiny_gpt2, tiny_dataset, training_args, tmp_path):
        custom_dir = str(tmp_path / "my_gradients")
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="mlp_io",
            gradient_save_dir=custom_dir,
        )
        trainer.train()
        assert os.path.isdir(custom_dir)
        assert len(os.listdir(custom_dir)) > 0


class TestGradientCollectingTrainerFull:
    def test_full_mode_gradient_files(self, tiny_gpt2, tiny_dataset, training_args):
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="full",
        )
        trainer.train()

        grad_dir = os.path.join(training_args.output_dir, "gradients")
        assert os.path.isdir(grad_dir)
        pt_files = [f for f in os.listdir(grad_dir) if f.endswith(".pt")]
        assert len(pt_files) > 0

    def test_full_mode_contains_param_names(self, tiny_gpt2, tiny_dataset, training_args):
        trainer = GradientCollectingTrainer(
            model=tiny_gpt2,
            args=training_args,
            train_dataset=tiny_dataset,
            gradient_collection_mode="full",
        )
        trainer.train()

        grad_dir = os.path.join(training_args.output_dir, "gradients")
        pt_file = sorted(os.listdir(grad_dir))[0]
        data = torch.load(os.path.join(grad_dir, pt_file), weights_only=True)

        # Keys should look like parameter names (contain dots)
        assert any("." in k for k in data.keys()), (
            "Expected dot-separated parameter names in full-mode output"
        )
