"""Unit tests for HookManager around the HuggingFace Transformers Trainer.

The training loop is NOT modified — gradient collection is added by wrapping
``trainer.train()`` in a :meth:`HookManager.collect` context (the integration
shown in ``examples/trainers/transformers_trainer.py``).  These tests use a
tiny GPT-2-config model so they run entirely on CPU in CI.
"""

from __future__ import annotations

import pytest
import torch

try:
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
    from transformers import set_seed
    import accelerate  # noqa: F401
    _HAS_TRANSFORMERS_STACK = True
except (ImportError, Exception):
    _HAS_TRANSFORMERS_STACK = False

pytestmark = pytest.mark.skipif(
    not _HAS_TRANSFORMERS_STACK,
    reason="transformers[torch] / accelerate not installed",
)

from dattri_llm.gradient.callbacks import HookManagerCallback, OffloadCallback  # noqa: E402
from dattri_llm.gradient.file_manager import GradientFileManager  # noqa: E402
from dattri_llm.gradient.gradient import Factorized, GradientRecord  # noqa: E402
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig  # noqa: E402

# Hook the decoder blocks only: wte/wpe see broadcast position ids (not
# per-sample) and lm_head shares wte's weight under GPT-2 weight tying.
HOOK_PATTERNS = [r"transformer\.h\."]

N_STEPS = 2
BATCH_SIZE = 2
N_SAMPLES = N_STEPS * BATCH_SIZE


class _Capture(HookManagerCallback):
    """Accumulates every GradientRecord emitted by HookManager."""

    def __init__(self) -> None:
        self.records: list[GradientRecord] = []

    def on_step_end(self, record: GradientRecord) -> None:
        self.records.append(record)


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
    """An iterable of 4 tiny token sequences (2 steps x 2 samples)."""
    from torch.utils.data import Dataset

    class TinyTokenDataset(Dataset):
        def __init__(self, n: int = N_SAMPLES, seq_len: int = 8, vocab_size: int = 64):
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
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=N_STEPS,
        use_cpu=True,
        logging_steps=1,
        save_steps=100,  # no checkpoint saves
        report_to="none",
    )


def _train_with_callbacks(model, training_args, dataset, callbacks) -> list[str]:
    """Run trainer.train() inside a collect() context; return the hooked names."""
    collector = HookManager(
        model,
        config=HookManagerConfig(linear_io=HOOK_PATTERNS),
        callbacks=callbacks,
    )
    layer_names = list(collector.layer_names)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    try:
        with collector.collect():
            trainer.train()
    finally:
        collector.remove()
    return layer_names


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


class TestTrainerCapture:
    """In-memory capture while the stock Trainer runs its unmodified loop."""

    def test_one_record_per_step(self, tiny_gpt2, tiny_dataset, training_args):
        capture = _Capture()
        layer_names = _train_with_callbacks(
            tiny_gpt2, training_args, tiny_dataset, [capture]
        )
        assert len(capture.records) == N_STEPS, (
            f"Expected {N_STEPS} records, got {len(capture.records)}"
        )
        hooked = set(layer_names)
        assert hooked, "No layers were hooked"
        for rec in capture.records:
            assert set(rec.gradient.data) == hooked

    def test_per_sample_batch_dim(self, tiny_gpt2, tiny_dataset, training_args):
        capture = _Capture()
        _train_with_callbacks(tiny_gpt2, training_args, tiny_dataset, [capture])
        for rec in capture.records:
            assert rec.gradient.batch_size == BATCH_SIZE
            assert isinstance(rec.input_hash, list)
            assert len(rec.input_hash) == BATCH_SIZE

    def test_factorized_and_finite(self, tiny_gpt2, tiny_dataset, training_args):
        capture = _Capture()
        _train_with_callbacks(tiny_gpt2, training_args, tiny_dataset, [capture])
        for rec in capture.records:
            for layer_name, val in rec.gradient.data.items():
                assert isinstance(val, Factorized), f"{layer_name}: expected Factorized"
                assert torch.isfinite(val.activation).all(), (
                    f"{layer_name}: activation contains non-finite values"
                )
                assert torch.isfinite(val.pre_activation_grad).all(), (
                    f"{layer_name}: pre_activation_grad contains non-finite values"
                )


class TestTrainerOffload:
    """Disk offload during training, then reload through a fresh manager."""

    def test_gradients_offloaded_and_reloadable(
        self, tiny_gpt2, tiny_dataset, training_args, tmp_path
    ):
        grad_dir = str(tmp_path / "gradients")
        offload = OffloadCallback(
            offload_interval=1,
            file_manager=GradientFileManager(grad_dir),
            recording_type="per_sample",
        )
        _train_with_callbacks(tiny_gpt2, training_args, tiny_dataset, [offload])

        fresh = GradientFileManager(grad_dir)
        assert len(fresh.available_steps()) == N_STEPS
        n_records = sum(len(entries) for entries in fresh.index.values())
        assert n_records == N_SAMPLES, (
            f"Expected {N_SAMPLES} per-sample records, got {n_records}"
        )
        for input_hash in fresh.index:
            for step, sample_idx in fresh.lookup_by_hash(input_hash):
                gradient = fresh.load_sample_by_hash(input_hash, step, sample_idx)
                assert gradient.batch_size == 1
                for layer_name, val in gradient.data.items():
                    assert isinstance(val, Factorized), (
                        f"{layer_name}: expected Factorized"
                    )
                    assert torch.isfinite(val.activation).all()
                    assert torch.isfinite(val.pre_activation_grad).all()
