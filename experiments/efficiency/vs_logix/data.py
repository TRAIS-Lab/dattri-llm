"""Shared data/model for the ours-vs-logix pair.

Reuses logix's own ``examples/language_modeling/utils.py`` (model wrapper with
Conv1D->Linear replacement, tokenizer, wikitext tokenize+group pipeline) so
both sides see the *identical* model and dataset.

Setting (logix's example defaults): GPT-2, wikitext-103-raw-v1, 0.5% train
sample, 512-token blocks.  Queries: first ``N_QUERY`` blocks of the test split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LOGIX_EXAMPLE = "/scratch/shixuanl/logix/examples/language_modeling"
sys.path.insert(0, LOGIX_EXAMPLE)

from utils import get_dataset, get_model, get_tokenizer, set_seed  # noqa: E402

MODEL_NAME = "gpt2"
DATA_PATH = "wikitext"
DATA_NAME = "wikitext-103-raw-v1"
CACHE_DIR = str(HERE / "data_cache")
N_QUERY = 16
BATCH = 8


def load_model_and_tokenizer():
    set_seed(0)
    model = get_model(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer = get_tokenizer(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    return model, tokenizer


def load_datasets(tokenizer):
    """Returns (train_blocks, query_blocks) HF datasets of 512-token blocks."""
    train = get_dataset(
        model_name=MODEL_NAME, tokenizer=tokenizer, data_path=DATA_PATH,
        data_name=DATA_NAME, split="train", cache_dir=CACHE_DIR,
    )
    test = get_dataset(
        model_name=MODEL_NAME, tokenizer=tokenizer, data_path=DATA_PATH,
        data_name=DATA_NAME, split="test", cache_dir=CACHE_DIR,
    )
    return train, test.select(range(N_QUERY))


class TensorBlocks(torch.utils.data.Dataset):
    """HF grouped dataset -> dict-of-tensors samples (default-collate ready)."""

    def __init__(self, hf_dataset) -> None:
        self.ds = hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


if __name__ == "__main__":
    model, tok = load_model_and_tokenizer()
    train, query = load_datasets(tok)
    print(f"train blocks: {len(train)}, query blocks: {len(query)}")
