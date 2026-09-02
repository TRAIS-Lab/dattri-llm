"""Shared tokenized-block dataset for the universal attribution benchmark.

`load_task_data(model_id, dataset, ...)` returns identical train/test token-block
splits usable by ANY library: tokenized with the model's own tokenizer, grouped
into fixed `block_size`-token blocks, with a seeded selection so every library
sees the same samples in the same order.  Tokenized block pools are cached per
(model, dataset, block_size) off the repo tree, so a large corpus is tokenized
once and every task/library reuses it.

    from data import load_task_data
    train_ds, test_ds = load_task_data("Qwen/Qwen2.5-0.5B", "wikitext103",
                                       block_size=1024, n_train=2000, n_test=64)
    # each item: {"input_ids", "attention_mask", "labels"}  (labels == input_ids)
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import torch
from torch.utils.data import Dataset

CACHE = pathlib.Path(
    os.environ.get("BENCH_CACHE", "/scratch/shixuanl/bench_cache"),
)

# name -> (hf_path, hf_config, train_split, test_split, text_column, streaming)
#
# modal/-only: the bare `wikitext` id is the legacy canonical-dataset form and
# current huggingface_hub rejects it -- `parse_hf_uri` requires `namespace/name`
# and raises HfUriError on a single-segment id.  `Salesforce/wikitext` is the
# same dataset at its namespaced home, ungated, carrying the identical
# `wikitext-103-raw-v1` / `wikitext-2-raw-v1` configs.  The cache key in
# `_pool` hashes the benchmark's dataset NAME ("wikitext103"), not this path,
# so no cached block pool is invalidated by the change.
DATASETS: dict[str, tuple] = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "test", "text", False),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "test", "text", False),
    "pile": ("NeelNanda/pile-10k", None, "train", "train", "text", False),
    "c4": ("allenai/c4", "en", "train", "validation", "text", True),
}

# Pool sizes tokenized+cached per split (tasks select n_train/n_test from these).
_TRAIN_POOL = 12_000
_TEST_POOL = 256


class Blocks(Dataset):
    """Fixed-length token blocks; yields the causal-LM training dict."""

    def __init__(self, ids: list[torch.Tensor]) -> None:
        self._ids = ids

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> dict:
        x = self._ids[i]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


def _tokenize_pool(model_id: str, dataset: str, block_size: int, split: str,
                   n_blocks: int, skip_docs: int = 0) -> list[list[int]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    path, config, tr_split, te_split, col, streaming = DATASETS[dataset]
    hf_split = tr_split if split == "train" else te_split
    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset(path, config, split=hf_split, streaming=streaming)
    buf: list[int] = []
    blocks: list[list[int]] = []
    for j, ex in enumerate(ds):
        if j < skip_docs:  # keep train/test disjoint on single-split corpora
            continue
        text = ex[col]
        if not text:
            continue
        buf.extend(tok(text)["input_ids"])
        while len(buf) >= block_size:
            blocks.append(buf[:block_size])
            buf = buf[block_size:]
            if len(blocks) >= n_blocks:
                return blocks
    return blocks


def _pool(model_id: str, dataset: str, block_size: int, split: str) -> list[torch.Tensor]:
    key = hashlib.md5(f"{model_id}|{dataset}|{block_size}".encode()).hexdigest()[:12]
    cdir = CACHE / key
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "meta.txt").write_text(f"{model_id}|{dataset}|block={block_size}\n")
    f = cdir / f"{split}.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    n = _TRAIN_POOL if split == "train" else _TEST_POOL
    # Single-split corpora (pile): carve test from the tail so it is disjoint.
    single = DATASETS[dataset][2] == DATASETS[dataset][3]
    skip = _TRAIN_POOL if (single and split == "test") else 0
    ids = [torch.tensor(b, dtype=torch.long)
           for b in _tokenize_pool(model_id, dataset, block_size, split, n, skip)]
    torch.save(ids, f)
    return ids


def load_task_data(model_id: str, dataset: str, block_size: int = 1024,
                   n_train: int = 2000, n_test: int = 64,
                   seed: int = 0) -> tuple[Blocks, Blocks]:
    """Return (train_ds, test_ds) of block_size token blocks for a task.

    Deterministic given (model_id, dataset, block_size, n_train, n_test, seed):
    the train pool is shuffled with `seed` and the first n_train taken; the test
    set is the pool's first n_test (fixed queries).  Same across every library.
    """
    if dataset not in DATASETS:
        msg = f"unknown dataset {dataset!r}; choices: {sorted(DATASETS)}"
        raise ValueError(msg)
    train_pool = _pool(model_id, dataset, block_size, "train")
    test_pool = _pool(model_id, dataset, block_size, "test")
    if n_train > len(train_pool) or n_test > len(test_pool):
        msg = (f"pool too small: have train={len(train_pool)} test={len(test_pool)}, "
               f"need train={n_train} test={n_test}")
        raise ValueError(msg)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_pool), generator=g).tolist()
    train = [train_pool[i] for i in order[:n_train]]
    test = test_pool[:n_test]
    return Blocks(train), Blocks(test)
