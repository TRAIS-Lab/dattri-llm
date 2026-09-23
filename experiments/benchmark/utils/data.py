"""Shared tokenized-block dataset for the attribution benchmark.

`load_task_data(model_id, dataset, ...)` returns the train/test token-block
splits every adapter uses: tokenized with the model's own tokenizer, grouped
into fixed `block_size`-token blocks, with a seeded selection so every library
sees the same samples in the same order.  Tokenized block pools are cached per
(model, dataset, block_size) in the directory named by the `BENCH_CACHE`
environment variable and shared by every task and library.  Models and
datasets are downloaded through Hugging Face (`HF_HOME` applies).

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
    os.environ.get("BENCH_CACHE", str(pathlib.Path.home() / ".cache" / "dattri_llm_bench")),
)

# name -> (hf_path, hf_config, train_split, test_split, text_column, streaming)
# The name (not the Hub path) enters the cache key in `_pool`.
DATASETS: dict[str, tuple] = {
    "wikitext2": (
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        "train",
        "test",
        "text",
        False,
    ),
    "wikitext103": (
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        "test",
        "text",
        False,
    ),
    "pile": ("NeelNanda/pile-10k", None, "train", "train", "text", False),
    "c4": ("allenai/c4", "en", "train", "validation", "text", True),
}

# Number of blocks tokenized and cached per split; tasks select n_train/n_test
# from these (`load_task_data` raises if a task asks for more).  The training
# set is a seeded draw from the train pool, and the pool size is part of the
# cache file's name.
_TRAIN_POOL = 40_000
_TEST_POOL = 256


class Blocks(Dataset):
    """Fixed-length token blocks; yields the causal-LM training dict."""

    def __init__(self, ids: list[torch.Tensor]) -> None:
        self._ids = ids

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> dict:
        x = self._ids[i]
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


def _tokenize_pool(
    model_id: str,
    dataset: str,
    block_size: int,
    split: str,
    n_blocks: int,
    skip_docs: int = 0,
) -> list[list[int]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    path, config, tr_split, te_split, col, streaming = DATASETS[dataset]
    hf_split = tr_split if split == "train" else te_split
    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset(path, config, split=hf_split, streaming=streaming)
    buf: list[int] = []
    blocks: list[list[int]] = []
    for j, ex in enumerate(ds):
        if j < skip_docs:  # test pool of a single-split corpus: skip leading docs
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


def _pool(
    model_id: str, dataset: str, block_size: int, split: str
) -> list[torch.Tensor]:
    key = hashlib.md5(f"{model_id}|{dataset}|{block_size}".encode()).hexdigest()[:12]
    cdir = CACHE / key
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "meta.txt").write_text(f"{model_id}|{dataset}|block={block_size}\n")
    n = _TRAIN_POOL if split == "train" else _TEST_POOL
    # One file per (split, pool size).
    f = cdir / f"{split}-{n}.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    # Under torchrun only rank 0 builds the pool; the other ranks wait for the
    # finished file.  The build writes to a temporary name and renames it, so
    # the file appears complete.
    import os
    import time

    if int(os.environ.get("RANK", "0")) != 0:
        deadline = time.monotonic() + 3600
        while not f.exists():
            if time.monotonic() > deadline:
                msg = f"rank {os.environ['RANK']}: pool {f} not built by rank 0 within an hour"
                raise TimeoutError(msg)
            time.sleep(5)
        return torch.load(f, weights_only=False)
    # Single-split corpora (pile): the test pool starts after the first
    # _TRAIN_POOL documents of the split.
    single = DATASETS[dataset][2] == DATASETS[dataset][3]
    skip = _TRAIN_POOL if (single and split == "test") else 0
    ids = [
        torch.tensor(b, dtype=torch.long)
        for b in _tokenize_pool(model_id, dataset, block_size, split, n, skip)
    ]
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    torch.save(ids, tmp)
    os.replace(tmp, f)
    return ids


def load_task_data(
    model_id: str,
    dataset: str,
    block_size: int = 1024,
    n_train: int = 2000,
    n_test: int = 64,
    seed: int = 0,
) -> tuple[Blocks, Blocks]:
    """Return (train_ds, test_ds) of block_size token blocks for a task.

    Deterministic given (model_id, dataset, block_size, n_train, n_test, seed):
    the train pool is shuffled with `seed` and the first n_train taken; the test
    set is the test pool's first n_test blocks.  `dataset` is a key of
    `DATASETS`.  Raises ValueError for an unknown dataset or when a pool holds
    fewer blocks than requested.
    """
    if dataset not in DATASETS:
        msg = f"unknown dataset {dataset!r}; choices: {sorted(DATASETS)}"
        raise ValueError(msg)
    train_pool = _pool(model_id, dataset, block_size, "train")
    test_pool = _pool(model_id, dataset, block_size, "test")
    if n_train > len(train_pool) or n_test > len(test_pool):
        msg = (
            f"pool too small: have train={len(train_pool)} test={len(test_pool)}, "
            f"need train={n_train} test={n_test}"
        )
        raise ValueError(msg)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_pool), generator=g).tolist()
    train = [train_pool[i] for i in order[:n_train]]
    test = test_pool[:n_test]
    return Blocks(train), Blocks(test)
