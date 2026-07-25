"""Shared data/model for the ours-vs-bergson pair.

Setting (bergson's README quickstart): EleutherAI/pythia-14m,
NeelNanda/pile-10k (10k documents), tokenized with truncation to the model's
context (2048), sum-CE loss over all next-token positions.  Queries: the
first ``N_QUERY`` documents.
"""

from __future__ import annotations

import torch

MODEL_NAME = "EleutherAI/pythia-14m"
DATASET = "NeelNanda/pile-10k"
MAX_LEN = 2048
N_QUERY = 16


def load_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    return model.eval()


def load_tokenized():
    """Tokenized pile-10k, sorted by length (descending) for batch packing."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET, split="train")

    def tok_fn(ex):
        out = tok(ex["text"], truncation=True, max_length=MAX_LEN)
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tok_fn, batched=True, num_proc=4, remove_columns=ds.column_names)
    ds = ds.map(lambda ex: {"n_tokens": [len(x) for x in ex["input_ids"]]},
                batched=True)
    return ds


class Docs(torch.utils.data.Dataset):
    def __init__(self, hf_dataset) -> None:
        self.ds = hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        ids = torch.tensor(self.ds[i]["input_ids"], dtype=torch.long)
        return {"input_ids": ids}


def pad_collate(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = b["input_ids"]
            mask[i, :k] = 1
        return {"input_ids": ids, "attention_mask": mask}
    return collate
