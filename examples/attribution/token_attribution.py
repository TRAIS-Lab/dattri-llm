"""This example shows token-level attribution: each training token's share of
its text's influence on a query, printed as a colored heatmap.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from dattri.task import AttributionTask
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dattri_llm.attribution.algorithm.kronecker import KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig

MODEL_ID = "sshleifer/tiny-gpt2"  # 2-layer GPT-2, runs on CPU
MAX_LENGTH = 16
TRAIN_TEXTS = [
    "The plain maskray feeds on caridean shrimp and polychaete worms.",
    "A quick brown fox jumps over the lazy dog.",
    "Gradient hooks capture per-sample signals efficiently.",
]
QUERY_TEXT = "The plain maskray is a stingray that feeds on caridean shrimp."
# the transformer blocks' linear layers (not the tied embedding / LM head)
HOOKS = HookManagerConfig(linear_io=[r"transformer\.h\."])


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts: list[str]) -> None:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids, self.attention_mask = enc["input_ids"], enc["attention_mask"]
        self.tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in self.input_ids]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict:
        labels = self.input_ids[i].masked_fill(self.attention_mask[i] == 0, -100)
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": labels,
        }


def paint(token: str, value: float) -> str:
    """The token on a red (positive) or blue (negative) background, |value| <= 1."""
    end = (215, 48, 39) if value > 0 else (69, 117, 180)
    r, g, b = (round(255 + abs(value) * (c - 255)) for c in end)
    return f"\x1b[48;2;{r};{g};{b}m{token}\x1b[0m" if sys.stdout.isatty() else token


def show(score, dataset: TextDataset, title: str) -> None:
    """One colored line per training text, plus its whole-text score."""
    print(f"\n{title}  (query: {QUERY_TEXT!r})")
    print("-" * 76)
    for train_hash, tokens, mask in zip(
        score.train_ids, dataset.tokens, dataset.attention_mask, strict=True
    ):
        _positions, per_token = score.token_scores(train_hash)
        heat = per_token[:, 0][mask.bool()]  # drop the padding positions
        scale = heat.abs().max().clamp_min(1e-12)
        line = "".join(
            paint(tok.replace("Ġ", " "), float(v / scale))
            for tok, v in zip(tokens, heat.tolist(), strict=False)
        )
        print(f"{line}   [score {heat.sum():+.4f}]")


if __name__ == "__main__":
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    train_ds = TextDataset(tokenizer, TRAIN_TEXTS)
    query_ds = TextDataset(tokenizer, [QUERY_TEXT])

    # the attribution target: the language-modeling loss of a batch
    def loss_func(params, batch):
        return torch.func.functional_call(model, params, args=(), kwargs=batch).loss

    checkpoint = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[checkpoint])

    with tempfile.TemporaryDirectory() as tmp:
        attr_args = AttributionArguments(
            output_dir=tmp,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        # attribution_granularity="token" gives one score row per training
        # token position instead of one per training text; the rows of a text
        # sum to its ordinary score.  Any inner-product attributor takes it.
        graddot = TracInAttributor(attr_args, task=task).attribute(
            train_ds, query_ds, hook_config=HOOKS, attribution_granularity="token"
        )
        show(graddot, train_ds, "GradDot")

        kfac = KFACAttributor(attr_args, task=task).attribute(
            train_ds,
            query_ds,
            hook_config=HOOKS,
            damping=1e-3,
            attribution_granularity="token",
        )
        show(kfac, train_ds, "K-FAC")
