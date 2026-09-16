"""The two settings of the fidelity study (Appendix D.1 of arXiv:2605.18814).

``mlp``: a 784-16-16-10 ReLU MLP on 6,000 random MNIST images, one epoch at
batch 64 under AdamW (betas 0.9/0.95, constant learning rate); TSLOO over
200 training images against 500 test images; every parameter attributed.

``gpt2``: GPT-2 (124M) continually pretrained on 512 random 128-token
WikiText-2 blocks, three epochs at batch 32 under AdamW (betas 0.9/0.999,
linear schedule, 10% warmup); TSLOO over 50 training blocks against 256
validation blocks (the first 64 are the queries every method is scored on);
a random-mask ensemble of 10 masks with 512 coordinates per layer, or every
coordinate with ``--n-masks 0`` (snapshots + replay).  Dropout is disabled
and gradients are not clipped so the leave-one-out reruns are exact
counterfactuals.

    python utils/ours.py --setting gpt2 --lr 1e-5 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn

from protocol import Setting

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["mlp", "gpt2"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the run directory")
    ap.add_argument("--methods", nargs="+", default=["adamw_influence", "dvemb"],
                    choices=["adamw_influence", "dvemb"])
    ap.add_argument("--n-masks", type=int, default=None,
                    help="random-mask ensemble size (mlp default: none; gpt2: 10; 0 = every coordinate)")
    ap.add_argument("--mask-dim", type=int, default=512, help="coordinates per layer per mask")
    ap.add_argument("--n-val", type=int, default=None, help="validation points (mlp 500, gpt2 256)")
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--tsloo-at", default=None, choices=[None, "first", "last", "each"],
                    help="which occurrence the ground truth removes (default: every one)")
    ap.add_argument("--tsloo-from", default=None, help="run directory whose ground truth is reused")
    ap.add_argument("--skip-tsloo", action="store_true", help="timing runs: capture + attribution only")
    ap.add_argument("--recompute", action="store_true",
                    help="snapshot the trajectory and recompute gradients at attribution (with --n-masks 0)")
    ap.add_argument("--device", default="cuda")
    return ap


def run_dir(setting: str, lr: float, seed: int, tag: str = "") -> pathlib.Path:
    return RESULTS / setting / f"lr{lr:g}_seed{seed}{tag}"


# --------------------------------------------------------------------------- #
# MNIST + MLP                                                                  #
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def _mnist(split: str, n: int, seed: int, device: str):
    ds = load_dataset("mnist", split=split)
    gen = torch.Generator().manual_seed(seed)
    sub = ds.select(torch.randperm(len(ds), generator=gen)[:n].tolist())
    x = torch.stack(
        [torch.tensor(list(img.getdata()), dtype=torch.float32) for img in sub["image"]]
    ) / 255.0
    return x.to(device), torch.tensor(sub["label"]).to(device)


def build_mlp(a: argparse.Namespace) -> Setting:
    n_train, n_val = 6000, a.n_val or 500
    x_tr, y_tr = _mnist("train", n_train, a.seed, a.device)
    x_va, y_va = _mnist("test", n_val, a.seed, a.device)

    def train_step(model, idx):
        return F.cross_entropy(model(x_tr[idx]), y_tr[idx])

    def val_sum_step(model, idx):
        return F.cross_entropy(model(x_va[idx]), y_va[idx], reduction="sum")

    @torch.no_grad()
    def val_losses(model):
        return F.cross_entropy(model(x_va), y_va, reduction="none")

    def make_optimizer(model, _n_steps):
        opt = torch.optim.AdamW(
            model.parameters(), lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        return opt, None

    gen = torch.Generator().manual_seed(a.seed + 7)
    selected = torch.randperm(n_train, generator=gen)[:200]
    n_masks = a.n_masks or None
    return Setting(
        name="mlp",
        out_dir=run_dir("mlp", a.lr, a.seed, a.tag),
        seed=a.seed,
        build_model=MLP,
        make_optimizer=make_optimizer,
        n_train=n_train,
        batch_size=64,
        epochs=1,
        train_step=train_step,
        val_sum_step=val_sum_step,
        val_losses=val_losses,
        n_val=n_val,
        val_batch_size=a.val_batch_size or 100,
        selected=selected,
        n_masks=n_masks,
        mask_dim=a.mask_dim,
        tsloo_from=pathlib.Path(a.tsloo_from) if a.tsloo_from else None,
        tsloo_at=a.tsloo_at,
        skip_tsloo=a.skip_tsloo,
        methods=tuple(a.methods),
        device=a.device,
        extra={"lr": a.lr, "mask": f"{n_masks}x{a.mask_dim}/layer" if n_masks else "none"},
        data={"x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "y_va": y_va},
        lr_factor=lambda _step, _n: 1.0,
    )


# --------------------------------------------------------------------------- #
# WikiText-2 + GPT-2                                                           #
# --------------------------------------------------------------------------- #


def _blocks(split: str, block: int, n: int, seed: int) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:  # newer hub clients need the namespaced id
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = torch.tensor(tok("\n\n".join(ds["text"]))["input_ids"])
    n_blocks = ids.numel() // block
    ids = ids[: n_blocks * block].view(n_blocks, block)
    gen = torch.Generator().manual_seed(seed)
    return ids[torch.randperm(n_blocks, generator=gen)[:n]]


def per_sample_loss(model, ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits[:, :-1].float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none"
    )
    return loss.view(ids.shape[0], -1).mean(1)


def lr_factor(step: int, n_steps: int, warmup: float = 0.1) -> float:
    """Linear warmup then linear decay to zero (0-based step)."""
    warm = round(warmup * n_steps)
    if step < warm:
        return (step + 1) / max(1, warm)
    return max(0.0, (n_steps - step) / max(1, n_steps - warm))


def build_gpt2(a: argparse.Namespace) -> Setting:
    from transformers import GPT2LMHeadModel

    n_train, block, epochs, n_val = 512, 128, 3, a.n_val or 256
    x_tr = _blocks("train", block, n_train, a.seed).to(a.device)
    x_va = _blocks("validation", block, n_val, a.seed).to(a.device)

    def build_model():
        return GPT2LMHeadModel.from_pretrained(
            "gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0
        )

    def train_step(model, idx):
        return per_sample_loss(model, x_tr[idx]).mean()

    def val_sum_step(model, idx):
        return per_sample_loss(model, x_va[idx]).sum()

    @torch.no_grad()
    def val_losses(model):
        return torch.cat(
            [per_sample_loss(model, x_va[i : i + 32]) for i in range(0, n_val, 32)]
        )

    def make_optimizer(model, n_steps):
        opt = torch.optim.AdamW(
            model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_factor(st, n_steps))
        return opt, sched

    gen = torch.Generator().manual_seed(a.seed + 7)
    selected = torch.randperm(n_train, generator=gen)[:50]
    n_masks = 10 if a.n_masks is None else (a.n_masks or None)
    return Setting(
        name="gpt2",
        out_dir=run_dir("gpt2", a.lr, a.seed, a.tag),
        seed=a.seed,
        build_model=build_model,
        make_optimizer=make_optimizer,
        n_train=n_train,
        batch_size=32,
        epochs=epochs,
        train_step=train_step,
        val_sum_step=val_sum_step,
        val_losses=val_losses,
        n_val=n_val,
        val_batch_size=a.val_batch_size or 16,
        selected=selected,
        # wpe is applied to a (1, T) position tensor shared by the batch:
        # its gradient is batch-collapsed, so it is trained but not hooked.
        hook_layers=[r"^(?!transformer\.wpe$)"],
        n_masks=n_masks,
        mask_dim=a.mask_dim,
        tsloo_from=pathlib.Path(a.tsloo_from) if a.tsloo_from else None,
        tsloo_at=a.tsloo_at,
        skip_tsloo=a.skip_tsloo,
        methods=tuple(a.methods),
        recompute=a.recompute,
        batch=lambda idx: {"input_ids": x_tr[idx]},
        loss_fn=lambda model, batch: per_sample_loss(model, batch["input_ids"]).mean(),
        device=a.device,
        extra={
            "lr": a.lr,
            "mask": f"{n_masks}x{a.mask_dim}/layer" if n_masks else "none",
            "recompute": a.recompute,
        },
        data={"x_tr": x_tr, "x_va": x_va},
        lr_factor=lr_factor,
    )


BUILDERS = {"mlp": build_mlp, "gpt2": build_gpt2}


def build(a: argparse.Namespace) -> Setting:
    return BUILDERS[a.setting](a)
