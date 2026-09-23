"""The setting of the fidelity study: a language model on WikiText-2.

The model at ``--scale`` (Qwen2.5 at 0.5b, 1.5b or 3b, or ``gpt2``, the
124M GPT-2) is continually pretrained on 512
random 128-token WikiText-2 blocks for one epoch at batch 32 (16 steps) under
AdamW (betas 0.9/0.999, no weight decay, linear schedule with 10% warmup);
every parameter is trained.  TSLOO removes each of 50 random training blocks
from its batch and records the change in loss on the validation blocks (256
are recorded; the first 64 are the queries every method is scored on).

Everything is fp32 with TF32 off, attention is the eager implementation,
every kernel is deterministic, dropout is off and gradients are not clipped.
The pretrained weights are loaded once and the same model instance is reset
from a CPU copy for each rerun.

    python utils/attribution/adamw_influence.py --scale 3b --lr 1e-5 --seed 0 --tsloo-only
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F
from datasets import load_dataset

from protocol import Setting

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
QWEN = {"0.5b": "Qwen/Qwen2.5-0.5B", "1.5b": "Qwen/Qwen2.5-1.5B", "3b": "Qwen/Qwen2.5-3B"}
OLMO = {"olmo2-1b": "allenai/OLMo-2-0425-1B", "olmo2-7b": "allenai/OLMo-2-1124-7B",
        "olmo3-7b": "allenai/Olmo-3-1025-7B"}  # trained by utils/olmocore.py
MODELS = {**QWEN, "gpt2": "gpt2", **OLMO}
# Every dropout rate of the model's configuration is set to zero.
NO_DROPOUT = {"gpt2": {"resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0}}


def setting_name(scale: str) -> str:
    """The results sub-directory of a scale: ``qwen<scale>`` or ``gpt2``."""
    return f"qwen{scale}" if scale in QWEN else scale


N_TRAIN, BLOCK, N_VAL, BATCH, N_SELECTED = 512, 128, 256, 32, 50


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=sorted(MODELS), help="Qwen2.5 size, or gpt2")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the run directory")
    ap.add_argument("--tsloo-only", action="store_true",
                    help="ground truth only: the reference trajectory, the validation losses and "
                         "the leave-one-out reruns; no capture or attribution")
    ap.add_argument("--tsloo-from", default=None, help="run directory whose ground truth is reused")
    ap.add_argument("--n-masks", type=int, default=10,
                    help="random-mask ensemble size (0 = every coordinate, with --recompute)")
    ap.add_argument("--mask-dim", type=int, default=512, help="coordinates per layer per mask")
    ap.add_argument("--n-val", type=int, default=N_VAL, help="validation blocks captured and scored")
    ap.add_argument("--val-batch-size", type=int, default=16)
    ap.add_argument("--recompute", action="store_true",
                    help="snapshot the trajectory and recompute gradients at attribution (with --n-masks 0)")
    ap.add_argument("--device", default="cuda")
    return ap


def run_dir(name: str, lr: float, seed: int, tag: str = "") -> pathlib.Path:
    return RESULTS / name / f"lr{lr:g}_seed{seed}{tag}"


def _blocks(model_id: str, split: str, n: int, seed: int) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:  # fall back to the namespaced dataset id
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = torch.tensor(tok("\n\n".join(ds["text"]))["input_ids"])
    n_blocks = ids.numel() // BLOCK
    ids = ids[: n_blocks * BLOCK].view(n_blocks, BLOCK)
    gen = torch.Generator().manual_seed(seed)
    return ids[torch.randperm(n_blocks, generator=gen)[:n]]


def per_sample_loss(model, ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits[:, :-1].float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none"
    )
    return loss.view(ids.shape[0], -1).mean(1)


class _Checkpoint:
    """The pretrained model, loaded once; every call returns the same instance
    reset to the pretrained weights (from a CPU copy)."""

    def __init__(self, load, device: str) -> None:
        self._load, self._device = load, device
        self._model, self._init = None, None

    def __call__(self):
        if self._model is None:
            self._model = self._load().to(self._device)
            self._init = {k: v.detach().to("cpu", copy=True).pin_memory() if self._device != "cpu"
                          else v.detach().clone() for k, v in self._model.state_dict().items()}
            return self._model
        self._model.zero_grad(set_to_none=True)
        current = self._model.state_dict()
        with torch.no_grad():
            for k, v in self._init.items():
                current[k].copy_(v, non_blocking=True)
        if self._device != "cpu":
            torch.cuda.synchronize()
        return self._model

    def fresh(self):
        """A separate, independently loaded instance (for a driver that
        wraps or re-registers the model, such as MAGIC's functional trainer)."""
        return self._load().to(self._device)


def lr_factor(step: int, n_steps: int, warmup: float = 0.1) -> float:
    """Linear warmup then linear decay to zero (0-based step)."""
    warm = round(warmup * n_steps)
    if step < warm:
        return (step + 1) / max(1, warm)
    return max(0.0, (n_steps - step) / max(1, n_steps - warm))


def build(a: argparse.Namespace) -> Setting:
    from transformers import AutoModelForCausalLM

    model_id = MODELS[a.scale]
    no_dropout = NO_DROPOUT.get(a.scale, {"attention_dropout": 0.0})

    def load():
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager", **no_dropout,
        )

    x_tr = _blocks(model_id, "train", N_TRAIN, a.seed).to(a.device)
    x_va = _blocks(model_id, "validation", a.n_val, a.seed).to(a.device)

    @torch.no_grad()
    def val_losses(model):
        return torch.cat([per_sample_loss(model, x_va[i : i + 32]) for i in range(0, a.n_val, 32)])

    def make_optimizer(model, n_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.0, fused=True)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_factor(st, n_steps))

    gen = torch.Generator().manual_seed(a.seed + 7)
    n_masks = a.n_masks or None
    name = setting_name(a.scale)
    return Setting(
        name=name,
        out_dir=run_dir(name, a.lr, a.seed, a.tag),
        seed=a.seed,
        build_model=_Checkpoint(load, a.device),
        make_optimizer=make_optimizer,
        n_train=N_TRAIN,
        batch_size=BATCH,
        epochs=1,
        train_step=lambda model, idx: per_sample_loss(model, x_tr[idx]).mean(),
        val_sum_step=lambda model, idx: per_sample_loss(model, x_va[idx]).sum(),
        val_losses=val_losses,
        n_val=a.n_val,
        val_batch_size=a.val_batch_size,
        selected=torch.randperm(N_TRAIN, generator=gen)[:N_SELECTED],
        # GPT-2's position embedding is applied to one (1, T) position tensor
        # shared by the batch, so it has no per-sample gradient: it is trained
        # but not hooked.
        hook_layers=[r"^(?!transformer\.wpe$)"] if a.scale == "gpt2" else None,
        n_masks=n_masks,
        mask_dim=a.mask_dim,
        tsloo_from=pathlib.Path(a.tsloo_from) if a.tsloo_from else None,
        tsloo_only=a.tsloo_only,
        recompute=a.recompute,
        batch=lambda idx: {"input_ids": x_tr[idx]},
        loss_fn=lambda model, batch: per_sample_loss(model, batch["input_ids"]).mean(),
        device=a.device,
        extra={"lr": a.lr, "model": model_id,
               "mask": f"{n_masks}x{a.mask_dim}/layer" if n_masks else "none", "recompute": a.recompute},
        data={"x_tr": x_tr, "x_va": x_va},
        lr_factor=lr_factor,
    )
