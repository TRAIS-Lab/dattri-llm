"""MAGIC (Bergson) on the fidelity trajectory.

Bergson's functional ``Trainer`` runs the setting's trajectory -- its model
initialization, batch order and learning-rate schedule, with Bergson's
weighted causal-LM cross-entropy and torchopt's AdamW (betas 0.9/0.999,
eps 1e-8, eps_root 1e-16, no weight decay) -- saving checkpoints, then
``Trainer.backward`` backpropagates each validation block's loss through
the whole run.  The training loss carries a weight per (step, position);
the negated gradient of a validation loss with respect to a weight is the
score of the training block at that step and position.

``train_s`` times ``Trainer.train`` (checkpoint saves included) and
``attribute_s`` the backward passes over all queries; the validation-loss
comparison with the reference trajectory lies between the two and is not
timed.  ``result.json`` holds the setting, the timing, the peak device
memory and, with ``--truth-dir``, the mean Spearman correlation under
``magic``; the scores are saved as ``magic_scores.pt`` with shape
``(n_queries, n_steps, batch_size)``.

    python utils/attribution/magic.py --scale 0.5b --lr 1e-5 --seed 0 --n-queries 64 --truth-dir results/qwen0.5b/lr1e-05_seed0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]  # protocol, settings, common

import torch  # noqa: E402
import torch.nn.functional as F
import torchopt
from torch import nn
from torchopt.pytree import tree_iter

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, Trainer
from bergson.utils.math import weighted_causal_lm_ce
from common import require_bergson
from protocol import make_batches, peak_gb, run_trajectory, spearman_per_column
from settings import build, parser


class IndexStream:
    """Bergson ``DataStream`` over the trajectory's index batches with one weight per (step, position)."""

    def __init__(self, s, batches, device):
        self.s = s
        self.batches = batches
        self.device = torch.device(device)
        self.weights = nn.Parameter(
            torch.ones(len(batches), s.batch_size, device=self.device)
        )

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.weights.requires_grad = value

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> dict:
        idx = self.batches[i]
        x = self.s.data["x_tr"][idx]
        return {"input_ids": x, "labels": x, "example_weight": self.weights[i, : idx.numel()]}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class WeightedLM(nn.Module):
    """Causal LM returning Bergson's weighted causal-LM cross-entropy.

    With ``tie_word_embeddings`` the output head is replaced by an identity
    and the logits are formed from the input-embedding weight, so the
    functional trainer holds one shared parameter; an untied head is used
    as it is.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.tied = bool(getattr(lm.config, "tie_word_embeddings", False))
        self.head = None if self.tied else lm.get_output_embeddings()
        self.lm = lm
        if self.tied:
            lm.set_output_embeddings(nn.Identity())

    def forward(self, input_ids, labels, example_weight=None):
        hidden = self.lm.base_model(input_ids=input_ids).last_hidden_state
        weight = self.lm.get_input_embeddings().weight if self.tied else self.head.weight
        logits = F.linear(hidden, weight)
        # Every position is a valid label: the token mean over the batch.
        return weighted_causal_lm_ce(logits, labels, example_weight=example_weight)


def magic_model(s):
    # A fresh instance of the setting's model, wrapped for the functional trainer.
    torch.manual_seed(s.seed)
    return WeightedLM(s.build_model.fresh()).to(s.device)


def query_batch(s, q: int) -> dict:
    x = s.data["x_va"][q : q + 1]
    return {"input_ids": x, "labels": x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    a, rest = ap.parse_known_args()
    require_bergson()
    s = build(parser().parse_args(rest))  # --scale / --lr / --seed / --tag
    out = s.out_dir.parent / f"{s.out_dir.name}_magic"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    lr = s.extra["lr"]
    log(f"== MAGIC {s.name} lr={lr} seed={s.seed} queries={a.n_queries}")
    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    n_steps = len(batches)
    opt = torchopt.adamw(
        lambda count: lr * s.lr_factor(int(count), n_steps),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        # sqrt(nu + eps_root) has a finite derivative where nu is zero.
        eps_root=1e-16,
    )
    model = magic_model(s)
    trainer, state = Trainer.initialize(model, opt)
    stream = IndexStream(s, batches, s.device)
    ckpt_dir = out / "ckpts"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    t0 = time.time()
    state = trainer.train(
        state, stream, save_dir=str(ckpt_dir), save_mode="sqrt", inplace=True
    )
    torch.cuda.synchronize()
    timing = {"train_s": round(time.time() - t0, 1), "phases": {}}
    peaks = [torch.cuda.max_memory_allocated() / 2**30]
    log(f"  forward: {timing['train_s']:.0f}s, {n_steps} steps, {peak_gb()}")
    torch.cuda.reset_peak_memory_stats()

    # Log the largest validation-loss difference between Bergson's final state
    # and the reference trajectory's final model (not timed).
    ref = torch.load(a.truth_dir + "/matrices.pt") if a.truth_dir else None

    def loss_of(q: int) -> torch.Tensor:
        outputs = model(**query_batch(s, q))
        return outputs.loss if hasattr(outputs, "loss") else outputs

    with state.activate(model), torch.no_grad():
        losses = torch.stack([loss_of(q) for q in range(min(s.n_val, 64))]).cpu()
    ref_model, _ = run_trajectory(s, batches)
    ref_losses = s.val_losses(ref_model).cpu()[: losses.numel()]
    del ref_model
    log(
        f"  val-loss max |magic - ours| = {(losses - ref_losses).abs().max():.2e}"
    )

    final = state.to("cpu")
    n_q = min(a.n_queries, s.n_val)
    w_grads = torch.zeros(n_q, n_steps, s.batch_size)
    t0 = time.time()
    for q in range(n_q):
        work = final.to(s.device)
        work.detach_()
        with work.activate(model) as params:
            outputs = model(**query_batch(s, q))
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(work.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        stream.weights.grad = None
        bwd = BackwardState(query_grads, opt_grads, torch.zeros_like(stream.weights))
        stream.requires_grad = True
        bwd = trainer.backward(
            str(ckpt_dir), stream, bwd, work, cleanup=False, inplace=True
        )
        w_grads[q] = bwd.weight_grads.detach().cpu()
        del bwd, work
        if (q + 1) % 10 == 0 or q + 1 == n_q:
            log(f"  query {q + 1}/{n_q}  {time.time() - t0:.0f}s, {peak_gb()}")
    torch.cuda.synchronize()
    timing["attribute_s"] = round(time.time() - t0, 1)
    timing["total_s"] = round(timing["train_s"] + timing["attribute_s"], 1)
    peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    timing["peak_gb"] = round(max(peaks), 2)
    timing["phases"] = {"train": timing["train_s"], "backward": timing["attribute_s"]}
    # Removing a sample is a weight change of -1: predicted loss change is -dL/dw.
    scores = -w_grads  # (n_q, n_steps, B)
    torch.save({"scores": scores, "batches": batches}, out / "magic_scores.pt")

    result = {
        "name": s.name,
        "seed": s.seed,
        "lr": lr,
        "n_queries": n_q,
        "method": "magic",
        "model": s.extra.get("model"),
        **timing,
    }
    if ref is not None:
        pairs = ref["pairs"]
        truth = ref["tsloo"][:, :n_q]
        pred = torch.stack([scores[:, t, int((batches[t] == i).nonzero()[0])] for i, t in pairs])
        result["magic"] = float(spearman_per_column(pred, truth).mean())
        torch.save({"pred": pred, "pairs": pairs}, out / "matrices.pt")
    log(f"  result: {json.dumps(result)}")
    with (out / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # the trainer's checkpoints


if __name__ == "__main__":
    main()
