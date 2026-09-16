"""MAGIC (Bergson) on the harness trajectories.

Bergson's functional ``Trainer`` replays our exact trajectory -- same init,
batch order, loss, and AdamW settings (torchopt, bias-corrected, eps outside
the root, no weight decay) -- saving checkpoints, then backpropagates each
validation point's loss through the whole run.  The training loss carries a
weight per (step, position), so the weight gradient is the per-(sample,
step) influence (the ``each`` protocol) and its sum over a sample's
occurrences is the whole-run influence (the ``all`` protocol).

    python utils/magic.py --setting mlp  --lr 1e-3 --seed 0 --n-queries 500
    python utils/magic.py --setting gpt2 --lr 1e-5 --seed 0 --n-queries 64 \
        --truth-dir results/gpt2/lr1e-05_seed0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import torch
import torch.nn.functional as F
import torchopt
from torch import nn
from torchopt.pytree import tree_iter

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, Trainer
from bergson.utils.math import weighted_causal_lm_ce
from protocol import make_batches, occurrences, peak_gb, run_trajectory, spearman_per_column
from settings import build, parser


class IndexStream:
    """Bergson ``DataStream`` over our index batches with (step, position) weights."""

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
        w = self.weights[i, : idx.numel()]
        if self.s.name == "mlp":
            return {
                "input_ids": self.s.data["x_tr"][idx],
                "labels": self.s.data["y_tr"][idx],
                "example_weight": w,
            }
        x = self.s.data["x_tr"][idx]
        return {
            "input_ids": x,
            "labels": x,
            "example_weight": w,
            "valid_mask": torch.ones_like(x, dtype=torch.bool),
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class WeightedMLP(nn.Module):
    """Raw-loss wrapper: mean cross-entropy with per-example weights."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, input_ids, labels, example_weight=None, valid_mask=None):
        ce = F.cross_entropy(self.mlp(input_ids), labels, reduction="none")
        if example_weight is None:
            return ce.mean()
        return (ce * example_weight).sum() / ce.numel()


class WeightedLM(nn.Module):
    """Raw-loss wrapper: Bergson's weighted causal-LM cross-entropy.

    GPT-2 ties ``lm_head.weight`` to ``wte.weight``; the functional trainer
    copies every named parameter separately, which would untie them.  The
    head is therefore dropped and the logits formed from ``wte`` directly.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        lm.lm_head = nn.Identity()
        self.lm = lm

    def forward(self, input_ids, labels, example_weight=None, valid_mask=None):
        hidden = self.lm.transformer(input_ids=input_ids).last_hidden_state
        logits = F.linear(hidden, self.lm.transformer.wte.weight)
        return weighted_causal_lm_ce(
            logits, labels, example_weight=example_weight, valid_mask=valid_mask
        )


def magic_model(s):
    torch.manual_seed(s.seed)
    if s.name == "mlp":
        return WeightedMLP(s.build_model()).to(s.device)
    from transformers import GPT2LMHeadModel

    lm = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager",
    )
    return WeightedLM(lm).to(s.device)


def query_batch(s, q: int) -> dict:
    if s.name == "mlp":
        return {
            "input_ids": s.data["x_va"][q : q + 1],
            "labels": s.data["y_va"][q : q + 1],
        }
    x = s.data["x_va"][q : q + 1]
    return {"input_ids": x, "labels": x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    a, rest = ap.parse_known_args()
    s = build(parser().parse_args(rest))  # --setting / --lr / --seed / --tag
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
    betas = (0.9, 0.95) if s.name == "mlp" else (0.9, 0.999)
    opt = torchopt.adamw(
        lambda count: lr * s.lr_factor(int(count), n_steps),
        betas=betas,
        eps=1e-8,
        weight_decay=0.0,
        # Not 0: the backward differentiates sqrt(nu), which is singular at
        # coordinates whose gradient is identically zero.  1e-16 shifts the
        # forward by ~1e-8 at those coordinates only.
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
    log(f"  forward: {time.time() - t0:.0f}s, {n_steps} steps, {peak_gb()}")
    torch.cuda.reset_peak_memory_stats()

    # Trajectory check against our reference run: the ground truth is only
    # valid if Bergson's replay lands on the same parameters.
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
        f"  val-loss max |magic - ours| = {(losses - ref_losses).abs().max():.2e} (tsloo effects ~1e-3)"
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
    # Removing a sample is a weight change of -1: predicted loss change is -dL/dw.
    scores = -w_grads  # (n_q, n_steps, B)
    torch.save({"scores": scores, "batches": batches}, out / "magic_scores.pt")

    result = {
        "name": s.name,
        "seed": s.seed,
        "lr": lr,
        "n_queries": n_q,
        "method": "magic",
    }
    if ref is not None:
        pairs = ref.get("pairs") or [(int(i), None) for i in ref["selected"].tolist()]
        truth = ref["tsloo"][:, :n_q]
        pred = torch.zeros(len(pairs), n_q)
        for k, (i, t) in enumerate(pairs):
            steps = [t] if t is not None else occurrences(batches, i)
            for tt in steps:
                pos = int((batches[tt] == i).nonzero()[0])
                pred[k] += scores[:, tt, pos]
        rho = spearman_per_column(pred, truth)
        result["magic"] = float(rho.mean())
        if all(t is not None for _, t in pairs):
            rank = {(i, t): occurrences(batches, i).index(t) for i, t in pairs}
            for e in sorted(set(rank.values())):
                sel = [k for k, pr in enumerate(pairs) if rank[pr] == e]
                result[f"magic_epoch{e}"] = float(
                    spearman_per_column(pred[sel], truth[sel]).mean()
                )
        torch.save({"pred": pred, "pairs": pairs}, out / "matrices.pt")
    log(f"  result: {json.dumps(result)}")
    with (out / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # ~20 GB of checkpoints


if __name__ == "__main__":
    main()
