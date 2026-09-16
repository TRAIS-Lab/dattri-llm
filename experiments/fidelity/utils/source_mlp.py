"""SOURCE (approximate unrolling, Bae et al. 2024) on the MNIST+MLP setting.

Bergson's pipeline only takes HF causal LMs, so this applies the same
segment-wise formulas -- imported from Bergson -- to the MLP with *exact*
curvature: the Gauss-Newton matrix of the training loss, averaged over the
checkpoints of each segment and eigendecomposed in full (13,002 parameters).
Everything else mirrors Bergson's ``approx_unrolling`` pipeline: walk the
query gradient backwards through the segments with ``f_backward``, apply
``f_segment`` per segment, and score against per-example training gradients
at the final checkpoint.

    python utils/source_mlp.py --setting mlp --lr 1e-3 --seed 0
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap

from bergson.approx_unrolling.approx_unrolling_math import f_backward, f_segment
from protocol import make_batches, run_trajectory, spearman_per_column
from settings import build, parser

SEGMENTS = 3
CKPTS_PER_SEGMENT = 2


def flat_params(model):
    return {k: v.detach() for k, v in model.named_parameters()}


def per_sample_grads(model, params, x, y):
    """(n, p) gradients of the per-sample cross-entropy."""

    def loss(p, xi, yi):
        return F.cross_entropy(functional_call(model, p, (xi[None],)), yi[None])

    g = vmap(jacrev(loss), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([v.reshape(x.shape[0], -1) for v in g.values()], dim=1)


def ggn(model, params, x, chunk=500):
    """Gauss-Newton matrix of the mean cross-entropy over x, (p, p)."""

    def logits(p, xi):
        return functional_call(model, p, (xi[None],))[0]

    n = x.shape[0]
    total = None
    for i in range(0, n, chunk):
        xb = x[i : i + chunk]
        jac = vmap(jacrev(logits), in_dims=(None, 0))(params, xb)
        j = torch.cat([v.reshape(xb.shape[0], 10, -1) for v in jac.values()], dim=2)
        with torch.no_grad():
            prob = torch.softmax(functional_call(model, params, (xb,)), dim=1)
        a = torch.diag_embed(prob) - prob[:, :, None] * prob[:, None, :]  # (n, 10, 10)
        ja = torch.einsum("ncd,ndk->nck", a, j)
        part = torch.einsum("nck,ncj->kj", j, ja)
        total = part if total is None else total + part
        del jac, j, ja
    return total / n


def main() -> None:
    a = parser().parse_args()
    s = build(a)
    if s.name != "mlp":
        raise SystemExit("source_mlp.py is the MLP variant; use source.py for GPT-2")
    out = s.out_dir.parent / f"{s.out_dir.name}_source"
    out.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr, x_va, y_va = (s.data[k] for k in ("x_tr", "y_tr", "x_va", "y_va"))
    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    n_steps = len(batches)
    n_ckpts = SEGMENTS * CKPTS_PER_SEGMENT
    bounds = [round(n_steps * (l + 1) / SEGMENTS) for l in range(SEGMENTS)]
    ckpt_steps = []
    for l in range(SEGMENTS):
        lo = 0 if l == 0 else bounds[l - 1]
        for c in range(CKPTS_PER_SEGMENT):
            ckpt_steps.append(lo + round((bounds[l] - lo) * (c + 1) / CKPTS_PER_SEGMENT))
    snaps: dict[int, dict] = {}
    holder: dict = {}
    orig_step = s.train_step

    def train_step(model, idx):
        holder["model"] = model
        return orig_step(model, idx)

    def after_step(t):
        if t + 1 in ckpt_steps:
            snaps[t + 1] = {k: v.detach().clone() for k, v in holder["model"].named_parameters()}

    s.train_step = train_step
    t0 = time.time()
    model, lrs = run_trajectory(s, batches, after_step=after_step)
    s.train_step = orig_step
    print(f"trajectory {time.time() - t0:.0f}s; checkpoints at steps {ckpt_steps}")

    # Segment curvature: mean GGN over the segment's checkpoints, eigendecomposed.
    t0 = time.time()
    eig = []
    for l in range(SEGMENTS):
        h = None
        for step in ckpt_steps[l * CKPTS_PER_SEGMENT : (l + 1) * CKPTS_PER_SEGMENT]:
            g = ggn(model, snaps[step], x_tr)
            h = g if h is None else h + g
        h /= CKPTS_PER_SEGMENT
        lam, vec = torch.linalg.eigh(h)
        eig.append((lam.clamp_min(0), vec))
        del h
    lr_times_steps = []
    for l in range(SEGMENTS):
        lo = 0 if l == 0 else bounds[l - 1]
        lr_times_steps.append(sum(lrs[t] for t in range(lo, bounds[l])))
    print(f"curvature {time.time() - t0:.0f}s; lr x steps per segment {lr_times_steps}")

    # Query gradients at the final checkpoint, walked back and per segment.
    params = flat_params(model)
    q = per_sample_grads(model, params, x_va, y_va)  # (n_val, p)
    q_at = [None] * SEGMENTS
    q_at[SEGMENTS - 1] = q
    for k in range(SEGMENTS - 1, 0, -1):
        lam, vec = eig[k]
        q_at[k - 1] = (q_at[k] @ vec) * f_backward(lr_times_steps[k])(lam) @ vec.T
    q_seg = []
    for l in range(SEGMENTS):
        lam, vec = eig[l]
        q_seg.append((q_at[l] @ vec) * f_segment(lr_times_steps[l])(lam) @ vec.T)
    # Train gradients at the final checkpoint; score = sum_l q_seg_l . g(z).
    sel = s.selected.tolist()
    g_tr = per_sample_grads(model, params, x_tr[sel], y_tr[sel])  # (n_sel, p)
    pred = sum(g_tr @ qs.T for qs in q_seg).cpu()  # (n_sel, n_val)

    truth_dir = s.out_dir
    ref = torch.load(truth_dir / "matrices.pt")
    assert torch.equal(ref["selected"], s.selected)
    rho = spearman_per_column(pred, ref["tsloo"])
    result = {
        "name": s.name, "seed": s.seed, "lr": a.lr, "n_queries": s.n_val,
        "method": "source", "source": float(rho.mean()),
    }
    torch.save({"pred": pred, "pairs": [(int(i), None) for i in sel]}, out / "matrices.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
