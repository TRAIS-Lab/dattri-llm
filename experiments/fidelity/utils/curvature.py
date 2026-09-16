"""AdamW-influence on MNIST with three curvature choices in the sweep:
none (push only), the empirical Fisher of the batch (the standard sweep),
and the exact Gauss-Newton matrix of the batch loss at each step.

Reads the run directory of ``ours.py`` for the same (lr, seed) -- its
stores, moments and ground truth -- and writes ``result.json`` under
``<run>_curvature``.

    python utils/curvature.py --setting mlp --lr 1e-3 --seed 0
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource
from dattri_llm.utils.hashing import hash_sample
from protocol import make_batches, run_trajectory, spearman_per_column
from settings import build, parser

LAYERS = ["fc1", "fc2", "fc3"]


def layout(grads: dict, n: int) -> torch.Tensor:
    """(n, p) in the library's flat layout: per layer [W[o,:], b[o]] rows."""
    parts = []
    for name in LAYERS:
        w = grads[f"{name}.weight"].reshape(n, grads[f"{name}.weight"].shape[-2], -1)
        b = grads[f"{name}.bias"].reshape(n, -1, 1)
        parts.append(torch.cat([w, b], dim=2).reshape(n, -1))
    return torch.cat(parts, dim=1)


def ggn_rows(model, params, x):
    """Rows f with sum_f f f^T = sum_i J_i^T (diag p_i - p_i p_i^T) J_i, (n*10, p)."""

    def logits(p, xi):
        return functional_call(model, p, (xi[None],))[0]

    n = x.shape[0]
    jac = vmap(jacrev(logits), in_dims=(None, 0))(params, x)
    # layout per class: reshape (n, 10, ...) -> (n*10, ...)
    j = layout({k: v.reshape(n * 10, *v.shape[2:]) for k, v in jac.items()}, n * 10)
    j = j.reshape(n, 10, -1)
    with torch.no_grad():
        prob = torch.softmax(functional_call(model, params, (x,)), dim=1)
    centered = j - (prob[:, :, None] * j).sum(1, keepdim=True)
    return (prob.sqrt()[:, :, None] * centered).reshape(n * 10, -1)


def main() -> None:
    a = parser().parse_args()
    s = build(a)
    if s.name != "mlp":
        raise SystemExit("the curvature ablation is defined for the MLP setting")
    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    x_tr = s.data["x_tr"]
    # Exact GGN factor rows at every step's pre-update parameters.
    holder, rows_at = {}, {}
    orig = s.train_step

    def train_step(model, idx):
        holder["model"] = model
        params = {k: v.detach() for k, v in model.named_parameters()}
        rows_at[len(rows_at)] = ggn_rows(model, params, x_tr[idx])
        return orig(model, idx)

    s.train_step = train_step
    run_trajectory(s, batches)
    s.train_step = orig

    args = AttributionArguments(output_dir=str(s.out_dir / "curv"), use_cpu=False)
    dyn = torch.load(s.out_dir / "train_grads" / "adamw_dynamics.pt", weights_only=False)
    dev = torch.device("cuda")

    def blocks(sub):
        out = {}
        for step, block, hashes in DiskGradientSource(GradientStorageManager(str(s.out_dir / sub)), args):
            g = torch.cat(
                [ops.materialize(block.data[n], block.layer_types[n]).reshape(block.batch_size, -1) for n in LAYERS], 1
            ).float().to(dev)
            out[step] = (g, list(hashes))
        return out

    train = blocks("train_grads")
    test_parts = blocks("test_grads")
    test_rep = torch.cat([g for g, _ in test_parts.values()])
    test_ids = [h for _, hs in test_parts.values() for h in hs]
    p = test_rep.shape[1]
    tmap = {hash_sample({"_arg0": s.data["x_va"][q].cpu()}): q for q in range(s.n_val)}
    col = torch.tensor([tmap[h] for h in test_ids]).argsort()
    truth = torch.load(s.out_dir / "matrices.pt")["tsloo"]
    hmap = {hash_sample({"_arg0": x_tr[i].cpu()}): i for i in range(s.n_train)}
    result = {"name": s.name, "seed": a.seed, "lr": a.lr, "method": "curvature"}
    for mode in ("none", "empirical_fisher", "exact_ggn"):
        w_theta, w_m, w_v = torch.eye(p, device=dev), torch.zeros(p, p, device=dev), torch.zeros(p, p, device=dev)
        scores = {}
        for step in sorted(train, reverse=True):
            g_z, ids = train[step]
            d = dyn[step]
            b1, b2 = d["betas"]
            count, lr, eps = int(d["step"]), float(d["lr"]), float(d["eps"])
            m_pre = torch.cat([d["pre"][n][0].reshape(-1) for n in LAYERS]).to(dev)
            m_post = torch.cat([d["post"][n][0].reshape(-1) for n in LAYERS]).to(dev)
            v_post = torch.cat([d["post"][n][1].reshape(-1) for n in LAYERS]).to(dev)
            g_t = (m_post - b1 * m_pre) / (1 - b1)
            dd, ss = ops.adam_preconditioner(m_post / (1 - b1**count), v_post / (1 - b2**count), eps=eps)
            common = {"lr": lr, "step": count, "beta1": b1, "beta2": b2}
            z = ops.adamw_influence_push(g_z, g_t, dd, ss, **common)
            inf = z[:, :p] @ w_theta.T + z[:, p : 2 * p] @ w_m.T + z[:, 2 * p :] @ w_v.T
            sc = -(inf @ test_rep.T).cpu()[:, col]
            for h, row in zip(ids, sc):
                scores[hmap[h]] = row
            if mode == "empirical_fisher":
                rows_h, scale = g_z, float(g_z.shape[0])  # mean loss: H ~ B sum g g^T
            elif mode == "exact_ggn":
                rows_h, scale = rows_at[step], 1.0 / g_z.shape[0]  # H = (1/B) sum f f^T
            else:
                rows_h = None
            if rows_h is not None:
                v = ops.adamw_influence_coupling(w_theta, w_m, w_v, rows_h, g_t, dd, ss, **common)
            w_theta, w_m, w_v = ops.adamw_influence_transition(w_theta, w_m, w_v, dd, ss, weight_decay=float(d["weight_decay"]), **common)
            if rows_h is not None:
                w_theta += scale * (v.T @ rows_h)
        pred = torch.stack([scores[int(i)] for i in s.selected.tolist()])
        result[mode] = float(spearman_per_column(pred, -truth).mean())
    out = s.out_dir.parent / f"{s.out_dir.name}_curvature"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
