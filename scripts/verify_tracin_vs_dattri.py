"""Cross-check the repo's on-disk TracIn against the canonical dattri TracIn.

This script runs three independent computations of the TracIn / GradCos
attribution matrix on the *same* small MLP, the *same* data, and the *same*
parameters, then asserts they agree:

1. **autograd oracle** — per-sample weight gradients via plain
   ``torch.autograd.grad``; the exact TracIn inner-product matrix.
2. **repo on-disk TracIn** (``dattri_llm.algorithm.tracin.TracInAttributor``,
   "workflow 2") — gradients are first *collected to disk* with the repo's
   ``HookManager`` + ``OffloadCallback`` pipeline (factorised act/grad pairs),
   then loaded back and dotted.
3. **reference dattri TracIn** (``dattri.algorithm.tracin.TracInAttributor``,
   "workflow 1") — gradients computed on the fly via ``torch.func`` per-sample
   vmap.  dattri *always* projects, so we pass ``proj_type="identity"`` (a true
   no-op: ``return features``) to recover exact gradients for the comparison.

Equivalence relies on the loss being a **sum over samples** so that each
sample's captured ``grad_output`` equals its own gradient, making the
factorised per-sample gradient exactly ``dL_i/dW``.

The ensemble uses **two checkpoints**, so the repo's ``HookManager`` completes
two collection steps inside a single ``collect()`` context (``step 0`` for
checkpoint 0, ``step 1`` for checkpoint 1, each holding all samples).  This
exercises the multi-step path: ``_on_step_complete`` firing twice (buffer
reset + step-counter increment) and the per-step weighted ensemble sum
``score = Σ_k w_k · ⟨g_train^(k), g_test^(k)⟩``.

Run::

    python scripts/verify_tracin_vs_dattri.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── repo (workflow 2) ────────────────────────────────────────────────────────
from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.tracin import TracInAttributor, GradCosAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

# ── reference (workflow 1) ───────────────────────────────────────────────────
from dattri.algorithm.tracin import TracInAttributor as DattriTracIn
from dattri.task import AttributionTask


# --------------------------------------------------------------------------- #
# Model and data                                                              #
# --------------------------------------------------------------------------- #

IN_DIM, HID_DIM, OUT_DIM = 8, 16, 4
N_TRAIN, N_TEST = 16, 8
SEED = 0

# Per-step (per-checkpoint) ensemble weights for the TracIn variant.  Two
# entries → two collected steps.  GradCos forces uniform weights, so it uses
# [1.0, 1.0] regardless (see ``step_weights``).
STEP_WEIGHTS = [1.0, 0.5]


class MLP(nn.Module):
    """Tiny 2-layer MLP with bias-free linears named ``mlp.fc1`` / ``mlp.fc2``.

    The ``mlp.*`` naming matches both the MLP-keyword hook heuristic and the
    keys of ``model.named_parameters()``.  ``forward`` accepts (and ignores) a
    ``y`` kwarg so the repo's input-hashing sees the same {x, y} dict that the
    attribution-side dataset yields.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID_DIM, OUT_DIM, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
    """Yields ``{"x": x_i, "y": y_i}`` — the format the repo attributor hashes."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return {"x": self.x[i], "y": self.y[i]}


class TupleDataset(Dataset):
    """Yields ``(x_i, y_i)`` — the format dattri's loss_func unpacks."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def make_data():
    g = torch.Generator().manual_seed(SEED)
    x_tr = torch.randn(N_TRAIN, IN_DIM, generator=g)
    y_tr = torch.randn(N_TRAIN, OUT_DIM, generator=g)
    x_te = torch.randn(N_TEST, IN_DIM, generator=g)
    y_te = torch.randn(N_TEST, OUT_DIM, generator=g)
    return x_tr, y_tr, x_te, y_te


def make_checkpoints(model: nn.Module):
    """Return two distinct state_dicts (initial + perturbed) as the ensemble."""
    sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(SEED + 1)
    sd1 = {k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in sd0.items()}
    return [sd0, sd1]


def step_weights(*, normalized: bool):
    """Ensemble weights: per-step for TracIn, uniform for GradCos."""
    return [1.0, 1.0] if normalized else STEP_WEIGHTS


# --------------------------------------------------------------------------- #
# 1. autograd oracle                                                          #
# --------------------------------------------------------------------------- #

WEIGHT_NAMES = ["mlp.fc1.weight", "mlp.fc2.weight"]


def per_sample_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return (N, D) flattened per-sample weight gradients of squared error."""
    params = [dict(model.named_parameters())[n] for n in WEIGHT_NAMES]
    rows = []
    for i in range(x.shape[0]):
        out = model(x[i : i + 1])
        loss = ((out - y[i : i + 1]) ** 2).sum()
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)  # (N, D)


def oracle_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, *, normalized: bool):
    """Weighted ensemble of per-checkpoint inner-product matrices."""
    weights = step_weights(normalized=normalized)
    score = torch.zeros(x_tr.shape[0], x_te.shape[0])
    for sd, w in zip(checkpoints, weights):
        model.load_state_dict(sd)
        g_tr = per_sample_grads(model, x_tr, y_tr)
        g_te = per_sample_grads(model, x_te, y_te)
        if normalized:
            g_tr, g_te = F.normalize(g_tr, dim=-1), F.normalize(g_te, dim=-1)
        score += w * (g_tr @ g_te.T)
    return score


# --------------------------------------------------------------------------- #
# 2. repo on-disk TracIn (collect gradients, then attribute)                  #
# --------------------------------------------------------------------------- #


def collect_to_disk(model: nn.Module, checkpoints, x: torch.Tensor, y: torch.Tensor,
                    out_dir: Path):
    """Collect the whole dataset once per checkpoint, all inside one context.

    Each checkpoint's full-set forward+backward completes one HookManager step,
    so two checkpoints produce ``step 0`` and ``step 1`` — the HookManager's
    step-completion logic fires twice within a single ``collect()`` block.
    """
    fm = GradientFileManager(str(out_dir))
    offload = OffloadCallback(
        offload_interval=1, file_manager=fm, recording_type="per_sample"
    )
    hm = HookManager(
        model,
        config=HookManagerConfig(mlp_name_patterns=[r"mlp\."]),
        callbacks=[offload],
    )
    with hm.collect():
        for sd in checkpoints:
            model.load_state_dict(sd)
            model.zero_grad(set_to_none=True)
            out = model(x=x, y=y)
            loss = ((out - y) ** 2).sum()  # sum over samples → exact per-sample grads
            loss.backward()
    hm.remove()
    model.zero_grad(set_to_none=True)


def repo_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, tmp: Path, *,
                normalized: bool):
    train_dir, test_dir, out_dir = tmp / "train_g", tmp / "test_g", tmp / "out"
    collect_to_disk(model, checkpoints, x_tr, y_tr, train_dir)
    collect_to_disk(model, checkpoints, x_te, y_te, test_dir)

    args = AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    # Sanity-check that the collector really wrote one record per step.
    train_steps = sorted(
        {e["step"] for es in GradientFileManager(str(train_dir)).index.values() for e in es}
    )
    assert train_steps == list(range(len(checkpoints))), (
        f"expected steps {list(range(len(checkpoints)))}, collected {train_steps}"
    )

    steps = list(range(len(checkpoints)))
    if normalized:
        # Forces normalized_grad and uniform weights.
        attr = GradCosAttributor(args, layer_name=["mlp.fc1", "mlp.fc2"], steps=steps)
    else:
        attr = TracInAttributor(
            args, layer_name=["mlp.fc1", "mlp.fc2"],
            weight_list=STEP_WEIGHTS, steps=steps,
        )
    return attr.attribute(
        train_dataset=DictDataset(x_tr, y_tr),
        test_dataset=DictDataset(x_te, y_te),
        train_gradients_dir=str(train_dir),
        test_gradients_dir=str(test_dir),
    )


# --------------------------------------------------------------------------- #
# 3. Dattri TracIn                                                   #
# --------------------------------------------------------------------------- #


def dattri_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, *, normalized: bool):
    def loss_func(params, data):
        x, y = data
        yhat = torch.func.functional_call(model, params, (x,))
        return ((yhat - y) ** 2).sum()

    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=checkpoints)

    full_dim = sum(dict(model.named_parameters())[n].numel() for n in WEIGHT_NAMES)
    identity_proj = {
        "proj_type": "identity",
        "proj_dim": full_dim,
        "proj_max_batch_size": 32,
        "proj_seed": 0,
        "device": "cpu",
    }
    attr = DattriTracIn(
        task=task,
        weight_list=torch.tensor(step_weights(normalized=normalized)),
        normalized_grad=normalized,
        projector_kwargs=identity_proj,
        layer_name=WEIGHT_NAMES,
        device="cpu",
    )
    train_loader = torch.utils.data.DataLoader(
        TupleDataset(x_tr, y_tr), batch_size=4, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        TupleDataset(x_te, y_te), batch_size=3, shuffle=False
    )
    return attr.attribute(train_loader, test_loader)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def run_variant(name: str, *, normalized: bool, tmp: Path) -> bool:
    torch.manual_seed(SEED)
    model = MLP().eval()
    checkpoints = make_checkpoints(model)
    x_tr, y_tr, x_te, y_te = make_data()

    oracle = oracle_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, normalized=normalized)
    repo = repo_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, tmp, normalized=normalized)
    ref = dattri_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, normalized=normalized)

    d_repo = (repo - oracle).abs().max().item()
    d_ref = (ref - oracle).abs().max().item()
    d_cross = (repo - ref).abs().max().item()

    ok = (
        torch.allclose(repo, oracle, atol=1e-4)
        and torch.allclose(ref, oracle, atol=1e-4)
        and torch.allclose(repo, ref, atol=1e-4)
    )
    print(f"\n=== {name} (shape {tuple(repo.shape)}) ===")
    print(f"  max|repo  - oracle| = {d_repo:.3e}")
    print(f"  max|dattri- oracle| = {d_ref:.3e}")
    print(f"  max|repo  - dattri| = {d_cross:.3e}")
    print(f"  oracle[0,:] = {oracle[0].tolist()}")
    print(f"  repo  [0,:] = {repo[0].tolist()}")
    print(f"  dattri[0,:] = {ref[0].tolist()}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        all_ok &= run_variant("TracIn (dot product)", normalized=False, tmp=root / "tracin")
        all_ok &= run_variant("GradCos (cosine)", normalized=True, tmp=root / "gradcos")
    print("\n" + ("ALL VARIANTS PASS ✅" if all_ok else "SOME VARIANTS FAILED ❌"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
