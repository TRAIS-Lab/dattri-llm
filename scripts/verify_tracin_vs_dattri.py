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

The ensemble uses **two checkpoints**.  The simplified repo attributor computes
an unweighted, single-checkpoint gradient cross-gram (no internal step ensemble
or step alignment), so the two-checkpoint TracIn ensemble
``score = Σ_k w_k · ⟨g_train^(k), g_test^(k)⟩`` is reconstructed by running
attribution **once per checkpoint** and combining the per-checkpoint matrices
with the step weights externally.

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
from dattri_llm.algorithm.tracin import TracInAttributor
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.utils import hash_sample

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


def oracle_step_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, *, normalized: bool):
    """Per-step (per-checkpoint) inner-product matrices, one per step.

    Returns a list ``[M_0, M_1, ...]`` where ``M_k = weight_k · ⟨g^k, g^k⟩`` is
    the un-summed ensemble term for step ``k``.  Summing the list reproduces
    :func:`oracle_scores`.
    """
    weights = step_weights(normalized=normalized)
    mats = []
    for sd, w in zip(checkpoints, weights):
        model.load_state_dict(sd)
        g_tr = per_sample_grads(model, x_tr, y_tr)
        g_te = per_sample_grads(model, x_te, y_te)
        if normalized:
            g_tr, g_te = F.normalize(g_tr, dim=-1), F.normalize(g_te, dim=-1)
        mats.append(w * (g_tr @ g_te.T))
    return mats


def oracle_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, *, normalized: bool):
    """Weighted ensemble of per-checkpoint inner-product matrices."""
    return sum(
        oracle_step_scores(
            model, checkpoints, x_tr, y_tr, x_te, y_te, normalized=normalized
        )
    )


# --------------------------------------------------------------------------- #
# 2. repo on-disk TracIn (collect gradients, then attribute)                  #
# --------------------------------------------------------------------------- #


def collect_to_disk(model: nn.Module, checkpoints, x: torch.Tensor, y: torch.Tensor,
                    out_dir: Path):
    """Collect the whole dataset for each given checkpoint inside one context.

    Each checkpoint's full-set forward+backward completes one HookManager step.
    Callers now pass a single checkpoint per dir (one collected step, ``step 0``)
    so that each attribution run is a single, unweighted cross-gram.
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


def sample_hashes(x_tr, y_tr, x_te, y_te):
    """The per-sample content hashes in oracle (0..N-1) order.

    These are the same identifiers the collector stored on disk, so they let us
    realign the hash-keyed :class:`AttributionScore` back to the oracle's
    sample order for a direct numeric comparison.
    """
    train_hashes = [hash_sample({"x": x_tr, "y": y_tr}, i) for i in range(x_tr.shape[0])]
    test_hashes = [hash_sample({"x": x_te, "y": y_te}, j) for j in range(x_te.shape[0])]
    return train_hashes, test_hashes


def repo_step_matrices(model, checkpoints, x_tr, y_tr, x_te, y_te, tmp: Path, *,
                       normalized: bool):
    """Run the (single-checkpoint) repo attributor once per checkpoint.

    The simplified ``TracInAttributor`` no longer ensembles or weights steps
    internally — it just computes one gradient cross-gram.  The two-checkpoint
    TracIn ensemble is therefore reconstructed here by attributing each
    checkpoint independently (its own single-step train/test dirs).  Returns the
    list of *unweighted* per-checkpoint matrices, each realigned to oracle
    (sample) order.
    """
    train_hashes, test_hashes = sample_hashes(x_tr, y_tr, x_te, y_te)
    mats = []
    for k, sd in enumerate(checkpoints):
        run = tmp / f"ckpt_{k}"
        train_dir, test_dir, out_dir = run / "train_g", run / "test_g", run / "out"
        collect_to_disk(model, [sd], x_tr, y_tr, train_dir)
        collect_to_disk(model, [sd], x_te, y_te, test_dir)

        # Each checkpoint is one collection step → one record per sample.
        train_steps = sorted(
            {e["step"] for es in GradientFileManager(str(train_dir)).index.values()
             for e in es}
        )
        assert train_steps == [0], f"expected a single step, collected {train_steps}"

        args = AttributionArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        # GradCos == TracIn with normalized_grad=True (the subclass was removed).
        attr = TracInAttributor(
            args, layer_name=["mlp.fc1", "mlp.fc2"], normalized_grad=normalized,
        )
        result = attr.attribute(
            train_gradients_dir=str(train_dir),
            test_gradients_dir=str(test_dir),
        )
        mats.append(result.query(train_hashes, test_hashes, trajectory="agnostic"))
    return mats


def repo_scores(model, checkpoints, x_tr, y_tr, x_te, y_te, tmp: Path, *,
                normalized: bool):
    """Weighted ensemble of the per-checkpoint repo matrices (weights applied
    externally now that the attributor itself is unweighted)."""
    weights = step_weights(normalized=normalized)
    mats = repo_step_matrices(
        model, checkpoints, x_tr, y_tr, x_te, y_te, tmp, normalized=normalized
    )
    return sum(w * m for w, m in zip(weights, mats))


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


def run_per_step_experiment(name: str, *, normalized: bool, tmp: Path) -> bool:
    """Attribute each checkpoint independently and check it per step.

    The simplified attributor has no internal step ensemble, so the per-step
    terms come from independent single-checkpoint runs (see
    :func:`repo_step_matrices`).  Each weighted per-checkpoint matrix is compared
    against the per-step autograd oracle, and we confirm they sum back to the
    aggregate (full-ensemble) score.
    """
    torch.manual_seed(SEED)
    model = MLP().eval()
    checkpoints = make_checkpoints(model)
    x_tr, y_tr, x_te, y_te = make_data()

    weights = step_weights(normalized=normalized)
    oracle_steps = oracle_step_scores(
        model, checkpoints, x_tr, y_tr, x_te, y_te, normalized=normalized
    )
    repo_mats = repo_step_matrices(
        model, checkpoints, x_tr, y_tr, x_te, y_te, tmp, normalized=normalized
    )

    print(f"\n=== {name}: per-step retrieval ===")
    print(f"  checkpoints attributed: {len(repo_mats)}")

    ok = True
    summed = torch.zeros_like(oracle_steps[0])
    for k, (w, mat) in enumerate(zip(weights, repo_mats)):
        repo_step = w * mat
        summed += repo_step
        diff = (repo_step - oracle_steps[k]).abs().max().item()
        step_ok = torch.allclose(repo_step, oracle_steps[k], atol=1e-4)
        ok &= step_ok
        print(f"  step {k}: shape {tuple(repo_step.shape)}  "
              f"max|repo - oracle| = {diff:.3e}  {'PASS' if step_ok else 'FAIL'}")

    # The weighted per-checkpoint matrices must sum to the full oracle ensemble.
    aggregate = oracle_scores(
        model, checkpoints, x_tr, y_tr, x_te, y_te, normalized=normalized
    )
    sum_ok = torch.allclose(summed, aggregate, atol=1e-5)
    ok &= sum_ok
    print(f"  Σ step matrices == aggregate score: {'PASS' if sum_ok else 'FAIL'}")
    return ok


def main() -> None:
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        all_ok &= run_variant("TracIn (dot product)", normalized=False, tmp=root / "tracin")
        all_ok &= run_variant("GradCos (cosine)", normalized=True, tmp=root / "gradcos")

        print("\n" + "-" * 78)
        print("Per-step attribution experiments")
        print("-" * 78)
        all_ok &= run_per_step_experiment(
            "TracIn (dot product)", normalized=False, tmp=root / "tracin_steps"
        )
        all_ok &= run_per_step_experiment(
            "GradCos (cosine)", normalized=True, tmp=root / "gradcos_steps"
        )
    print("\n" + ("ALL VARIANTS PASS ✅" if all_ok else "SOME VARIANTS FAILED ❌"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
