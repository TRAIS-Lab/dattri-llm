"""Ours vs dattri: mnist/mlp Grad-Dot efficiency (dattri's benchmark protocol).

Both sides compute the identical 5000x500 Grad-Dot score matrix at dattri's
first mnist/mlp checkpoint, batch 500 (dattri's own loader setting), fp32.

    python run_pair.py --side ours   --gpu 0
    python run_pair.py --side dattri --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))              # common.py
sys.path.insert(0, str(HERE.parent.parent / "benchmark"))  # _common.py bridge

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from common import Meter, env_info

SETTING = "mnist-mlp-graddot-5000x500"
N_TRAIN, N_TEST, BATCH = 5000, 500, 500


def load():
    from dattri.benchmark.load import load_benchmark
    md, _ = load_benchmark(model="mlp", dataset="mnist", metric="lds")
    model = md["model"]
    model.load_state_dict(torch.load(md["models_full"][0], map_location="cuda"))
    return md, model.cuda().eval()


def run_dattri(meter: Meter, projector_device: str) -> torch.Tensor:
    """dattri TracIn (Grad-Dot).  NOTE: dattri's TracInAttributor *always*
    random-projects gradients (DEFAULT_PROJECTOR_KWARGS, proj_dim=512, on CPU
    by default); ``projector_kwargs=None`` cannot disable it.  So dattri's
    score is a JL sketch of Grad-Dot.  ``projector_device`` picks their
    out-of-the-box CPU BasicProjector vs their best-case CudaProjector.
    """
    from dattri.algorithm.tracin import TracInAttributor
    from dattri.task import AttributionTask
    md, model = load()

    def loss_tracin(params, data_target_pair):
        image, label = data_target_pair
        yhat = torch.func.functional_call(model, params, image.unsqueeze(0))
        return nn.CrossEntropyLoss()(yhat, label.unsqueeze(0).long())

    task = AttributionTask(model=model, loss_func=loss_tracin,
                           checkpoints=md["models_full"][0])
    train_loader = DataLoader(Subset(md["train_dataset"], range(N_TRAIN)),
                              shuffle=False, batch_size=BATCH)
    test_loader = DataLoader(Subset(md["test_dataset"], range(N_TEST)),
                             shuffle=False, batch_size=BATCH)
    attributor = TracInAttributor(task=task, weight_list=torch.ones(1) * 1e-3,
                                  normalized_grad=False, device="cuda",
                                  projector_kwargs={"device": projector_device})
    with meter.phase("attribute", N_TRAIN + N_TEST):
        with torch.no_grad():
            score = attributor.attribute(train_loader, test_loader)
    return score.cpu().float() / 1e-3          # undo dattri's ensemble weight


def run_dattri_llm(meter: Meter) -> torch.Tensor:
    import _common as C
    from dattri.task import AttributionTask
    from torch.func import functional_call
    from dattri_llm.attribution.algorithm.tracin import TracInAttributor
    from dattri_llm.attribution.arguments import AttributionArguments

    md, model = load()

    def gl(p, b):
        x, y = b
        return F.cross_entropy(functional_call(model, p, (x,)), y,
                               reduction="sum")

    task = AttributionTask(loss_func=gl, model=model,
                           checkpoints=[md["models_full"][0]], target_func=gl)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="eff_dattri_"),
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    train_sub = Subset(md["train_dataset"], range(N_TRAIN))
    test_sub = Subset(md["test_dataset"], range(N_TEST))
    th = C.hashes_for(md["train_dataset"], list(range(N_TRAIN)), "cuda")
    vh = C.hashes_for(md["test_dataset"], list(range(N_TEST)), "cuda")
    with meter.phase("attribute", N_TRAIN + N_TEST):
        score = TracInAttributor(args, task=task).attribute(
            train_sub, test_sub, enable_update=False, normalized_grad=False)
    return score.query(th, vh, trajectory="agnostic").cpu().float()


SIDES = {
    # name -> (lib label, runner)
    "ours": ("dattri_llm", lambda m: run_dattri_llm(m)),
    "dattri": ("dattri(default,cpu-proj)", lambda m: run_dattri(m, "cpu")),
    "dattri-cuda": ("dattri(cuda-proj)", lambda m: run_dattri(m, "cuda")),
}


def main(a):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    lib, runner = SIDES[a.side]
    meter = Meter(lib, SETTING, HERE / "results.jsonl", meta=env_info())
    score = runner(meter)
    torch.save(score, HERE / f"score_{a.side}.pt")
    summary = {"score_shape": list(score.shape), "score_fro": float(score.norm())}
    # Fidelity vs the exact Grad-Dot matrix.  ``score_ours.pt`` is exact: our
    # factorized capture matches plain autograd dots to float precision
    # (spot-checked elementwise); dattri's is a proj_dim=512 JL sketch.
    p_exact = HERE / "score_ours.pt"
    if p_exact.exists():
        exact = torch.load(p_exact)
        corr = torch.corrcoef(
            torch.stack([score.flatten(), exact.flatten()]))[0, 1]
        summary["pearson_vs_exact"] = round(float(corr), 6)
        print(f"FIDELITY pearson({a.side}, exact) = {corr:.6f}")
    meter.note(**summary)
    meter.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=list(SIDES), required=True)
    p.add_argument("--gpu", default="0")
    main(p.parse_args())
