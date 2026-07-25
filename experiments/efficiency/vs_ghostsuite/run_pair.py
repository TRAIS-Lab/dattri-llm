"""ours-vs-GhostSuite: online train-vs-val gradient dot products during
training (In-Run Data Shapley setting, GhostSuite's ``graddotprod_lm``
example).

Every side trains the same GPT2-Small (124M, their own model class, seq 1024)
on the same seeded synthetic token stream, batch 16, fixed val batch 8, fp32,
AdamW, grad-clip 1.0, 30 steps.  The scored sides additionally compute, each
step, the dot product between every train sample's gradient and the val-batch
gradient (at the current weights) before updating on the full batch.

    python run_pair.py --side {ghost-regular,ghost-dotprod,ghost-dotprod-eager,
                               ours-baseline,ours-callback}
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py
GS = "/scratch/shixuanl/GhostSuite"
sys.path.insert(0, GS)
sys.path.insert(0, f"{GS}/examples/lm")
sys.path.insert(0, f"{GS}/examples/lm/graddotprod_lm")

import torch

from common import Meter, env_info

import os

MAX_STEPS = int(os.environ.get("PAIR_MAX_STEPS", "30"))
BATCH = 16
VAL_BATCH = 8
SETTING = "gpt2small-synthetic-online-graddot"


def build_their_stack(method: str, decoupled: bool):
    """Their example's own setup path (config, model, optimizer, data fns)."""
    from config_file import TrainingConfig, parse_arguments
    from shared.model_setup import setup_model_and_optimizer
    from shared.training_utils import (
        load_dataset_main,
        setup_data_functions,
        setup_distributed,
        setup_torch_backend,
    )
    from shared.utils import set_seed

    argv = [
        "--method", method, "--architecture", "GPT2-Small",
        "--train_set", "synthetic", "--val_set", "synthetic",
        "--batch_size", str(BATCH), "--val_batch_size", str(VAL_BATCH),
        "--max_steps", str(MAX_STEPS), "--seed", "42",
        "--model_dtype", "float32", "--train_dtype", "float32",
    ]
    if not decoupled:
        argv.append("--eager")
    old_argv = sys.argv
    sys.argv = ["run_pair.py", *argv]
    try:
        args = parse_arguments()
    finally:
        sys.argv = old_argv
    config = TrainingConfig(args)
    ddp_info = setup_distributed()
    set_seed(config.seed + ddp_info["seed_offset"])
    model, optimizer, scaler = setup_model_and_optimizer(
        config, ddp_info["device"], ddp_info)
    ctx = setup_torch_backend(config, model=model)
    dataset = load_dataset_main("synthetic", "synthetic")
    get_batch, get_val_batch = setup_data_functions(
        dataset, config, ddp_info["device"], ddp_info)
    return config, ddp_info, model, optimizer, scaler, ctx, get_batch, get_val_batch


def run_ghostsuite(meter: Meter, method: str, decoupled: bool):
    from ghostEngines import GhostEngineManager, NoSelection, UpdateAll
    from ghostEngines.selection_driver import online_selection_step
    from shared.training_utils import to_device

    (config, ddp_info, model, optimizer, scaler, ctx,
     get_batch, get_val_batch) = build_their_stack(method, decoupled)

    val_data = None
    if method == "GradDotProd":
        X_val, Y_val = get_val_batch(VAL_BATCH, return_idx=False)
        val_data = (to_device(X_val, ddp_info["device"]),
                    to_device(Y_val, ddp_info["device"]))
    manager = GhostEngineManager(config=config, model=model,
                                 optimizer=optimizer, ddp_info=ddp_info,
                                 val_data=val_data)
    policy = UpdateAll() if method == "GradDotProd" else NoSelection()
    forward_fn = lambda m, X, Y: m(X, Y).loss  # noqa: E731

    def draw(_micro):
        return get_batch("train", batch_size=BATCH, return_idx=True)

    all_scores, step_times = [], []
    with meter.phase("train", MAX_STEPS * BATCH, unit="train-samples"):
        for it in range(MAX_STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scores, _idx, _loss = online_selection_step(
                manager=manager, model=model, optimizer=optimizer,
                scaler=scaler, ctx=ctx, forward_fn=forward_fn, policy=policy,
                iter_num=it, grad_clip=config.grad_clip, grad_accum=1,
                draw_microbatch=draw, ddp=ddp_info["ddp"],
            )
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            if scores is not None:
                all_scores.append(scores.detach().float().cpu())
    return all_scores, step_times


def run_dattri_llm(meter: Meter, with_callback: bool):
    from dattri_llm.gradient.callbacks import DataSelectionCallback
    from dattri_llm.gradient.hooks import HookManager
    from shared.training_utils import to_device

    class GpuScoringCallback(DataSelectionCallback):
        """Score the cross-grams on the GPU.

        The HookManager's unprojected capture path buffers (a, g) on the CPU
        (tuned for the disk-offload workflow), so the stock callback's ghost
        grams run on CPU -- ~50s/step at this scale.  Moving the step gradient
        (and target) back to the GPU for scoring keeps the math identical.
        """

        def _compute_scores(self, record, target=None):
            gradient = record.gradient.to("cuda")
            other = gradient if target is None else target.to("cuda")
            per_layer = gradient.similarity(other, mode="factorized")
            B = gradient.batch_size
            scores = torch.zeros(B, device="cuda")
            for matrix in per_layer.values():
                layer_scores = matrix.sum(1)
                if layer_scores.shape[0] < B:
                    layer_scores = layer_scores.expand(B)
                scores += layer_scores
            return scores.cpu()

    (config, ddp_info, model, optimizer, _scaler, _ctx,
     get_batch, get_val_batch) = build_their_stack("Regular", True)

    all_scores, step_times = [], []

    def loop():
        for _it in range(MAX_STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            X, Y, _idx = get_batch("train", batch_size=BATCH, return_idx=True)
            loss = model(X, Y).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # callback scores + (no-op) filters here
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

    if not with_callback:
        with meter.phase("train", MAX_STEPS * BATCH, unit="train-samples"):
            loop()
        return all_scores, step_times

    X_val, Y_val = get_val_batch(VAL_BATCH, return_idx=False)
    X_val = to_device(X_val, ddp_info["device"])
    Y_val = to_device(Y_val, ddp_info["device"])

    def val_loss_fn(m, batch):
        Xv, Yv = batch
        return m(Xv, Yv).loss

    callback = GpuScoringCallback(
        model,
        threshold=0.0, threshold_mode="bottom_fraction",  # score, drop nothing
        score_mode="ghost",
        target="val_loader",
        val_loader=itertools.repeat((X_val, Y_val)),
        val_loss_fn=val_loss_fn,
    )
    hookmanager = HookManager(model, callbacks=[callback])

    def loop_scored():
        for _it in range(MAX_STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            X, Y, _idx = get_batch("train", batch_size=BATCH, return_idx=True)
            loss = model(X, Y).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            if callback.last_scores is not None:
                all_scores.append(callback.last_scores.detach().float().cpu())

    with meter.phase("train", MAX_STEPS * BATCH, unit="train-samples"):
        with hookmanager.collect():
            loop_scored()
    return all_scores, step_times


SIDES = {
    "ghost-regular": ("ghostsuite(regular)",
                      lambda m: run_ghostsuite(m, "Regular", True)),
    "ghost-dotprod": ("ghostsuite(graddotprod)",
                      lambda m: run_ghostsuite(m, "GradDotProd", True)),
    "ghost-dotprod-eager": ("ghostsuite(graddotprod,eager)",
                            lambda m: run_ghostsuite(m, "GradDotProd", False)),
    "ours-baseline": ("dattri_llm(baseline)", lambda m: run_dattri_llm(m, False)),
    "ours-callback": ("dattri_llm(ghost-callback)",
                      lambda m: run_dattri_llm(m, True)),
}


def main(a):
    lib, runner = SIDES[a.side]
    meter = Meter(lib, SETTING, HERE / "results.jsonl", meta=env_info())
    scores, step_times = runner(meter)
    ts = torch.tensor(step_times[1:])  # drop step 0 (warmup / compile)
    meter.note(
        n_steps=MAX_STEPS,
        step_time_first_s=round(step_times[0], 3),
        step_time_median_s=round(float(ts.median()), 4),
        step_time_p90_s=round(float(ts.quantile(0.9)), 4),
    )
    if scores:
        torch.save(torch.stack(scores), HERE / f"scores_{a.side}.pt")
    meter.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=list(SIDES), required=True)
    main(p.parse_args())
