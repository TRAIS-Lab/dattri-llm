"""SOURCE (Bergson's approximate unrolling) on the fidelity trajectory.

The trajectory is run once, saving HF checkpoints at evenly spaced steps
(``--segments`` x ``--ckpts-per-segment`` of them).  The factors are
query-independent and fitted once: ``precompute_checkpoint_hessians`` and
``precompute_checkpoint_averaged_lambdas`` fit EK-FAC (K-FAC with the
eigenvalue correction, fp32, full dimension) at each checkpoint, and
``aggregate_segment_covariances`` / ``aggregate_segment_lambdas`` aggregate
them per segment.  The queries are then processed ``--query-chunk`` at a
time with one gradient per validation block (``aggregation="none"``):
``bergson.build`` computes the chunk's gradients at the final checkpoint,
``walk_query_phase1`` / ``walk_query_phase2`` walk them backwards through
the segments, and ``score_per_segment_and_aggregate`` scores every training
block against them at each checkpoint of a segment (averaged, Bae et al.
Eq. 20).  Each segment enters with its mean learning rate times its number
of steps.

The timed phases are ``train`` (the trajectory with its checkpoint saves),
``factors`` and ``queries``; writing the token datasets is outside them.
``result.json`` holds the setting, the options, the timing of
``common.Run`` and, with ``--truth-dir``, the Spearman correlation under
``source``.  ``SOURCE_WORK_DIR``, when set, is the directory for the
checkpoints, datasets and Bergson's run files (default: the run
directory); they are removed at the end.

    python utils/attribution/source.py --scale 0.5b --lr 1e-5 --seed 0 --n-queries 64 --query-chunk 4 --truth-dir results/qwen0.5b/lr1e-05_seed0
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
from copy import deepcopy

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]  # protocol, settings, common

import numpy as np  # noqa: E402
import torch
from bergson.approx_unrolling.approx_unrolling_math import (
    score_per_segment_and_aggregate,
    walk_query_phase1,
    walk_query_phase2,
)
from bergson.approx_unrolling.precompute_checkpoints import (
    precompute_checkpoint_averaged_lambdas,
    precompute_checkpoint_hessians,
)
from bergson.approx_unrolling.segment_aggregation import (
    aggregate_segment_covariances,
    aggregate_segment_lambdas,
)
from bergson.build import build as bergson_build
from bergson.cli.commands import Build
from bergson.config import (
    ApproxUnrollingConfig,
    DataConfig,
    HessianConfig,
    IndexConfig,
    PreprocessConfig,
)
from bergson.config.config_io import save_run_config
from common import Run, batches_of, bergson_scores, fidelity, require_bergson, token_dataset
from protocol import run_trajectory
from settings import build, parser


def save_checkpoints(s, batches, out: pathlib.Path, every: int) -> list[pathlib.Path]:
    """Run the trajectory, saving an HF checkpoint after every *every* steps."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    paths: list[pathlib.Path] = []
    holder: dict = {}

    def after_step(t: int) -> None:
        if (t + 1) % every == 0:
            p = out / f"checkpoint-{t + 1}"
            holder["model"].save_pretrained(p, safe_serialization=True)
            paths.append(p)

    # The setting's train_step is wrapped for the run so that after_step can
    # reach the model that run_trajectory builds.
    orig_step = s.train_step

    def train_step(model, idx):
        holder["model"] = model
        return orig_step(model, idx)

    s.train_step = train_step
    run_trajectory(s, batches, after_step=after_step)
    s.train_step = orig_step
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--segments", type=int, default=4, help="segments of the 16-step trajectory")
    ap.add_argument("--ckpts-per-segment", type=int, default=2)
    ap.add_argument("--query-chunk", type=int, default=None,
                    help="queries walked and scored together (default: all)")
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    a, rest = ap.parse_known_args()
    require_bergson()
    s = build(parser().parse_args(rest))  # --scale / --lr / --seed / --tag
    lr = s.extra["lr"]
    run = Run(s, "source")
    out = run.out
    log = run.log
    # Checkpoints, datasets and Bergson's factors, indexes and memory-mapped
    # scores are written under SOURCE_WORK_DIR when it is set.
    work = pathlib.Path(os.environ.get("SOURCE_WORK_DIR", str(out)))
    work.mkdir(parents=True, exist_ok=True)

    batches = batches_of(s)
    n_steps = len(batches)
    segments = a.segments
    n_ckpts = segments * a.ckpts_per_segment
    assert n_steps % n_ckpts == 0, (n_steps, n_ckpts)
    every = n_steps // n_ckpts
    log(f"== SOURCE {s.name} lr={lr} seed={s.seed} {n_ckpts} checkpoints every {every} steps, {segments} segments")
    run.start("train")
    ckpts = save_checkpoints(s, batches, work / "ckpts", every)
    run.end()
    train_path = token_dataset(s.data["x_tr"], work / "train_ds")
    query_path = token_dataset(s.data["x_va"][: a.n_queries], work / "query_ds")
    chunk = a.query_chunk or a.n_queries
    chunk_paths = [token_dataset(s.data["x_va"][i : i + chunk], work / f"query_ds{i}")
                   for i in range(0, a.n_queries, chunk)]

    # Per-segment mean learning rate and step count of the setting's schedule.
    per_seg = n_steps // segments
    lrs = [lr * s.lr_factor(t, n_steps) for t in range(n_steps)]
    lr_list = [float(np.mean(lrs[l * per_seg : (l + 1) * per_seg])) for l in range(segments)]
    step_size_list = [per_seg] * segments

    run_path = work / "run"
    shutil.rmtree(run_path, ignore_errors=True)
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=s.extra["model"],
        tokenizer=s.extra["model"],
        precision="fp32",
        projection_dim=0,
        token_batch_size=1024,
        data=DataConfig(dataset=train_path, split="train", truncation=True),
    )
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True, hessian_dtype="fp32")
    au_cfg = ApproxUnrollingConfig(
        checkpoints=[str(p) for p in ckpts],
        segments=segments,
        lr_list=lr_list,
        step_size_list=step_size_list,
        query=DataConfig(dataset=query_path, split="train", truncation=True),
        query_aggregation="none",
    )
    segment_ckpts = [[str(p) for p in ckpts[l * a.ckpts_per_segment : (l + 1) * a.ckpts_per_segment]]
                     for l in range(segments)]
    lr_times_steps = [lr * k for lr, k in zip(lr_list, step_size_list, strict=True)]
    log(f"  lr x steps per segment: {lr_times_steps}")

    run.start("factors")
    precompute_checkpoint_hessians(index_cfg, hessian_cfg, au_cfg, overwrite=True)
    aggregate_segment_covariances(run_path=str(run_path), method="kfac", n_segments=segments,
                                  per_segment=a.ckpts_per_segment, distributed=index_cfg.distributed,
                                  resume=False)
    precompute_checkpoint_averaged_lambdas(index_cfg, hessian_cfg, au_cfg, resume=False)
    aggregate_segment_lambdas(run_path=str(run_path), method="kfac", n_segments=segments,
                              per_segment=a.ckpts_per_segment, distributed=index_cfg.distributed,
                              resume=False)
    run.end()

    # Per-row query gradients at the final checkpoint, walked back through the
    # segments and scored, one chunk of queries at a time.  The factors are
    # shared; the walk reads <run>/query and writes per-segment query
    # gradients and scores, which are removed after each chunk.
    run.start("queries")
    columns = []
    for chunk_path in chunk_paths:
        query_cfg = deepcopy(index_cfg)
        query_cfg.model = str(ckpts[-1])
        query_cfg.data = DataConfig(dataset=chunk_path, split="train", truncation=True)
        query_cfg.run_path = str(run_path / "query")
        query_cfg.projection_dim = 0
        shutil.rmtree(run_path / "query", ignore_errors=True)
        query_pre = PreprocessConfig(aggregation="none")
        save_run_config(Build(query_cfg, query_pre), query_cfg.partial_run_path)
        bergson_build(query_cfg, query_pre)
        q1 = walk_query_phase1(str(run_path), "kfac", lr_times_steps, index_cfg.distributed,
                               inversion_cfg=au_cfg.inversion_cfg)
        q2 = walk_query_phase2(str(run_path), "kfac", lr_times_steps, q1, index_cfg.distributed,
                               inversion_cfg=au_cfg.inversion_cfg)
        total = score_per_segment_and_aggregate(index_cfg, q2, segment_ckpts)
        # Bergson stores the summed scores with lower = more influential; flip
        # them to positive = removal raises the loss.
        columns.append(-bergson_scores(total))
        shutil.rmtree(total, ignore_errors=True)
        for l in range(segments):
            for d in ("query_grad_backward", "query_grad_segment",
                      *(f"scores_ckpt_{c}" for c in range(a.ckpts_per_segment))):
                shutil.rmtree(run_path / f"segment_{l}" / d, ignore_errors=True)
    scores = torch.cat(columns, dim=1)
    run.end(f"scores {tuple(scores.shape)}")

    result = {"name": s.name, "seed": s.seed, "lr": lr, "model": s.extra["model"], "n_queries": a.n_queries,
              "method": "source", "segments": segments, "ckpts_per_segment": a.ckpts_per_segment,
              "query_chunk": chunk}
    result.update(fidelity(s, batches, scores, a.truth_dir, run, "source", a.n_queries))
    run.finish(result)
    for d in (run_path, work / "ckpts", work / "train_ds", work / "query_ds", *chunk_paths):
        shutil.rmtree(d, ignore_errors=True)  # factors, indexes, checkpoints, datasets


if __name__ == "__main__":
    main()
