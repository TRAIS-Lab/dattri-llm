"""SOURCE (Bergson's approximate unrolling) on the harness GPT-2 trajectory.

Our training loop is rerun once, saving HF checkpoints at evenly spaced
steps; Bergson then fits EK-FAC at each checkpoint, aggregates them per
segment, walks the validation gradients backwards through the segments, and
scores every training block against every validation point at the final
checkpoint.  Steps 1-4 (the factors) are query-independent; step 5 is
replicated here with ``aggregation="none"`` so all validation points are
scored at once instead of a single mean query.

    python utils/source.py --setting gpt2 --lr 1e-5 --seed 0 --n-queries 64 \
        --truth-dir results/gpt2/lr1e-05_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time
from copy import deepcopy

import numpy as np
import torch
from datasets import Dataset

# Bergson passes ``feature=`` to ``Dataset.add_column``, which the installed
# datasets (2.21) does not accept: add the column, then cast it.
_add_column = Dataset.add_column


def _add_column_compat(self, name, column, new_fingerprint=None, feature=None):
    out = _add_column(self, name, column, new_fingerprint=new_fingerprint)
    return out.cast_column(name, feature) if feature is not None else out


Dataset.add_column = _add_column_compat

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
from bergson.build import build
from bergson.cli.commands import Build
from bergson.config import (
    ApproxUnrollingConfig,
    DataConfig,
    HessianConfig,
    IndexConfig,
    PreprocessConfig,
)
from bergson.config.config_io import save_run_config
from bergson.data import load_scores
from protocol import make_batches, occurrences, peak_gb, run_trajectory, spearman_per_column
from settings import build, parser


def save_checkpoints(s, batches, out: pathlib.Path, every: int) -> list[pathlib.Path]:
    """Rerun the trajectory, saving an HF checkpoint after every *every* steps."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    paths: list[pathlib.Path] = []
    holder: dict = {}

    def after_step(t: int) -> None:
        if (t + 1) % every == 0:
            p = out / f"checkpoint-{t + 1}"
            holder["model"].save_pretrained(p, safe_serialization=True)
            paths.append(p)

    # run_trajectory builds the model internally; capture it through a hook
    # on the setting's train_step.
    orig_step = s.train_step

    def train_step(model, idx):
        holder["model"] = model
        return orig_step(model, idx)

    s.train_step = train_step
    run_trajectory(s, batches, after_step=after_step)
    s.train_step = orig_step
    return paths


def token_dataset(x: torch.Tensor, path: pathlib.Path) -> str:
    ids = x.cpu().tolist()
    ds = Dataset.from_dict({"input_ids": ids, "length": [len(r) for r in ids]})
    shutil.rmtree(path, ignore_errors=True)
    ds.save_to_disk(str(path))
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--ckpts-per-segment", type=int, default=2)
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    a, rest = ap.parse_known_args()
    s = build(parser().parse_args(rest))  # --setting gpt2 / --lr / --seed / --tag
    if s.name != "gpt2":
        raise SystemExit("Bergson's pipeline takes HF causal LMs only; use source_mlp.py for the MLP")
    lr = s.extra["lr"]
    out = s.out_dir.parent / f"{s.out_dir.name}_source"
    out.mkdir(parents=True, exist_ok=True)
    # Bergson's factors, indexes and memory-mapped scores go to a fast local
    # directory when SOURCE_WORK_DIR is set (they crawl on a network filesystem).
    work = pathlib.Path(os.environ.get("SOURCE_WORK_DIR", str(out)))
    work.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    n_steps = len(batches)
    n_ckpts = a.segments * a.ckpts_per_segment
    assert n_steps % n_ckpts == 0, (n_steps, n_ckpts)
    every = n_steps // n_ckpts
    log(
        f"== SOURCE lr={lr} seed={s.seed} {n_ckpts} checkpoints every {every} steps, {a.segments} segments"
    )
    t0 = time.time()
    ckpts = save_checkpoints(s, batches, work / "ckpts", every)
    log(f"  trajectory + checkpoints: {time.time() - t0:.0f}s, {peak_gb()}")
    torch.cuda.reset_peak_memory_stats()
    train_path = token_dataset(s.data["x_tr"], work / "train_ds")
    query_path = token_dataset(s.data["x_va"][: a.n_queries], work / "query_ds")

    # Per-segment mean LR and step count of our schedule.
    per_seg = n_steps // a.segments
    lrs = [lr * s.lr_factor(t, n_steps) for t in range(n_steps)]
    lr_list = [
        float(np.mean(lrs[l * per_seg : (l + 1) * per_seg])) for l in range(a.segments)
    ]
    step_size_list = [per_seg] * a.segments

    run_path = work / "run"
    shutil.rmtree(run_path, ignore_errors=True)
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model="gpt2",
        tokenizer="gpt2",
        precision="fp32",
        projection_dim=0,
        token_batch_size=1024,
        data=DataConfig(dataset=train_path, split="train", truncation=True),
    )
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True, hessian_dtype="fp32")
    au_cfg = ApproxUnrollingConfig(
        checkpoints=[str(p) for p in ckpts],
        segments=a.segments,
        lr_list=lr_list,
        step_size_list=step_size_list,
        query=DataConfig(dataset=query_path, split="train", truncation=True),
    )
    lr_times_steps = [lr * k for lr, k in zip(lr_list, step_size_list, strict=True)]
    log(f"  lr x steps per segment: {lr_times_steps}")

    t0 = time.time()
    precompute_checkpoint_hessians(index_cfg, hessian_cfg, au_cfg, overwrite=True)
    aggregate_segment_covariances(
        run_path=str(run_path),
        method="kfac",
        n_segments=a.segments,
        per_segment=a.ckpts_per_segment,
        distributed=index_cfg.distributed,
        resume=False,
    )
    precompute_checkpoint_averaged_lambdas(index_cfg, hessian_cfg, au_cfg, resume=False)
    aggregate_segment_lambdas(
        run_path=str(run_path),
        method="kfac",
        n_segments=a.segments,
        per_segment=a.ckpts_per_segment,
        distributed=index_cfg.distributed,
        resume=False,
    )
    log(f"  factors (steps 1-4): {time.time() - t0:.0f}s, {peak_gb()}")
    torch.cuda.reset_peak_memory_stats()

    # Step 5 with per-row query gradients at the final checkpoint.
    t0 = time.time()
    query_cfg = deepcopy(index_cfg)
    query_cfg.model = str(ckpts[-1])
    query_cfg.data = au_cfg.query
    query_cfg.run_path = str(run_path / "query")
    query_cfg.projection_dim = 0
    query_pre = PreprocessConfig(aggregation="none")
    save_run_config(Build(query_cfg, query_pre, None), query_cfg.partial_run_path)
    build(query_cfg, query_pre)
    q1 = walk_query_phase1(str(run_path), "kfac", lr_times_steps, index_cfg.distributed)
    q2 = walk_query_phase2(
        str(run_path), "kfac", lr_times_steps, q1, index_cfg.distributed
    )
    score_per_segment_and_aggregate(index_cfg, q2, final_checkpoint=str(ckpts[-1]))
    log(f"  queries + walk + scoring (steps 5-8): {time.time() - t0:.0f}s, {peak_gb()}")
    # Bergson's own sum keeps score column 0 only; sum every query column.
    scores = None
    for l in range(a.segments):
        part = torch.as_tensor(
            np.array(load_scores(run_path / f"segment_{l}" / "scores")[:])
        ).float()
        scores = part if scores is None else scores + part
    assert scores is not None
    torch.save({"scores": scores}, out / "source_scores.pt")
    log(f"  scores shape {tuple(scores.shape)}")

    result = {
        "name": s.name,
        "seed": s.seed,
        "lr": lr,
        "n_queries": a.n_queries,
        "method": "source",
    }
    if a.truth_dir:
        ref = torch.load(a.truth_dir + "/matrices.pt")
        pairs = ref.get("pairs") or [(int(i), None) for i in ref["selected"].tolist()]
        n_q = min(a.n_queries, scores.shape[1])
        truth = ref["tsloo"][:, :n_q]
        # SOURCE scores the whole run per training example (its influence,
        # positive = removal raises the loss); per-pair rows share it.
        pred = torch.stack([scores[i, :n_q] for i, _ in pairs])
        rho = spearman_per_column(pred, truth)
        result["source"] = float(rho.mean())
        if all(t is not None for _, t in pairs):
            rank = {(i, t): occurrences(batches, i).index(t) for i, t in pairs}
            for e in sorted(set(rank.values())):
                sel = [k for k, pr in enumerate(pairs) if rank[pr] == e]
                result[f"source_epoch{e}"] = float(
                    spearman_per_column(pred[sel], truth[sel]).mean()
                )
        torch.save({"pred": pred, "pairs": pairs}, out / "matrices.pt")
    log(f"  result: {json.dumps(result)}")
    with (out / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    for d in (run_path, work / "ckpts", work / "train_ds", work / "query_ds"):
        shutil.rmtree(d, ignore_errors=True)  # factors, indexes, checkpoints


if __name__ == "__main__":
    main()
