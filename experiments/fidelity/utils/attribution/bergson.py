"""Bergson's single-checkpoint methods on the setting's final model.

``--method ekfac``: the four steps of Bergson's EK-FAC influence pipeline
(K-FAC factors with the eigenvalue correction, fp32, full dimension;
damped inverse with damping 0.1 of the mean eigenvalue), with one query
gradient per validation block (``aggregation="none"``) and the queries
processed ``--query-chunk`` at a time.  ``--method trackstar``: Bergson's
``trackstar`` pipeline (per-module random projection to
``--projection-dim``, autocorrelation Hessians of the training and query
gradients mixed and applied at scoring), also with one query gradient per
validation block.

The trajectory is trained once and saved as an HF checkpoint.  The timed
phases are ``train`` (the trajectory and the checkpoint save) and
``attribute`` (the Bergson pipeline); writing the token datasets is outside
them.  ``result.json`` holds the setting, the options, the timing of
``common.Run`` and, with ``--truth-dir``, the Spearman correlation under
``bergson_ekfac`` or ``trackstar``.  ``BERGSON_WORK_DIR``, when set, is the
directory for the checkpoint, datasets and Bergson's run files (default:
the run directory); they are removed at the end.  ``--final-model`` takes a trained model saved as a
Hugging Face checkpoint (``utils/attribution/olmo.py export``) instead of training the
setting's trajectory.

    python utils/attribution/bergson.py --method ekfac --scale 0.5b --lr 1e-5 --seed 0 --n-queries 64 --truth-dir results/qwen0.5b/lr1e-05_seed0
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

import torch  # noqa: E402
from bergson.build import build as bergson_build
from bergson.cli.commands import Build, Score
from bergson.config import (
    DataConfig,
    HessianConfig,
    IndexConfig,
    InversionConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config import TrackstarIndexConfig
from bergson.config.config_io import save_run_config
from common import (
    Run,
    batches_of,
    bergson_scores,
    fidelity,
    final_checkpoint,
    require_bergson,
    token_dataset,
)
from settings import build, parser

DAMPING = 0.1  # of the mean eigenvalue (Bergson's InversionConfig.damping_factor)


def ekfac(index_cfg: IndexConfig, run_path: pathlib.Path, queries: list[str]) -> torch.Tensor:
    """Bergson's four-step EK-FAC pipeline with per-row query gradients.

    ``approximate_hessians`` fits the factors once.  *queries* is one on-disk
    dataset per chunk; for each, ``bergson.build`` computes the query
    gradients, the ``apply_hessian`` worker preconditions them with the damped
    inverse, and ``score_dataset`` scores the training set against them
    (``higher_is_better=True``).  The chunk's query and preconditioned indexes
    are removed after scoring.  Returns the ``(n_train, n_queries)`` scores.
    """
    from bergson.distributed import launch_distributed_run
    from bergson.hessians.apply_hessian import EkfacConfig, apply_worker
    from bergson.hessians.hessian_approximations import approximate_hessians
    from bergson.score.score import score_dataset
    from bergson.utils.worker_utils import validate_run_path

    hessian_path = run_path / "hessian"
    hess_cfg = deepcopy(index_cfg)
    hess_cfg.run_path = str(hessian_path / "kfac")
    validate_run_path(hess_cfg)
    approximate_hessians(hess_cfg, HessianConfig(method="kfac", ev_correction=True, hessian_dtype="fp32"))
    inversion = InversionConfig(inversion="damped_inverse", damping_factor=DAMPING)
    pre = PreprocessConfig(aggregation="none")
    parts = []
    for c, query_ds in enumerate(queries):
        query_path, transformed, scores = run_path / f"query{c}", run_path / f"kfac_query{c}", run_path / f"scores{c}"
        query_cfg = deepcopy(index_cfg)
        query_cfg.run_path, query_cfg.projection_dim = str(query_path), 0
        query_cfg.data = DataConfig(dataset=query_ds, split="train", truncation=True)
        validate_run_path(query_cfg)
        save_run_config(Build(query_cfg, pre), query_cfg.partial_run_path)
        bergson_build(query_cfg, pre)
        launch_distributed_run(
            "apply_hessian", apply_worker,
            [EkfacConfig(hessian_method_path=str(hessian_path / "kfac"), gradient_path=str(query_path),
                         run_path=str(transformed), ev_correction=True), inversion],
            index_cfg.distributed,
        )
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path, score_index_cfg.projection_dim = str(scores), 0
        score_cfg = ScoreConfig(query_path=str(transformed), higher_is_better=True)
        validate_run_path(score_index_cfg)
        save_run_config(Score(score_cfg, score_index_cfg, pre), score_index_cfg.partial_run_path)
        score_dataset(score_index_cfg, score_cfg, pre)
        parts.append(bergson_scores(scores))
        for d in (query_path, transformed):
            shutil.rmtree(d, ignore_errors=True)
    return torch.cat(parts, dim=1)


def trackstar(index_cfg: IndexConfig, run_path: pathlib.Path, query_ds: str, proj_dim: int) -> pathlib.Path:
    """Bergson's ``trackstar`` pipeline at projection *proj_dim*; returns its ``scores`` directory."""
    from bergson.cli.trackstar import trackstar as pipeline
    from bergson.config import TrackstarConfig

    index_cfg = deepcopy(index_cfg)
    index_cfg.run_path, index_cfg.projection_dim = str(run_path), proj_dim
    cfg = TrackstarConfig(query=DataConfig(dataset=query_ds, split="train", truncation=True),
                          preprocess_cfg=PreprocessConfig(aggregation="none"),
                          score_cfg=ScoreConfig())
    pipeline(index_cfg, cfg)
    return run_path / "scores"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ekfac", "trackstar"], required=True)
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--projection-dim", type=int, default=64, help="trackstar: per-module projection")
    ap.add_argument("--query-chunk", type=int, default=64, help="ekfac: queries built, preconditioned and scored together")
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    ap.add_argument("--final-model", default=None,
                    help="Hugging Face checkpoint of the trained model (utils/olmocore.py); "
                         "default: train the setting's trajectory here")
    a, rest = ap.parse_known_args()
    require_bergson()
    s = build(parser().parse_args(rest))
    method = "bergson_ekfac" if a.method == "ekfac" else "trackstar"
    run = Run(s, method)
    work = pathlib.Path(os.environ.get("BERGSON_WORK_DIR", str(run.out)))
    work.mkdir(parents=True, exist_ok=True)
    batches = batches_of(s)
    run.log(f"== Bergson {a.method} {s.name} lr={s.extra['lr']} seed={s.seed} queries={a.n_queries}")

    final = pathlib.Path(a.final_model) if a.final_model else work / "final"
    run.start("train")
    if not a.final_model:
        final_checkpoint(s, batches, final)
    run.end(f"{len(batches)} steps" if not a.final_model else f"trained model {final}")
    train_ds = token_dataset(s.data["x_tr"], work / "train_ds")
    query_ds = token_dataset(s.data["x_va"][: a.n_queries], work / "query_ds")
    chunks = [token_dataset(s.data["x_va"][i : i + a.query_chunk], work / f"query_ds{i}")
              for i in range(0, a.n_queries, a.query_chunk)]
    run_path = work / "run"
    shutil.rmtree(run_path, ignore_errors=True)
    index_cfg = (IndexConfig if a.method == "ekfac" else TrackstarIndexConfig)(
        run_path=str(run_path), model=str(final), tokenizer=s.extra["model"],
        precision="fp32", projection_dim=0, token_batch_size=1024,
        data=DataConfig(dataset=train_ds, split="train", truncation=True),
    )
    run.start("attribute")
    if a.method == "ekfac":
        scores = ekfac(index_cfg, run_path, chunks)
    else:
        scores = bergson_scores(trackstar(index_cfg, run_path, query_ds, a.projection_dim))
    run.end(f"scores {tuple(scores.shape)}")
    result = {"name": s.name, "seed": s.seed, "lr": s.extra["lr"], "model": s.extra["model"],
              "method": method, "n_queries": a.n_queries,
              **({"projection_dim": a.projection_dim} if a.method == "trackstar"
                 else {"damping": DAMPING, "query_chunk": a.query_chunk})}
    result.update(fidelity(s, batches, scores, a.truth_dir, run, method, a.n_queries))
    run.finish(result)
    for d in (run_path, work / "final", work / "train_ds", work / "query_ds", *chunks):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
