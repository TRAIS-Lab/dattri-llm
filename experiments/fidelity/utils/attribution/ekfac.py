"""dattri-llm's EK-FAC on the setting's final model.

The trajectory is trained once; ``EKFACAttributor.attribute`` then captures
the training and query gradients live (every linear layer of the
transformer blocks, factorized), fits the K-FAC eigenbases and the
corrected spectrum, and scores every training block against every query.
Damping is 0.1 of each layer's mean corrected eigenvalue.

``--projection none`` (the default, run directory ``_ekfac``) works at full
dimension.  With ``--query-chunk 0`` the query blocks are streamed again
for every training block (``loop_over_test``) and ``--eval-batch`` of them
are resident at a time; with ``--query-chunk N`` the queries are scored N
at a time, each chunk's preconditioned gradients held on the device for one
pass over the training set, and the fit of the first chunk, written to a
``fisher_dir``, is reused.  ``--projection 64`` (``_ekfac_k64``) captures both sides through a
rank-64 LoGra projection.  ``--factor-cache-residency`` holds the fitted factors
in a cache of that residency, one layer on the device at a time.
``--final-model`` scores a trained model saved as a Hugging Face checkpoint
(``utils/attribution/olmo.py export``) instead of training the setting's trajectory.

The timed phases are ``train`` (the trajectory) and ``attribute`` (the
capture, the fit and the scoring).  ``result.json`` holds the setting, the
options, the timing of ``common.Run`` and, with ``--truth-dir``, the
Spearman correlation under ``ekfac`` or ``ekfac_k64``.

    python utils/attribution/ekfac.py --scale 0.5b --lr 1e-5 --seed 0 --n-queries 64 --truth-dir results/qwen0.5b/lr1e-05_seed0
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
from pathlib import Path

HERE = pathlib.Path(__file__).resolve().parent
sys.path[: 1] = [str(HERE.parent)]  # protocol, settings, common

import torch  # noqa: E402
from torch.utils.data import Dataset

from common import Run, batches_of, fidelity
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.task import AttributionTask
from dattri_llm.utils.hashing import hash_sample
from protocol import run_trajectory
from settings import build, parser

DAMPING = 0.1  # of the mean corrected eigenvalue, per layer


class Blocks(Dataset):
    """Token blocks as ``{"input_ids": block}`` samples."""

    def __init__(self, x: torch.Tensor) -> None:
        self.x = x

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.x[i]}


def linear_layers(model) -> list[str]:
    """Every linear layer of the transformer blocks (not the embedding or head);
    GPT-2's are ``transformers.pytorch_utils.Conv1D``."""
    from transformers.pytorch_utils import Conv1D

    return [f"{n}$" for n, m in model.named_modules()
            if isinstance(m, (torch.nn.Linear, Conv1D)) and "lm_head" not in n and "embed" not in n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--projection", default="none", choices=["none", "64"])
    ap.add_argument("--batch", type=int, default=8, help="training blocks per capture step (speed only)")
    ap.add_argument("--eval-batch", type=int, default=8, help="queries per capture step (memory only)")
    ap.add_argument("--query-chunk", type=int, default=0,
                    help="full dimension: score the queries in chunks of this many, each chunk's "
                         "preconditioned gradients computed once and held on the device for one pass "
                         "over the training set (the factors are fitted once).  0 (default): hold "
                         "only --eval-batch queries and rebuild them for every training block")
    ap.add_argument("--factor-cache-residency", default=None, choices=["memory", "tiered", "disk"],
                    help="hold the fitted factors in a cache of this residency, one layer on the device at a time")
    ap.add_argument("--truth-dir", default=None, help="run directory with matrices.pt")
    ap.add_argument("--final-model", default=None,
                    help="Hugging Face checkpoint of the trained model (utils/attribution/olmo.py export); "
                         "default: train the setting's trajectory here")
    a, rest = ap.parse_known_args()
    s = build(parser().parse_args(rest))
    projected = a.projection != "none"
    run = Run(s, "ekfac_k64" if projected else "ekfac")
    batches = batches_of(s)
    run.log(f"== EK-FAC (dattri-llm, projection={a.projection or 'none'}) {s.name} lr={s.extra['lr']} seed={s.seed} queries={a.n_queries}")
    run.start("train")
    if a.final_model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            a.final_model, torch_dtype=torch.float32, attn_implementation="eager").to(s.device)
        run.end(f"trained model {a.final_model}")
    else:
        model, _ = run_trajectory(s, batches)
        run.end(f"{len(batches)} steps")
    x_tr, x_va = s.data["x_tr"], s.data["x_va"][: a.n_queries]

    def loss(model, batch):
        logits = model(input_ids=batch["input_ids"]).logits[:, :-1].float()
        ids = batch["input_ids"]
        tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        return tok.view(ids.shape[0], -1).mean(1).sum()  # per-block mean-token loss, summed

    task = AttributionTask(loss, model)  # the trained model's current parameters
    args = AttributionArguments(output_dir=tempfile.mkdtemp(prefix="ekfac_"),
                                per_device_train_batch_size=a.batch,
                                per_device_eval_batch_size=min(a.n_queries, a.eval_batch),
                                dataloader_pin_memory=False)
    proj = None if not projected else {"__default__": {
        "style": "logra", "proj_dim": 64, "proj_max_batch_size": 32, "proj_type": "rademacher", "proj_seed": 0}}
    # The invasive hook is for nn.Linear; GPT-2's Conv1D layers take the plain one.
    hook = "linear_io" if s.name == "gpt2" else "invasive_linear_io"
    config = HookManagerConfig(**{hook: linear_layers(model)}, projection_kwargs=proj,
                               capture_style="factorized")
    run.start("attribute")
    attributor = EKFACAttributor(args, task=task)

    def score_queries(x_q: torch.Tensor, loop_over_test: bool, fisher_dir: str | None = None) -> torch.Tensor:
        """``(n_train, len(x_q))`` scores, rows in training order, columns in query order."""
        score = attributor.attribute(Blocks(x_tr), Blocks(x_q), hook_config=config,
                                     damping=DAMPING, relative_damping=True,
                                     factor_cache_residency=a.factor_cache_residency,
                                     fisher_dir=fisher_dir, loop_over_test=loop_over_test)
        ids, matrix = score.agnostic_matrix()
        pos = {h: i for i, h in enumerate(ids)}  # rows back into training order by content hash
        order = [pos[hash_sample({"input_ids": x_tr[i]})] for i in range(x_tr.shape[0])]
        cols = [score.test_ids.index(hash_sample({"input_ids": x_q[q]})) for q in range(x_q.shape[0])]
        return matrix.cpu().float()[order][:, cols]

    if projected or not a.query_chunk:
        scores = score_queries(x_va, loop_over_test=not projected)
    else:  # the first chunk's fit is written to ``fisher_dir`` and reused for the others
        fisher_dir = str(Path(args.output_dir) / "fisher")
        scores = torch.cat([score_queries(x_va[i : i + a.query_chunk], loop_over_test=False, fisher_dir=fisher_dir)
                            for i in range(0, x_va.shape[0], a.query_chunk)], dim=1)
    run.end(f"scores {tuple(scores.shape)}")
    shutil.rmtree(args.output_dir, ignore_errors=True)  # the persisted fit
    result = {"name": s.name, "seed": s.seed, "lr": s.extra["lr"], "model": s.extra["model"],
              "method": "ekfac_k64" if projected else "ekfac", "n_queries": a.n_queries, "damping": DAMPING,
              "projection": a.projection, "query_chunk": a.query_chunk,
              "factor_cache_residency": a.factor_cache_residency}
    result.update(fidelity(s, batches, scores, a.truth_dir, run, result["method"], a.n_queries))
    run.finish(result)


if __name__ == "__main__":
    main()
