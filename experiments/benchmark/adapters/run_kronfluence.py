"""kronfluence adapter for the universal benchmark (EK-FAC column).

Runs kronfluence's Analyzer (fit_all_factors strategy="ekfac" + pairwise scores)
on a universal (HF model, dataset) task, tracking every attention/MLP linear
(auto-detected, family-agnostic).  kronfluence fits *full-dimension* EK-FAC
factors (no projection) -- a different strategy from our proj-64 EK-FAC, which is
exactly the kind of strategy difference the benchmark is meant to expose.

    method -> only "ekfac".  Run in the ``dattri`` conda env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

import models
from data import load_task_data
from log import BenchRun

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

LIB = "kronfluence"


class LMTask(Task):
    """kronfluence LM task with auto-detected tracked linears."""

    def __init__(self, tracked: list[str]) -> None:
        self._tracked = tracked

    def compute_train_loss(self, batch, model, sample=False):
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        if not sample:
            labels = batch["labels"][..., 1:].contiguous()
            return F.cross_entropy(logits, labels.view(-1), reduction="sum")
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled, reduction="sum")

    def compute_measurement(self, batch, model):
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self):
        return self._tracked

    def get_attention_mask(self, batch):
        return batch["attention_mask"]


def build_model(model_id: str, params_b: float):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[models.dtype_for(params_b)]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def tracked_linears(model) -> list[str]:
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "lm_head" not in n and "embed" not in n]


def run(task: dict, out_root: Path) -> None:
    _STRATEGY = {"graddot": "identity", "kfac": "kfac", "ekfac": "ekfac"}
    if task["method"] not in _STRATEGY:
        msg = f"kronfluence adapter covers graddot/kfac/ekfac, not {task['method']!r}"
        raise ValueError(msg)
    strategy = _STRATEGY[task["method"]]
    n_train = task.get("n_train", 1024)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)

    tag = f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}-{task['method']}"
    run_dir = out_root / "runs" / f"kronfluence-{tag}"
    store = run_dir / "kf_store"
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch, "strategy": "full-dim"},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB)

    with bench.phase("build_model"):
        model, _ = build_model(task["model"], task["params_b"])
        tracked = tracked_linears(model)
        kf_task = LMTask(tracked)
        model = prepare_model(model, kf_task).cuda()
        bench.set(n_tracked_modules=len(tracked))
    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)

    analyzer = Analyzer(analysis_name=tag, model=model, task=kf_task,
                        output_dir=str(store))
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(collate_fn=default_data_collator))

    with bench.phase("fit_factors", n_train):
        analyzer.fit_all_factors(
            factors_name=strategy, dataset=train_ds, per_device_batch_size=batch,
            factor_args=FactorArguments(strategy=strategy), overwrite_output_dir=True)
    with bench.phase("pairwise_scores", n_train + n_test):
        analyzer.compute_pairwise_scores(
            scores_name=strategy, score_args=ScoreArguments(), factors_name=strategy,
            query_dataset=test_ds, train_dataset=train_ds,
            per_device_query_batch_size=n_test, per_device_train_batch_size=batch,
            overwrite_output_dir=True)

    scores = analyzer.load_pairwise_scores(strategy)["all_modules"].T.cpu().float()
    bench.record_disk("store", store)
    torch.save({"score": scores}, run_dir / "score.pt")
    bench.set(score_shape=list(scores.shape))
    bench.finish(status="ok")
    print(f"[done] kronfluence {tag}: score {tuple(scores.shape)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task")
    g.add_argument("--task-file", dest="task_file")
    ap.add_argument("--out", default=str(BENCH / "out"))
    a = ap.parse_args()
    if a.task_file:
        payload = json.loads(Path(a.task_file).read_text())
        task = payload.get("task", payload)
    else:
        task = json.loads(a.task)
    run(task, Path(a.out))


if __name__ == "__main__":
    main()
