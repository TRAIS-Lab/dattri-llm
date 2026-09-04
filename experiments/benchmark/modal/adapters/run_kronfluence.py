"""kronfluence adapter for the universal benchmark (EK-FAC column).

Runs kronfluence's Analyzer (fit_all_factors strategy="ekfac" + pairwise scores)
on a universal (HF model, dataset) task, tracking every attention/MLP linear
(auto-detected, family-agnostic).  kronfluence fits *full-dimension* EK-FAC
factors (no projection) -- a different strategy from our proj-64 EK-FAC, which is
exactly the kind of strategy difference the benchmark is meant to expose.

    method -> strategy:  graddot -> "identity", kfac -> "kfac", ekfac -> "ekfac".
    proj_mode is ignored -- kronfluence has no rank-64 mode and always fits
    full-dimension factors, so its cells belong in the full-dim column.
"""

from __future__ import annotations

import argparse
import contextlib
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

from accelerate import Accelerator

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


def build_model(model_id: str, params_b: float, dtype_override: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_name = models.dtype_for(params_b, dtype_override)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
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
    # Distributed via Accelerate, which is how kronfluence's own examples do it
    # (openwebtext/fit_factors.py): Accelerator() initializes the process group
    # and places the model, and it is a no-op at world size 1, so the same code
    # serves single-GPU and torchrun launches.
    accelerator = Accelerator()
    main = accelerator.is_main_process

    # Only rank 0 records.  Every rank writing would give N concurrent appenders
    # to one results.jsonl and N copies of every row.
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch,
                      "strategy": "full-dim",
                      "world_size": accelerator.num_processes,
                      # Accelerator() with no plugin gives DistributedDataParallel:
                      # every rank holds a FULL copy of the model. That multiplies
                      # throughput and does NOT reduce per-device memory, so this
                      # path cannot cross a single-card memory wall -- at 7.62B all
                      # four ranks OOMed at the same 139 GB the 1-GPU cell hit.
                      "distributed_mode": ("ddp" if accelerator.num_processes > 1
                                           else "single")},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB) if main else None

    def phase(name: str, units: int | None = None):
        return bench.phase(name, units) if bench else contextlib.nullcontext()

    def record(**kv) -> None:
        if bench:
            bench.set(**kv)

    with phase("build_model"):
        # The experiment's dtype, not models.dtype_for's size rule.  Without the
        # override this adapter built bf16 at >=1.0B while run_ours.py honoured
        # an fp32 experiment, so three of four scales in a cross-library ladder
        # compared our fp32 against kronfluence's bf16 -- roughly a 2x throughput
        # advantage, invisible because the record copied the task's dtype rather
        # than the one actually built.
        dtype_name = models.dtype_for(task["params_b"], task.get("dtype"))
        record(dtype=dtype_name)
        model, _ = build_model(task["model"], task["params_b"], task.get("dtype"))
        tracked = tracked_linears(model)
        kf_task = LMTask(tracked)
        model = prepare_model(model, kf_task)
        # accelerator.prepare_model, not .cuda(): under torchrun this wraps the
        # model for the active backend and pins it to this rank's device.
        model = accelerator.prepare_model(model)
        record(n_tracked_modules=len(tracked))
    with phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)

    analyzer = Analyzer(analysis_name=tag, model=model, task=kf_task,
                        output_dir=str(store))
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(collate_fn=default_data_collator))

    with phase("fit_factors", n_train):
        analyzer.fit_all_factors(
            factors_name=strategy, dataset=train_ds, per_device_batch_size=batch,
            factor_args=FactorArguments(strategy=strategy), overwrite_output_dir=True)
    with phase("pairwise_scores", n_train + n_test):
        analyzer.compute_pairwise_scores(
            scores_name=strategy, score_args=ScoreArguments(), factors_name=strategy,
            query_dataset=test_ds, train_dataset=train_ds,
            per_device_query_batch_size=n_test, per_device_train_batch_size=batch,
            overwrite_output_dir=True)

    # Every rank must reach here before rank 0 reads the merged scores.
    accelerator.wait_for_everyone()
    if not main:
        return
    scores = analyzer.load_pairwise_scores(strategy)["all_modules"].T.cpu().float()
    bench.record_disk("store", store)
    torch.save({"score": scores}, run_dir / "score.pt")
    bench.set(score_shape=list(scores.shape))
    bench.finish(status="ok")
    print(f"[done] kronfluence {tag}: score {tuple(scores.shape)} "
          f"on {accelerator.num_processes} process(es)", flush=True)


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
