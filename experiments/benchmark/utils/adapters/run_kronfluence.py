"""Kronfluence adapter for the benchmark.

Runs Kronfluence's ``Analyzer`` (``fit_all_factors``, then
``compute_pairwise_scores``) on a (HF model, dataset) task and logs through
``log.BenchRun``.  The tracked modules are every ``nn.Linear`` outside the
embedding and the LM head.  Kronfluence fits full-dimension factors: the row
records ``proj_mode="full"`` and the task's ``proj_mode`` is not read.

    method -> strategy:  graddot -> "identity",  kfac -> "kfac",  ekfac -> "ekfac"

Task fields: ``model``, ``params_b``, ``dataset``, ``method``, and optionally
``dtype``, ``n_train``, ``n_test``, ``block_size``, ``batch``, ``seed``,
``warmup_train``, ``measure_train`` and ``parallelism``.  The timed phases are
``fit_factors`` and ``pairwise_scores``.

Multi-GPU (under torchrun): ``parallelism="ddp"`` (default) replicates the
model through Accelerate's ``prepare_model``; ``"fsdp"`` shards it with
``FullyShardedDataParallel`` after Kronfluence's ``prepare_model``, with
``use_orig_params=True`` and one FSDP unit per transformer block.  Rank 0
records the row and saves the scores.

    python run_kronfluence.py --task-file plan.json --out <dir>
    torchrun --nproc_per_node=4 run_kronfluence.py --task-file plan.json --out <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

# TF32 matrix multiplies for float32 runs; every adapter sets this.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
from torch.utils.data import Subset
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

import models
from data import load_task_data
from log import BenchRun
from versions import require

from accelerate import Accelerator

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

LIB = "kronfluence"


class LMTask(Task):
    """Kronfluence ``Task`` for a causal LM: token-summed next-token loss."""

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
    # Accelerator() initializes the process group under torchrun and does
    # nothing at world size 1.
    accelerator = Accelerator()
    main = accelerator.is_main_process

    # Rank 0 records the row.
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch, "proj_mode": "full",
                      "strategy": "full-dim",
                      "world_size": accelerator.num_processes,
                      # Updated to the task's ``parallelism`` once the model
                      # is built.
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
        # The task's ``dtype`` takes precedence over models.dtype_for's size
        # rule; the row records the dtype the model is built in.
        dtype_name = models.dtype_for(task["params_b"], task.get("dtype"))
        record(dtype=dtype_name)
        model, _ = build_model(task["model"], task["params_b"], task.get("dtype"))
        tracked = tracked_linears(model)
        kf_task = LMTask(tracked)
        model = prepare_model(model, kf_task)
        parallelism = str(task.get("parallelism", "ddp")) if accelerator.num_processes > 1 else "single"
        if parallelism == "fsdp":
            import functools

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            from run_ours_fsdp import transformer_block_classes

            model = FSDP(
                model.to(accelerator.device),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_block_classes(model),
                ),
                use_orig_params=True,
                device_id=accelerator.device,
            )
        else:
            # Places the model on this rank's device and, under torchrun,
            # wraps it in DistributedDataParallel.
            model = accelerator.prepare_model(model)
        record(n_tracked_modules=len(tracked), distributed_mode=parallelism)
    with phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)

    analyzer = Analyzer(analysis_name=tag, model=model, task=kf_task,
                        output_dir=str(store))
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(collate_fn=default_data_collator))

    # Warm-up / measured split, as in run_ours.py: the first ``warmup_train``
    # samples go through the full fit + score once, untimed and under separate
    # factor/score names; the next ``measure_train`` samples are timed.
    # ``measure_train`` defaults to ``n_train - warmup_train``.
    n_warm = int(task.get("warmup_train", 0) or 0)
    n_meas = int(task.get("measure_train") or (n_train - n_warm))
    if n_warm + n_meas > n_train:
        msg = f"n_train={n_train} < warmup_train + measure_train = {n_warm + n_meas}"
        raise ValueError(msg)
    record(warmup_train=n_warm, measure_train=n_meas)

    def fit_and_score(name: str, ds):
        analyzer.fit_all_factors(
            factors_name=name, dataset=ds, per_device_batch_size=batch,
            factor_args=FactorArguments(strategy=strategy), overwrite_output_dir=True)
        analyzer.compute_pairwise_scores(
            scores_name=name, score_args=ScoreArguments(), factors_name=name,
            query_dataset=test_ds, train_dataset=ds,
            per_device_query_batch_size=n_test, per_device_train_batch_size=batch,
            overwrite_output_dir=True)

    if n_warm:
        fit_and_score(f"{strategy}_warm", Subset(train_ds, range(n_warm)))
        # The warm-up's factors and scores are removed, so the recorded disk
        # usage is the measured run's.
        accelerator.wait_for_everyone()
        if main:
            for warm in store.rglob(f"*_{strategy}_warm"):
                shutil.rmtree(warm, ignore_errors=True)
        torch.cuda.empty_cache()
    measured = Subset(train_ds, range(n_warm, n_warm + n_meas))

    with phase("fit_factors", n_meas):
        analyzer.fit_all_factors(
            factors_name=strategy, dataset=measured, per_device_batch_size=batch,
            factor_args=FactorArguments(strategy=strategy), overwrite_output_dir=True)
    with phase("pairwise_scores", n_meas + n_test):
        analyzer.compute_pairwise_scores(
            scores_name=strategy, score_args=ScoreArguments(), factors_name=strategy,
            query_dataset=test_ds, train_dataset=measured,
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
    # Requires the Kronfluence version pinned in versions.py.
    require("kronfluence")
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
