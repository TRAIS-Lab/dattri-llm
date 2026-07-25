"""dattri_llm side: KFAC influence with LoGRA-style capture-time projection
(proj_dim=64 per factor side, same 64x64=4096-dim per-layer projected gradient
space as logix's rank-64 LoRA), on the identical model, data, and hooked
layers.

Two workflows:

* ``--workflow disk`` (default) -- store-then-attribute: one collection pass
  writes projected gradients to disk, scoring reads them back.  This matches
  the *shape* of logix's extract + compute_influence pipeline.
* ``--workflow otf`` -- one ``attribute`` call, nothing persisted (streams the
  train set twice: Fisher fit + scoring).

    python run_dattri_llm.py [--workflow disk|otf]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py

import torch
import torch.nn.functional as F
from torch.func import functional_call

from common import Meter, dir_bytes, env_info
from data import BATCH, N_QUERY, TensorBlocks, load_datasets, load_model_and_tokenizer

from dattri.task import AttributionTask
from dattri_llm.attribution.algorithm.kronecker import KFACAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.utils.hashing import hash_sample

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

PROJ_DIM = 64  # per factor side -> 64*64 = 4096 per layer, = logix rank 64


def main(a):
    model, tokenizer = load_model_and_tokenizer()
    train_hf, query_hf = load_datasets(tokenizer)
    train_ds, query_ds = TensorBlocks(train_hf), TensorBlocks(query_hf)
    model = model.cuda().eval()

    def gl(params, batch):
        logits = functional_call(
            model, params, args=(),
            kwargs={"input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]},
        )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum", ignore_index=-100,
        )

    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=gl, model=model, checkpoints=[ckpt],
                           target_func=gl)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="eff_logix_"),
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=1,
        dataloader_pin_memory=False,  # num_workers defaults to 0 (fastest here)
    )
    hook_config = HookManagerConfig(
        linear_io=[r"attn\.", r"mlp\."],
        projection={"__default__": {
            "style": "logra_factorized", "proj_dim": PROJ_DIM,
            "proj_max_batch_size": 32, "proj_type": "rademacher",
            "proj_seed": 0,
        }},
    )

    meter = Meter(f"dattri_llm(kfac-proj64,{a.workflow})",
                  "gpt2-wikitext103-logra64",
                  HERE / "results.jsonl", meta=env_info())
    attributor = KFACAttributor(args, task=task)
    if a.workflow in ("otf", "otf_cached"):
        # otf: re-stream the model per Fisher sweep (KFAC = 2 model passes).
        # otf_cached (P2): collect once into an in-RAM residency store, replay
        # the sweeps from RAM -- projected blocks fit, so this is the win.
        residency = "tiered" if a.workflow == "otf_cached" else None
        with meter.phase("attribute", len(train_ds) + N_QUERY):
            score = attributor.attribute(
                train_ds, query_ds, damping=1e-10, hook_config=hook_config,
                residency=residency)
        meter.note(store_bytes=0)
    elif a.workflow == "refit":
        # Persisted-fit re-attribution: fit the Fisher factors once, then score
        # from them (skipping the fit pre-pass).  "score_full" is the per-query
        # cost today (fit + score inside attribute_from_cache); "fit" is the
        # one-time factor fit; "score_refit" is the re-attribution cost with the
        # factors loaded.  A second query set reuses the same fit for free, so
        # the win over N query sets is (N-1) x the fit saved (= score_full -
        # score_refit).
        cache_dir = HERE / "ours_grads"
        shutil.rmtree(cache_dir, ignore_errors=True)
        with meter.phase("cache", len(train_ds) + N_QUERY):
            (train_dir, test_dir), = attributor.cache(
                train_ds, query_ds, cache_dir=str(cache_dir),
                hook_config=hook_config)
        meter.note(store_bytes=dir_bytes(cache_dir))
        with meter.phase("score_full", len(train_ds) + N_QUERY):
            attributor.attribute_from_cache(train_dir, test_dir, damping=1e-10)
        with meter.phase("fit", len(train_ds)):
            fisher_dir = attributor.fit(train_dir)
        meter.note(fisher_bytes=dir_bytes(fisher_dir))
        with meter.phase("score_refit", len(train_ds) + N_QUERY):
            score = attributor.attribute_from_cache(
                train_dir, test_dir, damping=1e-10, fisher_dir=fisher_dir)
    elif a.workflow == "compact":
        # logix-style compact K-FAC: store the token-summed projected outer
        # product (logra_materialized), fit the projected (A, G) covariances at
        # capture with a KroneckerCovarianceCallback, and precondition the
        # compact store two-sided.  Compact store + no Fisher re-pass.
        from dattri_llm.attribution.utils import collect_to_disk, task_loss_fn
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import GradientStreamer

        mat_config = HookManagerConfig(
            linear_io=[r"attn\.", r"mlp\."],
            projection={"__default__": {
                "style": "logra_materialized", "proj_dim": PROJ_DIM,
                "proj_max_batch_size": 32, "proj_type": "rademacher",
                "proj_seed": 0,
            }},
        )
        cache_dir = HERE / "ours_grads_compact"
        shutil.rmtree(cache_dir, ignore_errors=True)
        train_dir = str(cache_dir / "train")
        test_dir = str(cache_dir / "test")
        cov = KroneckerCovarianceCallback()
        with meter.phase("cache", len(train_ds) + N_QUERY):
            task._load_checkpoints(0)
            tr = GradientStreamer(
                task.get_model(), train_ds, args, batch_size=BATCH,
                loss_fn=task_loss_fn(task.original_loss_func), config=mat_config)
            tr.hook_manager.add_callback(cov)  # projected (A, G) at capture
            collect_to_disk(tr, GradientStorageManager(train_dir))
            te = GradientStreamer(
                task.get_model(), query_ds, args, batch_size=1,
                loss_fn=task_loss_fn(task.original_target_func), config=mat_config)
            collect_to_disk(te, GradientStorageManager(test_dir))
        meter.note(store_bytes=dir_bytes(cache_dir))
        fisher_dir = attributor.save_fisher(cov.result())
        meter.note(fisher_bytes=dir_bytes(fisher_dir))
        # Scoring batches the train side at per_device_train_batch_size (collection
        # above used an explicit batch_size=BATCH, so this only affects scoring):
        # "auto" = whole store as one batch, or an int K to match logix's 64-doc
        # scoring batch so the memory is aligned for a like-for-like comparison.
        sb = len(train_ds) if a.score_batch == "auto" else int(a.score_batch)
        args.per_device_train_batch_size = sb
        meter.note(score_batch=a.score_batch)
        with meter.phase("score", len(train_ds) + N_QUERY):
            score = attributor.attribute_from_cache(
                train_dir, test_dir, damping=1e-10, fisher_dir=fisher_dir)
    else:
        cache_dir = HERE / "ours_grads"
        shutil.rmtree(cache_dir, ignore_errors=True)
        with meter.phase("cache", len(train_ds) + N_QUERY):
            (train_dir, test_dir), = attributor.cache(
                train_ds, query_ds, cache_dir=str(cache_dir),
                hook_config=hook_config)
        meter.note(store_bytes=dir_bytes(cache_dir))
        with meter.phase("score", len(train_ds) + N_QUERY):
            score = attributor.attribute_from_cache(
                train_dir, test_dir, damping=1e-10)

    th = [hash_sample({"input_ids": train_ds[i]["input_ids"],
                       "attention_mask": train_ds[i]["attention_mask"]})
          for i in range(len(train_ds))]
    vh = [hash_sample({"input_ids": query_ds[i]["input_ids"],
                       "attention_mask": query_ds[i]["attention_mask"]})
          for i in range(N_QUERY)]
    matrix = score.query(th, vh).cpu().float()
    torch.save({"score": matrix}, HERE / f"score_ours_{a.workflow}.pt")
    meter.note(n_train=len(train_ds), score_shape=list(matrix.shape))
    meter.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--workflow",
        choices=["disk", "otf", "otf_cached", "refit", "compact"],
        default="disk",
    )
    p.add_argument("--score-batch", dest="score_batch", default="auto",
                   help="'auto' (whole store) or an int K (K-doc scoring batch)")
    main(p.parse_args())
