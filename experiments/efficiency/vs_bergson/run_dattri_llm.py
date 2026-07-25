"""dattri_llm side: store-then-attribute gradient index on the identical
setup -- pythia-14m, pile-10k, sum-CE, per-module rank-16 double-sided random
projection (capture-time ``factorize proj_dim=16`` == bergson's
``projection_dim 16`` per-module sketch), same 25 Linear modules, token-packed
batches of <= 2048 tokens.

Phases: ``build`` (HookManager + OffloadCallback collection pass over all
10k documents -> on-disk index) and ``query`` (collect 16 query docs' projected
gradients the same way, then ``TracInAttributor.attribute_from_cache`` scores
them against the whole index).

    python run_dattri_llm.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F

from common import Meter, dir_bytes, env_info
from data import MAX_LEN, N_QUERY, load_model, load_tokenized

from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

TOKEN_BUDGET = 2048  # matches bergson (their code caps it at model max len)
# bergson's index covers the 24 base-transformer linears (no embed_out).
LINEAR_PATTERNS = [
    r"attention\.query_key_value$", r"attention\.dense$",
    r"mlp\.dense_h_to_4h$", r"mlp\.dense_4h_to_h$",
]


def make_proj(style):
    return {"__default__": {
        "style": style, "proj_dim": 16, "proj_max_batch_size": 32,
        "proj_type": "rademacher", "proj_seed": 0,
    }}


def token_packed_batches(ds, indices):
    """Greedy <=TOKEN_BUDGET-token batches over length-sorted docs (like
    bergson's token batching; sorting keeps padding waste minimal)."""
    order = sorted(indices, key=lambda i: ds[i]["n_tokens"], reverse=True)
    batch, batches, maxlen = [], [], 0
    for i in order:
        n = min(ds[i]["n_tokens"], MAX_LEN)
        new_max = max(maxlen, n)
        if batch and new_max * (len(batch) + 1) > TOKEN_BUDGET:
            batches.append(batch)
            batch, maxlen = [i], n
        else:
            batch.append(i)
            maxlen = new_max
    if batch:
        batches.append(batch)
    return batches


def collect(model, ds, indices, out_dir, config, disk_format="pickle"):
    """One HookManager + OffloadCallback collection pass (workflow 2)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    callback = OffloadCallback(
        offload_interval=16,
        file_manager=GradientStorageManager(str(out_dir), disk_format=disk_format))
    hookmanager = HookManager(model, config=config, callbacks=[callback])
    pad_id = 0
    with hookmanager.collect(), torch.enable_grad():
        for batch_idx in token_packed_batches(ds, indices):
            rows = [torch.tensor(ds[i]["input_ids"][:MAX_LEN]) for i in batch_idx]
            n = max(len(r) for r in rows)
            ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
            mask = torch.zeros((len(rows), n), dtype=torch.long)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = r
                mask[j, :len(r)] = 1
            ids, mask = ids.cuda(), mask.cuda()
            logits = model(input_ids=ids, attention_mask=mask).logits
            labels = ids.masked_fill(mask == 0, -100)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(),
                reduction="sum", ignore_index=-100)
            model.zero_grad(set_to_none=True)
            loss.backward()
    hookmanager.remove()


def realign(matrix, row_order, col_order):
    """Scatter disk-order rows/cols back to dataset order."""
    out = torch.empty_like(matrix)
    out[torch.tensor(row_order)] = matrix
    out2 = torch.empty_like(out)
    out2[:, torch.tensor(col_order)] = out
    return out2


def main(a):
    ds = load_tokenized()
    model = load_model().cuda()
    config = HookManagerConfig(
        linear_io=LINEAR_PATTERNS, projection=make_proj(a.style))

    tag = "logra" if a.style == "logra_factorized" else "logramat"
    if a.disk_format == "memmap":
        tag += "-mmap"
    train_dir = HERE / f"ours_index_{tag}" / "train"
    test_dir = HERE / f"ours_index_{tag}" / "test"
    meter = Meter(f"dattri_llm({tag})", "pythia14m-pile10k-proj16-index",
                  HERE / "results.jsonl",
                  meta=env_info({"style": a.style, "disk_format": a.disk_format}))

    with meter.phase("build", 10_000, unit="docs"):
        collect(model, ds, range(len(ds)), train_dir, config, a.disk_format)
    meter.note(store_bytes=dir_bytes(train_dir))

    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="eff_bergson_"),
        dataloader_pin_memory=False,  # num_workers defaults to 0 (fastest here)
        # Scoring batches the train side at per_device_train_batch_size.  The
        # store is already built above (collect() takes its own config), so this
        # only sizes the scoring GEMMs: the whole store as one batch, matching
        # bergson's single batched matmul over its flat gradients.bin.
        per_device_train_batch_size=len(ds),
    )
    with meter.phase("query", N_QUERY, unit="queries"):
        collect(model, ds, range(N_QUERY), test_dir, config, a.disk_format)
        score = TracInAttributor(args).attribute_from_cache(
            str(train_dir), str(test_dir))
        _, matrix = score.agnostic_matrix()  # rows/cols in disk order
        train_order = [i for b in token_packed_batches(ds, range(len(ds)))
                       for i in b]
        test_order = [i for b in token_packed_batches(ds, range(N_QUERY))
                      for i in b]
        matrix = realign(matrix.cpu().float(), train_order, test_order)
    torch.save({"score": matrix}, HERE / f"score_ours_{tag}.pt")
    meter.note(score_shape=list(matrix.shape))
    meter.flush()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="logra_factorized",
                   choices=["logra_factorized", "logra_materialized"])
    p.add_argument("--disk-format", dest="disk_format", default="pickle",
                   choices=["pickle", "memmap"])
    main(p.parse_args())
