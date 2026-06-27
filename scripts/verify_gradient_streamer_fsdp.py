"""verify_gradient_streamer_fsdp.py — check GradientStreamer under DDP / FSDP.

The streamer wraps the model itself (``_wrap_model``) and shards the dataset
across ranks (``DistributedSampler``).  This script verifies the core
correctness invariant: the per-sample ``(activation, grad_output)`` the streamer
yields under DDP/FSDP must match a **single-device, no-parallelism reference**
for the *same* samples.

Matching is by **content hash** (the streamer's row identity), so no sampler /
ordering assumptions are needed: each rank streams its shard, the per-rank
results are ``all_gather``-ed, and rank 0 compares every hash against the
reference computed with a plain :class:`HookManager` over the full dataset.

Usage::

    # Single-process sanity check (CPU, no torchrun): streamer == reference
    python scripts/verify_gradient_streamer_fsdp.py

    # DDP (2 processes)
    torchrun --nproc_per_node=2 scripts/verify_gradient_streamer_fsdp.py --setup ddp

    # FSDP (2 processes) — requires CUDA; see note below
    torchrun --nproc_per_node=2 scripts/verify_gradient_streamer_fsdp.py --setup fsdp

Note:
    ``single`` and ``ddp`` run on CPU (gloo).  ``fsdp`` requires a CUDA
    environment: PyTorch FSDP resolves its compute device to the local
    accelerator, and on a CPU/MPS-only host (e.g. macOS) initialization fails
    with ``torch.mps.current_device`` not implemented — an environment
    limitation, not a streamer bug.  Run the FSDP setup on a GPU node.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dattri_llm.algorithm.arguments import AttributionArguments  # noqa: E402
from dattri_llm.algorithm.streaming import EvalGradientStreamer  # noqa: E402
from dattri_llm.gradient.callbacks import HookManagerCallback  # noqa: E402
from dattri_llm.gradient.gradient import Factorized  # noqa: E402
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
N_SAMPLES = 8          # full dataset (divisible by world_size → no sampler padding)
PER_DEVICE_BATCH = 2
ATOL = 2e-4            # FSDP all-gather/reshard introduces small numerical drift
RTOL = 1e-3

GPT2_CFG = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=SEQ_LEN,
    n_embd=64,
    n_layer=2,
    n_head=2,
)


# --------------------------------------------------------------------------- #
# Data / helpers                                                                #
# --------------------------------------------------------------------------- #


class FixedTokenDataset(Dataset):
    """Deterministic {input_ids, labels} dataset — identical on every rank."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(SEED)
        self._ids = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN), generator=g)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self._ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class _Capture(HookManagerCallback):
    def __init__(self) -> None:
        self.record = None

    def on_step_end(self, record) -> None:
        self.record = record


def loss_fn(model, batch) -> torch.Tensor:
    """Causal-LM loss **summed** over tokens (and samples).

    A sum (not mean) reduction is what makes a sample's per-layer gradient
    independent of batch size/composition, so the batch-of-1 reference and the
    batched stream are directly comparable.  Called identically on both sides so
    the captured inputs (and therefore the content hashes) match.
    """
    logits = model(input_ids=batch["input_ids"]).logits  # (B, T, V)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )


def _check(cond: bool, msg: str) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    return cond


def _to_device(batch: dict, device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


# {hash: {layer: {"activation": (T,in), "grad_output": (T,out)}}} (per sample, on CPU).
PerSample = dict


def _add_block(out: PerSample, grad, hashes: list[str]) -> None:
    """Slice a batched factorized Gradient block into per-sample CPU tensors.

    Skips batch-collapsed / broadcast layers (e.g. GPT-2's position embedding,
    whose activation has batch dim 1 regardless of batch size) — those cannot be
    attributed per sample.
    """
    n = len(hashes)
    for i, h in enumerate(hashes):
        layers = {}
        for name, fac in grad.data.items():
            if not isinstance(fac, Factorized):
                continue
            if fac.activation.shape[0] != n:  # broadcast/batch-collapsed layer
                continue
            layers[name] = {
                "activation": fac.activation[i].detach().cpu().float(),
                "grad_output": fac.pre_activation_grad[i].detach().cpu().float(),
            }
        out[h] = layers


def reference_per_sample(device) -> PerSample:
    """No-parallelism reference: plain HookManager over the full dataset.

    Run one sample at a time (batch_size=1) so each record maps to a single
    hash; returns ``{hash: {layer: {activation, grad_output}}}``.
    """
    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG).to(device).eval()
    cap = _Capture()
    hm = HookManager(model, config=HookManagerConfig(), callbacks=[cap])

    out: PerSample = {}
    loader = DataLoader(FixedTokenDataset(), batch_size=1, shuffle=False)
    with hm.collect():
        for batch in loader:
            batch = _to_device(batch, device)
            model.zero_grad(set_to_none=True)
            loss_fn(model, batch).backward()
            rec = cap.record
            hashes = rec.input_hash if isinstance(rec.input_hash, list) else [rec.input_hash]
            _add_block(out, rec.gradient, hashes)
    hm.remove()
    return out


def stream_per_sample(setup: str, device, tmp_dir: str) -> PerSample:
    """Stream this rank's shard with GradientStreamer; return per-sample (a,g)."""
    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG).to(device)

    fsdp = "full_shard" if setup == "fsdp" else ""
    args = AttributionArguments(
        output_dir=tmp_dir,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fsdp=fsdp,
        fsdp_config={"use_orig_params": True} if fsdp else None,
        ddp_find_unused_parameters=False,
    )

    # The streamer wraps the model (DDP/FSDP per args) and shards the dataset.
    streamer = EvalGradientStreamer(
        model, FixedTokenDataset(), args, batch_size=PER_DEVICE_BATCH, loss_fn=loss_fn
    )
    out: PerSample = {}
    with streamer:
        for _step, grad, hashes in streamer:
            _add_block(out, grad, hashes)
    return out


# --------------------------------------------------------------------------- #
# Comparison                                                                    #
# --------------------------------------------------------------------------- #


def _compare(ref: PerSample, test: PerSample, label: str) -> bool:
    print(f"\n  --- {label} ---")
    ok = True
    ok &= _check(
        set(test.keys()) >= set(ref.keys()),
        f"all {len(ref)} reference samples present in streamed output "
        f"(streamed {len(test)} unique)",
    )
    worst = 0.0
    compared = 0  # per-sample layer comparisons actually performed
    for h in sorted(ref):
        if h not in test:
            print(f"  FAIL  sample {h[:12]}… missing from streamed output")
            ok = False
            continue
        # Compare the per-sample layers present in both (broadcast layers are
        # legitimately absent from batched output and are skipped, not failed).
        for layer in sorted(set(ref[h]) & set(test[h])):
            for key in ("activation", "grad_output"):
                r = ref[h][layer][key]
                d = test[h][layer][key]
                if r.shape != d.shape:
                    print(f"  FAIL  {h[:12]}… [{layer}/{key}] shape {list(r.shape)} vs {list(d.shape)}")
                    ok = False
                    continue
                diff = (r - d).abs().max().item()
                worst = max(worst, diff)
                compared += 1
                if not torch.allclose(r, d, atol=ATOL, rtol=RTOL):
                    print(f"  FAIL  {h[:12]}… [{layer}/{key}] max_diff={diff:.2e}")
                    ok = False
    ok &= _check(compared > 0, f"compared at least one per-sample layer (did {compared})")
    print(f"  worst max|ref - stream| across all samples/layers = {worst:.2e}")
    return ok


# --------------------------------------------------------------------------- #
# Runners                                                                       #
# --------------------------------------------------------------------------- #


def run_single(tmp_dir: str) -> bool:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[single] device={device}  (streamer with no wrapping vs reference)")
    ref = reference_per_sample(device)
    test = stream_per_sample("single", device, tmp_dir)
    return _compare(ref, test, "single-process streamer vs HookManager reference")


def run_distributed(setup: str, tmp_dir: str) -> bool:
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    print(f"\n[{setup} rank {rank}/{world_size}] device={device}")

    # Each rank streams its shard.
    local = stream_per_sample(setup, device, tmp_dir)
    print(f"  [rank {rank}] streamed {len(local)} local samples")

    # Gather per-rank per-sample dicts (tensors are on CPU → picklable).
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, local)
    dist.barrier()

    ok = True
    if rank == 0:
        merged: PerSample = {}
        for part in gathered:
            merged.update(part)  # disjoint hashes across ranks (dupes identical)
        print(f"  [rank 0] merged {len(merged)} unique samples across ranks")
        ref = reference_per_sample(device)
        ok = _compare(ref, merged, f"{setup} streamer (all ranks) vs single-device reference")

    dist.destroy_process_group()
    return ok


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="GradientStreamer FSDP/DDP verification")
    parser.add_argument("--setup", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    if args.tmp_dir:
        tmp_dir = args.tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        cleanup = False
    else:
        _tmpobj = tempfile.TemporaryDirectory()
        tmp_dir = _tmpobj.name
        cleanup = True

    try:
        if args.setup == "single":
            ok = run_single(tmp_dir)
        else:
            ok = run_distributed(args.setup, tmp_dir)

        if rank == 0:
            print("\n" + "=" * 55)
            print(f"  {'PASS' if ok else 'FAIL'}  GradientStreamer [{args.setup}]")
            print("=" * 55)
            sys.exit(0 if ok else 1)
    finally:
        if cleanup and rank == 0:
            _tmpobj.cleanup()


if __name__ == "__main__":
    main()
