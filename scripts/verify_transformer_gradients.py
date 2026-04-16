"""verify_trainer_gradients.py — smoke-test GradientCollectingTrainer under DDP / FSDP.

Runs one training step with ``GradientCollectingTrainer`` and verifies that:

1. Gradient files (``step_*.pt``) are written to disk.
2. Every file contains ``activation`` and ``grad_output`` tensors for each
   hooked MLP layer, with the expected shapes.
3. Values are finite (no NaN / Inf).
4. (DDP / FSDP) The ``a`` and ``g`` tensors collected on each rank agree with
   a single-GPU reference run on the same fixed batch.

Usage::

    # Single-GPU smoke test (CPU, no torchrun)
    python scripts/verify_trainer_gradients.py

    # DDP smoke test (2 GPUs)
    torchrun --nproc_per_node=2 scripts/verify_trainer_gradients.py --setup ddp

    # FSDP smoke test (2 GPUs)
    torchrun --nproc_per_node=2 scripts/verify_trainer_gradients.py --setup fsdp
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402
from dattri_llm.gradient.trainers.transformers import GradientCollectingTrainer  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
N_SAMPLES = 4          # total dataset size (must be ≥ 2 × world_size)
PER_DEVICE_BATCH = 2   # samples per GPU per step
ATOL = 1e-5
RTOL = 1e-4

GPT2_CFG = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=SEQ_LEN,
    n_embd=64,
    n_layer=2,
    n_head=2,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


class FixedTokenDataset(Dataset):
    """Deterministic token dataset — same tokens regardless of worker / rank."""

    def __init__(self) -> None:
        torch.manual_seed(SEED)
        self._ids = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self._ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _check(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {msg}")
    return cond


def _tensor_stats(t: torch.Tensor) -> str:
    t = t.float()
    return f"shape={list(t.shape)} min={t.min():.3e} max={t.max():.3e} mean={t.mean():.3e}"


def _compare_ag(
    ref: dict[str, dict[str, torch.Tensor]],
    test: dict[str, dict[str, torch.Tensor]],
    label: str,
) -> bool:
    """Compare activation and grad_output tensors layer by layer."""
    all_pass = True
    width = max((len(k) for k in ref), default=30)
    print(f"\n  --- {label} ---")
    for layer in sorted(ref.keys()):
        lbl = layer.ljust(width)
        if layer not in test:
            print(f"  FAIL  {lbl}  — layer missing")
            all_pass = False
            continue
        for key in ("activation", "grad_output"):
            r = ref[layer][key].cpu().float()
            d = test[layer][key].cpu().float()
            if r.shape != d.shape:
                print(f"  FAIL  {lbl} [{key}]  shape: ref={list(r.shape)} test={list(d.shape)}")
                all_pass = False
                continue
            ok = torch.allclose(r, d, atol=ATOL, rtol=RTOL)
            diff = (r - d).abs()
            flat_idx = int(diff.view(-1).argmax())
            worst = list(map(int, torch.unravel_index(torch.tensor(flat_idx), diff.shape)))
            print(
                f"  {'PASS' if ok else 'FAIL'}  {lbl} [{key}]  "
                f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}  "
                f"worst_at={worst}  "
                f"ref={r.view(-1)[flat_idx]:.4e}  test={d.view(-1)[flat_idx]:.4e}"
            )
            if not ok:
                B = r.shape[0]
                per_sample = [f"s{b}:{diff[b].max():.2e}" for b in range(B)]
                print(f"         per-sample max_diff: {', '.join(per_sample)}")
                print(f"         ref  — {_tensor_stats(r)}")
                print(f"         test — {_tensor_stats(d)}")
                all_pass = False
    return all_pass


# --------------------------------------------------------------------------- #
# Reference: collect a/g with the raw GradientCollector (no Trainer)           #
# --------------------------------------------------------------------------- #


def collect_reference(
    device: torch.device,
    sample_ids: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run one forward+backward on the given token IDs and return a/g.

    Args:
        device: Device to run on.
        sample_ids: Token ID tensor of shape ``(B, T)`` — the exact batch
            saved by the trainer under ``"_sample_ids"``.
    """
    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG).to(device)

    ids = sample_ids.to(device)
    labels = ids.clone()

    collector = GradientCollector(model)
    model.train()
    with collector:
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()

    ag = collector.get_activations_and_grad_outputs()
    collector.remove()
    return ag


# --------------------------------------------------------------------------- #
# Smoke-test helpers                                                            #
# --------------------------------------------------------------------------- #


def _smoke_check_files(grad_dir: str, expected_layers: int) -> bool:
    """Verify that saved .pt files have the expected structure.

    The ``"_sample_ids"`` key (added by the trainer for traceability) is
    excluded from the layer count and from activation checks.
    """
    pt_files = sorted(f for f in os.listdir(grad_dir) if f.endswith(".pt"))
    ok = True

    ok &= _check(len(pt_files) > 0, f"at least one .pt file written  (found {len(pt_files)})")
    if not pt_files:
        return False

    path = os.path.join(grad_dir, pt_files[0])
    data = torch.load(path, weights_only=True)

    ok &= _check(isinstance(data, dict), "saved object is a dict")

    # Separate gradient layers from the optional metadata key.
    layer_data = {k: v for k, v in data.items() if not k.startswith("_")}
    sample_ids = data.get("_sample_ids")
    ok &= _check(
        len(layer_data) == expected_layers,
        f"correct layer count  (expected {expected_layers}, got {len(layer_data)})",
    )
    ok &= _check(
        sample_ids is not None,
        "_sample_ids key present (batch token IDs saved for traceability)",
    )

    for layer, entry in layer_data.items():
        ok &= _check(isinstance(entry, dict), f"{layer}: entry is a dict")
        for key in ("activation", "grad_output"):
            t = entry.get(key)
            ok &= _check(t is not None, f"{layer}[{key}]: key present")
            if t is not None:
                ok &= _check(isinstance(t, torch.Tensor), f"{layer}[{key}]: is Tensor")
                ok &= _check(
                    torch.isfinite(t).all().item(),
                    f"{layer}[{key}]: all finite  {_tensor_stats(t)}",
                )

    return ok


# --------------------------------------------------------------------------- #
# Per-setup runners                                                             #
# --------------------------------------------------------------------------- #


def run_single_gpu(tmp_dir: str) -> bool:
    """Run trainer on CPU/GPU0 for 1 step, smoke-check output."""
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Single-GPU] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)

    args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        max_steps=1,
        no_cuda=(not torch.cuda.is_available()),
        use_cpu=(not torch.cuda.is_available()),
        logging_steps=1,
        save_steps=100,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = GradientCollectingTrainer(
        model=model,
        args=args,
        train_dataset=FixedTokenDataset(),
        gradient_collection_mode="mlp_io",
    )
    trainer.train()

    grad_dir = os.path.join(tmp_dir, "gradients")
    print(f"  Checking files in {grad_dir} …")

    # Determine expected layer count from a fresh collector.
    set_seed(SEED)
    probe = GPT2LMHeadModel(GPT2_CFG)
    coll = GradientCollector(probe)
    expected_layers = len(coll.layer_names)
    coll.remove()

    ok = _smoke_check_files(grad_dir, expected_layers)

    # --- Compare against raw GradientCollector reference ---
    print("\n  Comparing trainer output against raw GradientCollector reference …")
    pt_files = sorted(f for f in os.listdir(grad_dir) if f.endswith(".pt"))
    saved = torch.load(os.path.join(grad_dir, pt_files[0]), weights_only=True)
    sample_ids = saved.pop("_sample_ids", None)
    trainer_ag = saved  # remaining keys are layer names

    if sample_ids is None:
        print("  SKIP  _sample_ids not found — cannot build reference batch")
        return ok

    print(f"  sample_ids shape: {list(sample_ids.shape)}")
    device = torch.device(device_str)
    ref_ag = collect_reference(device, sample_ids)
    ok &= _compare_ag(ref_ag, trainer_ag, "trainer vs raw collector")

    return ok


def run_ddp(tmp_dir: str) -> bool:
    """Run trainer with DDP via torchrun; rank 0 validates output."""
    import torch.distributed as dist

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"\n[DDP rank {rank}/{world_size}] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)

    args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        max_steps=1,
        no_cuda=(not torch.cuda.is_available()),
        use_cpu=(not torch.cuda.is_available()),
        logging_steps=1,
        save_steps=100,
        report_to="none",
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )

    trainer = GradientCollectingTrainer(
        model=model,
        args=args,
        train_dataset=FixedTokenDataset(),
        gradient_collection_mode="mlp_io",
    )
    trainer.train()
    dist.barrier()

    ok = True
    if rank == 0:
        grad_dir = os.path.join(tmp_dir, "gradients")
        print(f"\n  [rank 0] Checking files in {grad_dir} …")

        set_seed(SEED)
        probe = GPT2LMHeadModel(GPT2_CFG)
        coll = GradientCollector(probe)
        expected_layers = len(coll.layer_names)
        coll.remove()

        ok &= _smoke_check_files(grad_dir, expected_layers)

        # --- Compare trainer a/g against raw collector reference ---
        # Load _sample_ids saved by the trainer to identify which samples
        # rank 0 actually processed — no sampler ordering assumptions needed.
        pt_files = sorted(f for f in os.listdir(grad_dir) if f.endswith(".pt"))
        saved = torch.load(os.path.join(grad_dir, pt_files[0]), weights_only=True)
        sample_ids = saved.pop("_sample_ids", None)
        trainer_ag = saved

        if sample_ids is None:
            print("\n  [rank 0] SKIP  _sample_ids not found — cannot build reference batch")
        else:
            print(f"\n  [rank 0] sample_ids shape: {list(sample_ids.shape)}")
            device = torch.device(device_str)
            ref_ag = collect_reference(device, sample_ids)
            ok &= _compare_ag(ref_ag, trainer_ag, "DDP rank-0 trainer vs raw collector")

    dist.destroy_process_group()
    return ok


def run_fsdp(tmp_dir: str) -> bool:
    """Run trainer with FSDP via torchrun; rank 0 validates output."""
    import torch.distributed as dist

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"\n[FSDP rank {rank}/{world_size}] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)

    args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        max_steps=1,
        no_cuda=(not torch.cuda.is_available()),
        use_cpu=(not torch.cuda.is_available()),
        logging_steps=1,
        save_steps=100,
        report_to="none",
        dataloader_drop_last=True,
        fsdp="full_shard",
        fsdp_config={"use_orig_params": True},
    )

    trainer = GradientCollectingTrainer(
        model=model,
        args=args,
        train_dataset=FixedTokenDataset(),
        gradient_collection_mode="mlp_io",
    )
    trainer.train()
    dist.barrier()

    ok = True
    if rank == 0:
        grad_dir = os.path.join(tmp_dir, "gradients")
        print(f"\n  [rank 0] Checking files in {grad_dir} …")

        set_seed(SEED)
        probe = GPT2LMHeadModel(GPT2_CFG)
        coll = GradientCollector(probe)
        expected_layers = len(coll.layer_names)
        coll.remove()

        ok &= _smoke_check_files(grad_dir, expected_layers)

        pt_files = sorted(f for f in os.listdir(grad_dir) if f.endswith(".pt"))
        saved = torch.load(os.path.join(grad_dir, pt_files[0]), weights_only=True)
        sample_ids = saved.pop("_sample_ids", None)
        trainer_ag = saved

        if sample_ids is None:
            print("\n  [rank 0] SKIP  _sample_ids not found — cannot build reference batch")
        else:
            print(f"\n  [rank 0] sample_ids shape: {list(sample_ids.shape)}")
            device = torch.device(device_str)
            ref_ag = collect_reference(device, sample_ids)
            ok &= _compare_ag(ref_ag, trainer_ag, "FSDP rank-0 trainer vs raw collector (informational)")

    dist.destroy_process_group()
    return ok


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainer gradient smoke test")
    parser.add_argument(
        "--setup",
        choices=["single", "ddp", "fsdp"],
        default="single",
        help="Training setup to test (default: single)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for trainer output (default: auto temp dir)",
    )
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
            ok = run_single_gpu(tmp_dir)
        elif args.setup == "ddp":
            ok = run_ddp(tmp_dir)
        elif args.setup == "fsdp":
            ok = run_fsdp(tmp_dir)
        else:
            ok = False

        if rank == 0:
            print("\n" + "=" * 55)
            print("SUMMARY")
            print("=" * 55)
            print(f"  {'PASS' if ok else 'FAIL'}  {args.setup}")
            sys.exit(0 if ok else 1)

    finally:
        if cleanup and rank == 0:
            _tmpobj.cleanup()


if __name__ == "__main__":
    main()
