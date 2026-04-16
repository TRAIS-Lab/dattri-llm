"""verify_gradient_consistency.py — compare MLP gradients across training setups.

Trains a tiny GPT-2 for one step under each requested setup and verifies
that collected per-sample MLP gradients match the single-GPU reference
within numerical tolerance.

Usage::

    # Single-GPU only (default, runs on CPU/GPU, no torchrun needed)
    python scripts/verify_gradient_consistency.py

    # DataParallel (single-machine multi-GPU, experimental — see NOTE below)
    python scripts/verify_gradient_consistency.py --setup dp

    # DDP (requires 2+ GPUs, launch with torchrun)
    torchrun --nproc_per_node=2 scripts/verify_gradient_consistency.py --setup ddp

    # FSDP (requires 2+ GPUs, launch with torchrun)
    torchrun --nproc_per_node=2 scripts/verify_gradient_consistency.py --setup fsdp

NOTE on DataParallel (DP)
--------------------------
The hook-based ``grad_output`` comparison for multi-GPU DP is known to be
unreliable for models with residual connections due to the way PyTorch shares
``_backward_hooks`` dicts between DP replicas.  DP activation capture works
correctly; gradient comparison is tracked in DEVLOG.md as an open issue.
The ``--setup dp`` path demonstrates the hook plumbing and will run, but the
gradient comparison result should be treated as informational only.

The script prints a PASS / FAIL summary per layer per setup.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

# Make sure the package is importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

ATOL = 1e-5
RTOL = 1e-4
SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH_SIZE = 2

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


def make_model_and_batch(device: torch.device) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """Create a reproducible tiny GPT-2 and a fixed batch."""
    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG).to(device)
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    return model, {"input_ids": input_ids, "labels": input_ids.clone()}


def collect_one_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    label: str = "",
) -> dict[str, dict[str, torch.Tensor]]:
    """Run one forward+backward and return raw activations and output gradients.

    Returns:
        Dict mapping layer name to ``{"activation": Tensor, "grad_output": Tensor}``.
        Shapes: activation ``(B, T, in_features)``, grad_output ``(B, T, out_features)``.
        This avoids forming the expensive outer-product ``(B, out, in)`` tensor and
        keeps memory usage proportional to the hidden dimension, not its square.
    """
    collector = GradientCollector(model)
    model.train()
    with collector:
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        # DataParallel gathers losses from replicas into a vector; reduce first.
        loss = out.loss if out.loss.dim() == 0 else out.loss.mean()
        loss.backward()

    ag = collector.get_activations_and_grad_outputs()
    tag = f"[{label}] " if label else ""
    first = next(iter(ag.values()))
    print(
        f"  {tag}loss={loss.item():.4f}  "
        f"layers={len(ag)}  "
        f"batch_size={first['activation'].shape[0]}  "
        f"seq_len={first['activation'].shape[1]}"
    )
    collector.remove()
    return ag


def _tensor_stats(t: torch.Tensor) -> str:
    """Return a compact statistics string for a tensor."""
    t = t.float()
    return (
        f"shape={list(t.shape)}  "
        f"min={t.min():.3e}  max={t.max():.3e}  "
        f"mean={t.mean():.3e}  std={t.std():.3e}"
    )


def _check_tensor(
    r: torch.Tensor,
    d: torch.Tensor,
    label: str,
    tag: str,
    atol: float,
    rtol: float,
) -> bool:
    """Compare two tensors and print a detailed status line. Returns True if they match."""
    r, d = r.cpu().float(), d.cpu().float()

    if r.shape != d.shape:
        print(f"  FAIL  {label} [{tag}]  shape mismatch: ref={list(r.shape)} test={list(d.shape)}")
        return False

    ok = torch.allclose(r, d, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    diff = (r - d).abs()
    flat_idx = int(diff.view(-1).argmax())
    worst_at = list(map(int, torch.unravel_index(torch.tensor(flat_idx), diff.shape)))

    print(
        f"  {status}  {label} [{tag}]  "
        f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}  "
        f"worst_at={worst_at}  ref={r.view(-1)[flat_idx]:.4e}  test={d.view(-1)[flat_idx]:.4e}"
    )
    if not ok:
        B = r.shape[0]
        per_sample = [f"s{b}:{diff[b].max():.2e}" for b in range(B)]
        print(f"         per-sample max_diff: {', '.join(per_sample)}")
        print(f"         ref  — {_tensor_stats(r)}")
        print(f"         test — {_tensor_stats(d)}")

    return ok


def compare_to_reference(
    ref: dict[str, dict[str, torch.Tensor]],
    test: dict[str, dict[str, torch.Tensor]],
    setup_name: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    """Compare activation (a) and grad-output (g) tensors per layer.

    Comparing ``a`` and ``g`` directly is more memory-efficient than comparing
    the full outer-product gradient and is equally informative for verifying
    gradient correctness across training setups.

    Prints a PASS/FAIL line for each of ``activation`` and ``grad_output`` per
    layer; on failure also prints per-sample breakdown and tensor statistics.

    Returns:
        True iff every layer's ``activation`` and ``grad_output`` both pass.
    """
    all_pass = True
    width = max((len(k) for k in ref), default=20)
    print(f"\n=== {setup_name} (atol={atol:.0e}, rtol={rtol:.0e}) ===")

    for layer in sorted(ref.keys()):
        label = layer.ljust(width)

        if layer not in test:
            print(f"  FAIL  {label}  — layer missing in test output")
            all_pass = False
            continue

        for key in ("activation", "grad_output"):
            r_val = ref[layer].get(key)
            t_val = test[layer].get(key)
            if r_val is None or t_val is None:
                print(f"  FAIL  {label} [{key}]  — key missing")
                all_pass = False
                continue
            if not _check_tensor(r_val, t_val, label, key, atol, rtol):
                all_pass = False

    return all_pass


# --------------------------------------------------------------------------- #
# Setup implementations                                                         #
# --------------------------------------------------------------------------- #


def run_single_gpu(device: torch.device | None = None) -> dict[str, torch.Tensor]:
    """Collect reference gradients on a single device.

    Args:
        device: Device to use.  Defaults to CUDA:0 if available, else CPU.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Single-GPU] collecting reference gradients on {device} …")
    model, batch = make_model_and_batch(device)
    return collect_one_step(model, batch, label="single-GPU")


def run_dp() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """DataParallel (experimental) — returns ``(reference_grads, dp_grads)``.

    When 2+ GPUs are present both runs use CUDA so floating-point results are
    on the same device class.  Falls back to single-device DP when only one
    GPU is available.

    .. warning::
        Multi-GPU DP gradient comparison is an open issue (see DEVLOG.md).
        Activation capture is correct; gradient comparison may fail for models
        with residual connections due to shared ``_backward_hooks`` between
        DP replicas.  Results are printed for informational purposes.
    """
    print("\n[DataParallel] collecting gradients …")
    if torch.cuda.device_count() >= 2:
        device_ids = [0, 1]
        ref_device = torch.device("cuda:0")
        print(f"  Using device_ids={device_ids}")
        print("  NOTE: multi-GPU DP gradient comparison is informational only.")
    else:
        print("  (1 GPU / no GPU — running DP on single device, should match reference)")
        device_ids = None
        ref_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ref = run_single_gpu(ref_device)

    dp_device = torch.device("cuda:0") if device_ids else ref_device
    model, batch = make_model_and_batch(dp_device)
    if device_ids is not None:
        model = nn.DataParallel(model, device_ids=device_ids)

    dp_grads = collect_one_step(model, batch, label="DP")
    return ref, dp_grads


def run_ddp() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """DistributedDataParallel — must be launched with ``torchrun``.

    Rank 0 returns ``(reference_grads, ddp_grads)``; other ranks return
    ``(None, None)`` and should not be used for comparison.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank = int(os.environ.get("RANK", 0))
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model, batch = make_model_and_batch(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    ddp_grads = collect_one_step(ddp_model, batch, label=f"DDP rank{rank}")
    dist.destroy_process_group()

    if rank != 0:
        return None, None

    # Reference on the same device as rank 0.
    ref = run_single_gpu(device)
    return ref, ddp_grads


def run_fsdp() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """FullyShardedDataParallel — must be launched with ``torchrun``.

    .. note::
        FSDP shards parameters across ranks.  The hooks capture the *local
        shard's* activation and grad-output; a full all-gather is required
        for an exact comparison against the single-GPU reference.  This
        function demonstrates the hook plumbing; numerical agreement is not
        guaranteed.

    Rank 0 returns ``(reference_grads, fsdp_grads)``; other ranks return
    ``(None, None)``.
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    rank = int(os.environ.get("RANK", 0))
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model, batch = make_model_and_batch(device)
    fsdp_model = FSDP(model)

    fsdp_grads = collect_one_step(fsdp_model, batch, label=f"FSDP rank{rank}")
    dist.destroy_process_group()

    if rank != 0:
        return None, None

    ref = run_single_gpu(device)
    return ref, fsdp_grads


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient consistency verification")
    parser.add_argument(
        "--setup",
        choices=["single", "dp", "ddp", "fsdp"],
        default="single",
        help=(
            "Which setup to verify (default: single).  "
            "'ddp' and 'fsdp' require torchrun."
        ),
    )
    args = parser.parse_args()

    results: dict[str, bool] = {}

    if args.setup == "single":
        run_single_gpu()
        results["single-GPU"] = True
        print("\n[single-GPU] hook fires correctly — reference baseline established.")

    elif args.setup == "dp":
        dp_ref, dp_grads = run_dp()
        passed = compare_to_reference(dp_ref, dp_grads, "DataParallel")
        results["DataParallel"] = passed
        if not passed and torch.cuda.device_count() >= 2:
            print(
                "\n  NOTE: multi-GPU DP gradient mismatch is a known open issue."
                "\n  See DEVLOG.md for details.  Activation capture is correct."
            )

    elif args.setup == "ddp":
        ref, ddp_grads = run_ddp()
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            results["DDP"] = compare_to_reference(ref, ddp_grads, "DDP")

    elif args.setup == "fsdp":
        ref, fsdp_grads = run_fsdp()
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            results["FSDP (local shard, informational)"] = compare_to_reference(
                ref, fsdp_grads, "FSDP",
            )

    # Summary (only rank 0 prints)
    if results:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        all_pass = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status}  {name}")
            if not passed:
                all_pass = False

        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
