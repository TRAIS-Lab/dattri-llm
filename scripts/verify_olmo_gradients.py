"""verify_olmo_gradients.py — verify GradientCollector hooks work with OLMo models.

Checks that ``GradientCollector`` correctly captures MLP activations and output
gradients from OLMo's ``OLMoSequentialBlock`` feed-forward layers in two
parallelism settings:

* **single** — plain ``OLMo`` on CPU / single GPU.
* **fsdp**   — model wrapped with ``torch.distributed.fsdp.FullyShardedDataParallel``
  (``full_shard``, ``use_orig_params=True``).

Why ``OLMO_MLP_PATTERNS`` is needed
-------------------------------------
OLMo names its block-level feed-forward linears ``ff_proj`` and ``ff_out``.
Their parent path is ``transformer.blocks.<i>``, which does not contain any of
the default heuristic keywords (``mlp``, ``ffn``, ``dense``, …), so
``GradientCollector`` must be given explicit ``name_patterns`` via
:data:`~dattri_llm.gradient.trainers.olmo.OLMO_MLP_PATTERNS`.

The top-level ``transformer.ff_out`` (vocabulary projection, present only when
``weight_tying=False``) is excluded by anchoring patterns to ``blocks.\\d+``.

FSDP notes
----------
The collector is registered on the bare ``OLMo`` module **before** FSDP
wrapping.  ``use_orig_params=True`` tells FSDP to keep the original parameter
objects in place, which preserves the hooked modules and allows the forward /
backward hooks to fire normally.

Usage::

    # Single GPU / CPU
    python scripts/verify_olmo_gradients.py

    # FSDP (2 processes, adjust nproc_per_node as needed)
    torchrun --nproc_per_node=2 scripts/verify_olmo_gradients.py --setup fsdp

Requirements::

    pip install ai2-olmo
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from olmo.config import ActivationType, ModelConfig
    from olmo.model import OLMo
except ImportError as exc:
    raise SystemExit(
        "OLMo is required for this script.\n"
        "Install with:  pip install ai2-olmo\n"
        f"Original error: {exc}"
    ) from exc

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402
from dattri_llm.gradient.trainers.olmo import OLMO_MLP_PATTERNS  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH_SIZE = 2
N_LAYERS = 2
D_MODEL = 64
N_HEADS = 2
MLP_RATIO = 4
ATOL = 1e-5
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Tiny OLMo model
# ---------------------------------------------------------------------------


def make_tiny_olmo(seed: int = SEED) -> OLMo:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        mlp_ratio=MLP_RATIO,
        activation_type=ActivationType.swiglu,
        rope=True,
        flash_attention=False,
        include_bias=False,
        weight_tying=True,
    )
    model = OLMo(cfg)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Reference: raw GradientCollector, no distributed wrapping
# ---------------------------------------------------------------------------


def collect_reference(
    device: torch.device,
    sample_ids: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run forward+backward on *sample_ids* with a fresh model and return a/g.

    This is the ground truth we compare distributed runs against.  A new model
    is built from the same seed so weights are identical.

    Args:
        device: Device to run the reference on.
        sample_ids: Token IDs ``(B, T)`` — the exact same tokens that the
            distributed run processed.
    """
    ref_model = make_tiny_olmo(seed=SEED).to(device)
    ref_model.train()
    ref_collector = GradientCollector(ref_model, name_patterns=OLMO_MLP_PATTERNS)
    ids = sample_ids.to(device)
    ref_model.zero_grad()
    with ref_collector:
        out = ref_model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()
    ag = ref_collector.get_activations_and_grad_outputs()
    ref_collector.remove()
    return ag


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _check(cond: bool, msg: str) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    return cond


def _tensor_stats(t: torch.Tensor) -> str:
    t = t.float()
    return f"shape={list(t.shape)} min={t.min():.3e} max={t.max():.3e} mean={t.mean():.3e}"


def _compare_ag(
    ref: Dict[str, Dict[str, torch.Tensor]],
    test: Dict[str, Dict[str, torch.Tensor]],
    label: str,
) -> bool:
    all_pass = True
    width = max((len(k) for k in ref), default=30)
    print(f"\n  --- {label} ---")
    for layer in sorted(ref.keys()):
        lbl = layer.ljust(width)
        if layer not in test:
            print(f"  FAIL  {lbl}  — layer missing in test output")
            all_pass = False
            continue
        for key in ("activation", "grad_output"):
            r = ref[layer][key].cpu().float()
            d = test[layer][key].cpu().float()
            if r.shape != d.shape:
                print(
                    f"  FAIL  {lbl} [{key}]  "
                    f"shape mismatch: ref={list(r.shape)} test={list(d.shape)}"
                )
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


def _sanity_checks(
    ag: Dict[str, Dict[str, torch.Tensor]],
    *,
    batch_size: int = BATCH_SIZE,
) -> bool:
    """Layer count, finiteness, and SwiGLU shape checks."""
    ok = True
    ok &= _check(len(ag) > 0, f"at least one layer hooked  (found {len(ag)})")
    ok &= _check(
        len(ag) == N_LAYERS * 2,
        f"correct layer count (expected {N_LAYERS * 2}, got {len(ag)})",
    )
    print(f"  layers collected: {sorted(ag.keys())}")

    for layer, entry in ag.items():
        for key in ("activation", "grad_output"):
            t = entry[key]
            ok &= _check(
                torch.isfinite(t).all().item(),
                f"{layer}[{key}]: all finite  {_tensor_stats(t)}",
            )

    # Expected SwiGLU shapes:
    #   ff_proj  activation : (B, T, D)    grad_output : (B, T, 4D)
    #   ff_out   activation : (B, T, 2D)   grad_output : (B, T, D)
    for layer, entry in ag.items():
        act = entry["activation"]
        go = entry["grad_output"]
        if "ff_proj" in layer:
            ok &= _check(
                act.shape == (batch_size, SEQ_LEN, D_MODEL),
                f"{layer}  ff_proj activation shape == (B, T, D_MODEL)",
            )
            ok &= _check(
                go.shape == (batch_size, SEQ_LEN, MLP_RATIO * D_MODEL),
                f"{layer}  ff_proj grad_output shape == (B, T, 4*D_MODEL)",
            )
        elif "ff_out" in layer:
            hidden = MLP_RATIO * D_MODEL // 2  # SwiGLU halves
            ok &= _check(
                act.shape == (batch_size, SEQ_LEN, hidden),
                f"{layer}  ff_out activation shape == (B, T, 2*D_MODEL)",
            )
            ok &= _check(
                go.shape == (batch_size, SEQ_LEN, D_MODEL),
                f"{layer}  ff_out grad_output shape == (B, T, D_MODEL)",
            )
    return ok


# ---------------------------------------------------------------------------
# Setup 1: single GPU / CPU
# ---------------------------------------------------------------------------


def run_single() -> bool:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Single] device={device_str}")

    device = torch.device(device_str)
    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # ---- Primary run ----
    model = make_tiny_olmo(seed=SEED).to(device)
    collector = GradientCollector(model, name_patterns=OLMO_MLP_PATTERNS)

    model.zero_grad()
    with collector:
        out = model(input_ids=ids.to(device))
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids.to(device)[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()

    ok = True
    ok &= _check(collector.get_activations_and_grad_outputs() is not None,
                 "collector populated after forward/backward")

    ag = collector.get_activations_and_grad_outputs()
    ok &= _sanity_checks(ag)
    collector.remove()

    # ---- Compare against a fresh reference model on the same tokens ----
    print("\n  Comparing against raw GradientCollector reference ...")
    ref_ag = collect_reference(device, ids)
    ok &= _compare_ag(ref_ag, ag, "single-GPU run vs raw collector reference")

    # ---- Confirm a different batch yields different activations ----
    print("\n  Checking that a different batch produces different activations ...")
    torch.manual_seed(SEED + 1)
    ids_diff = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    model2 = make_tiny_olmo(seed=SEED).to(device)
    collector2 = GradientCollector(model2, name_patterns=OLMO_MLP_PATTERNS)
    model2.zero_grad()
    with collector2:
        out2 = model2(input_ids=ids_diff.to(device))
        sl2 = out2.logits[:, :-1, :].contiguous()
        lb2 = ids_diff.to(device)[:, 1:].contiguous()
        F.cross_entropy(sl2.view(-1, sl2.shape[-1]), lb2.view(-1)).backward()
    ag2 = collector2.get_activations_and_grad_outputs()
    collector2.remove()

    any_differ = any(
        not torch.allclose(ag[l][k].float(), ag2[l][k].float(), atol=ATOL, rtol=RTOL)
        for l in ag
        if l in ag2
        for k in ("activation", "grad_output")
    )
    ok &= _check(any_differ, "different input produces different activations/gradients")

    return ok


# ---------------------------------------------------------------------------
# Setup 2: FSDP-wrapped model
# ---------------------------------------------------------------------------


def run_fsdp() -> bool:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"\n[FSDP rank {rank}/{world_size}] device={device_str}")

    device = torch.device(device_str)

    # --- Build bare model and register hooks BEFORE FSDP wrapping ---
    # use_orig_params=True preserves original module objects (and their hooks).
    torch.manual_seed(SEED)
    model = make_tiny_olmo(seed=SEED).to(device)
    collector = GradientCollector(model, name_patterns=OLMO_MLP_PATTERNS)

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device if torch.cuda.is_available() else None,
    )

    # --- Forward / backward ---
    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    fsdp_model.zero_grad()
    with collector:
        out = fsdp_model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()

    dist.barrier()

    ok = True
    if rank == 0:
        ok &= _check(
            len(collector.get_activations_and_grad_outputs()) > 0,
            "[rank 0] collector populated after FSDP forward/backward",
        )
        ag = collector.get_activations_and_grad_outputs()

        print(f"\n  [rank 0] sample ids shape: ({BATCH_SIZE}, {SEQ_LEN})")
        ok &= _sanity_checks(ag)

        # Compare rank-0 tensors against a single-GPU reference.
        # Tiny floating-point differences from shard-level all-reduce are expected;
        # the comparison is informational.
        print("\n  [rank 0] Comparing against raw GradientCollector reference ...")
        ref_ag = collect_reference(device, ids.cpu())
        ok &= _compare_ag(
            ref_ag,
            ag,
            "FSDP rank-0 vs raw collector reference (informational)",
        )

    collector.remove()
    dist.destroy_process_group()
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify GradientCollector hooks work with OLMo models"
    )
    parser.add_argument(
        "--setup",
        choices=["single", "fsdp"],
        default="single",
        help="Parallelism setup to test (default: single)",
    )
    cli = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print("=" * 60)
        print("OLMo GradientCollector — model-level hook verification")
        print("=" * 60)
        print(f"  d_model={D_MODEL}  n_layers={N_LAYERS}  mlp_ratio={MLP_RATIO}")
        print(f"  vocab={VOCAB_SIZE}  seq_len={SEQ_LEN}  batch={BATCH_SIZE}")
        print(f"  OLMO_MLP_PATTERNS = {OLMO_MLP_PATTERNS}")

    if cli.setup == "single":
        ok = run_single()
    elif cli.setup == "fsdp":
        ok = run_fsdp()
    else:
        ok = False

    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  {'PASS' if ok else 'FAIL'}  {cli.setup}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
