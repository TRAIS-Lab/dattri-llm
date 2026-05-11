"""verify_verl_gradients.py — verify GradientCollector works with verl-style RL training.

Checks that :class:`~dattri_llm.gradient.collector.GradientCollector` correctly
captures MLP activations and output gradients when the backward pass is driven by
a *response-masked* NLL loss, as used in verl's actor policy-gradient update.

Unlike standard language model training (cross-entropy over all tokens), verl's
actor computes the loss only over *response* tokens.  This script verifies that
the collector captures the correct gradient signal under that regime.

GPT-2 is used as a stand-in for the transformer families verl typically operates
on (LLaMA-2/3, Qwen-2, Mistral, DeepSeek).  GPT-2's ``mlp.c_fc`` and
``mlp.c_proj`` layers share the same parent-path naming convention as those
families, so the default MLP heuristics apply and no explicit patterns are needed.

Two parallelism settings are verified:

* **single** — plain GPT-2 on CPU / single GPU (the reference baseline).
* **fsdp**   — FSDP-wrapped model with ``full_shard`` + ``use_orig_params=True``,
  which is verl's primary distributed strategy.

Response-mask design
---------------------
Each batch has ``SEQ_LEN = PROMPT_LEN + RESPONSE_LEN`` tokens.  The response
mask is 0 for the first ``PROMPT_LEN`` positions and 1 for the remaining
``RESPONSE_LEN`` positions.  The loss (and hence gradient) is restricted to
those response positions, just as in verl's actor update.

Usage::

    # Single GPU / CPU
    python scripts/verify_verl_gradients.py

    # FSDP (2 processes, adjust nproc_per_node as needed)
    torchrun --nproc_per_node=2 scripts/verify_verl_gradients.py --setup fsdp

Requirements::

    pip install transformers torch
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError as exc:
    raise SystemExit(
        "transformers is required for this script.\n"
        "Install with:  pip install transformers\n"
        f"Original error: {exc}"
    ) from exc

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402
from dattri_llm.trainers.verl import (  # noqa: E402
    VERL_MLP_PATTERNS,
    compute_response_nll_loss,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
VOCAB_SIZE = 64
PROMPT_LEN = 8
RESPONSE_LEN = 8
SEQ_LEN = PROMPT_LEN + RESPONSE_LEN   # 16
BATCH_SIZE = 2
N_LAYERS = 2
N_EMBD = 64
N_HEAD = 2
MLP_HIDDEN = N_EMBD * 4               # GPT-2 default 4x MLP expansion
ATOL = 1e-5
RTOL = 1e-4

GPT2_CFG = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=SEQ_LEN,
    n_embd=N_EMBD,
    n_layer=N_LAYERS,
    n_head=N_HEAD,
)


def _make_response_mask(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a response mask: 0 for prompt tokens, 1 for response tokens."""
    mask = torch.zeros(batch_size, SEQ_LEN)
    mask[:, PROMPT_LEN:] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Tiny GPT-2 model factory
# ---------------------------------------------------------------------------


def make_tiny_gpt2(seed: int = SEED) -> GPT2LMHeadModel:
    """Create a reproducible tiny GPT-2 model for testing."""
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2_CFG)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Reference: raw GradientCollector, no distributed wrapping
# ---------------------------------------------------------------------------


def collect_reference(
    device: torch.device,
    sample_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run a verl-style forward+backward on *sample_ids* and return a/g.

    A fresh model is built from the same seed so weights are identical to
    the primary run being compared.

    Args:
        device: Device to run the reference on.
        sample_ids: Token IDs ``(B, T)`` — the same tokens used by the
            primary run.
        response_mask: Float mask ``(B, T)`` — 1 for response tokens.
    """
    ref_model = make_tiny_gpt2(seed=SEED).to(device)
    ref_model.train()
    ref_collector = GradientCollector(ref_model, name_patterns=VERL_MLP_PATTERNS)

    ids = sample_ids.to(device)
    mask = response_mask.to(device)

    ref_model.zero_grad()
    with ref_collector:
        out = ref_model(input_ids=ids)
        loss = compute_response_nll_loss(out.logits, ids, mask)
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
    """Layer count, finiteness, and GPT-2 MLP shape checks."""
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

    # Expected GPT-2 MLP shapes:
    #   c_fc   activation : (B, T, N_EMBD)     grad_output : (B, T, 4*N_EMBD)
    #   c_proj activation : (B, T, 4*N_EMBD)   grad_output : (B, T, N_EMBD)
    for layer, entry in ag.items():
        act = entry["activation"]
        go = entry["grad_output"]
        if "c_fc" in layer:
            ok &= _check(
                act.shape == (batch_size, SEQ_LEN, N_EMBD),
                f"{layer}  c_fc activation shape == (B, T, N_EMBD)",
            )
            ok &= _check(
                go.shape == (batch_size, SEQ_LEN, MLP_HIDDEN),
                f"{layer}  c_fc grad_output shape == (B, T, 4*N_EMBD)",
            )
        elif "c_proj" in layer:
            ok &= _check(
                act.shape == (batch_size, SEQ_LEN, MLP_HIDDEN),
                f"{layer}  c_proj activation shape == (B, T, 4*N_EMBD)",
            )
            ok &= _check(
                go.shape == (batch_size, SEQ_LEN, N_EMBD),
                f"{layer}  c_proj grad_output shape == (B, T, N_EMBD)",
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
    response_mask = _make_response_mask(BATCH_SIZE)

    # ---- Primary run ----
    model = make_tiny_gpt2(seed=SEED).to(device)
    collector = GradientCollector(model, name_patterns=VERL_MLP_PATTERNS)

    model.zero_grad()
    with collector:
        out = model(input_ids=ids.to(device))
        loss = compute_response_nll_loss(
            out.logits, ids.to(device), response_mask.to(device)
        )
        loss.backward()

    ok = True
    ok &= _check(
        collector.get_activations_and_grad_outputs() is not None,
        "collector populated after verl-style forward/backward",
    )

    ag = collector.get_activations_and_grad_outputs()
    ok &= _sanity_checks(ag)
    collector.remove()

    # ---- Compare against a fresh reference model on the same tokens ----
    print("\n  Comparing against raw GradientCollector reference ...")
    ref_ag = collect_reference(device, ids, response_mask)
    ok &= _compare_ag(ref_ag, ag, "single-GPU run vs raw collector reference")

    # ---- Verify that a response-masked loss gives non-zero gradients ----
    print("\n  Checking response mask effect: response grad_outputs must be non-zero ...")
    any_nonzero = any(
        entry["grad_output"].abs().max().item() > 0
        for entry in ag.values()
    )
    ok &= _check(any_nonzero, "at least one layer has non-zero grad_output (response mask applied)")

    # ---- Confirm a different batch yields different activations ----
    print("\n  Checking that a different batch produces different activations ...")
    torch.manual_seed(SEED + 1)
    ids_diff = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    model2 = make_tiny_gpt2(seed=SEED).to(device)
    collector2 = GradientCollector(model2, name_patterns=VERL_MLP_PATTERNS)
    model2.zero_grad()
    with collector2:
        out2 = model2(input_ids=ids_diff.to(device))
        loss2 = compute_response_nll_loss(
            out2.logits, ids_diff.to(device), response_mask.to(device)
        )
        loss2.backward()
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
    # use_orig_params=True preserves original module objects (and their hooks),
    # matching the recommended verl FSDP configuration.
    torch.manual_seed(SEED)
    model = make_tiny_gpt2(seed=SEED).to(device)
    collector = GradientCollector(model, name_patterns=VERL_MLP_PATTERNS)

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device if torch.cuda.is_available() else None,
    )

    # --- verl-style forward / backward ---
    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    response_mask = _make_response_mask(BATCH_SIZE).to(device)

    fsdp_model.zero_grad()
    with collector:
        out = fsdp_model(input_ids=ids)
        loss = compute_response_nll_loss(out.logits, ids, response_mask)
        loss.backward()

    dist.barrier()

    ok = True
    if rank == 0:
        ok &= _check(
            len(collector.get_activations_and_grad_outputs()) > 0,
            "[rank 0] collector populated after FSDP verl-style forward/backward",
        )
        ag = collector.get_activations_and_grad_outputs()

        print(f"\n  [rank 0] sample ids shape: ({BATCH_SIZE}, {SEQ_LEN})")
        ok &= _sanity_checks(ag)

        # Compare rank-0 tensors against a single-GPU reference.
        print("\n  [rank 0] Comparing against raw GradientCollector reference ...")
        ref_ag = collect_reference(device, ids.cpu(), response_mask.cpu())
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
        description="Verify GradientCollector hooks work with verl-style RL training"
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
        print("verl GradientCollector — response-masked gradient verification")
        print("=" * 60)
        print(f"  model=GPT-2  n_embd={N_EMBD}  n_layers={N_LAYERS}")
        print(f"  vocab={VOCAB_SIZE}  seq_len={SEQ_LEN}  batch={BATCH_SIZE}")
        print(f"  prompt_len={PROMPT_LEN}  response_len={RESPONSE_LEN}")
        print(f"  VERL_MLP_PATTERNS = {VERL_MLP_PATTERNS!r}  (None → default heuristics)")

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
