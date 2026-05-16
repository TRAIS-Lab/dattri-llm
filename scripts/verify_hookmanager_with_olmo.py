"""verify_hookmanager_with_olmo.py — verify HookManager works with OLMo models.

Checks that :class:`~dattri_llm.gradient.hooks.HookManager` correctly captures
MLP activations and output gradients from OLMo's ``OLMoSequentialBlock``
feed-forward layers across several scenarios:

* **single**     — plain ``OLMo`` on CPU / single GPU, single forward+backward.
* **fsdp**       — model wrapped with ``torch.distributed.fsdp.FullyShardedDataParallel``
  (``full_shard``, ``use_orig_params=True``).
* **identity**   — for each MLP linear layer, ``Σ_{b,t} g ⊗ a`` reconstructs
  ``param.weight.grad`` (the fundamental TDA correctness identity).
* **offload**    — multi-step training cycle (``optimizer.step()`` between steps)
  with :class:`OffloadCallback` writing batch files; reload from disk and
  verify tensor-level equality plus correct flush count.
* **per_sample** — ``per_sample`` and ``per_batch`` modes on identical inputs
  agree: per-sample record ``i`` carries the same factors and hash as
  ``per_batch.gradient.slice(dim="batch", index=i)``.

The first port of ``verify_olmo_gradients.py`` (which targets the legacy
``GradientCollector`` API) covered only the single-step capture comparison.
The new modes exercise the multi-step pipeline, the materialized identity,
and the per-sample slicing path.

Why ``OLMO_MLP_PATTERNS`` is needed
-------------------------------------
OLMo names its block-level feed-forward linears ``ff_proj`` and ``ff_out``.
Their parent path is ``transformer.blocks.<i>``, which does not contain any of
the default heuristic keywords (``mlp``, ``ffn``, ``dense``, …), so
:class:`HookManagerConfig` must be given explicit ``mlp_name_patterns`` via
:data:`~dattri_llm.trainers.olmo.OLMO_MLP_PATTERNS`.

The top-level ``transformer.ff_out`` (vocabulary projection, present only when
``weight_tying=False``) is excluded by anchoring patterns to ``blocks.\\d+``.

FSDP notes
----------
The hook manager is registered on the bare ``OLMo`` module **before** FSDP
wrapping.  ``use_orig_params=True`` tells FSDP to keep the original parameter
objects in place, which preserves the hooked modules and allows the forward /
backward hooks to fire normally.

Scripts Usage::

    # Single GPU / CPU — capture sanity + reference comparison
    python scripts/verify_olmo_gradients_hooks.py

    # Factorized identity check
    python scripts/verify_olmo_gradients_hooks.py --setup identity

    # OffloadCallback round-trip in a training cycle
    python scripts/verify_olmo_gradients_hooks.py --setup offload

    # per_sample vs per_batch equivalence
    python scripts/verify_olmo_gradients_hooks.py --setup per_sample

    # Run every non-distributed check back-to-back
    python scripts/verify_olmo_gradients_hooks.py --setup all

    # FSDP (2 processes, adjust nproc_per_node as needed)
    torchrun --nproc_per_node=2 scripts/verify_olmo_gradients_hooks.py --setup fsdp

Requirements::

    pip install ai2-olmo
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
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

from dattri_llm.gradient.hooks import (  # noqa: E402
    HookManager,
    HookManagerCallback,
    HookManagerConfig,
    OffloadCallback,
)
from dattri_llm.gradient.file_manager import GradientFileManager  # noqa: E402
from dattri_llm.gradient.gradient import Factorized, GradientRecord  # noqa: E402
from dattri_llm.trainers.olmo import OLMO_MLP_PATTERNS  # noqa: E402

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
# Helpers: HookManager wiring
# ---------------------------------------------------------------------------


class _InMemoryCallback(HookManagerCallback):
    """Capture every emitted :class:`GradientRecord` in a list."""

    def __init__(self) -> None:
        self.records: list[GradientRecord] = []

    def on_collect_end(self, record: GradientRecord) -> None:
        self.records.append(record)


def _make_hook_manager(model: nn.Module) -> tuple[HookManager, _InMemoryCallback]:
    """Build a per-batch, mlp_io-only HookManager configured for OLMo."""
    cb = _InMemoryCallback()
    cfg = HookManagerConfig(
        recording_type="per_batch",
        hook_types=["mlp_io"],
        mlp_name_patterns=OLMO_MLP_PATTERNS,
    )
    manager = HookManager(model, config=cfg, callbacks=[cb])
    return manager, cb


def _records_to_ag(
    records: list[GradientRecord],
) -> dict[str, dict[str, torch.Tensor]]:
    """Convert per-batch mlp_io records into the legacy ``{layer: {a, g}}`` shape.

    For a single-step verification ``records`` has exactly one entry; the
    factorized data for each MLP layer becomes ``{"activation": ...,
    "grad_output": ...}``.  Per-batch recording preserves the batch and token
    dims, so the resulting tensors match what the legacy ``GradientCollector``
    used to expose and the existing comparison helpers can stay unchanged.
    """
    if not records:
        raise RuntimeError(
            "HookManager emitted no records. "
            "Was backward() called inside the collect() context?"
        )
    if len(records) > 1:
        raise RuntimeError(
            f"Expected exactly one per-batch record for single-step verification, "
            f"got {len(records)}."
        )

    record = records[0]
    ag: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, value in record.gradient.data.items():
        if record.gradient.layer_types and (
            record.gradient.layer_types.get(layer_name) != "mlp_io"
        ):
            continue
        if not isinstance(value, Factorized):
            raise TypeError(
                f"Layer {layer_name!r}: expected Factorized, got {type(value).__name__}."
            )
        ag[layer_name] = {
            "activation": value.activation,
            "grad_output": value.pre_activation_grad,
        }
    return ag


# ---------------------------------------------------------------------------
# Helpers shared by the new modes (identity, offload, per_sample)
# ---------------------------------------------------------------------------


def _materialize_sum(factorized: Factorized) -> torch.Tensor:
    """Return ``Σ_{b,t} (g ⊗ a)``: the materialized weight gradient.

    For ``nn.Linear`` with no bias, ``Y = X @ W.T``, so
    ``dL/dW = Σ_{b,t} g[b,t] ⊗ a[b,t]``, which the einsum ``"bto,bti->oi"``
    computes directly.  The resulting shape matches ``param.weight.grad``.
    """
    a = factorized.activation              # (B, T, in_features)
    g = factorized.pre_activation_grad     # (B, T, out_features)
    return torch.einsum("bto,bti->oi", g.float(), a.float())


def _make_hook_manager_for_mode(
    model: nn.Module, recording_type: str
) -> tuple[HookManager, _InMemoryCallback]:
    """Build a HookManager with the requested recording type, OLMo-targeted."""
    cb = _InMemoryCallback()
    cfg = HookManagerConfig(
        recording_type=recording_type,
        hook_types=["mlp_io"],
        mlp_name_patterns=OLMO_MLP_PATTERNS,
    )
    manager = HookManager(model, config=cfg, callbacks=[cb])
    return manager, cb


# ---------------------------------------------------------------------------
# Reference: raw HookManager, no distributed wrapping
# ---------------------------------------------------------------------------


def collect_reference(
    device: torch.device,
    sample_ids: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
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
    manager, cb = _make_hook_manager(ref_model)
    ids = sample_ids.to(device)
    ref_model.zero_grad()
    with manager.collect():
        out = ref_model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()
    ag = _records_to_ag(cb.records)
    manager.remove()
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
    ref: dict[str, dict[str, torch.Tensor]],
    test: dict[str, dict[str, torch.Tensor]],
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
    ag: dict[str, dict[str, torch.Tensor]],
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
# Setup 1: single GPU
# ---------------------------------------------------------------------------


def run_single() -> bool:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Single] device={device_str}")

    device = torch.device(device_str)
    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # ---- Primary run ----
    model = make_tiny_olmo(seed=SEED).to(device)
    manager, cb = _make_hook_manager(model)

    model.zero_grad()
    with manager.collect():
        out = model(input_ids=ids.to(device))
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids.to(device)[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()

    ok = True
    ok &= _check(
        len(cb.records) == 1,
        f"manager emitted exactly one per-batch record (got {len(cb.records)})",
    )

    ag = _records_to_ag(cb.records)
    ok &= _sanity_checks(ag)
    manager.remove()

    # ---- Compare against a fresh reference model on the same tokens ----
    print("\n  Comparing against raw HookManager reference ...")
    ref_ag = collect_reference(device, ids)
    ok &= _compare_ag(ref_ag, ag, "single-GPU run vs raw HookManager reference")

    # ---- Confirm a different batch yields different activations ----
    print("\n  Checking that a different batch produces different activations ...")
    torch.manual_seed(SEED + 1)
    ids_diff = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    model2 = make_tiny_olmo(seed=SEED).to(device)
    manager2, cb2 = _make_hook_manager(model2)
    model2.zero_grad()
    with manager2.collect():
        out2 = model2(input_ids=ids_diff.to(device))
        sl2 = out2.logits[:, :-1, :].contiguous()
        lb2 = ids_diff.to(device)[:, 1:].contiguous()
        F.cross_entropy(sl2.view(-1, sl2.shape[-1]), lb2.view(-1)).backward()
    ag2 = _records_to_ag(cb2.records)
    manager2.remove()

    any_differ = any(
        not torch.allclose(ag[l][k].float(), ag2[l][k].float(), atol=ATOL, rtol=RTOL)
        for l in ag
        if l in ag2
        for k in ("activation", "grad_output")
    )
    ok &= _check(any_differ, "different input produces different activations/gradients")

    return ok


# ---------------------------------------------------------------------------
# Setup 3: factorized identity vs param.grad
# ---------------------------------------------------------------------------


def run_identity() -> bool:
    """Verify ``Σ_{b,t} (g ⊗ a) == param.weight.grad`` for every OLMo MLP layer.

    For each block's ``ff_proj`` and ``ff_out``, materialize the captured
    factorized form and compare to the layer's ``weight.grad`` set by the same
    backward pass.  Equality here is the fundamental TDA correctness
    property: every downstream attribution method assumes the factorized
    representation is a faithful drop-in for the full weight gradient.
    """
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Identity] device={device_str}")
    device = torch.device(device_str)

    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    model = make_tiny_olmo(seed=SEED).to(device)
    manager, cb = _make_hook_manager_for_mode(model, recording_type="per_batch")

    model.zero_grad()
    with manager.collect():
        out = model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()

    ok = True
    ok &= _check(
        len(cb.records) == 1,
        f"manager emitted exactly one per-batch record (got {len(cb.records)})",
    )
    if not cb.records:
        manager.remove()
        return ok

    record = cb.records[0]
    name_to_param = dict(model.named_parameters())

    # Match the worst tolerance we'd accept from a fused matmul vs a separate
    # einsum on the same float32 tensors.  In practice on these tiny shapes
    # the diff is dominated by summation order, not precision loss.
    identity_atol = 1e-4
    identity_rtol = 1e-4

    width = max((len(k) for k in record.gradient.data), default=30)
    print("\n  --- factorized identity (Σ_{b,t} g⊗a == param.grad) ---")
    for layer_name, fac in record.gradient.data.items():
        lbl = layer_name.ljust(width)
        if not isinstance(fac, Factorized):
            ok &= _check(False, f"{lbl}  expected Factorized, got {type(fac).__name__}")
            continue

        weight_key = f"{layer_name}.weight"
        if weight_key not in name_to_param:
            ok &= _check(False, f"{lbl}  no parameter {weight_key!r}")
            continue
        weight_grad = name_to_param[weight_key].grad
        if weight_grad is None:
            ok &= _check(False, f"{lbl}  param.weight.grad is None — backward did not flow")
            continue

        reconstructed = _materialize_sum(fac)            # CPU, fp32
        expected = weight_grad.detach().cpu().float()
        if reconstructed.shape != expected.shape:
            ok &= _check(
                False,
                f"{lbl}  shape mismatch: reconstructed={list(reconstructed.shape)} "
                f"expected={list(expected.shape)}",
            )
            continue

        diff = (reconstructed - expected).abs()
        passed = torch.allclose(reconstructed, expected, atol=identity_atol, rtol=identity_rtol)
        print(
            f"  {'PASS' if passed else 'FAIL'}  {lbl}  "
            f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}  "
            f"|grad|={expected.abs().mean():.2e}"
        )
        ok &= passed

    manager.remove()
    return ok


# ---------------------------------------------------------------------------
# Setup 4: OffloadCallback round-trip in a training cycle
# ---------------------------------------------------------------------------


def run_offload() -> bool:
    """Multi-step training with OffloadCallback; reload and check tensor equality.

    Drives a real training loop (``optimizer.zero_grad`` → forward → backward →
    ``optimizer.step``) for several steps so the model state actually changes
    between captures.  Verifies:

    * ``mgr.steps_collected`` equals the number of driven steps.
    * The expected number of ``batch_*.pt`` files is on disk, including the
      remainder of an uneven flush interval written on context exit.
    * Every in-memory record reloads from disk with bit-identical factor
      tensors and matching ``input_hash`` / ``step``.
    * Records from different steps actually differ (training is doing
      something, not stuck at the initial weights).
    """
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Offload] device={device_str}")
    device = torch.device(device_str)

    n_steps = 5
    every_n = 2
    # Expected files: floor(n_steps / every_n) full batches + remainder on
    # context exit = 5/2 → 2 full + 1 remainder = 3.
    expected_files = (n_steps // every_n) + (1 if n_steps % every_n else 0)

    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    model = make_tiny_olmo(seed=SEED).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        fm = GradientFileManager(tmpdir)
        offload = OffloadCallback(every_n_steps=every_n, file_manager=fm)
        captured = _InMemoryCallback()

        cfg = HookManagerConfig(
            recording_type="per_batch",
            hook_types=["mlp_io"],
            mlp_name_patterns=OLMO_MLP_PATTERNS,
        )
        manager = HookManager(model, config=cfg, callbacks=[offload, captured])

        with manager.collect():
            for _ in range(n_steps):
                optimizer.zero_grad()
                out = model(input_ids=ids)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                )
                loss.backward()
                optimizer.step()

        # ---- 1. Step count ----
        ok &= _check(
            manager.steps_collected == n_steps,
            f"manager.steps_collected == {n_steps} (got {manager.steps_collected})",
        )
        ok &= _check(
            len(captured.records) == n_steps,
            f"captured {n_steps} records in memory (got {len(captured.records)})",
        )

        # ---- 2. File count ----
        files = sorted(Path(tmpdir).glob("batch_*.pt"))
        ok &= _check(
            len(files) == expected_files,
            f"expected {expected_files} batch files for {n_steps} steps "
            f"with every_n_steps={every_n} (got {len(files)}: {[f.name for f in files]})",
        )
        ok &= _check(
            (Path(tmpdir) / "index.json").exists(),
            "index.json written to save_dir",
        )

        # ---- 3. Reload via fresh manager (exercises on-disk index path) ----
        fresh = GradientFileManager(tmpdir)
        for rec in captured.records:
            sample_hash = rec.input_hash[0]  # per_batch → list[str]
            loaded = fresh.load_by_hash(rec.step, sample_hash)
            same_step = loaded.step == rec.step
            same_hash = loaded.input_hash == rec.input_hash
            same_layers = set(loaded.gradient.data) == set(rec.gradient.data)

            tensors_equal = True
            for layer in rec.gradient.data:
                a_mem = rec.gradient.data[layer].activation
                a_disk = loaded.gradient.data[layer].activation
                g_mem = rec.gradient.data[layer].pre_activation_grad
                g_disk = loaded.gradient.data[layer].pre_activation_grad
                if not (torch.equal(a_mem, a_disk) and torch.equal(g_mem, g_disk)):
                    tensors_equal = False
                    break

            ok &= _check(
                same_step and same_hash and same_layers and tensors_equal,
                f"step {rec.step}: loaded record bit-equal to in-memory "
                f"(step={same_step} hash={same_hash} layers={same_layers} tensors={tensors_equal})",
            )

        # ---- 4. Training actually did something (records differ across steps) ----
        if len(captured.records) >= 2:
            first = captured.records[0].gradient
            last = captured.records[-1].gradient
            any_layer = next(iter(first.data))
            differs = not torch.allclose(
                first.data[any_layer].pre_activation_grad,
                last.data[any_layer].pre_activation_grad,
                atol=0.0,
                rtol=0.0,
            )
            ok &= _check(
                differs,
                f"records from step 0 and step {n_steps - 1} differ "
                "(optimizer.step actually changed the model)",
            )

        manager.remove()

    return ok


# ---------------------------------------------------------------------------
# Setup 5: per_sample vs per_batch equivalence
# ---------------------------------------------------------------------------


def run_per_sample() -> bool:
    """Verify ``per_sample`` record ``i`` == ``per_batch.gradient.slice(i)``.

    The per-sample slicing path in :meth:`HookManager._on_step_complete`
    fires only with ``recording_type="per_sample"`` and is not exercised by
    ``single``/``fsdp``.  This check runs the same input through both modes
    on independent (but seed-identical) models and confirms:

    * Hash typing: ``per_sample.input_hash`` is a ``str``; ``per_batch.input_hash``
      is a ``list[str]`` of length ``B``.
    * Hash equality: ``per_sample[i].input_hash == per_batch.input_hash[i]``.
    * Factor equality: per-sample activation/grad tensors equal the
      corresponding slice of the per-batch tensors (bit-exact).
    """
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[PerSample] device={device_str}")
    device = torch.device(device_str)

    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    # Two fresh seed-identical models so the captures are independent.
    model_batch = make_tiny_olmo(seed=SEED).to(device)
    model_sample = copy.deepcopy(model_batch)

    mgr_b, cb_b = _make_hook_manager_for_mode(model_batch, recording_type="per_batch")
    mgr_s, cb_s = _make_hook_manager_for_mode(model_sample, recording_type="per_sample")

    def _one_step(m, mgr):
        # Reset global RNG before each forward.  OLMo's forward can consume
        # RNG state (dropout layers, etc., even at p=0) so the second run
        # would otherwise see a different RNG stream and produce tensors that
        # differ by float-noise — enough to break bit-exact equality.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        m.zero_grad()
        with mgr.collect():
            out = m(input_ids=ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            loss.backward()

    _one_step(model_batch, mgr_b)
    _one_step(model_sample, mgr_s)

    ok = True
    ok &= _check(
        len(cb_b.records) == 1,
        f"per_batch emitted 1 record (got {len(cb_b.records)})",
    )
    ok &= _check(
        len(cb_s.records) == BATCH_SIZE,
        f"per_sample emitted B={BATCH_SIZE} records (got {len(cb_s.records)})",
    )
    if not cb_b.records or len(cb_s.records) != BATCH_SIZE:
        mgr_b.remove()
        mgr_s.remove()
        return ok

    per_batch_rec = cb_b.records[0]

    # ---- Hash typing ----
    ok &= _check(
        isinstance(per_batch_rec.input_hash, list)
        and len(per_batch_rec.input_hash) == BATCH_SIZE
        and all(isinstance(h, str) and len(h) == 64 for h in per_batch_rec.input_hash),
        f"per_batch.input_hash is list[str] of length {BATCH_SIZE}",
    )
    ok &= _check(
        all(isinstance(r.input_hash, str) and len(r.input_hash) == 64 for r in cb_s.records),
        "per_sample.input_hash is str (sha256 hex) on every record",
    )

    # ---- Hash equality ----
    ok &= _check(
        per_batch_rec.input_hash == [r.input_hash for r in cb_s.records],
        "per_sample[i].input_hash == per_batch.input_hash[i] for all i",
    )

    # ---- Factor equality (per-sample == per-batch sliced) ----
    width = max((len(k) for k in per_batch_rec.gradient.data), default=30)
    print("\n  --- per_sample[i] vs per_batch.slice(i) factors ---")
    for i in range(BATCH_SIZE):
        sliced = per_batch_rec.gradient.slice(dim="batch", index=i)
        per_sample_grad = cb_s.records[i].gradient

        same_layers = sliced.layer_names == per_sample_grad.layer_names
        ok &= _check(same_layers, f"sample {i}: layer sets equal")
        if not same_layers:
            continue

        for layer in sorted(sliced.layer_names):
            lbl = layer.ljust(width)
            a_slice = sliced.data[layer].activation
            g_slice = sliced.data[layer].pre_activation_grad
            a_samp = per_sample_grad.data[layer].activation
            g_samp = per_sample_grad.data[layer].pre_activation_grad

            shapes_ok = a_slice.shape == a_samp.shape and g_slice.shape == g_samp.shape
            tensors_ok = torch.equal(a_slice, a_samp) and torch.equal(g_slice, g_samp)
            a_max_diff = (a_slice - a_samp).abs().max().item() if shapes_ok else float("nan")
            g_max_diff = (g_slice - g_samp).abs().max().item() if shapes_ok else float("nan")
            ok &= _check(
                shapes_ok and tensors_ok,
                f"sample {i}  {lbl}  "
                f"shapes={shapes_ok} tensors_bit_equal={tensors_ok}  "
                f"max_diff(a)={a_max_diff:.2e} max_diff(g)={g_max_diff:.2e}",
            )

    mgr_b.remove()
    mgr_s.remove()
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
    manager, cb = _make_hook_manager(model)

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
    with manager.collect():
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
            len(cb.records) == 1,
            f"[rank 0] manager emitted exactly one per-batch record (got {len(cb.records)})",
        )
        ag = _records_to_ag(cb.records)

        print(f"\n  [rank 0] sample ids shape: ({BATCH_SIZE}, {SEQ_LEN})")
        ok &= _sanity_checks(ag)

        # Compare rank-0 tensors against a single-GPU reference.
        # Tiny floating-point differences from shard-level all-reduce are expected;
        # the comparison is informational.
        print("\n  [rank 0] Comparing against raw HookManager reference ...")
        ref_ag = collect_reference(device, ids.cpu())
        ok &= _compare_ag(
            ref_ag,
            ag,
            "FSDP rank-0 vs raw HookManager reference (informational)",
        )

    manager.remove()
    dist.destroy_process_group()
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_SINGLE_PROCESS_MODES: dict[str, Callable[[], bool]] = {
    "single": run_single,
    "identity": run_identity,
    "offload": run_offload,
    "per_sample": run_per_sample,
}


def run_all() -> bool:
    """Run every non-distributed check in sequence.

    Each sub-check is independent (its own model, its own HookManager); a
    failure in one does not short-circuit the rest, so the summary lists the
    full set of outcomes.
    """
    results: dict[str, bool] = {}
    for name, fn in _SINGLE_PROCESS_MODES.items():
        print("\n" + "=" * 60)
        print(f"  RUNNING: {name}")
        print("=" * 60)
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 — surface unexpected failures
            print(f"  ERROR in {name}: {exc!r}")
            results[name] = False

    print("\n" + "=" * 60)
    print("ALL-MODE RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(results.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify HookManager works with OLMo models"
    )
    parser.add_argument(
        "--setup",
        choices=["single", "fsdp", "identity", "offload", "per_sample", "all"],
        default="single",
        help="Which check to run (default: single).  'all' runs every "
        "non-distributed check back-to-back; 'fsdp' must be launched via torchrun.",
    )
    cli = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print("=" * 60)
        print("OLMo HookManager — model-level hook verification")
        print("=" * 60)
        print(f"  d_model={D_MODEL}  n_layers={N_LAYERS}  mlp_ratio={MLP_RATIO}")
        print(f"  vocab={VOCAB_SIZE}  seq_len={SEQ_LEN}  batch={BATCH_SIZE}")
        print(f"  OLMO_MLP_PATTERNS = {OLMO_MLP_PATTERNS}")

    if cli.setup == "fsdp":
        ok = run_fsdp()
    elif cli.setup == "all":
        ok = run_all()
    elif cli.setup in _SINGLE_PROCESS_MODES:
        ok = _SINGLE_PROCESS_MODES[cli.setup]()
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
