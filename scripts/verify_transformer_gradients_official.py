"""verify_hooks_with_official_trainer.py — verify GradientCollector hooks work
with the official ``transformers.Trainer`` (no GradientCollectingTrainer).

This script attaches a :class:`~dattri_llm.gradient.collector.GradientCollector`
directly to a plain ``transformers.Trainer`` instance to confirm that the hook
machinery is compatible with the stock training loop.

Two lightweight pieces of glue bridge the gap — neither requires subclassing
``Trainer``:

* **GradCollectCallback** — a ``TrainerCallback`` that resets the collector
  buffers in ``on_step_begin`` and reads them in ``on_step_end``.
* **A one-line monkey-patch** on ``trainer.training_step`` that captures the
  raw ``input_ids`` from the batch so the test can identify which samples were
  seen at each step.

After one training step the script compares the collected ``activation`` and
``grad_output`` tensors against a raw :class:`GradientCollector` reference run
on the same token IDs and asserts exact agreement.

Usage::

    # Single-GPU (restrict to 1 GPU to avoid HF Trainer's DataParallel mode)
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_hooks_with_official_trainer.py

    # DDP (2 GPUs)
    torchrun --nproc_per_node=2 scripts/verify_hooks_with_official_trainer.py --setup ddp

    # FSDP (2 GPUs)
    torchrun --nproc_per_node=2 scripts/verify_hooks_with_official_trainer.py --setup fsdp
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
N_SAMPLES = 4
PER_DEVICE_BATCH = 2
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
# Dataset                                                                       #
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


# --------------------------------------------------------------------------- #
# Callback: reset / read collector buffers around each training step           #
# --------------------------------------------------------------------------- #


class GradCollectCallback(TrainerCallback):
    """Brackets each training step with the collector context.

    ``on_step_begin`` resets internal buffers so stale data is never returned.
    ``on_step_end`` assembles and stores the collected activations and output
    gradients.

    The collected a/g are available via :attr:`ag` after training completes.
    The raw ``input_ids`` of the last processed batch are available via
    :attr:`sample_ids`; they are populated by a monkey-patch on
    ``trainer.training_step`` (see :func:`attach_to_trainer`).
    """

    def __init__(self, collector: GradientCollector) -> None:
        self.collector = collector
        self.ag: dict[str, dict[str, torch.Tensor]] | None = None
        self.sample_ids: torch.Tensor | None = None  # set by monkey-patch

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.collector.__enter__()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.collector.__exit__(None, None, None)
        try:
            self.ag = self.collector.get_activations_and_grad_outputs()
        except RuntimeError:
            # No data — backward was not called (e.g. gradient accumulation
            # step where the optimiser has not yet fired).
            pass


# --------------------------------------------------------------------------- #
# Glue: attach collector + callback to an existing Trainer instance            #
# --------------------------------------------------------------------------- #


def attach_to_trainer(
    trainer: Trainer,
    collector: GradientCollector,
) -> GradCollectCallback:
    """Register collector hooks on *trainer* without subclassing it.

    Two things happen:

    1. A :class:`GradCollectCallback` is added to the trainer's callback list.
    2. ``trainer.training_step`` is monkey-patched to capture the ``input_ids``
       from each batch so the callback can expose them for downstream
       traceability (same role as ``"_sample_ids"`` in
       ``GradientCollectingTrainer``).

    Args:
        trainer: Any ``transformers.Trainer`` instance (plain, DDP, or FSDP).
        collector: A :class:`GradientCollector` already registered on
            ``trainer.model``.

    Returns:
        The :class:`GradCollectCallback` that was added, so callers can read
        :attr:`~GradCollectCallback.ag` and
        :attr:`~GradCollectCallback.sample_ids` after training.
    """
    callback = GradCollectCallback(collector)
    trainer.add_callback(callback)

    # Monkey-patch training_step to intercept the batch dict.
    # We bind to the *instance* so the patch does not affect other trainers.
    _original_ts = trainer.training_step

    def _patched_ts(model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        if "input_ids" in inputs:
            callback.sample_ids = inputs["input_ids"].detach().cpu()
        return _original_ts(model, inputs, *args, **kwargs)

    trainer.training_step = _patched_ts  # type: ignore[method-assign]
    return callback


# --------------------------------------------------------------------------- #
# Reference: raw GradientCollector (no Trainer)                                #
# --------------------------------------------------------------------------- #


def collect_reference(
    device: torch.device,
    sample_ids: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run forward+backward on *sample_ids* and return a/g.

    Args:
        device: Device for the reference run.
        sample_ids: Token IDs of shape ``(B, T)`` — the exact same tokens
            that the trainer processed, extracted from the monkey-patch.
    """
    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG).to(device)
    collector = GradientCollector(model)
    ids = sample_ids.to(device)
    model.train()
    with collector:
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()
    ag = collector.get_activations_and_grad_outputs()
    collector.remove()
    return ag


# --------------------------------------------------------------------------- #
# Reporting helpers                                                             #
# --------------------------------------------------------------------------- #


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
# Per-setup runners                                                             #
# --------------------------------------------------------------------------- #


def _make_training_args(tmp_dir: str, **overrides: Any) -> TrainingArguments:
    defaults: dict[str, Any] = dict(
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
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def run_single_gpu(tmp_dir: str) -> bool:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Single-GPU] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)
    collector = GradientCollector(model)

    args = _make_training_args(tmp_dir)
    trainer = Trainer(model=model, args=args, train_dataset=FixedTokenDataset())
    callback = attach_to_trainer(trainer, collector)

    trainer.train()

    ok = True
    ok &= _check(callback.ag is not None, "collector populated after 1 training step")
    ok &= _check(callback.sample_ids is not None, "sample_ids captured via monkey-patch")
    if not ok:
        return False

    ok &= _check(
        len(callback.ag) > 0,
        f"at least one MLP layer hooked  (found {len(callback.ag)})",
    )

    print(f"\n  sample_ids shape: {list(callback.sample_ids.shape)}")
    print(f"  layers collected: {sorted(callback.ag.keys())}")

    # Finiteness check
    for layer, entry in callback.ag.items():
        for key in ("activation", "grad_output"):
            t = entry[key]
            ok &= _check(
                torch.isfinite(t).all().item(),
                f"{layer}[{key}]: all finite  {_tensor_stats(t)}",
            )

    # Compare against raw reference on the same tokens.
    print("\n  Comparing against raw GradientCollector reference …")
    device = torch.device(device_str)
    ref_ag = collect_reference(device, callback.sample_ids)
    ok &= _compare_ag(ref_ag, callback.ag, "official Trainer hooks vs raw collector")

    collector.remove()
    return ok


def run_ddp(tmp_dir: str) -> bool:
    import torch.distributed as dist

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"\n[DDP rank {rank}/{world_size}] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)
    collector = GradientCollector(model)

    args = _make_training_args(tmp_dir, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=args, train_dataset=FixedTokenDataset())
    callback = attach_to_trainer(trainer, collector)

    trainer.train()
    dist.barrier()

    ok = True
    if rank == 0:
        ok &= _check(callback.ag is not None, "[rank 0] collector populated")
        ok &= _check(callback.sample_ids is not None, "[rank 0] sample_ids captured")
        if ok:
            print(f"\n  [rank 0] sample_ids shape: {list(callback.sample_ids.shape)}")
            device = torch.device(device_str)
            ref_ag = collect_reference(device, callback.sample_ids)
            ok &= _compare_ag(ref_ag, callback.ag, "DDP rank-0 official Trainer vs raw collector")

    collector.remove()
    dist.destroy_process_group()
    return ok


def run_fsdp(tmp_dir: str) -> bool:
    import torch.distributed as dist

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"\n[FSDP rank {rank}/{world_size}] device={device_str}")

    set_seed(SEED)
    model = GPT2LMHeadModel(GPT2_CFG)
    collector = GradientCollector(model)

    args = _make_training_args(
        tmp_dir,
        fsdp="full_shard",
        fsdp_config={"use_orig_params": True},
    )
    trainer = Trainer(model=model, args=args, train_dataset=FixedTokenDataset())
    callback = attach_to_trainer(trainer, collector)

    trainer.train()
    dist.barrier()

    ok = True
    if rank == 0:
        ok &= _check(callback.ag is not None, "[rank 0] collector populated")
        ok &= _check(callback.sample_ids is not None, "[rank 0] sample_ids captured")
        if ok:
            print(f"\n  [rank 0] sample_ids shape: {list(callback.sample_ids.shape)}")
            device = torch.device(device_str)
            ref_ag = collect_reference(device, callback.sample_ids)
            ok &= _compare_ag(
                ref_ag, callback.ag,
                "FSDP rank-0 official Trainer vs raw collector (informational)",
            )

    collector.remove()
    dist.destroy_process_group()
    return ok


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify GradientCollector hooks work with the official transformers.Trainer"
    )
    parser.add_argument(
        "--setup",
        choices=["single", "ddp", "fsdp"],
        default="single",
        help="Training setup to test (default: single)",
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))

    with tempfile.TemporaryDirectory() as tmp_dir:
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
            import sys
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import tempfile
    main()
