"""Fidelity protocol: one AdamW trajectory, captured gradients, TSLOO truth.

One deterministic training trajectory (fixed batch order, no dropout) is run
three ways:

1. **Reference**, wrapped in a ``HookManager`` so the per-sample gradients
   and the optimizer's moments around every update are cached (the
   store-then-attribute workflow), or snapshotted for a later replay
   (``recompute``, the full-dimension setting).
2. **Query capture** at the final model: raw gradients of the validation
   blocks.
3. **TSLOO retraining**: for each selected training block the same
   trajectory is rerun with that block dropped from its batch; the ground
   truth is the change in each validation block's loss.

AdamW-influence is then scored from the caches and compared with the ground
truth by the Spearman correlation across the selected blocks, averaged over
validation blocks.  ``run`` drives one run end to end; ``fidelity.py`` lists
the runs.
"""

from __future__ import annotations

import hashlib
import os
import json
import pathlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from dattri_llm.attribution import AdamWInfluenceAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.callbacks import (
    HookManagerCallback,
    OffloadCallback,
    OptimizerStateCallback,
    ParameterSnapshotCallback,
)
from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.gradient.optimizer_state import OptimizerSnapshot
from dattri_llm.gradient.snapshots import TrajectorySnapshots
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource, ReplayGradientSource

DYNAMICS_FILE = "adamw_dynamics.pt"
METHOD = "adamw_influence"
# TSLOO is the loss without the block minus the reference loss;
# AdamW-influence scores ``-grad^T W Z``, the opposite sign.
SIGN = -1.0

# A loss over the samples at ``idx`` of a dataset: (model, idx) -> scalar.
StepFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass
class Setting:
    """Everything one (scale, lr, seed) run needs."""

    name: str
    out_dir: pathlib.Path
    seed: int
    build_model: Callable[[], nn.Module]
    make_optimizer: Callable[[nn.Module, int], tuple[torch.optim.Optimizer, object]]
    n_train: int
    batch_size: int
    epochs: int
    train_step: StepFn  # mean loss over the batch (as trained)
    val_sum_step: StepFn  # sum over the batch of per-sample val losses
    val_losses: Callable[[nn.Module], torch.Tensor]  # (n_val,) under no_grad
    n_val: int
    val_batch_size: int
    selected: torch.Tensor  # train indices with TSLOO ground truth
    n_masks: int | None = None  # random-mask ensemble (None: every coordinate)
    mask_dim: int = 512  # coordinates per mask and layer
    hook_layers: list[str] | None = None  # linear_io regexes (None: every layer)
    tsloo_from: pathlib.Path | None = None  # reuse another run's ground truth
    tsloo_only: bool = False  # ground truth only: no capture, no attribution
    # Snapshot the trajectory (parameters, batches, moments) instead of
    # storing gradients, and recompute them at attribution time through the
    # library's replay source -- the no-mask (full-dimension) setting.
    recompute: bool = False
    batch: Callable[[torch.Tensor], object] | None = None  # indices -> batch
    loss_fn: Callable[[nn.Module, object], torch.Tensor] | None = None
    device: str = "cuda"
    extra: dict = field(default_factory=dict)
    # Raw tensors and the LR schedule factor, for drivers that replay the
    # trajectory with another trainer (Bergson's MAGIC).
    data: dict = field(default_factory=dict)
    lr_factor: Callable[[int, int], float] | None = None  # (step, n_steps) -> factor


# --------------------------------------------------------------------------- #
# Trajectory                                                                   #
# --------------------------------------------------------------------------- #


def make_batches(n: int, batch_size: int, epochs: int, seed: int) -> list[torch.Tensor]:
    out = []
    for epoch in range(epochs):
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(n, generator=gen)
        out.extend(perm[i : i + batch_size] for i in range(0, n, batch_size))
    return out


def run_trajectory(
    s: Setting,
    batches: list[torch.Tensor],
    *,
    exclude: torch.Tensor | None = None,
    exclude_at: int | None = None,
    hooks: HookManager | None = None,
    after_step: Callable[[int], None] | None = None,
) -> tuple[nn.Module, dict[int, float]]:
    """Train from the seeded init along *batches*; *exclude* drops those
    sample indices from every batch (TSLOO), or only from the batch at step
    *exclude_at*.  Returns the model and the learning rate applied at each
    step.
    """
    torch.manual_seed(s.seed)
    model = s.build_model().to(s.device)
    optimizer, scheduler = s.make_optimizer(model, len(batches))
    lrs: dict[int, float] = {}
    ctx = hooks.collect() if hooks is not None else _null()
    with ctx:
        for t, idx in enumerate(batches):
            if exclude is not None and exclude_at in (None, t):
                idx = idx[~torch.isin(idx, exclude)]
            lrs[t] = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            s.train_step(model, idx).backward()
            optimizer.step()
            if after_step is not None:
                after_step(t)
            if scheduler is not None:
                scheduler.step()
    return model, lrs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _IdRecorder(HookManagerCallback):
    """The per-step sample hashes, in batch order."""

    def __init__(self) -> None:
        self.hashes: dict[int, list[str]] = {}

    def on_step_end(self, record: GradientRecord) -> None:
        h = record.input_hash
        self.hashes[record.step] = list(h) if isinstance(h, list) else [h]


def _linear_io(s: Setting):
    """The ``linear_io`` selector of the setting."""
    return REGISTER_ALL if s.hook_layers is None else list(s.hook_layers)


def hooked_layers(s: Setting) -> tuple[list[str], dict[str, int]]:
    """Names and flat widths of the layers a default HookManager hooks."""
    torch.manual_seed(s.seed)
    model = s.build_model()
    opt, _ = s.make_optimizer(model, 1)
    hm = HookManager(model, config=HookManagerConfig(linear_io=_linear_io(s)))
    names = list(hm.layer_names)
    hm.remove()
    snap = OptimizerSnapshot(model, opt)
    return names, {n: snap.width(n) for n in names}


def mask_projection(s: Setting) -> dict[str, dict] | None:
    """One ``"mask"`` capture holding all ``n_masks`` masks: ``n_masks *
    mask_dim`` random coordinates of every hooked layer.
    """
    if s.n_masks is None:
        return None
    names, widths = hooked_layers(s)
    return {
        n: {
            "style": "mask",
            "proj_dim": min(widths[n], s.n_masks * s.mask_dim),
            "proj_seed": s.seed,
        }
        for n in names
    }


def snapshot_dir(s: Setting) -> pathlib.Path:
    """Where a recompute run keeps its snapshots: under the run directory, or
    under ``$FIDELITY_SNAPSHOT_DIR`` when it is set."""
    root = os.environ.get("FIDELITY_SNAPSHOT_DIR")
    if root:
        return pathlib.Path(root) / s.out_dir.name / "train_snapshots"
    return s.out_dir / "train_snapshots"


def capture_reference(
    s: Setting, batches: list[torch.Tensor], projection: dict | None
) -> tuple[nn.Module, dict[str, int]]:
    """The reference run under hooks; caches the train gradients and the optimizer dynamics."""
    train_dir = snapshot_dir(s) if s.recompute else s.out_dir / "train_grads"
    shutil.rmtree(train_dir, ignore_errors=True)
    torch.manual_seed(s.seed)
    model = s.build_model().to(s.device)
    optimizer, scheduler = s.make_optimizer(model, len(batches))
    ids = _IdRecorder()
    if s.recompute:
        snaps = TrajectorySnapshots(train_dir)
        recorder = OptimizerStateCallback(
            model, optimizer, projection_kwargs=projection, snapshots=snaps
        )
        callbacks = [ParameterSnapshotCallback(model, snaps), recorder, ids]
    else:
        snaps = None
        store = GradientStorageManager(str(train_dir))
        recorder = OptimizerStateCallback(
            model, optimizer, projection_kwargs=projection
        )
        callbacks = [OffloadCallback(1, store), recorder, ids]
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=_linear_io(s), projection_kwargs=projection),
        callbacks=callbacks,
    )
    with hm.collect():
        for t, idx in enumerate(batches):
            if snaps is not None:
                snaps.save_batch(t, s.batch(idx))
            optimizer.zero_grad(set_to_none=True)
            s.train_step(model, idx).backward()
            optimizer.step()
            recorder.record_post(t)
            if scheduler is not None:
                scheduler.step()
    hm.remove()
    if snaps is None:
        torch.save(recorder.dynamics(), train_dir / DYNAMICS_FILE)
    hash_to_index: dict[str, int] = {}
    for t, idx in enumerate(batches):
        for h, i in zip(ids.hashes[t], idx.tolist(), strict=True):
            hash_to_index.setdefault(h, i)
    return model, hash_to_index


def capture_queries(
    s: Setting, model: nn.Module, projection: dict | None
) -> dict[str, int]:
    """Raw gradients of every validation point at the final model."""
    test_dir = s.out_dir / "test_grads"
    shutil.rmtree(test_dir, ignore_errors=True)
    store = GradientStorageManager(str(test_dir))
    ids = _IdRecorder()
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=_linear_io(s), projection_kwargs=projection),
        callbacks=[OffloadCallback(1, store), ids],
    )
    order = []
    with hm.collect():
        for start in range(0, s.n_val, s.val_batch_size):
            idx = torch.arange(start, min(start + s.val_batch_size, s.n_val))
            model.zero_grad(set_to_none=True)
            s.val_sum_step(model, idx).backward()
            order.append(idx)
    hm.remove()
    out: dict[str, int] = {}
    for t, idx in enumerate(order):
        for h, i in zip(ids.hashes[t], idx.tolist(), strict=True):
            out.setdefault(h, i)
    return out


# --------------------------------------------------------------------------- #
# Ground truth                                                                 #
# --------------------------------------------------------------------------- #


def occurrences(batches: list[torch.Tensor], i: int) -> list[int]:
    return [t for t, idx in enumerate(batches) if bool((idx == i).any())]


def tsloo_pairs(s: Setting, batches: list[torch.Tensor]) -> list[tuple[int, int]]:
    """``(block, step)`` pairs of the ground truth: each selected block and
    the step it is removed at (its last occurrence; with one epoch, its only
    one)."""
    return [(i, occurrences(batches, i)[-1]) for i in s.selected.tolist()]


def tsloo(
    s: Setting, batches: list[torch.Tensor], ref_losses: torch.Tensor, log: Callable
) -> torch.Tensor:
    """``(n_selected, n_val)``: val-loss change from dropping each sample."""
    pairs = tsloo_pairs(s, batches)
    out = torch.zeros(len(pairs), ref_losses.numel())
    t0 = time.time()
    for k, (i, t) in enumerate(pairs):
        model, _ = run_trajectory(s, batches, exclude=torch.tensor([i]), exclude_at=t)
        out[k] = s.val_losses(model).cpu() - ref_losses
        if (k + 1) % 10 == 0 or k + 1 == len(pairs):
            log(f"  tsloo {k + 1}/{len(pairs)}  {time.time() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #


def _args(s: Setting) -> AttributionArguments:
    return AttributionArguments(
        output_dir=str(s.out_dir / "scores"),
        per_device_train_batch_size=s.batch_size,
        per_device_eval_batch_size=s.val_batch_size,
        use_cpu=s.device == "cpu",
        dataloader_pin_memory=False,
    )


@dataclass
class Rows:
    """Row-level scores: one row per (train sample, step)."""

    scores: torch.Tensor
    train_ids: list[str]
    steps: list[int]
    test_ids: list[str]

    @classmethod
    def of(cls, score) -> Rows:
        return cls(
            score.scores.clone(),
            list(score.row_train_ids),
            list(score.row_steps),
            list(score.test_ids),
        )

    def add(self, other: Rows) -> None:
        assert self.train_ids == other.train_ids and self.steps == other.steps
        assert self.test_ids == other.test_ids
        self.scores += other.scores


def _matrix(
    rows: Rows, train_map, test_map, pairs: list[tuple[int, int | None]], n_val
) -> torch.Tensor:
    """``(n_pairs, n_val)``: for each ``(sample, step)`` the score row at that
    step, or the sample's rows summed over its steps when the step is None."""
    col = {test_map[h]: c for c, h in enumerate(rows.test_ids) if h in test_map}
    cols = [col[v] for v in range(n_val)]
    out = torch.zeros(len(pairs), n_val)
    by_sample: dict[int, list[tuple[int, int]]] = {}
    for r, (h, t) in enumerate(zip(rows.train_ids, rows.steps, strict=True)):
        if h in train_map:
            by_sample.setdefault(train_map[h], []).append((t, r))
    for k, (i, step) in enumerate(pairs):
        for t, r in by_sample[i]:
            if step is None or step == t:
                out[k] += rows.scores[r][cols]
    return out


def _stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha1(name.encode()).digest()[:4], "little")


def _slice_mask(train_dir, test_dir, dynamics, layers, k_of, mask, n_masks, root):
    """Per-mask memory stores holding one disjoint column group per layer.

    The groups are drawn from a seeded random permutation of the captured
    columns.
    """
    cols = {}
    for n in layers:
        perm = torch.randperm(
            k_of[n] * n_masks, generator=torch.Generator().manual_seed(_stable_seed(n))
        )
        cols[n] = perm[mask * k_of[n] : (mask + 1) * k_of[n]].sort().values

    def cut(block):
        return block.map_layers(lambda n, v, _lt: v[:, cols[n]])

    args = AttributionArguments(output_dir=str(root), use_cpu=True)
    stores = []
    for src_dir, tag in ((train_dir, "train"), (test_dir, "test")):
        store = GradientStorageManager(str(root / f"{tag}_{mask}"), residency="memory")
        for step, block, hashes in DiskGradientSource(
            GradientStorageManager(str(src_dir)), args
        ):
            store.save_bulk(
                [
                    GradientRecord(
                        step=step, input_hash=list(hashes), gradient=cut(block)
                    )
                ]
            )
        stores.append(store)
    dyn = {}
    for step, d in dynamics.items():
        dyn[step] = {
            **d,
            "pre": {n: tuple(x[cols[n]] for x in d["pre"][n]) for n in layers},
            "post": {n: tuple(x[cols[n]] for x in d["post"][n]) for n in layers},
        }
    return stores[0], stores[1], dyn


def attribute(
    s: Setting,
    train_map: dict[str, int],
    test_map: dict[str, int],
    projection: dict | None,
    log: Callable,
    model: nn.Module | None = None,
) -> Rows:
    """AdamW-influence row-level scores (averaged over the masks)."""
    args = _args(s)
    test_dir = s.out_dir / "test_grads"
    if s.recompute:
        # The trajectory is replayed from its snapshots by the library: the
        # attributors pull each step's gradients through the same hooks the
        # capture used, and read the moments from the snapshot store.
        train_dir = snapshot_dir(s)
        replay = ReplayGradientSource(
            model,
            args,
            TrajectorySnapshots(train_dir),
            loss_fn=s.loss_fn,
            config=HookManagerConfig(
                linear_io=_linear_io(s), projection_kwargs=projection
            ),
        )
        dynamics = None
    else:
        train_dir = s.out_dir / "train_grads"
        dynamics = torch.load(train_dir / DYNAMICS_FILE, weights_only=False)

    def score(train_src, test_src, dyn) -> Rows:
        # The recurrence is carried on the query side ("test").
        t0 = time.time()
        rows = Rows.of(AdamWInfluenceAttributor(args).attribute_from_cache(
            train_src,
            test_src,
            dynamics=dyn,
            loss_reduction="mean",
            propagation="test",
            loop_over_test=s.recompute,
            block_dtype=torch.bfloat16 if s.recompute else torch.float32,
        ))
        log(f"  attribution: {time.time() - t0:.1f}s")
        return rows

    if s.recompute:
        with replay:
            return score(replay, str(test_dir), None)
    if projection is None:
        return score(str(train_dir), str(test_dir), dynamics)
    layers = sorted(projection)
    k_of = {n: projection[n]["proj_dim"] // s.n_masks for n in layers}
    total: Rows | None = None
    for mask in range(s.n_masks):
        tr, te, dyn = _slice_mask(
            train_dir,
            test_dir,
            dynamics,
            layers,
            k_of,
            mask,
            s.n_masks,
            s.out_dir / "masks",
        )
        with tr, te:
            part = score(tr, te, dyn)
        if total is None:
            total = part
        else:
            total.add(part)
        log(f"  mask {mask + 1}/{s.n_masks} done")
    total.scores /= s.n_masks
    return total


# --------------------------------------------------------------------------- #
# Metric and driver                                                            #
# --------------------------------------------------------------------------- #


def _rank(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(dim=0).argsort(dim=0).double()


def spearman_per_column(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman across rows, one value per column."""
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(0, keepdim=True)
    rb = rb - rb.mean(0, keepdim=True)
    return (ra * rb).sum(0) / (ra.norm(dim=0) * rb.norm(dim=0) + 1e-12)


def peak_gb() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"


def run_tsloo_only(s: Setting, batches: list[torch.Tensor], log: Callable) -> dict:
    """The ground truth alone: reference trajectory, validation losses and the
    leave-one-out reruns; ``matrices.pt`` then serves later attribution runs
    through ``--tsloo-from``."""
    t0 = time.time()
    model, _ = run_trajectory(s, batches)
    ref_losses = s.val_losses(model).cpu()
    log(f"  reference: {time.time() - t0:.0f}s, {len(batches)} steps, {peak_gb()}")
    pairs = tsloo_pairs(s, batches)
    t0 = time.time()
    truth = tsloo(s, batches, ref_losses, log)
    torch.save(
        {"tsloo": truth, "selected": s.selected, "pairs": pairs, "ref_losses": ref_losses},
        s.out_dir / "matrices.pt",
    )
    result = {
        "name": s.name, "seed": s.seed, **s.extra,
        "n_selected": int(s.selected.numel()), "n_val": int(ref_losses.numel()),
        "tsloo_s": round(time.time() - t0, 1), "tsloo_abs_mean": float(truth.abs().mean()),
        "tsloo_abs_max": float(truth.abs().max()), "tsloo_only": True,
    }
    log(f"  result: {json.dumps(result)}")
    (s.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def run(s: Setting) -> dict:
    s.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = s.out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"== {s.name} seed={s.seed} {s.extra}")
    if s.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    if s.tsloo_only:
        return run_tsloo_only(s, batches, log)
    projection = mask_projection(s)
    if projection is not None:
        log(f"  masks: {s.n_masks} x {s.mask_dim} over {len(projection)} layers")
    t0 = time.time()
    model, train_map = capture_reference(s, batches, projection)
    ref_losses = s.val_losses(model).cpu()
    test_map = capture_queries(s, model, projection)
    if s.device != "cpu":
        torch.cuda.synchronize()
    timing = {"train_s": round(time.time() - t0, 1)}
    peaks = [torch.cuda.max_memory_allocated() / 2**30 if s.device != "cpu" else 0.0]
    log(
        f"  reference + capture: {time.time() - t0:.0f}s, {len(batches)} steps, {peak_gb()}"
    )
    # Keep the selected samples whose content hash maps back to them.
    ids = {v: k for k, v in train_map.items()}
    keep = [i for i in s.selected.tolist() if i in ids]
    s.selected = torch.tensor(keep)
    if s.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t_attr = time.time()
    rows = attribute(s, train_map, test_map, projection, log, model=model)
    if s.device != "cpu":
        torch.cuda.synchronize()
    timing["attribute_s"] = round(time.time() - t_attr, 1)
    timing["total_s"] = round(timing["train_s"] + timing["attribute_s"], 1)
    peaks.append(torch.cuda.max_memory_allocated() / 2**30 if s.device != "cpu" else 0.0)
    timing["peak_gb"] = round(max(peaks), 2)
    log(f"  attribution total: {time.time() - t_attr:.0f}s, {peak_gb()}")
    if s.recompute:  # delete the intermediate snapshots and query gradients
        shutil.rmtree(snapshot_dir(s), ignore_errors=True)
        shutil.rmtree(s.out_dir / "test_grads", ignore_errors=True)
    pairs = tsloo_pairs(s, batches)
    scores = _matrix(rows, train_map, test_map, pairs, s.n_val)
    if s.tsloo_from is not None:
        saved = torch.load(s.tsloo_from / "matrices.pt")
        assert torch.equal(saved["selected"], s.selected)
        truth = saved["tsloo"][:, : s.n_val]  # the first n_val queries
        log(f"  tsloo reused from {s.tsloo_from}")
    else:
        truth = tsloo(s, batches, ref_losses, log)
    torch.save(
        {"tsloo": truth, "selected": s.selected, "pairs": pairs, METHOD: scores},
        s.out_dir / "matrices.pt",
    )
    rho = spearman_per_column(scores, SIGN * truth)
    result = {
        "name": s.name, "seed": s.seed, **s.extra, "n_selected": len(keep), **timing,
        METHOD: float(rho.mean()), f"{METHOD}_std_over_val": float(rho.std()),
    }
    log(f"  result: {json.dumps(result)}")
    with (s.out_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result
