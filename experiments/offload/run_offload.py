"""Baseline for the gradient-offload write path.

Unlike ``experiments/efficiency`` this is not a head-to-head against another
library -- it measures *our own* store under the settings that actually change
what offloading costs, so that later work on the write path (async write-behind,
pinned staging, memmap for factorized groups) has a before/after number instead
of an argument.

What it measures, per configuration:

* the per-phase save breakdown from ``GradientStorageManager.timing``
  (``to_cpu`` / ``write_group`` / ``index_update`` / ``index_write`` / ``spill``),
* total training walltime and the **share of it spent inside offloading** --
  the number that decides whether moving writes off the training thread is
  worth anything,
* bytes actually written to disk.

Every config differs from the ``factorized/pickle/raw`` baseline along exactly
one axis, so any pair including the baseline isolates that axis:

===================  ==========================  ============================
axis                 values                      what it changes
===================  ==========================  ============================
``proj_dim``         raw, 64, 16                 how many bytes reach the
                                                 store at all
``disk_format``      pickle, memmap              serializer cost per byte
representation       factorized, materialized    payload size and shape
``offload_to_cpu``   False, True                 whether the device->host copy
                                                 lands in ``to_cpu`` or in the
                                                 (uninstrumented) hooks
``offload_interval`` 1, 8, 32                    stall frequency vs stall size
``residency``        disk, tiered                exercises the spill path
===================  ==========================  ============================

Usage::

    python run_offload.py                    # the default sweep
    python run_offload.py --repeat 3         # medians over 3 runs per config
    python run_offload.py --quick            # a fast smoke sweep
    python run_offload.py --steps 200 --batch-size 16
    python run_offload.py --list             # show the configs, run nothing

On a disk with variable throughput (cloud VMs) use ``--repeat 3`` or more: the
``x`` column reports max/min walltime across repeats, and anything above ~1.5
means the disk, not the config, is setting the pace.

Results are appended to ``results.jsonl`` next to this file, one line per
(config, phase) plus a ``summary`` line per config.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
# Run from a checkout without `pip install -e .`, like the other runners assume.
sys.path.insert(0, str(HERE.parents[1]))

from dattri_llm.gradient.callbacks.base import HookManagerCallback  # noqa: E402
from dattri_llm.gradient.gradient import GradientRecord  # noqa: E402


def available_ram_bytes() -> int:
    """Host RAM available for page cache, or 0 when it cannot be read."""
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:  # macOS / BSD
        import subprocess

        out = subprocess.run(  # noqa: S603
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def describe_filesystem(path: Path) -> str:
    """Mount point and fs type backing *path*, e.g. ``"/scratch (ext4)"``.

    A store written to a tmpfs mount measures memory bandwidth, not disk, so
    which filesystem was used has to travel with the result.
    """
    try:
        target = path.resolve()
        best = ("", "")
        with Path("/proc/mounts").open(encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                mount, fstype = parts[1], parts[2]
                if str(target).startswith(mount) and len(mount) > len(best[0]):
                    best = (mount, fstype)
        if best[0]:
            return f"{best[0]} ({best[1]})"
    except (OSError, IndexError):
        pass
    return "unknown"


def dir_bytes(path: str | Path) -> int:
    """Total size of every file under *path*."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class Results:
    """Collects rows and appends them to ``results.jsonl``.

    Deliberately local rather than reusing ``efficiency/common.Meter``: that
    harness measures phases itself, whereas the phase timings here come from
    ``GradientStorageManager.timing``, so all it would contribute is the JSON
    shape -- not worth a cross-folder import.
    """

    def __init__(self, out_json: Path, meta: dict) -> None:
        self.out_json = out_json
        self.meta = meta
        self.rows: list[dict] = []

    def add(self, **kv: object) -> None:
        self.rows.append({**kv, **self.meta})

    def flush(self) -> None:
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        with self.out_json.open("a") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        self.rows.clear()


class MaterializeCallback(HookManagerCallback):
    """Materializes each record before handing it to the wrapped callback.

    The capture path emits factorized gradients, so this is how the sweep gets
    a materialized arm to compare against: same steps, same layers, but the
    outer product expanded (~2.8x the bytes).  Both representations reach the
    memmap writer, so the comparison isolates payload size and shape rather
    than which serializer ran.
    """

    def __init__(self, inner: HookManagerCallback) -> None:
        self._inner = inner

    def on_step_end(self, record: GradientRecord) -> None:
        self._inner.on_step_end(
            GradientRecord(
                step=record.step,
                input_hash=record.input_hash,
                gradient=record.gradient.materialize(),
                sample_id_key=record.sample_id_key,
            ),
        )

    def on_context_end(self) -> None:
        self._inner.on_context_end()


@dataclass(frozen=True)
class Config:
    """One point in the sweep."""

    disk_format: str = "pickle"
    representation: str = "factorized"
    offload_to_cpu: bool = False
    offload_interval: int = 8
    residency: str = "disk"
    # Capture-time random projection width, or None for raw factors.  The
    # library's main lever on how many bytes reach the store at all.
    proj_dim: int | None = None
    # Tiered only.  None takes the library default (~half of available RAM),
    # which a short benchmark never reaches; the sweep pins 0 so every group is
    # evicted and the spill path is exercised whatever the payload size.
    budget_bytes: int | None = None

    @property
    def name(self) -> str:
        # o2c = offload_to_cpu (where the device->host copy happens), NOT the
        # compute device -- that is reported separately as `device`.
        return (
            f"{self.representation}/{self.disk_format}/"
            f"proj={self.proj_dim or 'raw'}/o2c={int(self.offload_to_cpu)}/"
            f"int={self.offload_interval}/{self.residency}"
        )

    def as_dict(self) -> dict:
        return {
            "disk_format": self.disk_format,
            "representation": self.representation,
            "offload_to_cpu": self.offload_to_cpu,
            "offload_interval": self.offload_interval,
            "residency": self.residency,
            "proj_dim": self.proj_dim,
            "budget_bytes": self.budget_bytes,
        }


class TinyTransformer(nn.Module):
    """A transformer-shaped stack, so layer count and shapes resemble a real
    capture without pulling in ``transformers`` (not a dependency here).
    """

    def __init__(self, vocab: int = 512, d: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attn": nn.Linear(d, d, bias=False),
                    "proj": nn.Linear(d, d, bias=False),
                    "fc_in": nn.Linear(d, 4 * d, bias=False),
                    "fc_out": nn.Linear(4 * d, d, bias=False),
                },
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for block in self.blocks:
            h = h + block["proj"](torch.relu(block["attn"](h)))
            h = h + block["fc_out"](torch.relu(block["fc_in"](h)))
        return self.head(self.norm(h))


# The model's outer-product layers -- the only ones ``logra_factorized`` is
# defined for.  Hooking just these keeps every config's layer set identical.
_LINEAR_PATTERNS = (r"blocks\.\d+\.(attn|proj|fc_in|fc_out)$", r"^head$")


def _hook_config(cfg: Config) -> object:
    """HookManagerConfig for *cfg*: the linear layer set, plus projection."""
    from dattri_llm.gradient.hooks import HookManagerConfig

    kwargs: dict = {"linear_io": list(_LINEAR_PATTERNS)}
    if cfg.proj_dim is not None:
        kwargs["projection"] = {
            "__default__": {
                "style": "logra_factorized",
                "proj_dim": cfg.proj_dim,
                "proj_max_batch_size": 32,
                "proj_type": "rademacher",
                "proj_seed": 0,
            },
        }
    return HookManagerConfig(**kwargs)


# Projection widths for --proj-sweep, coarse to fine.  Spans both sides of the
# compute/write crossover: wide values cost projection compute without shrinking
# much, narrow ones remove the bytes entirely.
_PROJ_LADDER = (None, 512, 256, 128, 64, 32, 16, 8)


def _build_proj_sweep() -> list[Config]:
    """One axis only: projection width, everything else at the baseline."""
    return [Config(proj_dim=p) for p in _PROJ_LADDER]


# Default projection width for this model.  The knee of the --proj-sweep curve:
# 128->64 still buys 1.63x walltime, 64->32 only 1.07x, so below 64 you trade
# sketch fidelity for almost no speed.  d=256 here, so this is d/4 (64*64 =
# 4096 dims/layer, the rank LoGRA and logix use).  It does not transfer to
# another model -- re-run --proj-sweep when the hidden size changes.
DEFAULT_PROJ_DIM = 64


def _build_sweep(quick: bool, proj_dim: int | None) -> list[Config]:
    """The configurations to run, all at *proj_dim* except the projection axis.

    Every entry differs from the baseline along exactly one axis, so any pair
    including the baseline isolates that axis.
    """
    base = {"proj_dim": proj_dim}
    if quick:
        # One axis only, so even the smoke run is interpretable.
        return [
            Config(disk_format="pickle", **base),
            Config(disk_format="memmap", **base),
        ]
    return [
        Config(**base),
        Config(disk_format="memmap", **base),
        Config(representation="materialized", **base),
        # Projection axis: one coarser and one finer than the default.
        Config(proj_dim=None),
        Config(proj_dim=16),
        # Where the device->host copy happens.
        Config(offload_to_cpu=True, **base),
        # Stall frequency vs stall size.
        Config(offload_interval=1, **base),
        Config(offload_interval=32, **base),
        # The spill path.  A zero budget evicts every group as it lands, so
        # the row measures spilling at any payload size -- a fixed budget stops
        # firing as soon as projection shrinks the store under it.
        Config(residency="tiered", budget_bytes=0, **base),
    ]


def _run_one(
    cfg: Config,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
    results: Results,
    store_dir: str | None = None,
) -> dict:
    """Train ``steps`` steps under *cfg*, returning the measured row."""
    from dattri_llm.gradient.callbacks import OffloadCallback
    from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
    from dattri_llm.gradient.storage_manager import GradientStorageManager

    on_cuda = device.startswith("cuda")  # matches "cuda" and "cuda:N"
    if on_cuda:
        # Per-config peak, not the running max since process start -- without
        # this every row after the first reports the previous config's peak.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    model = TinyTransformer().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    save_dir = Path(tempfile.mkdtemp(prefix="offload_bench_", dir=store_dir))

    try:
        store = GradientStorageManager(
            str(save_dir),
            residency=cfg.residency,
            disk_format=cfg.disk_format,
            budget_bytes=cfg.budget_bytes,
        )
        offload: HookManagerCallback = OffloadCallback(
            offload_interval=cfg.offload_interval,
            file_manager=store,
        )
        if cfg.representation == "materialized":
            offload = MaterializeCallback(offload)
        hooks = HookManager(
            model,
            config=_hook_config(cfg),
            callbacks=[offload],
            offload_to_cpu=cfg.offload_to_cpu,
        )

        batches = [
            torch.randint(0, 512, (batch_size, seq_len), device=device)
            for _ in range(steps)
        ]

        # One warm-up step outside the measurement: first-touch allocation and
        # lazy CUDA init would otherwise land entirely in step 0's phases.
        with hooks.collect():
            logits = model(batches[0])
            logits.float().pow(2).mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        store.reset_timing()

        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with hooks.collect():
            for ids in batches:
                logits = model(ids)
                logits.float().pow(2).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if on_cuda:
            torch.cuda.synchronize()
        wall = time.monotonic() - t0
        hooks.remove()

        timing = store.timing
        offload_s = sum(p["seconds"] for p in timing.values())
        store_bytes = dir_bytes(save_dir)
        # Which serializer actually ran: a memmap store falls back to pickle
        # for a payload its writer does not know.
        written = (
            "pt"
            if not list(save_dir.rglob("*.mmap.bin"))
            else ("mmap" if not list(save_dir.rglob("*.pt")) else "mixed")
        )
        if cfg.residency != "disk":
            store.close()

        row = {
            "config": cfg.name,
            "wall_s": round(wall, 3),
            "offload_s": round(offload_s, 3),
            "offload_share": round(offload_s / wall, 4) if wall > 0 else None,
            "store_bytes": store_bytes,
            "written_as": written,
            # Disk throughput actually achieved.  When every config lands on the
            # same number, storage bandwidth is the constraint and no amount of
            # serializer work will move it.
            # Compute is whatever walltime is not offloading.  The async
            # ceiling follows: overlapping writes with compute leaves
            # max(C, W), so the best a writer thread can ever do is
            # (C + W) / max(C, W) -- 2x when balanced, ~1x at either extreme.
            "compute_s": round(max(wall - offload_s, 0.0), 3),
            "async_ceiling": (
                round((wall) / max(wall - offload_s, offload_s), 2)
                if max(wall - offload_s, offload_s) > 0
                else None
            ),
            "write_mbps": (
                round(store_bytes / 2**20 / timing["write_group"]["seconds"], 1)
                if timing["write_group"]["seconds"] > 0
                else None
            ),
            "samples": steps * batch_size,
            "phases": {k: round(v["seconds"], 4) for k, v in timing.items()},
            "saves": max((p["calls"] for p in timing.values()), default=0),
            **cfg.as_dict(),
        }
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if on_cuda else 0.0
        for phase, stats in timing.items():
            seconds = stats["seconds"]
            results.add(
                config=cfg.name,
                phase=phase,
                wall_s=round(seconds, 4),
                saves=stats["calls"],
                peak_mem_gb=round(peak_gb, 3),
                **cfg.as_dict(),
            )
        results.add(
            config=cfg.name,
            phase="summary",
            wall_s=row["wall_s"],
            offload_s=row["offload_s"],
            offload_share=row["offload_share"],
            store_bytes=store_bytes,
            written_as=written,
            write_mbps=row["write_mbps"],
            compute_s=row["compute_s"],
            async_ceiling=row["async_ceiling"],
            peak_mem_gb=round(peak_gb, 3),
            **cfg.as_dict(),
        )
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
    return row


def _median_row(repeats: list[dict]) -> dict:
    """Collapse repeated runs of one config into per-field medians.

    Medians resist a single stalled write (page-cache writeback, a throttled
    cloud disk).  ``runs`` and the walltime spread ride along so a reader can
    tell when the repeats disagreed.
    """
    head = repeats[0]
    if len(repeats) == 1:
        return {**head, "runs": 1, "wall_spread": 0.0}
    walls = sorted(r["wall_s"] for r in repeats)
    row = {
        **head,
        "runs": len(repeats),
        "wall_s": round(statistics.median(walls), 3),
        "offload_s": round(statistics.median(r["offload_s"] for r in repeats), 3),
        "phases": {
            phase: round(statistics.median(r["phases"][phase] for r in repeats), 4)
            for phase in head["phases"]
        },
        # max/min: >1.5 means the disk, not the config, is setting the pace.
        "wall_spread": round(walls[-1] / walls[0], 2) if walls[0] > 0 else 0.0,
    }
    row["offload_share"] = (
        round(row["offload_s"] / row["wall_s"], 4) if row["wall_s"] else None
    )
    row["compute_s"] = round(max(row["wall_s"] - row["offload_s"], 0.0), 3)
    denom = max(row["compute_s"], row["offload_s"])
    row["async_ceiling"] = round(row["wall_s"] / denom, 2) if denom > 0 else None
    return row


def _print_table(rows: list[dict]) -> None:
    phases = ["to_cpu", "write_group", "index_update", "index_write", "spill"]
    header = (
        f"{'config':<58} {'as':>5} {'wall':>7} {'compute':>8} {'offload':>8} "
        f"{'share':>6} {'async':>6} {'MB/s':>7} {'x':>5}  "
        + " ".join(f"{p[:11]:>11}" for p in phases)
        + f" {'store':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        share = f"{r['offload_share'] * 100:5.1f}%" if r["offload_share"] else "    -"
        cells = " ".join(f"{r['phases'].get(p, 0.0):11.4f}" for p in phases)
        spread = f"{r.get('wall_spread', 0.0):.2f}" if r.get("runs", 1) > 1 else "-"
        mbps = f"{r['write_mbps']:.0f}" if r.get("write_mbps") else "-"
        ceil = f"{r['async_ceiling']:.2f}x" if r.get("async_ceiling") else "-"
        print(
            f"{r['config']:<58} {r['written_as']:>5} {r['wall_s']:7.2f} "
            f"{r.get('compute_s', 0.0):8.3f} {r['offload_s']:8.3f} "
            f"{share:>6} {ceil:>6} {mbps:>7} {spread:>5}  {cells} "
            f"{r['store_bytes'] / 2**20:9.1f}M",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults sized so one config writes a few hundred MB (the store is
    # removed between configs, so this is peak, not cumulative).  Scale up with
    # --steps / --batch-size / --seq-len to stress the write path harder.
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="short smoke sweep")
    parser.add_argument(
        "--proj-sweep",
        action="store_true",
        help="vary only projection width, to locate the compute/write crossover",
    )
    parser.add_argument(
        "--proj-dim",
        default=str(DEFAULT_PROJ_DIM),
        help=(
            f"projection width every config runs at (default {DEFAULT_PROJ_DIM}); "
            "'raw' disables projection.  Use 'raw' when comparing serializers -- "
            "projection shrinks the write so far that the other axes get lost in "
            "the noise.  Ignored by --proj-sweep"
        ),
    )
    parser.add_argument("--list", action="store_true", help="show configs only")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "runs per config; the table reports medians.  Disks with variable "
            "throughput (cloud VMs) need >=3 before any comparison is meaningful"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help=(
            "filesystem to write gradient stores to (default: TMPDIR).  Point "
            "this at the node-local scratch you actually train against -- "
            "TMPDIR is tmpfs on many clusters, which measures RAM, not disk"
        ),
    )
    args = parser.parse_args()

    proj_dim = None if args.proj_dim.lower() in ("raw", "none") else int(args.proj_dim)
    configs = (
        _build_proj_sweep()
        if args.proj_sweep
        else _build_sweep(args.quick, proj_dim)
    )
    if args.list:
        for cfg in configs:
            print(cfg.name)
        return 0

    out = HERE / "results.jsonl"
    out.unlink(missing_ok=True)
    store_root = Path(args.store_dir) if args.store_dir else Path(tempfile.gettempdir())
    ram = available_ram_bytes()
    fs = describe_filesystem(store_root)
    results = Results(
        out,
        meta={
            "device": args.device,
            "torch": torch.__version__,
            "store_root": str(store_root),
            "store_fs": fs,
            "ram_bytes": ram,
        },
    )
    print(f"store -> {store_root}  fs={fs}  RAM={ram / 2**30:.1f} GiB")
    if "tmpfs" in fs or "ramfs" in fs:
        print(
            "  WARNING: that mount is RAM-backed -- this measures memory "
            "bandwidth, not disk.  Pass --store-dir pointing at real storage.",
        )

    # Throwaway config: absorbs one-time process costs (filesystem cache,
    # allocator, torch.save code paths) that would otherwise all land on
    # config #1 and penalise it ~12%.  Its rows are discarded.
    print("[warm-up] priming process-level caches", flush=True)
    _run_one(
        configs[0],
        steps=max(2, args.steps // 4),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        results=Results(out, meta={}),
        store_dir=args.store_dir,
    )

    rows = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name}", flush=True)
        repeats = [
            _run_one(
                cfg,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=args.device,
                results=results,
                store_dir=args.store_dir,
            )
            for _ in range(args.repeat)
        ]
        rows.append(_median_row(repeats))
    results.flush()
    _print_table(rows)
    biggest = max((r["store_bytes"] for r in rows), default=0)
    if ram and biggest < ram * 0.5:
        print(
            f"\n  WARNING: the largest store ({biggest / 2**30:.1f} GiB) fits "
            f"well inside available RAM ({ram / 2**30:.1f} GiB), so writes may "
            "never reach the platter and write_group measures page cache.  "
            "Scale up with --steps / --batch-size / --seq-len.",
        )
    print(f"\nwrote {out}")
    print(json.dumps({"device": args.device, "configs": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
