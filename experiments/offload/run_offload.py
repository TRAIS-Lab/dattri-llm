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

The axes are the ones that move those numbers:

===================  ==========================  ============================
axis                 values                      why it is in the matrix
===================  ==========================  ============================
``disk_format``      pickle, memmap              pickle holds the GIL, the
                                                 memmap write releases it
representation       factorized, materialized    both memmap; materialized is
                                                 larger on disk (outer product
                                                 expanded)
``offload_to_cpu``   False, True                 moves the device->host copy
                                                 between ``to_cpu`` and the
                                                 forward/backward hooks
``offload_interval`` 1, 8, 32                    trades stall frequency
                                                 against stall size
``residency``        disk, tiered                exercises the spill path
===================  ==========================  ============================

Usage::

    python run_offload.py                    # the default sweep
    python run_offload.py --quick            # a fast smoke sweep
    python run_offload.py --steps 200 --batch-size 16
    python run_offload.py --list             # show the configs, run nothing

Results are appended to ``results.jsonl`` next to this file, one line per
(config, phase) plus a ``summary`` line per config.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    # Tiered only.  None takes the library default (~half of available RAM),
    # which a short benchmark never reaches -- so the sweep pins a small budget
    # to actually exercise the spill path rather than reporting 0 for it.
    budget_bytes: int | None = None

    @property
    def name(self) -> str:
        return (
            f"{self.representation}/{self.disk_format}/"
            f"cpu={int(self.offload_to_cpu)}/int={self.offload_interval}/"
            f"{self.residency}"
        )

    def as_dict(self) -> dict:
        return {
            "disk_format": self.disk_format,
            "representation": self.representation,
            "offload_to_cpu": self.offload_to_cpu,
            "offload_interval": self.offload_interval,
            "residency": self.residency,
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


def _build_sweep(quick: bool) -> list[Config]:
    """The configurations to run, coarsest axis first."""
    if quick:
        return [
            Config(disk_format="pickle", representation="factorized"),
            Config(disk_format="memmap", representation="materialized"),
        ]
    configs: list[Config] = []
    # Format x representation: the pair that decides whether the memmap fast
    # path is reachable at all for the library's default capture.
    for representation in ("factorized", "materialized"):
        for disk_format in ("pickle", "memmap"):
            configs.append(
                Config(disk_format=disk_format, representation=representation),
            )
    # Where the device->host copy happens.
    configs.append(Config(offload_to_cpu=True))
    # Stall frequency vs stall size.
    configs.extend(Config(offload_interval=n) for n in (1, 32))
    # The spill path: a 64 MiB budget so groups are actually evicted to disk.
    configs.append(Config(residency="tiered", budget_bytes=64 * 2**20))
    return configs


def _run_one(
    cfg: Config,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
    results: Results,
) -> dict:
    """Train ``steps`` steps under *cfg*, returning the measured row."""
    from dattri_llm.gradient.callbacks import OffloadCallback
    from dattri_llm.gradient.hooks import HookManager
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
    save_dir = Path(tempfile.mkdtemp(prefix="offload_bench_"))

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
            peak_mem_gb=round(peak_gb, 3),
            **cfg.as_dict(),
        )
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
    return row


def _print_table(rows: list[dict]) -> None:
    phases = ["to_cpu", "write_group", "index_update", "index_write", "spill"]
    header = (
        f"{'config':<46} {'as':>5} {'wall':>7} {'offload':>8} {'share':>6}  "
        + " ".join(f"{p[:11]:>11}" for p in phases)
        + f" {'store':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        share = f"{r['offload_share'] * 100:5.1f}%" if r["offload_share"] else "    -"
        cells = " ".join(f"{r['phases'].get(p, 0.0):11.4f}" for p in phases)
        print(
            f"{r['config']:<46} {r['written_as']:>5} {r['wall_s']:7.2f} "
            f"{r['offload_s']:8.3f} "
            f"{share:>6}  {cells} {r['store_bytes'] / 2**20:9.1f}M",
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
    parser.add_argument("--list", action="store_true", help="show configs only")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    configs = _build_sweep(args.quick)
    if args.list:
        for cfg in configs:
            print(cfg.name)
        return 0

    out = HERE / "results.jsonl"
    out.unlink(missing_ok=True)
    results = Results(
        out,
        meta={"device": args.device, "torch": torch.__version__},
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
    )

    rows = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name}", flush=True)
        rows.append(
            _run_one(
                cfg,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=args.device,
                results=results,
            ),
        )
    results.flush()
    _print_table(rows)
    print(f"\nwrote {out}")
    print(json.dumps({"device": args.device, "configs": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
