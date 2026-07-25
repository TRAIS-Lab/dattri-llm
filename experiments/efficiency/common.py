"""Shared measurement harness for the pairwise efficiency benchmarks.

Every pair folder (``vs_*``) runs both libraries under identical settings and
records phases through :class:`Meter`:

* walltime per phase (``time.monotonic``),
* peak GPU memory per phase (``torch.cuda.max_memory_allocated``, reset at
  phase start) plus the process-wide reserved peak,
* throughput = caller-declared work units / walltime,
* optional on-disk store size for store-based pipelines,
* environment capture (torch / transformers / library versions, GPU name).

One JSON line per (library, phase) is appended to the pair's ``results.jsonl``.
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def env_info(extra: dict | None = None) -> dict:
    import transformers
    info = dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    if extra:
        info.update(extra)
    return info


def dir_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class Meter:
    """Collects per-phase measurements for one library run."""

    def __init__(self, lib: str, setting: str, out_json: str | Path,
                 meta: dict | None = None) -> None:
        self.lib = lib
        self.setting = setting
        self.out_json = Path(out_json)
        self.meta = meta or {}
        self.rows: list[dict] = []

    @contextlib.contextmanager
    def phase(self, name: str, work_units: int, unit: str = "samples"):
        """Measure one phase.  ``work_units`` defines throughput denominator."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        row = dict(
            lib=self.lib, setting=self.setting, phase=name,
            wall_s=round(wall, 2),
            throughput=round(work_units / wall, 2) if wall > 0 else None,
            unit=f"{unit}/s", work_units=work_units,
            peak_mem_gb=round(peak, 3),
            **self.meta,
        )
        self.rows.append(row)
        print(f"[{self.lib}] {name}: {wall:.2f}s, "
              f"{row['throughput']} {unit}/s, peak {peak:.2f} GB")

    def note(self, **kv) -> None:
        """Attach extra fields (e.g. store_bytes, score checksum) to a summary row."""
        self.rows.append(dict(lib=self.lib, setting=self.setting, phase="summary",
                              **kv, **self.meta))

    def flush(self) -> None:
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        with self.out_json.open("a") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        self.rows.clear()
