"""Result logging for the attribution benchmark.

Every (task, library) run is wrapped in a :class:`BenchRun`, which records a
single self-describing JSON line:

* **runtime** -- wall time per phase and total, GPU-synced.
* **GPU memory** -- peak allocated + reserved per visible device.
* **CPU memory** -- peak process RSS (the kernel's high-water mark) + host RAM.
* **disk** -- bytes written to the run's cache/output dirs.
* **device details** -- GPU name/count/capability/VRAM, CPU count, host RAM,
  CUDA/cuDNN/torch/transformers versions, hostname, job id, world size.
* **task** -- family/scale/model/dataset/method/parallelism/n_train/... verbatim.

Usage::

    run = BenchRun(task, results_path="results.jsonl", run_dir="runs/qwen-0.5b-kfac")
    with run.phase("cache", work_units=n_train):
        ...
    with run.phase("attribute", work_units=n_train + n_test):
        ...
    run.record_disk("store", cache_dir)
    run.finish(score_shape=[n_test, n_train])
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import time
from pathlib import Path

import psutil
import torch

from versions import BASELINE_VERSIONS, installed

_GB = 1024 ** 3


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def device_details() -> dict:
    """Device and environment details for the run record."""
    rank, world = _dist_rank_world()
    info: dict = {
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_gpus": os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_JOB_GPUS"),
        "rank": rank,
        "world_size": world,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
        "host_ram_gb": round(psutil.virtual_memory().total / _GB, 1),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001 -- optional dependency
        info["transformers"] = None
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i, "name": p.name,
                "vram_gb": round(p.total_memory / _GB, 1),
                "capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
            })
    info["gpu_count"] = len(gpus)
    info["gpus"] = gpus
    info["gpu_name"] = gpus[0]["name"] if gpus else "cpu"
    # the installed version of every baseline library
    info["baseline_versions"] = {lib: installed(lib) for lib in BASELINE_VERSIONS}
    return info


class _PeakRSS:
    """Peak host memory (RSS) of the process and of its finished children.

    Read from the kernel's high-water marks (``ru_maxrss`` of
    ``RUSAGE_SELF`` plus ``RUSAGE_CHILDREN``).  The marks cover the lifetime of
    the process, so the value a phase records is the peak reached from process
    start to the end of that phase.
    """

    @staticmethod
    def _peak() -> int:
        import resource

        kb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
              + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return kb * 1024  # Linux reports kilobytes

    def start(self) -> None:
        """No-op; the kernel tracks the high-water mark."""

    def reset(self) -> None:
        """No-op; the high-water mark covers the process lifetime."""

    @property
    def peak_gb(self) -> float:
        return round(self._peak() / _GB, 3)

    def stop(self) -> None:
        """No-op."""


def _gpu_peaks() -> list[dict]:
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            out.append({
                "index": i,
                "alloc_gb": round(torch.cuda.max_memory_allocated(i) / _GB, 3),
                "reserved_gb": round(torch.cuda.max_memory_reserved(i) / _GB, 3),
            })
    return out


def dir_bytes(path: str | Path) -> int:
    """Total size in bytes of the files under *path* (0 if it does not exist)."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class BenchRun:
    """One (task, library) benchmark run: timing, memory, and disk.

    Args:
        task: the task dict, stored verbatim in the record.
        results_path: the JSON-lines file the record is appended to (rank 0 only).
        run_dir: optional directory that also receives ``record.json``.
        lib: library name; defaults to ``task["lib"]``.
    """

    def __init__(self, task: dict, results_path: str | Path,
                 run_dir: str | Path | None = None,
                 lib: str | None = None) -> None:
        self.task = dict(task)
        self.lib = lib or task.get("lib") or task.get("library")
        self.results_path = Path(results_path)
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device_details()
        self.phases: list[dict] = []
        self.disk: dict[str, float] = {}
        self.extra: dict = {}
        self._sampler = _PeakRSS()
        self._sampler.start()
        self._t_start = time.monotonic()

    # -- timing -------------------------------------------------------------
    @contextlib.contextmanager
    def phase(self, name: str, work_units: int | None = None, unit: str = "samples"):
        """Time the enclosed block as phase *name*.

        Synchronizes CUDA and resets the GPU peak-memory counters on entry;
        on exit records wall time, per-device peak GPU memory and peak RSS.
        With *work_units*, also records throughput and ``s_per_sample``.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        self._sampler.reset()
        t0 = time.monotonic()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.monotonic() - t0
        row = {
            "phase": name, "wall_s": round(wall, 3),
            "gpu_peak": _gpu_peaks(), "cpu_rss_peak_gb": self._sampler.peak_gb,
        }
        if work_units:
            row["work_units"] = work_units
            row["throughput"] = round(work_units / wall, 3) if wall > 0 else None
            row["unit"] = f"{unit}/s"
            # Seconds per work unit (per training sample in the adapters).
            row["s_per_sample"] = round(wall / work_units, 6) if work_units else None
        self.phases.append(row)
        peak = row["gpu_peak"][0]["alloc_gb"] if row["gpu_peak"] else 0.0
        print(f"[{self.lib}] {name}: {wall:.2f}s  gpu {peak:.2f}GB  "
              f"cpu {row['cpu_rss_peak_gb']:.2f}GB", flush=True)

    # -- annotations --------------------------------------------------------
    def record_disk(self, label: str, path: str | Path) -> None:
        """Record the size in GB of all files under *path* as ``disk_gb[label]``."""
        self.disk[label] = round(dir_bytes(path) / _GB, 4)

    def set(self, **kv) -> None:
        """Add top-level fields to the record."""
        self.extra.update(kv)

    # -- output -------------------------------------------------------------
    def finish(self, **summary) -> dict:
        """Assemble the record, append it to ``results_path`` (rank 0) and return it."""
        self._sampler.stop()
        self.extra.update(summary)
        record = {
            "lib": self.lib,
            "task": self.task,
            "total_wall_s": round(time.monotonic() - self._t_start, 3),
            "phases": self.phases,
            "disk_gb": self.disk,
            "disk_total_gb": round(sum(self.disk.values()), 4),
            "device": self.device,
            **self.extra,
        }
        rank = self.device["rank"]
        if rank == 0:  # only rank 0 writes the shared results file
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if self.run_dir:
            (self.run_dir / "record.json").write_text(json.dumps(record, indent=2))
        return record
