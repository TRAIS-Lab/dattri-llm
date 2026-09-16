"""Cross-library scaling figure: three libraries up the Qwen ladder on H200s.

Qwen2.5 0.5B to 110B, WikiText-103, bf16, batch 1, 512-token sequences,
rank-64 projection wherever the library supports it, one query.  Every library
runs 8 warm-up samples through its full pipeline untimed, then 64 measured
samples; model build and dataset load are excluded from every timing.  One
experiment per method and hardware tier:

    scaling-<method>         one H200: dattri-llm, Bergson and LogIX, 0.5B-32B
    scaling-<method>-fsdp4   four H200s: dattri-llm at 72B and 110B (FSDP);
                             Bergson's own sharding for K-FAC/EK-FAC from 7B
                             and for GradDot at 72B/110B (see below)

Run anywhere with the right GPUs:

    python scaling.py --experiment scaling-graddot --run

or on Modal, which provisions the cards, caches weights and token pools on
Volumes and pulls every results.jsonl back into results/:

    modal run scaling.py::bench --experiment scaling-graddot
    modal run scaling.py::bench --all

Then draw the figure from results/scaling-*.jsonl:

    python scaling.py --figure
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "utils").is_dir():
    # Modal imports the entrypoint from a copy in /root; the benchmark tree
    # itself is mounted at REMOTE_ROOT (see the Modal section below).
    HERE = Path("/root/dattri-llm/experiments/benchmark")
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

METHODS = ("graddot", "kfac", "ekfac")
SINGLE = ("0.5b", "1b", "3b", "7b", "14b", "32b")

BASE = dict(family="qwen", dataset="wikitext103", block_size=512, seed=0,
            dtype="bfloat16", proj_mode="rank64", batch=1,
            n_train=72, warmup_train=8, measure_train=64, n_test=1)
SHARDED = dict(parallelism="fsdp", n_gpus=4)


def single(method: str) -> list[dict]:
    cells = runner.grid(["dattri_llm", "logix"], [method], SINGLE, **BASE)
    # Bergson's K-FAC / EK-FAC keep full-dimension factors whatever projection
    # is asked for, and exhaust one H200 from 7B; those scales run sharded.
    scales = SINGLE if method == "graddot" else ("0.5b", "1b", "3b")
    cells += runner.grid(["bergson"], [method], scales, **BASE)
    # Kronfluence has no capture-time projection, so it runs every method at
    # full dimension (its only mode) up the same ladder; a cell that neither
    # finishes nor OOMs within the limit is recorded as a timeout.
    return cells + runner.grid(["kronfluence"], [method], SINGLE,
                               **{**BASE, "time_limit_s": 1800})


def sharded(method: str) -> list[dict]:
    cells = runner.grid(["dattri_llm"], [method], ["72b", "110b"], **BASE, **SHARDED)
    if method == "graddot":
        # Bergson builds its query index in one process and only shards it when
        # the query set has at least four chunks: at 72B+ the 1-query build no
        # longer fits one card, so these two cells score four queries.
        return cells + runner.grid(["bergson"], [method], ["72b", "110b"],
                                   **{**BASE, "n_test": 4}, **SHARDED)
    return cells + runner.grid(["bergson"], [method], ["7b", "14b", "32b"], **BASE, **SHARDED)


EXPERIMENTS = {}
for _m in METHODS:
    EXPERIMENTS[f"scaling-{_m}"] = single(_m)
    EXPERIMENTS[f"scaling-{_m}-fsdp4"] = sharded(_m)


def n_gpus(experiment: str) -> int:
    return max(t["n_gpus"] for t in EXPERIMENTS[experiment])


if __name__ == "__main__":
    if "--figure" in sys.argv:
        import figure

        figure.main(HERE / "results")
    else:
        runner.main(EXPERIMENTS)


# --------------------------------------------------------------------------- #
# Modal                                                                        #
# --------------------------------------------------------------------------- #
try:
    import modal
except ImportError:  # the launcher also works without Modal
    modal = None

if modal is not None:
    from versions import BASELINE_VERSIONS, DIST_NAMES  # noqa: E402

    REMOTE_ROOT = "/root/dattri-llm"
    RUN_DIR = f"{REMOTE_ROOT}/experiments/benchmark"
    LOCAL = modal.is_local()
    ROOT = HERE.parent.parent if LOCAL else Path(REMOTE_ROOT)
    HOUR = 60 * 60

    # One image for every library (python 3.10: logix needs <3.11), so each
    # experiment's libraries share one container -- host speed varies between
    # containers, and a cross-library number means nothing across two of them.
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .uv_pip_install("torch", "transformers>=4.40", "accelerate>=0.28", "datasets",
                        "dattri>=0.3.0", "numpy", "psutil", "tqdm",
                        *[f"{DIST_NAMES.get(lib, lib)}=={ver}"
                          for lib, ver in BASELINE_VERSIONS.items()])
        .env({"PYTHONPATH": REMOTE_ROOT, "BENCH_CACHE": "/cache", "HF_HOME": "/hf",
              "TOKENIZERS_PARALLELISM": "false"})
    )
    if LOCAL:
        image = image.add_local_dir(ROOT / "dattri_llm", f"{REMOTE_ROOT}/dattri_llm",
                                    copy=True, ignore=["**/__pycache__"])
        image = image.add_local_dir(HERE, RUN_DIR, copy=True,
                                    ignore=["**/__pycache__", "**/out", "**/results/**"])

    results_vol = modal.Volume.from_name("dattri-bench-results", create_if_missing=True)
    cache_vol = modal.Volume.from_name("dattri-bench-cache", create_if_missing=True)
    hf_vol = modal.Volume.from_name("dattri-bench-hf", create_if_missing=True)
    VOLUMES = {"/results": results_vol, "/cache": cache_vol, "/hf": hf_vol}
    SCRATCH = "/scratch"   # container-local: gradient stores must not sit on a Volume
    SHARDED_KW = dict(memory=256 * 1024, ephemeral_disk=1024 * 1024)

    app = modal.App("dattri-benchmark")

    def _publish(local_out: Path, experiment: str) -> None:
        """Copy the small artifacts to the results Volume; leave the stores."""
        import shutil

        dest = Path("/results") / experiment
        dest.mkdir(parents=True, exist_ok=True)
        if (local_out / "results.jsonl").exists():
            shutil.copy2(local_out / "results.jsonl", dest / "results.jsonl")
        if (local_out / "plans").is_dir():
            shutil.copytree(local_out / "plans", dest / "plans", dirs_exist_ok=True)
        runs = local_out / "runs"
        if runs.is_dir():
            for d in sorted(x for x in runs.iterdir() if x.is_dir()):
                for name in ("record.json", "score.pt"):
                    if (d / name).exists():
                        (dest / "runs" / d.name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(d / name, dest / "runs" / d.name / name)

    def _partial_name(experiment: str, libs: str) -> str:
        """Volume/result name of a run restricted to *libs* (comma-separated)."""
        return f"{experiment}~{libs.replace(',', '+')}" if libs else experiment

    def _bench(experiment: str, libs: str = "") -> None:
        import os
        import subprocess

        name = _partial_name(experiment, libs)
        local_out = Path(SCRATCH) / name
        local_out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "BERGSON_STORE": str(local_out / "bergson")}
        cmd = [sys.executable, "-u", "scaling.py", "--experiment", experiment,
               "--out_dir", str(local_out), "--run"]
        if libs:
            cmd += ["--libs", libs]
        try:
            subprocess.run(cmd, cwd=RUN_DIR, check=True, env=env)  # noqa: S603
        finally:
            _publish(local_out, name)
            for vol in VOLUMES.values():
                vol.commit()

    @app.function(image=image, volumes={"/hf": hf_vol, "/cache": cache_vol}, timeout=6 * HOUR)
    def prefetch(cells: list[dict]) -> None:
        """Cache weights and token pools on CPU so GPU time is not spent downloading."""
        from huggingface_hub import snapshot_download

        sys.path.insert(0, f"{RUN_DIR}/utils")
        from data import load_task_data

        seen = set()
        for t in cells:
            key = (t["model"], t["dataset"], t["block_size"], t["n_train"], t["n_test"], t["seed"])
            if key in seen:
                continue
            seen.add(key)
            snapshot_download(t["model"], allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
            hf_vol.commit()
            load_task_data(*key)
            cache_vol.commit()

    @app.function(image=image, gpu="H200", volumes=VOLUMES, timeout=12 * HOUR,
                  ephemeral_disk=2048 * 1024)
    def run_h200(experiment: str, libs: str = "") -> None:
        _bench(experiment, libs)

    @app.function(image=image, gpu="H200:4", volumes=VOLUMES, timeout=24 * HOUR, **SHARDED_KW)
    def run_h200_x4(experiment: str, libs: str = "") -> None:
        _bench(experiment, libs)

    def _fetch(experiment: str, out_dir: Path) -> bool:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{experiment}.jsonl"
        try:
            with dest.open("wb") as fh:
                for chunk in results_vol.read_file(f"{experiment}/results.jsonl"):
                    fh.write(chunk)
        except Exception as e:  # noqa: BLE001 -- absent means every cell failed
            dest.unlink(missing_ok=True)
            print(f"  [fetch] {experiment}: no results.jsonl ({type(e).__name__})")
            return False
        print(f"  [fetch] {experiment}: {sum(1 for _ in dest.open())} rows -> {dest}")
        return True

    def _merge_partial(experiment: str, libs: str, out: Path) -> None:
        """Fold a ``--libs`` re-run into ``results/<experiment>.jsonl``: the
        re-run's libraries replace their old rows, every other library's rows
        are kept, and the previous file is archived beside it."""
        import json
        import shutil
        import time

        partial = out / f"{_partial_name(experiment, libs)}.jsonl"
        full = out / f"{experiment}.jsonl"
        new_rows = [json.loads(line) for line in partial.open() if line.strip()]
        redone = {r["lib"] for r in new_rows}
        kept = []
        if full.exists():
            # Archive outside the ``scaling-*.jsonl`` glob the figure loads.
            archive = out / "archive"
            archive.mkdir(exist_ok=True)
            shutil.copy2(full, archive / f"{experiment}.pre-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
            kept = [r for r in (json.loads(line) for line in full.open() if line.strip())
                    if r["lib"] not in redone]
        with full.open("w") as fh:
            for r in kept + new_rows:
                fh.write(json.dumps(r) + "\n")
        partial.unlink()
        print(f"  [merge] {full}: replaced {sorted(redone)} ({len(new_rows)} rows), "
              f"kept {len(kept)} rows of the other libraries")

    @app.local_entrypoint()
    def bench(experiment: str = "", all: bool = False, warm_first: bool = True,  # noqa: A002
              libs: str = "") -> None:
        """Run one experiment (or every one) on the GPUs it needs, then fetch results.

        ``--libs a,b`` re-runs only those libraries' cells and merges them into
        the existing ``results/<experiment>.jsonl`` (other libraries kept).
        """
        names = sorted(EXPERIMENTS) if all else [experiment]
        if not all and experiment not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {experiment!r}; choices: {sorted(EXPERIMENTS)}")
        for exp in names:
            gpus = n_gpus(exp)
            print(f"\n--- {exp} on {gpus}x H200{f' ({libs} only)' if libs else ''} ---")
            if warm_first:
                prefetch.remote(EXPERIMENTS[exp])
            try:
                (run_h200_x4 if gpus > 1 else run_h200).remote(exp, libs)
            except Exception as e:  # noqa: BLE001 -- the others are independent
                print(f"  [run] {exp} FAILED: {type(e).__name__}: {e}")
        out = HERE / "results"
        got = [e for e in names if _fetch(_partial_name(e, libs), out)]
        if libs:
            for e in got:
                _merge_partial(e, libs, out)
        print(f"\n{len(got)}/{len(names)} experiments produced results in {out}/; "
              "next: python scaling.py --figure")

    @app.local_entrypoint()
    def warm(experiment: str) -> None:
        """Cache an experiment's weights and token pools without a GPU."""
        prefetch.remote(EXPERIMENTS[experiment])
