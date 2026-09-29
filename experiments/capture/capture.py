"""Ordinary versus invasive capture.

``invasive_linear_io`` replaces each hooked ``nn.Linear`` forward so its
backward skips the weight-gradient matmul; ``linear_io`` leaves the forward
alone and hooks the same layers.  Each pair runs one of our methods through the
same pipeline with only the hook family changed, measuring how much of the
attribution time the skipped matmul accounts for and whether scores move.

Workload: Pythia-0.5B, one A40, fp32, WikiText-103, 1024 training
sequences of 512 tokens, batch 8, every method in both projection regimes.
Each cell warms up on 8 further sequences untimed; five repetitions per pair,
alternating which family runs first.

    capture-query16     sixteen queries, 60 cells
    capture-query1      one query, 60 cells
    capture-shared-fit  K-FAC/EK-FAC at sixteen queries, one repetition: the
                        invasive run scores against the linear_io run's fit,
                        isolating capture from curvature fitting

    python capture.py --experiment capture-query16 --dry-run
    python capture.py --experiment capture-query16 --run
    python capture.py --table        # results/<experiment>/ or out/<experiment>/

or on Modal, one L40S (48 GB, as the A40, which Modal does not offer) per
experiment and method, so both runs of a pair share a card; results land in
results/<experiment>/ with the report in report.txt:

    modal run capture.py::bench --experiment capture-query1 --smoke   # one pair each
    modal run --detach capture.py::bench --all
    modal run capture.py::fetch --all          # after a detached run

Scores are compared outside the timed region, from each cell's saved matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "utils").is_dir():
    # Modal imports the entrypoint from a copy in /root.
    HERE = Path("/root/dattri-llm/experiments/capture")
# This benchmark shares the cell runner, the data, the model registry and the
# result logger of ``experiments/benchmark``; its adapter and report are its own.
sys.path.insert(0, str(HERE.parent / "benchmark" / "utils"))
sys.path.insert(0, str(HERE / "utils"))

import runner  # noqa: E402

runner.ADAPTERS = HERE / "utils" / "adapters"

METHODS = ("graddot", "kfac", "ekfac")
FAMILIES = ("linear_io", "invasive_linear_io")
REPEATS = 5

BASE = dict(family="pythia", dataset="wikitext103", block_size=512, seed=0,
            dtype="float32", batch=8, n_train=1032, warmup_train=8, measure_train=1024)


def pairs(n_test: int, methods=METHODS, repeats: int = REPEATS, alternate: bool = True,
          **kw) -> list[dict]:
    cells = []
    for method in methods:
        for proj_mode in ("full", "rank64"):
            for r in range(repeats):
                order = FAMILIES[::-1] if alternate and r % 2 else FAMILIES
                for fam in order:
                    cells += runner.grid(["dattri_llm"], [method], ["0.5b"],
                                         proj_mode=proj_mode, n_test=n_test,
                                         hook_family=fam, repeat=r, **BASE, **kw)
    return cells


EXPERIMENTS = {
    "capture-query16": pairs(n_test=16),
    "capture-query1": pairs(n_test=1),
    # linear_io writes the fit, so it always runs first.
    "capture-shared-fit": pairs(n_test=16, methods=("kfac", "ekfac"), repeats=1,
                                alternate=False, shared_fit=True),
}


if __name__ == "__main__":
    if "--table" in sys.argv:
        import capture_report

        capture_report.main(HERE, EXPERIMENTS)
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
    REMOTE_ROOT = "/root/dattri-llm"
    RUN_DIR = f"{REMOTE_ROOT}/experiments/capture"
    ROOT = HERE.parent.parent if modal.is_local() else Path(REMOTE_ROOT)
    HOUR = 60 * 60
    GPU = "L40S"

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .uv_pip_install("torch", "transformers>=4.40", "accelerate>=0.28", "datasets",
                        "dattri>=0.3.0", "numpy", "psutil", "tqdm")
        .env({"PYTHONPATH": REMOTE_ROOT, "BENCH_CACHE": "/cache", "HF_HOME": "/hf",
              "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    )
    if modal.is_local():
        image = image.add_local_dir(ROOT / "dattri_llm", f"{REMOTE_ROOT}/dattri_llm",
                                    copy=True, ignore=["**/__pycache__"])
        image = image.add_local_dir(HERE, RUN_DIR, copy=True,
                                    ignore=["**/__pycache__", "**/out", "**/results/**"])

    # The scaling benchmark's volumes: weights and token pools are shared.
    results_vol = modal.Volume.from_name("dattri-bench-results", create_if_missing=True)
    cache_vol = modal.Volume.from_name("dattri-bench-cache", create_if_missing=True)
    hf_vol = modal.Volume.from_name("dattri-bench-hf", create_if_missing=True)
    VOLUMES = {"/results": results_vol, "/cache": cache_vol, "/hf": hf_vol}

    app = modal.App("dattri-capture")

    @app.function(image=image, volumes={"/hf": hf_vol, "/cache": cache_vol}, timeout=2 * HOUR)
    def prefetch(task: dict) -> None:
        """Cache the weights and token pools on CPU, off the GPU clock."""
        from huggingface_hub import snapshot_download

        from data import load_task_data

        snapshot_download(task["model"], allow_patterns=["*.safetensors", "*.json", "*.txt"])
        hf_vol.commit()
        load_task_data(task["model"], task["dataset"], task["block_size"], task["n_train"],
                       task["n_test"], task["seed"])
        cache_vol.commit()

    @app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=12 * HOUR)
    def run_shard(experiment: str, method: str, dest: str, smoke: bool = False) -> None:
        """Run one method's cells in order, then publish rows and scores to *dest*."""
        import shutil

        cells = [c for c in EXPERIMENTS[experiment] if c["method"] == method
                 and not (smoke and c["repeat"])]
        out = Path("/scratch") / dest  # container-local: stores stay off the Volume
        try:
            runner.execute(runner.runs_for(cells), out, experiment)
        finally:
            pub = Path("/results") / dest
            shutil.rmtree(pub, ignore_errors=True)
            pub.mkdir(parents=True)
            if (out / "results.jsonl").exists():
                shutil.copy2(out / "results.jsonl", pub / "results.jsonl")
            for run in sorted((out / "runs").glob("*/")):
                for name in ("record.json", "score.pt"):
                    if (run / name).exists():
                        (pub / "runs" / run.name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(run / name, pub / "runs" / run.name / name)
            results_vol.commit()

    def _fetch(prefix: str, local: Path) -> int:
        """Merge the method shards under *prefix* into *local*; archive what was there."""
        import shutil
        import time

        if local.exists():
            archive = local.parent / "archive" / f"{local.name}.pre-{time.strftime('%Y%m%d-%H%M%S')}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(local, archive)
        rows = []
        for entry in results_vol.listdir(prefix, recursive=True):
            rel = Path(entry.path).relative_to(prefix)  # <method>/...
            if rel.name == "results.jsonl":
                rows += b"".join(results_vol.read_file(entry.path)).decode().splitlines()
            elif rel.parts[1:2] == ("runs",) and rel.name in ("record.json", "score.pt"):
                target = local.joinpath(*rel.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as fh:
                    for chunk in results_vol.read_file(entry.path):
                        fh.write(chunk)
        local.mkdir(parents=True, exist_ok=True)
        (local / "results.jsonl").write_text("".join(f"{r}\n" for r in rows if r.strip()))
        return len(rows)

    def _collect(names: list[str], suffix: str) -> None:
        import contextlib
        import io

        import capture_report

        for name in names:
            local = HERE / "results" / f"{name}{suffix}"
            try:
                print(f"  [fetch] {name}{suffix}: {_fetch(name + suffix, local)} rows -> {local}")
            except Exception as e:  # noqa: BLE001 -- absent means no shard published
                print(f"  [fetch] {name}{suffix}: nothing on the volume ({type(e).__name__})")
                continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                capture_report.report(name, local, timed="shared-fit" not in name)
            (local / "report.txt").write_text(buf.getvalue())
            print(buf.getvalue())

    def _names(experiment: str, run_all: bool) -> list[str]:
        if not run_all and experiment not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {experiment!r}; choices: {sorted(EXPERIMENTS)}")
        return list(EXPERIMENTS) if run_all else [experiment]

    @app.local_entrypoint()
    def bench(experiment: str = "", all: bool = False, smoke: bool = False) -> None:  # noqa: A002
        """Run on L40S, one container per (experiment, method), then fetch and report.

        ``--smoke`` runs only the first repetition of each pair, into ``<experiment>~smoke``.
        """
        names = _names(experiment, all)
        suffix = "~smoke" if smoke else ""
        prefetch.remote(EXPERIMENTS[names[0]][0])
        jobs = [(e, m, f"{e}{suffix}/{m}", smoke)
                for e in names for m in dict.fromkeys(c["method"] for c in EXPERIMENTS[e])]
        print(f"--- {len(jobs)} {GPU} containers: {', '.join(j[2] for j in jobs)} ---")
        for job, res in zip(jobs, run_shard.starmap(jobs, return_exceptions=True), strict=True):
            if isinstance(res, Exception):
                print(f"  [run] {job[2]} FAILED: {type(res).__name__}: {res}")
        _collect(names, suffix)

    @app.local_entrypoint()
    def fetch(experiment: str = "", all: bool = False, smoke: bool = False) -> None:  # noqa: A002
        """Fetch and report published results (after ``modal run --detach``)."""
        _collect(_names(experiment, all), "~smoke" if smoke else "")
