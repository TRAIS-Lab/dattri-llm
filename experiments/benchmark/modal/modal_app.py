"""Run the efficiency and scaling benchmark on Modal GPUs.

    modal run modal_app.py::main --experiment scaling-pythia-A40 --dry-run
    modal run modal_app.py::main --experiment scaling-pythia-A40
    modal run modal_app.py::main --experiment scaling-qwen-H200

This directory is a copy of ``experiments/benchmark`` plus this file.  The
harness itself is unmodified: ``modal_app.py`` provisions the right GPU, points
the two cache paths at Volumes, and shells out to ``run.py --experiment X
--run``, which expands the experiment into plans and executes the cells in
order.  Everything about the measurement -- the fixed
``n_train=64 / n_test=1 / batch 1 / 512-token`` workload, the phase timing, the
peak-memory accounting -- is whatever ``run.py`` and the adapters do locally.

**The device is enforced here, not by convention.**  ``scaling-pythia-A40`` and
``scaling-pythia-H200`` are the same four models and differ only in the card
they run on; the local README notes that "the tree cannot enforce that, so run
each where its name says."  On Modal it *is* enforced: ``GPU_FOR`` maps each
experiment to its GPU and the wrong pairing is not expressible.

Three Volumes, because all three are expensive to rebuild and cheap to keep:

    /results   plans, per-run records, results.jsonl   (the actual output)
    /cache     BENCH_CACHE -- tokenized WikiText block pools, per (model, ds)
    /hf        HF_HOME -- model weights (Qwen-72B alone is ~145 GB)

Each experiment gets its own ``--out_dir`` under ``/results``.  The local
harness appends every run to one ``results.jsonl``; on a Volume that would mean
two containers appending to one file, so the split is per experiment and the
files are concatenated when analyzed.

Baselines: bergson and kronfluence run on ``baseline_image``.  logix is not
carried -- it keeps per-sample gradients at full parameter width and OOMs on
Pythia-410m, and needs its own Python 3.10 image besides.
"""

from __future__ import annotations

from pathlib import Path

import modal

REMOTE_ROOT = "/root/dattri-llm"
RUN_DIR = f"{REMOTE_ROOT}/experiments/benchmark/modal"

LOCAL = modal.is_local()
ROOT = (
    next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "dattri_llm").is_dir()
    )
    if LOCAL
    else Path(REMOTE_ROOT)
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "datasets",
        "dattri>=0.3.0",
        "numpy",
        "psutil",      # log.py samples process-tree RSS through it
        "tqdm",
    )
    .env({
        # `dattri_llm` is mounted as a source tree, not pip-installed, and the
        # adapters only put their own directory on sys.path (for data/models/
        # log).  REMOTE_ROOT is where the package lives, so it has to be on the
        # path too -- and via the environment rather than sys.path, because
        # run.py launches each cell as a subprocess and torchrun forks a rank
        # per GPU; only an inherited PYTHONPATH reaches all of them.  Verified
        # that every symbol the adapters import resolves from a bare path, so
        # no editable install is needed.
        "PYTHONPATH": REMOTE_ROOT,
        "BENCH_CACHE": "/cache",   # data.py reads this; default is a cluster path
        "HF_HOME": "/hf",
        "TOKENIZERS_PARALLELISM": "false",
    })
)
if LOCAL:
    image = image.add_local_dir(
        ROOT / "dattri_llm",
        f"{REMOTE_ROOT}/dattri_llm",
        copy=True,
        ignore=["**/__pycache__"],
    ).add_local_dir(
        Path(__file__).parent,
        RUN_DIR,
        copy=True,
        ignore=["**/__pycache__", "**/out", "**/*.jsonl", "**/*.pt"],
    )

# bergson and kronfluence install onto the main image; both are real PyPI
# packages (`bergson`, `kronfluence`) with Python floors this image clears.
baseline_image = image.uv_pip_install("bergson", "kronfluence")

results_vol = modal.Volume.from_name("dattri-bench-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("dattri-bench-cache", create_if_missing=True)
hf_vol = modal.Volume.from_name("dattri-bench-hf", create_if_missing=True)
VOLUMES = {"/results": results_vol, "/cache": cache_vol, "/hf": hf_vol}

app = modal.App("dattri-benchmark")

# experiment -> the GPU it runs on.
#
# **Modal has no A40.**  The supported types are T4, L4, A10, L40S, A100,
# A100-40GB, A100-80GB, RTX-PRO-6000, H100, H200, B200 and B300; `gpu="A40"` is
# rejected at Function creation.  The `*-A40` experiment therefore runs on an
# **L40S**, the closest available card: 48 GB against the A40's 46 GB usable, so
# the memory ceiling that makes the ladder stop where it does is preserved
# (Pythia-6.9B peaks near 40 GB and still fits).  L40S is Ada rather than Ampere
# and has ~1.2x the memory bandwidth, so it is NOT a timing-identical stand-in --
# the compute curve will sit slightly lower than an A40's.  `log.py` records the
# real `gpu_name` in every row, so results are self-labelling and no row can be
# mistaken for an A40 measurement.  The experiment name is left alone because it
# is the paper's identifier.
#
# The sharded ladder asks for 4 GPUs, matching the `-fsdp4` name: modal/'s
# `models.n_gpus` adds an (80, 4) rung that upstream lacks (see models.py).
# `run.py` passes that count to `torchrun --nproc_per_node`, so this MUST equal
# what n_gpus returns or the launch fails.
# The efficiency-* cells sit on H200, not L40S.  The un-projected regime is
# large: logix OOMed a 44.4 GB card here before being dropped, and bergson's
# index and kronfluence's full-dimension factors are in the same family.  The
# whole cell shares one card -- a cross-library number means nothing if the
# libraries did not run on the same hardware.
GPU_FOR = {
    "efficiency-dattri-llm-rank64": "H200",
    "efficiency-dattri-llm-full": "H200",
    "scaling-pythia-A40": "L40S",
    "scaling-pythia-H200": "H200",
    "scaling-qwen-H200": "H200",
    "scaling-families-H200": "H200",
    "scaling-families-H200-ekfac": "H200",
    "scaling-qwen-H200-fsdp4": "H200:4",
    "scaling-qwen-H200-fsdp4-110b": "H200:4",
    "efficiency-baselines-rank64": "H200-baseline",
    "efficiency-baselines-full": "H200-baseline",
    "scaling-crosslib-H200": "H200-baseline",
    "scaling-crosslib-H200-ekfac": "H200-baseline",
}

# Qwen-72B is ~145 GB of bf16 weights and every FSDP rank calls
# `from_pretrained` independently, so host RAM scales with world size, not with
# the shard: 4 x 145 GB = ~580 GB worst case.  The checkpoint is natively BF16
# and we request bfloat16, so safetensors can mmap it without a conversion copy
# and true peak RSS should sit below that.
#
# `::check` settled how much this number matters.  It reported
# `cgroup_ram == host_ram`, i.e. **no hard cap is enforced**: a scalar `memory=`
# is a scheduling and billing request, and only the tuple form
# `memory=(request, limit)` sets a ceiling the kernel enforces.  So this value
# cannot OOM-kill the run -- the container may use what the node has, which the
# two preflights measured at 1992 GB and 2905 GB (node size varies).  What it
# does do is set a billing floor at $0.00000222/GiB/sec: 720 GiB was $5.75/hr
# whether or not it was touched.  256 GiB is $2.05/hr and still an order above
# any single rank's share, with the node's own multiple-TB pool absorbing the
# peak.  Raise it only to reserve capacity, never for safety.
SHARDED_MEMORY_MIB = 256 * 1024      # ~256 GiB -- billing floor, not a ceiling
SHARDED_DISK_MIB = 1024 * 1024       # 1 TiB (default quota is 512 GiB)

HOUR = 60 * 60

# The gradient store must NOT live on a Volume.  A modal.Volume is
# network-backed: writing the per-sample store there would measure Modal's
# storage fabric rather than the local-disk IO a training node sees, and
# `BenchRun.record_disk` would report those bytes as the run's disk cost.  So
# run.py writes everything to container-local disk and only the small artifacts
# (results.jsonl, plans, per-run record.json + score.pt) are published to the
# Volume afterwards.
SCRATCH = "/scratch"


def _publish(local_out: Path, experiment: str) -> None:
    """Copy the small artifacts to the results Volume; leave the store behind.

    `store/` is the per-sample gradient store -- the thing whose *local* write
    cost the benchmark is measuring.  It is deliberately not published: it can
    be hundreds of GB, it is regenerated by any re-run, and copying it to a
    network Volume is exactly the IO the local-disk placement exists to avoid.
    """
    import shutil

    dest = Path("/results") / experiment
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "plan_index.jsonl"):
        src = local_out / name
        if src.exists():
            shutil.copy2(src, dest / name)
    if (local_out / "plans").is_dir():
        shutil.copytree(local_out / "plans", dest / "plans", dirs_exist_ok=True)

    # runs/ is NOT copied wholesale.  Adapters put their stores inside their own
    # run directory -- kronfluence writes kf_store/ there, bergson its index --
    # so a recursive copy ships gigabytes of factors to a network volume and
    # fills it (this is what raised ENOSPC and killed a 22-of-24 run at the
    # publish step, after every measurement had already succeeded).  Only the
    # two small per-run artifacts are published; the stores stay on scratch and
    # die with the container, which is what the disk numbers already measured.
    KEEP = ("record.json", "score.pt")
    runs = local_out / "runs"
    n = 0
    if runs.is_dir():
        for d in sorted(x for x in runs.iterdir() if x.is_dir()):
            for name in KEEP:
                src = d / name
                if src.exists():
                    (dest / "runs" / d.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest / "runs" / d.name / name)
                    n += 1
    print(f"[publish] {local_out} -> {dest} "
          f"(results + plans + {n} run artifacts; stores omitted)", flush=True)


def _bench(experiment: str, dry_run: bool) -> None:
    """Drive run.py for one experiment; identical on every GPU flavor."""
    import os
    import subprocess
    import sys

    local_out = Path(SCRATCH) / experiment
    local_out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # bergson builds its gradient index from this rather than from run.py's
    # --out_dir, so it needs pointing at container-local disk explicitly or it
    # inherits BENCH_CACHE and writes >100 GB to a network volume.
    env["BERGSON_STORE"] = str(local_out / "bergson")
    if "fsdp" in experiment:
        # Without this the FSDP adapter does `model.to(dev)` BEFORE FSDP shards,
        # which materializes the whole model on one card: 72B in bf16 is ~145 GB
        # against a 141 GB H200, so it OOMs during init and never reaches the
        # capture.  CPU init lets FSDP's device_id move each unit to the device
        # as it shards.
        env["DATTRI_FSDP_CPU_INIT"] = "1"

    cmd = [
        sys.executable, "-u", "run.py",
        "--experiment", experiment,
        "--out_dir", str(local_out),
        "--dry-run" if dry_run else "--run",
    ]
    print(f"$ {' '.join(cmd)}  (cwd={RUN_DIR})", flush=True)
    try:
        subprocess.run(cmd, cwd=RUN_DIR, check=True, env=env)  # noqa: S603
    finally:
        # Publish and commit even on failure: cells that did finish are already
        # appended to results.jsonl and are worth keeping.
        _publish(local_out, experiment)
        results_vol.commit()
        cache_vol.commit()
        hf_vol.commit()


@app.function(image=image, volumes=VOLUMES, timeout=HOUR)
def plan_only(experiment: str) -> None:
    """List an experiment's cells and write its plans.  **No GPU.**

    A dry run does no attribution work, so provisioning the experiment's real
    GPU for it -- 4x H200 for the sharded ladder -- would burn accelerator time
    to print a table.  `run.py --dry-run` only imports tasks/models, neither of
    which touches torch.cuda, so this runs on CPU.
    """
    _bench(experiment, dry_run=True)


@app.function(image=image, gpu="H200:4", timeout=HOUR,
              memory=SHARDED_MEMORY_MIB, ephemeral_disk=SHARDED_DISK_MIB)
def preflight() -> None:
    """Prove the sharded container is grantable and NCCL works.  No model.

    `memory=` is a *request*: Modal enforces an unpublished maximum at Function
    creation, so 720 GiB may simply be refused, and there is no way to find out
    short of asking for it.  This starts the exact 72B container -- same GPU
    count, same memory, same disk -- reports what was actually granted, and
    initializes a 4-rank process group.  It downloads nothing and runs in
    about a minute, so a refusal or a NCCL problem costs a minute of 4x H200
    instead of surfacing after a 145 GB download.
    """
    import os
    import subprocess

    # torchrun takes a script PATH (or `-m module`); it has no `-c` flag, so the
    # probe is written to a file rather than passed inline.
    probe = """
import os, shutil
import psutil, torch
import torch.distributed as dist

lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{lr}"))

# ranks contribute 1..world; the sum proves every rank joined the collective.
t = torch.ones(1, device=f"cuda:{lr}") * (lr + 1)
dist.all_reduce(t)

p = torch.cuda.get_device_properties(lr)

# psutil reports the HOST's total RAM, not this container's limit -- on a
# shared 8-GPU node that reads ~2 TB while the cgroup cap is whatever
# `memory=` asked for.  The cgroup is the number that decides whether a 72B
# FSDP init survives, so read it directly and report both.
host_ram = psutil.virtual_memory().total / 2**30
cap = None
for f in ("/sys/fs/cgroup/memory.max",                      # cgroup v2
          "/sys/fs/cgroup/memory/memory.limit_in_bytes"):   # cgroup v1
    try:
        raw = open(f).read().strip()
        if raw != "max":
            cap = int(raw) / 2**30
        break
    except OSError:
        continue
# A cgroup file that just echoes the host size means no limit was applied.
if cap is None:
    cap_s = "unreadable"
elif cap >= host_ram * 0.95:
    cap_s = f"uncapped(={cap:.0f}GB=host)"
else:
    cap_s = f"{cap:.0f}GB"

# An overlay/ephemeral mount can answer 2**63 bytes, meaning "unbounded" --
# report that as such instead of printing 8.6e9 GB.
probe_dir = "/scratch" if os.path.isdir("/scratch") else "/"
free = shutil.disk_usage(probe_dir).free
disk_s = "unbounded" if free >= 2**62 else f"{free / 2**30:.0f}GB"

world = dist.get_world_size()
expect = world * (world + 1) // 2
print(
    f"rank {dist.get_rank()}/{world} {p.name} "
    f"vram={p.total_memory / 2**30:.0f}GB "
    f"cgroup_ram={cap_s} host_ram={host_ram:.0f}GB "
    f"free_disk={disk_s} ({probe_dir}) "
    f"allreduce={t.item():.0f} (expect {expect})",
    flush=True,
)
dist.destroy_process_group()
"""
    os.makedirs("/scratch", exist_ok=True)
    probe_path = "/tmp/preflight_probe.py"  # noqa: S108
    with open(probe_path, "w") as fh:
        fh.write(probe)
    subprocess.run(  # noqa: S603
        ["torchrun", "--nproc_per_node=4", "--master_port=29511", probe_path],
        check=True,
    )
    print("[preflight] container granted, 4 ranks up, NCCL all-reduce OK", flush=True)


@app.function(image=image, volumes={"/hf": hf_vol, "/cache": cache_vol},
              timeout=6 * HOUR)
def prefetch(experiment: str) -> None:
    """Warm both caches for an experiment: weights and token pools.  **No GPU.**

    Two costs move off the accelerator here.  Weight download happens inside
    `build_model`, which on the sharded ladder means pulling ~210 GB (Qwen-32B +
    -72B) while four H200s bill at the full rate.  Tokenization happens inside
    `load_data`, and `data.py` caches block pools per (model, dataset,
    block_size) -- so a 9-model experiment tokenizes nine times, all of it CPU
    work that would otherwise run with the GPU idle.

    Both land in Volumes, so every later run finds them ready.  Skipping this
    only costs money; the experiments work either way.
    """
    import sys

    from huggingface_hub import snapshot_download

    # The entrypoint file lands at /root/modal_app.py while the harness is
    # mounted at RUN_DIR, so `run`/`tasks` are not importable by default.
    # `_bench` sidesteps this by shelling out with cwd=RUN_DIR; this function
    # imports them in-process, so it has to put RUN_DIR on the path itself.
    if RUN_DIR not in sys.path:
        sys.path.insert(0, RUN_DIR)

    import run as R
    import tasks as T

    from data import load_task_data

    spec = R.EXPERIMENTS[experiment]
    pairs = {(t["model"], t["dataset"]) for t in T.grid(
        families=spec["families"], scales=spec["scales"],
        datasets=spec["datasets"], methods=spec["methods"],
        parallelisms=spec.get("parallelisms"))}
    wl = {**R.WORKLOAD, **spec.get("workload", {})}

    for i, (model_id, dataset) in enumerate(sorted(pairs), 1):
        print(f"[{i}/{len(pairs)}] {model_id} + {dataset}", flush=True)
        snapshot_download(model_id, allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model"])
        hf_vol.commit()
        # Builds and caches the block pool; same arguments the adapters use, so
        # the cache key matches and they find it rather than rebuilding it.
        load_task_data(model_id, dataset, wl["block_size"],
                       wl["n_train"], wl["n_test"], wl["seed"])
        cache_vol.commit()
    print(f"[prefetch] {len(pairs)} (model, dataset) pairs cached", flush=True)


@app.function(image=image, gpu="L40S", volumes=VOLUMES, timeout=6 * HOUR)
def run_l40s(experiment: str, dry_run: bool = False) -> None:
    _bench(experiment, dry_run)


# Panel (c) runs the same libraries up the Qwen ladder, so it needs an H200 for
# headroom -- kronfluence's full-dimension factors and bergson's index both grow
# with the model, and fp32 doubles every captured byte.
@app.function(image=baseline_image, gpu="H200", volumes=VOLUMES, timeout=12 * HOUR,
              ephemeral_disk=2048 * 1024)
def run_baselines_h200(experiment: str, dry_run: bool = False) -> None:
    _bench(experiment, dry_run)


@app.function(image=image, gpu="H200", volumes=VOLUMES, timeout=12 * HOUR)
def run_h200(experiment: str, dry_run: bool = False) -> None:
    _bench(experiment, dry_run)


@app.function(image=image, gpu="H200:4", volumes=VOLUMES, timeout=24 * HOUR,
              memory=SHARDED_MEMORY_MIB, ephemeral_disk=SHARDED_DISK_MIB)
def run_h200_sharded(experiment: str, dry_run: bool = False) -> None:
    _bench(experiment, dry_run)


@app.local_entrypoint()
def main(experiment: str = "scaling-pythia-A40", dry_run: bool = False) -> None:
    if experiment not in GPU_FOR:
        raise SystemExit(
            f"unknown experiment {experiment!r}\n"
            f"choices: {', '.join(GPU_FOR)}",
        )
    gpu = GPU_FOR[experiment]
    if dry_run:
        # No GPU: a plan listing does not need one, least of all 4x H200.
        print(f"=== {experiment} (plan only, CPU) ===")
        plan_only.remote(experiment)
        return
    fn = {"L40S": run_l40s, "H200": run_h200, "H200:4": run_h200_sharded,
          "H200-baseline": run_baselines_h200}[gpu]
    print(f"=== {experiment} on {gpu} ===")
    fn.remote(experiment, dry_run)
    print(f"=== results under the dattri-bench-results volume, {experiment}/ ===")


# Named sets, so a figure is one command instead of a remembered sequence.
# Order matters inside a group: cheaper and more likely to fail first, so a
# broken adapter surfaces before the expensive cells are paid for.
GROUPS: dict[str, list[str]] = {
    "panels-ab": ["scaling-families-H200", "scaling-qwen-H200-fsdp4"],
    "panel-c": ["scaling-crosslib-H200", "scaling-crosslib-H200-ekfac"],
    "efficiency": ["efficiency-dattri-llm-rank64", "efficiency-dattri-llm-full",
                   "efficiency-baselines-rank64", "efficiency-baselines-full"],
    "pythia": ["scaling-pythia-A40", "scaling-pythia-H200"],
}


def _fetch(experiment: str, out_dir: Path) -> bool:
    """Pull one experiment's results.jsonl off the volume, if it produced one."""
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


@app.local_entrypoint()
def bench(group: str = "", experiment: str = "", warm_first: bool = True,
          dry_run: bool = False) -> None:
    """Run a group or a single experiment end to end.

        modal run modal_app.py::bench --group panel-c
        modal run modal_app.py::bench --experiment scaling-crosslib-H200-ekfac

    Both spellings work because `main` takes --experiment and this used to take
    only --group, which is an easy and unhelpful thing to get wrong. Caches
    weights and token pools on CPU, runs each experiment in order, then pulls
    every results.jsonl into ./results/. One experiment failing does not stop
    the rest: each writes its own file, and a group is usually several
    independent figures' worth of cells.
    """
    if group and experiment:
        raise SystemExit("pass --group or --experiment, not both")
    if experiment:
        if experiment not in GPU_FOR:
            raise SystemExit(f"unknown experiment {experiment!r}\n"
                             f"choices: {', '.join(sorted(GPU_FOR))}")
        group, names = experiment, [experiment]
    else:
        group = group or "panels-ab"
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r}\n"
                             f"groups: {', '.join(GROUPS)}\n"
                             f"or pass --experiment <name>")
        names = GROUPS[group]
    print(f"=== {group}: {len(names)} experiment(s) ===")
    for i, exp in enumerate(names, 1):
        gpu = GPU_FOR[exp]
        print(f"\n--- [{i}/{len(names)}] {exp} on {gpu} ---")
        if dry_run:
            plan_only.remote(exp)
            continue
        if warm_first:
            try:
                prefetch.remote(exp)
            except Exception as e:  # noqa: BLE001 -- warming only saves money
                print(f"  [warm] skipped: {type(e).__name__}: {e}")
        fn = {"L40S": run_l40s, "H200": run_h200, "H200:4": run_h200_sharded,
              "L40S-baseline": run_baselines,
              "H200-baseline": run_baselines_h200,
          "H200:4-baseline": run_baselines_h200_x4}[gpu]
        try:
            fn.remote(exp, False)
        except Exception as e:  # noqa: BLE001 -- keep going; others are independent
            print(f"  [run] {exp} FAILED: {type(e).__name__}: {e}")
    if dry_run:
        return
    print("\n=== fetching results ===")
    out = Path("results")
    got = [e for e in names if _fetch(e, out)]
    print(f"\n{len(got)}/{len(names)} experiments produced results in {out}/")
    if got:
        print("next: python summarize.py results/*.jsonl")


@app.local_entrypoint()
def warm(experiment: str = "scaling-qwen-H200-fsdp4") -> None:
    """`modal run modal_app.py::warm --experiment X` -- cache weights, no GPU."""
    if experiment not in GPU_FOR:
        raise SystemExit(f"unknown experiment {experiment!r}")
    prefetch.remote(experiment)


@app.local_entrypoint()
def check() -> None:
    """`modal run modal_app.py::check` -- provision the sharded container only."""
    preflight.remote()
