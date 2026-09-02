# Benchmark on Modal

A copy of `experiments/benchmark` plus `modal_app.py`. `modal_app.py` provisions
the GPU, mounts the caches, and runs `run.py --experiment X --run`; the
measurement itself is unchanged.

Three local entrypoints, so every invocation must be qualified with `::name`.

## Run

```bash
modal run modal_app.py::main --experiment scaling-families-H200 --dry-run
```

```bash
modal run modal_app.py::warm --experiment scaling-families-H200
```

```bash
modal run modal_app.py::main --experiment scaling-families-H200
```

```bash
modal run modal_app.py::check
```

`--dry-run` lists cells and writes plans on CPU. `warm` caches weights and
tokenized block pools on CPU. `check` provisions the sharded container and runs
a 4-rank NCCL all-reduce.

## Experiments

| experiment | models | cells | GPU |
|---|---|---|---|
| `scaling-families-H200` | Pythia 0.41–6.9B, Qwen 0.49–14.8B | 27 | H200 |
| `scaling-families-H200-ekfac` | same, EK-FAC only | 9 | H200 |
| `scaling-qwen-H200` | Qwen 0.49–14.8B | 15 | H200 |
| `scaling-qwen-H200-fsdp4` | Qwen 32.5B, 72.7B | 6 | H200 ×4 |
| `scaling-pythia-H200` | Pythia 0.41–6.9B | 12 | H200 |
| `scaling-pythia-A40` | Pythia 0.41–6.9B | 12 | L40S |
| `efficiency-dattri-llm-rank64` | Pythia-0.5B, rank-64 | 3 | L40S |
| `efficiency-dattri-llm-full` | Pythia-0.5B, full dim | 3 | L40S |

Workload throughout: `n_train=64`, `n_test=1`, batch 1, 512-token sequences,
wikitext-103. `scaling-*` is bf16; `efficiency-*` is fp32. Methods are GradDot,
K-FAC and EK-FAC. Cells run one at a time.

Only `dattri_llm` is carried here; the logix, kronfluence and bergson adapters
stay in the parent tree.

## Analysis

```bash
modal volume get dattri-bench-results <experiment>/results.jsonl results/
```

```bash
python summarize.py results/aggregate.jsonl
```

```bash
python summarize.py results/aggregate.jsonl --memory --methods=graddot,kfac
```

```bash
python plot.py
```

`summarize.py` prints per-method tables (`--by-family=<method>`, `--memory`,
`--methods=`). `plot.py` reads `results/aggregate.jsonl` and writes
`results/panel_b.png` by default, or `results/panel_a.png` with `--panel a`
(`--panel both` for both; only panel B is tracked). Both flag
mixed device or dtype. `plot.py` needs matplotlib, which the repo does not
depend on.

Single-device and FSDP timings are not comparable: `run_ours_fsdp.py` has no
warm-up and times `n_train + n_test` samples where `run_ours.py` times `n_meas`
after one. Panel A therefore uses single-device rows only. Peak memory is
unaffected at batch 1.

## Environment

| path | volume | holds |
|---|---|---|
| `/results` | `dattri-bench-results` | `results.jsonl`, plans, per-run records |
| `/cache` | `dattri-bench-cache` | `BENCH_CACHE` — tokenized block pools |
| `/hf` | `dattri-bench-hf` | `HF_HOME` — model weights |
| `/scratch` | container-local | gradient stores |

`PYTHONPATH=/root/dattri-llm` — `dattri_llm` is mounted as a source tree, not
pip-installed, and `run.py` launches each cell as a subprocess.

The gradient store stays on container-local disk. A `modal.Volume` is
network-backed, so a store written there measures Modal's storage fabric rather
than local-disk IO, and `BenchRun.record_disk` reports those bytes as the run's
disk cost. Only `results.jsonl`, `plans/` and `runs/` are published to the
volume.

Models are ungated; no HF token needed. The image is unpinned, so a rebuild can
change torch and shift the numbers.

## Constraints

**Modal has no A40.** Supported types are T4, L4, A10, L40S, A100, A100-40GB,
A100-80GB, RTX-PRO-6000, H100, H200, B200, B300. `scaling-pythia-A40` runs on an
L40S: 48 GB against the A40's 46 GB usable, so the ladder still stops where it
did, but L40S is Ada with ~1.2× the bandwidth and is not a timing-identical
stand-in. `log.py` records the real `gpu_name` in every row.

**The sharded ladder uses 4 GPUs.** `models.n_gpus` here carries an `(80, 4)`
rung the parent lacks. `run.py` feeds that count to `torchrun --nproc_per_node`,
so `GPU_FOR` and `n_gpus` must agree or the launch fails.

**`DATTRI_FSDP_CPU_INIT=1`** is set for sharded runs. Without it the FSDP adapter
calls `model.to(dev)` before sharding, materializing ~145 GB on a 141 GB card.

**Host RAM scales with world size**, not with the shard: every FSDP rank calls
`from_pretrained` independently, so 4 × 145 GB worst case at 72B. `memory=` is a
billing request, not an enforced cap — `check` reports `cgroup_ram` equal to host
RAM. The sharded function requests 256 GiB and 1 TiB of disk.

**Precision is per experiment.** `models.dtype_for` takes an override and every
experiment states its dtype; the size-based fallback (fp32 below 1.0B) crosses
its threshold between the ladder's first and second rung.

**EK-FAC captures factorized.** K-FAC keeps `logra_materialized` — its Fisher
comes from `KroneckerCovarianceCallback` at capture — but EK-FAC's `fit()` needs
`(a, g)` factors and falls back to a dense empirical Fisher without them.

**`Salesforce/wikitext`** rather than the bare id, which current
`huggingface_hub` rejects. The `_pool` cache key hashes the dataset name, not
the path.

## Cost

H200 SXM $0.001261/sec, L40S $0.000542/sec, memory $0.00000222/GiB/sec; disk
bills as memory at 20:1.

| run | resources | rate |
|---|---|---|
| `scaling-families-H200` | 1× H200 | $4.55/hr |
| `scaling-qwen-H200-fsdp4` | 4× H200 + 256 GiB + 1 TiB | $20.61/hr |
| `scaling-pythia-A40` | 1× L40S | $1.96/hr |

`scaling-families-H200` measured 0.80 hr of compute. Run `warm` first: weight
download otherwise happens inside `build_model` at GPU rates (~210 GB for the
sharded ladder).
