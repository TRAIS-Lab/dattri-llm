# Benchmark on Modal

A copy of `experiments/benchmark` plus `modal_app.py`. `modal_app.py` provisions
the GPU, mounts the caches, and runs `run.py --experiment X --run`; the
measurement itself is unchanged.

Four local entrypoints, so every invocation must be qualified with `::name`.

## Run

```bash
modal run modal_app.py::bench --experiment scaling-crosslib-H200
```

```bash
modal run modal_app.py::bench --group panel-c
```

`bench` caches weights and token pools on CPU, runs each experiment in order,
then pulls every `results.jsonl` into `./results/`. It takes `--group` or
`--experiment`, plus `--dry-run` (plans only, CPU) and `--warm-first false`.

`::main` runs one experiment without the caching or fetching, `::warm` does the
caching alone, `::check` provisions the sharded container and runs a 4-rank
NCCL all-reduce.

| group | experiments |
|---|---|
| `panels-ab` | `scaling-families-H200`, `scaling-qwen-H200-fsdp4` |
| `panel-c` | `scaling-crosslib-H200`, `scaling-crosslib-H200-ekfac` |
| `efficiency` | the four `efficiency-*` |
| `pythia` | `scaling-pythia-A40`, `scaling-pythia-H200` |

## Experiments

| experiment | models | libs | cells | GPU |
|---|---|---|---|---|
| `scaling-families-H200` | Pythia 0.41–6.9B, Qwen 0.49–14.8B | ours | 27 | H200 |
| `scaling-families-H200-ekfac` | same, EK-FAC only | ours | 9 | H200 |
| `scaling-qwen-H200` | Qwen 0.49–14.8B | ours | 15 | H200 |
| `scaling-qwen-H200-fsdp4` | Qwen 32.5B, 72.7B | ours | 6 | H200 x4 |
| `scaling-qwen-H200-fsdp4-110b` | Qwen 111.2B | ours | 3 | H200 x4 |
| `scaling-pythia-H200` | Pythia 0.41–6.9B | ours | 12 | H200 |
| `scaling-pythia-A40` | Pythia 0.41–6.9B | ours | 12 | L40S |
| `scaling-crosslib-H200` | Qwen 0.49–7.6B | all 3 | 24 | H200 |
| `scaling-crosslib-H200-ekfac` | same, EK-FAC only | all 3 | 12 | H200 |
| `efficiency-dattri-llm-{rank64,full}` | Pythia-0.5B | ours | 3 each | H200 |
| `efficiency-baselines-rank64` | Pythia-0.5B | bergson | 2 | H200 |
| `efficiency-baselines-full` | Pythia-0.5B | bergson, kronfluence | 6 | H200 |

Workload throughout: `n_train=64`, `n_test=1`, batch 1, 512-token sequences,
wikitext-103. `scaling-{families,qwen,pythia}-*` are bf16; every cross-library
cell is fp32. Methods are GradDot, K-FAC and EK-FAC. Cells run one at a time.

Libraries: `dattri_llm`, `bergson`, `kronfluence`. logix is not carried — it
keeps per-sample gradients at full parameter width (~98 GB for the 64-sample set
on Pythia-410m) and OOMed, and needs its own Python 3.10 image besides.

## Analysis

```bash
python summarize.py results/scaling-crosslib-H200.jsonl
```

```bash
python plot.py --panel c --results results/scaling-crosslib-H200.jsonl results/crosslib-ekfac.jsonl
```

```bash
python plot.py --panel b --results results/aggregate.jsonl
```

`plot.py` takes several result files and concatenates them, which is how a panel
spanning experiments is drawn. `--panel b` (default) is memory vs scale for one
library; `--panel a` is time vs scale per method; `--panel c` is the
cross-library comparison, memory by default and `--metric time` on request.
`summarize.py` prints per-library tables (`--by-family=<method>`, `--memory`,
`--methods=`). Both need matplotlib, which the repo does not depend on.

## Comparability rules the tooling enforces

**Precision is per experiment.** `models.dtype_for` takes an override and every
experiment states its dtype; the size-based fallback (fp32 below 1.0B) crosses
its threshold between the ladder's first and second rung. Every adapter records
the dtype it actually built, and `summarize.py` warns when a row has none —
without that, an adapter ignoring the override reads identically to one honoring
it.

**Peak memory is measured two ways, so cross-library panels convert.** Adapters
that run in-process report `torch.cuda.max_memory_allocated`; bergson runs
through its own CLI in a subprocess, where those counters see nothing, so it
polls `nvidia-smi memory.used` — allocator bytes plus reserved-but-unallocated
plus the CUDA context. `comparable_peak()` puts everything on whole-process
memory (`reserved_gb` for torch rows) for panel C; `run_peak()` keeps allocated
bytes for the single-library panels. A residual bias favours the torch-measured
libraries, since reserved still excludes the CUDA context.

**Attribution time is every phase except `build_model` and `load_data`.** The
names differ per library — ours `attribute`, bergson `fit`+`score`, kronfluence
`fit_factors`+`pairwise_scores` — and reading one hard-coded name scored
kronfluence at 0 s.

**Wall-clock is not comparable across libraries.** `run_ours.py` warms up and
times `n_meas` (32); the baseline adapters time the full `n_train` (64–65) with
no warm-up, and bergson records no `work_units`. Compare `s/smp`, or use the
memory axis.

**Strategies are not matched, by design.** The benchmark records end-to-end
totals. Ours captures at rank 64; kronfluence has no rank-64 mode and always
fits full-dimension factors; bergson forces `projection_dim=0` for EK-FAC. Part
of any K-FAC or EK-FAC gap is projection, not implementation.

## Environment

| path | volume | holds |
|---|---|---|
| `/results` | `dattri-bench-results` | `results.jsonl`, plans, per-run records |
| `/cache` | `dattri-bench-cache` | `BENCH_CACHE` — tokenized block pools |
| `/hf` | `dattri-bench-hf` | `HF_HOME` — model weights |
| `/scratch` | container-local | gradient stores, `BERGSON_STORE` |

`PYTHONPATH=/root/dattri-llm` — `dattri_llm` is mounted as a source tree, not
pip-installed, and `run.py` launches each cell as a subprocess.

Stores stay on container-local disk. A `modal.Volume` is network-backed, so a
store written there measures Modal's storage fabric rather than local-disk IO.
Only `results.jsonl`, `plans/` and the small per-run artifacts are published;
`runs/` is not copied wholesale, because adapters put their stores inside their
own run directory.

Models are ungated; no HF token needed. The image is unpinned, so a rebuild can
change torch and shift the numbers.

## Constraints

**Modal has no A40.** `scaling-pythia-A40` runs on an L40S: 48 GB against the
A40's 46 GB usable, so the ladder stops where it did, but L40S is Ada with ~1.2x
the bandwidth and is not a timing-identical stand-in. `log.py` records the real
`gpu_name` in every row.

**Only our own adapter shards.** `LAUNCH` in `run.py` records how each adapter
must be launched and what parallelism that gives: `run_ours_fsdp.py` is
`torchrun-fsdp`; `run_kronfluence.py` is `torchrun-ddp` (Accelerate with no
plugin replicates, so every rank holds a full model and per-device memory is
unchanged); `run_bergson.py` is `self-ddp` (it spawns its own workers from
`--nproc_per_node`, and `--fsdp` is incompatible with its single-process
query-build path). A sharded task refuses an adapter that can only replicate.

**`DATTRI_FSDP_CPU_INIT=1`** is set for sharded runs. Without it the FSDP
adapter calls `model.to(dev)` before sharding, materializing ~145 GB on a
141 GB card.

**Host RAM scales with world size**, not with the shard: every FSDP rank calls
`from_pretrained` independently. `memory=` is a billing request, not an enforced
cap — `check` reports `cgroup_ram` equal to host RAM.

**`Salesforce/wikitext`** rather than the bare id, which current
`huggingface_hub` rejects. Both `data.py` and `run_bergson.py` carry their own
dataset tables; the `_pool` cache key hashes the dataset name, not the path.

## Cost

H200 SXM $0.001261/sec, L40S $0.000542/sec, memory $0.00000222/GiB/sec; disk
bills as memory at 20:1.

| run | resources | rate |
|---|---|---|
| single H200 | 1x H200 | $4.55/hr |
| baselines | 1x H200 + 2 TiB disk | $5.37/hr |
| sharded | 4x H200 + 256 GiB + 1 TiB | $20.61/hr |
| L40S | 1x L40S | $1.96/hr |

`scaling-families-H200` measured 0.80 hr of compute. Run `warm` first: weight
download otherwise happens inside `build_model` at GPU rates.
