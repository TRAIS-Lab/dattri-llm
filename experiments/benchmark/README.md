# Efficiency benchmark

Time and memory of four attribution libraries (dattri-llm, Bergson,
Kronfluence, LogIX). Every launcher produces a `results.jsonl` file of
measured rows.

| launcher | what it measures |
|---|---|
| `benchmark.py` | four libraries on Pythia-0.5B, one A40, both projection regimes, 1 and 16 queries |
| `scaling.py` | four libraries up the Qwen ladder (0.5B to 110B) on H200s, batch 1 |
| `throughput.py` | four libraries on four H200s, each cell at its largest batch, up the Qwen ladder |
| `routing.py` | GradDot at full dimension against the sequence length: dattri-llm's cost model and each pinned route, with Bergson and Kronfluence, on Pythia-0.5B and one A40 |

## Protocol

A scale label is a nominal size that selects a released model of the family
(`utils/models.py`): `0.5b` is Pythia-410M in the Pythia family and
Qwen2.5-0.5B in the Qwen family, `1b` is Qwen2.5-1.5B, and `110b` is
Qwen1.5-110B. Every row records the model id and its parameter count
(`params_b`).

Every library scores the same token blocks in the same order
(`utils/data.py`, seeded). Each adapter records `build_model`, `load_data` and
its attribution phases separately. *Attribution time* is the sum of every
recorded phase except `build_model` and `load_data`; peak memory is the maximum
over the same phases. The attribution phases are `attribute` for dattri-llm,
`fit` + `score` for Bergson, `extract` + `score` for LogIX, and `fit_factors` +
`pairwise_scores` for Kronfluence. All adapters enable TF32 matmuls. Host
memory is read from the kernel's peak-RSS counter when a phase ends. Bergson
runs through its Python entry points in-process; its sharded runs spawn worker
processes, and their peak memory is read from NVML. A cell that fails is
recorded with `status: oom`, `timeout` or `error`.

**`benchmark.py`.** Pythia-0.5B (`EleutherAI/pythia-410m`), fp32, WikiText-103, 1024 training
sequences of 512 tokens, batch 8, one A40. `query1` scores one query, `query16`
sixteen. Rank-64 uses each library's own projection (LoGra for dattri-llm and
LogIX, Bergson's projected index); full dimension uses none. Kronfluence runs
at full dimension only and Bergson's EK-FAC accepts no projection, so those
rank-64 cells are not listed (`NOT_EXPRESSIBLE` in `benchmark.py`).

**`routing.py`.** The setting of `benchmark.py` with the sequence length T swept
from 32 to 2048 tokens (and, at one sequence per step, to 16384): GradDot at
full dimension, one query.
`--experiment routing-steps-b8` uses batch 8 at every T and times 128 steps
after 8 warm-up steps, so time is reported per step;
`routing-steps-b{1,2,4,16}` use the other batch sizes. dattri-llm runs three
times per T: with its cost model choosing per layer (`route: auto`) and with each
route pinned for every layer (`factorized`, `materialized`; the capture
representation follows the pinned route). Bergson materializes the training
side; Kronfluence holds a dense query and chooses by a contraction-order
search between materializing the training side and applying the query to the
factors. `--experiment routing` is the fixed-token protocol: 524,288 training
tokens at every T, batch 8 up to T = 512 and 4096 tokens per batch beyond it
(its T = 512 cells are the GradDot full-dimension cells of `benchmark.py`).
`routing-batch1` runs that protocol at one sequence per step on an eighth of
the tokens.

**`scaling.py`.** Qwen2.5 0.5B to 72B and Qwen1.5-110B, bf16, batch 1,
512-token sequences, one query per device, rank-64 projection wherever the library
supports it (Kronfluence runs at full dimension). Each library runs 8 warm-up
samples untimed and is timed on the next 64. `scaling-<method>` runs 0.5B to
32B on one H200; a Kronfluence cell is cut off after 1800 s and recorded as
`timeout`. `scaling-<method>-fsdp4` runs on four H200s: dattri-llm with FSDP
at 72B and 110B, and Bergson with its own `--fsdp`. Bergson's K-FAC and EK-FAC
keep full-dimension factors, and their 7B to 32B cells run sharded.
Kronfluence runs under FSDP from the first scale that does not fit one H200:
32B for GradDot, 7B to 32B for K-FAC and EK-FAC. Bergson's
query build shards a query set across its ranks, from four queries on, so its
72B and 110B GradDot cells are given four queries: one per device, as for
dattri-llm, which captures its one query on every rank. LogIX runs one replica per GPU
and is listed up to 32B.

**`throughput.py`.** Qwen ladder, bf16, 512-token sequences, four H200s.
Every library runs every method (GradDot, K-FAC, EK-FAC) with rank-64
projection wherever the library supports it and full dimension elsewhere
(`PROJECTION` in `throughput.py`). Each library uses its own multi-GPU mode:
dattri-llm FSDP with frozen capture and live sharded scoring; Bergson
`--fsdp`; Kronfluence FSDP; LogIX data-parallel replicas. Each (library,
method, scale) cell runs at the largest power-of-two per-GPU batch at which
the library completes the workload on the four cards within the time limit
(`BATCHES`; 0 means out of memory at batch 1 and is recorded as `oom` without
running). The workload is fixed: every cell attributes the same 512 training
samples (`WORKLOAD`) in as many steps of its own batch as that takes, after 2
warm-up steps. The time is the whole attribution call, Kronecker-factor fit
included, and excludes model loading, data preparation and process start-up
for every library. Every library of a scale scores the same number of queries
(`n_test`): one through 32B, four at 72B and 110B. All libraries of a scale
run on one host, one after another.

Bergson spawns worker processes in every pipeline step, and each worker loads
the model; the model loads and the start-up before them (interpreter, imports,
CUDA context, NCCL rendezvous) are timed inside each worker and subtracted. A
Bergson row holds `time_raw_s` (the summed phases),
`time_no_worker_load_s` (model loads subtracted) and `time_s` (model loads and
start-up subtracted); `time_s` is the reported time.

**Capture paths.** `invasive_linear_io` replaces each hooked `nn.Linear`
forward so its backward skips the weight-gradient matmul; it produces no
`weight.grad`, so it serves attribution only, never training. The adapter's
`hook_family` task key assigns one family to every hooked layer by name. Unset,
GradDot and full-dimension K-FAC/EK-FAC use `invasive_linear_io` and the
rank-64 K-FAC/EK-FAC store uses `linear_io` (the paths of `benchmark.py`); every row
records `hook_family`.

`capture.py` runs each method and projection regime with both families on the
workload of `benchmark.py` (1024 measured sequences after 8 warm-up ones, batch 8, fp32
with TF32 matmuls, one A40): five repetitions per pair, alternating which
family runs first, 60 cells per query count. The model runs in eval mode with
trainable parameters and no optimizer, and the streamer clears parameter
gradients each step in both families. Scores are compared offline after
aligning rows and columns by sample hash: maximum absolute and relative
Frobenius error, per-query Spearman and top-10 overlap. A pair agrees when its
relative error is within the precision tolerance or ten times the run-to-run
floor of the ordinary path, whichever is larger. `capture-shared-fit` is the
secondary K-FAC/EK-FAC check: the invasive run scores against the fit of the
`linear_io` run, so any remaining difference comes from capture alone.

## Running

Results and caches are written next to the launcher.

`benchmark.py`, `scaling.py` and `routing.py` share one command line:
`--experiment NAME` with `--dry-run` (list the cells and write the plan files)
or `--run` (execute them in order), and optionally `--out_dir DIR`,
`--libs a,b` (keep these libraries) and `--cells 0-4,7` (keep these indices of
the `--dry-run` listing, to split an experiment over several jobs).

| launcher | experiments |
|---|---|
| `benchmark.py` | `query1`, `query16` |
| `scaling.py` | `scaling-<method>`, `scaling-<method>-fsdp4`, with `<method>` one of `graddot`, `kfac`, `ekfac` |
| `routing.py` | `routing-steps-b{1,2,4,8,16}`, `routing`, `routing-batch1` |

```bash
# one A40; cells run one at a time
python benchmark.py --experiment query1 --dry-run
python benchmark.py --experiment query1 --run
python benchmark.py --experiment query16 --run
python benchmark.py --experiment query16 --run --libs dattri_llm   # dattri-llm cells only

# one A40
python routing.py --experiment routing-steps-b8 --run

python scaling.py --experiment scaling-graddot --run          # one H200
python scaling.py --experiment scaling-graddot-fsdp4 --run    # four H200s

# four H200s; the libraries of a scale run in one process, one after another
python throughput.py --run 0.5b,1b,3b,7b,14b,32b,72b,110b     # -> out/throughput/<scale>/results.jsonl
python throughput.py --run 7b --libs bergson --methods graddot
```

Runs append to `out/<experiment>/results.jsonl` (`throughput.py`:
`out/throughput/<scale>/results.jsonl`); `--out_dir` changes the directory.

Environment:

* `PYTHONPATH` must reach the repository root (`dattri_llm` runs from the
  working tree).
* `BENCH_CACHE` points at the tokenized-block cache and `HF_HOME` at the model
  weights.
* `BERGSON_STORE` is the directory of Bergson's gradient stores.
* `THROUGHPUT_N_GPUS` sets the GPU count of `throughput.py` (default 4).
* Baselines are pinned in `utils/versions.py` (Bergson 0.26.1, Kronfluence
  1.0.1, LogIX 0.1.1) and the adapters refuse other versions. LogIX needs
  Python < 3.11.

Settings for the timed region, the same for every library:

* every gradient store, index or log is written under `--out_dir` (Bergson's
  under `BERGSON_STORE`); point both at local disk;
* read the checkpoint once before the first cell (`cat model.safetensors >
  /dev/null`) so the page cache is warm;
* set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and the
  `PYTORCH_ALLOC_CONF` spelling of newer PyTorch versions).
