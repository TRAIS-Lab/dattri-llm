# Efficiency benchmark

The two efficiency experiments in the paper, and nothing else:

| launcher | paper | what it measures |
|---|---|---|
| `benchmark.py` | Tables 1 and 6 | four libraries on Pythia-0.5B, one A40, both projection regimes, 1 and 16 queries |
| `scaling.py` | scaling figure | three libraries up the Qwen ladder (0.5B to 110B) on H200s |

Both share `utils/`: one adapter per library, the tokenized-block dataset, the
model registry, the result logger, the pinned baseline versions, and the
sequential cell runner. A *cell* is one (task, library) run; every cell appends
one self-describing JSON line (task, per-phase wall-clock and peak memory,
device fingerprint, baseline versions) to `results.jsonl`.

```
benchmark.py            Tables 1/6: cells, --run, --table
scaling.py              scaling figure: cells, --run, --figure, Modal entrypoints
utils/runner.py         expand cells -> plan files -> run them one at a time
utils/adapters/         run_ours.py, run_ours_fsdp.py, run_bergson.py, run_logix.py, run_kronfluence.py
utils/data.py           WikiText-103 token blocks; identical inputs and order for every library
utils/models.py         family/scale -> HF id, parameter count
utils/log.py            BenchRun: phase timing, peak memory, disk, device details
utils/versions.py       pinned baseline versions, asserted at adapter start
utils/tables.py         results/query{1,16}.jsonl -> the two tables
utils/figure.py         results/scaling-*.jsonl  -> results/scaling.{pdf,png}
results/                the measured rows behind the paper, and the rendered figure
```

## Protocol

Every library scores the same token blocks in the same order
(`utils/data.py`, seeded). Attribution time is every recorded phase except
`build_model` and `load_data`; peak memory is the maximum over those phases.
Phase names differ per library (ours `attribute`; Bergson `fit` + `score`;
LogIX `extract` + `score`; Kronfluence `fit_factors` + `pairwise_scores`), so
setup is excluded rather than attribution named. Bergson runs through its
Python entry points in-process; its sharded runs spawn worker processes, so
their peak memory is read from NVML rather than torch's allocator. A cell that
dies is recorded with `status: oom` or `error`, so a library that stops
climbing a ladder leaves a row.

**Tables 1 and 6.** Pythia-0.5B, fp32, WikiText-103, 1024 training
sequences of 512 tokens, batch 8, one A40. Table 1 scores one query, Table 6
sixteen. Rank-64 uses each library's own projection (LoGra for ours and
LogIX, Bergson's projected index); full dimension uses none. Cells a library
cannot express are `n/a`: Kronfluence has no projected mode, and Bergson's
EK-FAC refuses projection. Every cell runs at batch 8; at sixteen queries
the resident query representation is 1.1 GB per query, which is where our
full-dimension peaks at 16 queries come from.

**Scaling figure.** Qwen2.5 0.5B to 72B and Qwen1.5-110B, bf16, batch 1,
512-token sequences, rank 64, one query. Each library warms up on 8 samples
untimed and is timed on the next 64. Models up to 32B run on one H200; 72B
and 110B run sharded over four (ours with FSDP; Bergson with its own `--fsdp`,
scoring four queries there because its query build only shards with at least
four chunks). Bergson's K-FAC and EK-FAC keep full-dimension factors and
exhaust one card from 7B, so those cells run sharded. LogIX has no sharded
path, so its curves end at 32B. Repeated cells reduce to the median-time run.

## Running

Run from the copy of this directory under `experiments_exe/` (see
`experiments/README.md`): results, caches and figures are written next to
the launcher and stay out of the tree.

`--experiment` accepts `query1` and `query16` (`benchmark.py`: the one- and
sixteen-query tables) and, for
`scaling.py`, `scaling-<method>` and `scaling-<method>-fsdp4` with `<method>`
one of `graddot`, `kfac`, `ekfac`: eight experiments, each a list of cells
(`--dry-run` prints them).

```bash
# Tables 1 and 6 (one A40; cells run one at a time, on purpose)
python benchmark.py --experiment query1 --dry-run
python benchmark.py --experiment query1 --run
python benchmark.py --experiment query16 --run
python benchmark.py --experiment query16 --run --libs dattri_llm   # our cells only
python benchmark.py --table                      # reads results/query{1,16}.jsonl

# scaling figure, locally with the right GPUs ...
python scaling.py --experiment scaling-graddot --run          # one H200
python scaling.py --experiment scaling-graddot-fsdp4 --run    # four H200s
# ... or on Modal (provisions the GPUs, caches weights, fetches results/)
modal run scaling.py::bench --experiment scaling-graddot
modal run scaling.py::bench --all
python scaling.py --figure                       # results/scaling-*.jsonl -> results/scaling.{pdf,png}
```

Runs write to `out/<experiment>/results.jsonl` (appending, never
overwriting). To make a run the source of a table or the figure, copy it to
`results/<experiment>.jsonl`; `results/scaling.pdf` is the paper's
`figures/scaling_crosslib.pdf`.

Environment: `PYTHONPATH` must reach the repository root (`dattri_llm` runs
from the working tree), `BENCH_CACHE` points at the tokenized-block cache and
`HF_HOME` at the model weights. Baselines are pinned in `utils/versions.py`
and the adapters refuse other versions. LogIX needs Python < 3.11.

The tables report *attribution time*: each adapter records `build_model`,
`load_data` and its attribution phases separately, and the two setup phases
are excluded (as in the scaling figure), so cold checkpoint reads and each
library's own data pipeline do not enter the comparison. All four adapters
enable TF32 matmuls. Three things about the machine matter for the timed
region and must be the same for every library:

* every gradient store, index or log is written under `--out_dir` (Bergson's
  under `BERGSON_STORE`), so point both at **node-local** disk, not a shared
  filesystem, and copy `results.jsonl` back afterwards;
* read the checkpoint once before the first cell (`cat model.safetensors >
  /dev/null`) so `build_model` reflects a warm page cache;
* set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and the newer
  `PYTORCH_ALLOC_CONF` spelling): two Bergson cells sit within 1 GB of the
  A40's capacity and fragment without it.

On a SLURM cluster wrap the same command:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
export BERGSON_STORE=/tmp/bergson_$SLURM_JOB_ID TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
cat $HF_HOME/hub/models--EleutherAI--pythia-410m/snapshots/*/model.safetensors > /dev/null
python benchmark.py --experiment query1 --run --out_dir /tmp/bench_$SLURM_JOB_ID/query1
cp /tmp/bench_$SLURM_JOB_ID/query1/results.jsonl results/query1.jsonl
```

## Results in this directory

`results/query1.jsonl` and `results/query16.jsonl` hold the A40 rows behind the
tables; `results/scaling-<method>.jsonl` and `results/scaling-<method>-fsdp4.jsonl`
hold the H200 rows behind the figure, including the recorded OOM rows. Every
cell of the tables traces to a row in these files (the previous run sets are
kept under `results/archive/`). Every cell runs at batch 8.
