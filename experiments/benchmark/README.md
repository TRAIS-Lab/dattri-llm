# Efficiency and scaling benchmark

Everything needed to reproduce the efficiency and scaling results in the paper,
and nothing else.

## Layout

```
run.py       experiment -> one plan per (task, library) cell -> run them in order
log.py       BenchRun: phase timing + peak memory, appends to out/results.jsonl
models.py    model registry (family/scale -> HF id, params, dtype)
data.py      WikiText-103 blocks; identical inputs and order for every library
tasks.py     task grid construction
adapters/    one per library; all report through log.BenchRun
```

No scheduler is required. `run.py --run` executes the cells sequentially, one at
a time on purpose: two runs sharing a GPU would contaminate every timing and
memory number. To use a cluster, wrap the same command in your own job script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
python run.py --experiment crosslib-16query-dattri-llm-rank64 --run
```

## Experiments

One workload throughout — `n_train=64`, `n_test=1`, batch 1, 512-token
sequences — so every number is directly comparable. Two prefixes:
`efficiency-*` compares libraries at a fixed model scale; `scaling-*` varies
model scale for `dattri_llm`, one ladder per (family, device, parallelism).

| `--experiment` | What it measures | Models |
|---|---|---|
| `efficiency-dattri-llm-rank64` | our library, rank-64 projection | Pythia-0.5B |
| `efficiency-dattri-llm-full` | our library, full dimension | Pythia-0.5B |
| `efficiency-baselines-rank64` | LogIX, Bergson, Kronfluence, rank-64 | Pythia-0.5B |
| `efficiency-baselines-full` | the same three, full dimension | Pythia-0.5B |
| `scaling-qwen-H200` | one H200, up to the largest that fits | Qwen 0.5B–14B |
| `scaling-qwen-H200-fsdp4` | 4x H200 with FSDP, from the smallest that does not fit on one | Qwen 32B, 72B |
| `scaling-pythia-A40` | one A40 | Pythia 0.41B–6.9B |
| `scaling-pythia-H200` | one H200 (same models, different device) | Pythia 0.41B–6.9B |

Scale boundaries are measured, not guessed: at this workload Qwen-32B exhausts
one H200 (139.8 GB used, 786 MiB short), so the single-device ladder stops at
14B and the sharded one starts at 32B. Pythia's largest release is 6.9B, which
fits on one A40, so both Pythia ladders span the same models and differ only in
the device you run them on -- the tree cannot enforce that, so run each where
its name says.

```bash
python run.py --experiment scaling-qwen-H200 --dry-run   # list cells, write plans
python run.py --experiment scaling-qwen-H200 --run       # execute them in order
```

Sharded experiments (`scaling-qwen-H200-fsdp4`) need nothing extra: the plan
records `parallelism: fsdp`, so `run.py` selects the FSDP adapter and launches
it under `torchrun` automatically.

## Output

Results are appended to `out/results.jsonl`, one JSON line per run: the full task
spec, per-phase wall-clock and peak memory, and a `device` record (hostname, GPU,
torch version). Nothing is ever overwritten, so re-running a cell adds a row
rather than replacing one — which is how repeated timings are collected.

Memory is recorded two ways: `alloc_gb` (`max_memory_allocated`) and
`reserved_gb` (`max_memory_reserved`). **The paper reports allocated.** The two
agree to under a gigabyte on single-device runs but diverge by tens of gigabytes
under FSDP, where the all-gather/reshard cycle leaves the allocator holding freed
blocks — reserved is retention, not requirement.

## Batch size

Batch 1 everywhere. Nothing to tune and nothing to sweep.
