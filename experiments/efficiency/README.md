# Pairwise efficiency benchmarks

Head-to-head efficiency comparisons of `dattri_llm` against five existing TDA
libraries, one folder per pair.  Each pair runs both libraries under one of
**their** flagship/documented settings (same model, data, loss, batch sizes,
projection ranks, device), measures walltime / throughput / peak GPU memory /
store size per phase with the shared `common.Meter`, and cross-checks that the
two sides compute the same (or an equivalent) score -- see each folder's
README for the exact setting and fairness notes.

## Running

```bash
python run.py logix              # run one pair here, serially
python run.py logix --slurm      # submit it as one SLURM job (one GPU)
python run.py all --slurm        # every pair, chained so none overlap
python run.py --list             # show the pairs and their steps
python summarize.py              # aggregate every results.jsonl into a table
```

Pair names: `dattri`, `logix`, `kronfluence`, `ghostsuite`, `bergson`.

Each run first clears that pair's previous `results.jsonl`, score matrices and
gradient stores, then runs its steps in order and prints the results.

**Always run pairs serially.**  Two benchmarks sharing a GPU contend and
silently inflate each other's walltimes (this corrupted a published number
once); `--slurm` therefore chains jobs with `--dependency=afterany` rather than
submitting them side by side.

## Pairs

| pair | setting (theirs) | algorithm matched |
|---|---|---|
| `dattri` | dattri benchmark: mnist/mlp Grad-Dot, 5000x500 | grad-dot (ours exact; theirs proj-512 sketch, cannot be disabled) |
| `logix` | LoGRA: GPT-2, wikitext-103 0.5%, rank-64 | KFAC in the 4096-d/layer projected space |
| `kronfluence` | GPT-2 fine-tuned on wikitext-2, EKFAC | EKFAC, full parameter space, damping 1e-8 |
| `ghostsuite` | In-Run Data Shapley: online train-vs-val grad-dots during GPT2-Small training | per-step ghost grad-dot vs val gradient |
| `bergson` | pythia-14m + pile-10k rank-16 projected gradient index | grad-dot over per-module double-sided sketches |

`kronfluence` additionally needs `vs_kronfluence/checkpoints/model.pth` from
their `train.py` before it will run -- see that folder's README.

## Layout

```
run.py          # the entry point: pair definitions + local/SLURM execution
common.py       # Meter: per-phase walltime, peak GPU memory, throughput, store size
summarize.py    # all results.jsonl -> one table
REPORT.md       # findings write-up
vs_<lib>/       # per pair: run_dattri_llm.py + run_<lib>.py (or one run_pair.py
                #   with --side), data.py, README.md
archive/        # one-off probes, profilers and superseded launchers (safe to delete)
```

Each side's runner is named for the library it drives -- `run_dattri_llm.py`
for ours, `run_logix.py` / `run_bergson.py` / `run_kronfluence.py` for theirs.
`vs_dattri` and `vs_ghostsuite` instead use a single `run_pair.py --side ...`,
since their sides share almost all setup.

These benchmarks measure **efficiency only** -- walltime, throughput, memory and
store size.  They do not cross-check scores; each pair's README records the
score-agreement findings from when that was verified separately.

Generated artifacts (`results.jsonl`, `score_*.pt`, gradient stores, SLURM logs)
are gitignored -- every run regenerates them.

Environment: torch 2.11.0+cu130, transformers 4.55.4, single NVIDIA A40
(44 GB) per run, fp32 (TF32 matmuls where a side enables it).
