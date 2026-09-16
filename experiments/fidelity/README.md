# Fidelity of optimizer-aware attribution

The experiment behind the fidelity section and its appendix: how well DVEmb and
AdamW-influence (the `dattri_llm` implementations on captured gradients)
track trajectory-specific leave-one-out retraining, against MAGIC and SOURCE
(Bergson's implementations on the same trajectory), in the two settings of
Deng et al. (arXiv:2605.18814).

```
fidelity.py             the runs behind the paper: --experiment, --run, --collect, --table
utils/protocol.py       one AdamW trajectory under hooks, query capture, TSLOO reruns, scoring
utils/settings.py       the two settings: mlp (MNIST + MLP), gpt2 (WikiText-2 + GPT-2)
utils/ours.py           DVEmb and AdamW-influence on one (setting, lr, seed), with the ground truth
utils/magic.py          MAGIC through Bergson's functional trainer on the same trajectory
utils/source.py         SOURCE through Bergson's approximate-unrolling pipeline (GPT-2)
utils/source_mlp.py     SOURCE with exact Gauss-Newton curvature (the MLP, which Bergson cannot take)
utils/curvature.py      the curvature-term ablation on the MLP
utils/timing.py         a plain training run, the unit of the timing comparison
utils/tables.py         run directories -> results/fidelity.jsonl -> the tables
results/fidelity.jsonl  the collected rows behind the paper (run directories are not tracked)
```

## Protocol

One deterministic trajectory per (setting, learning rate, seed): seeded
initialization and batch order, no dropout, no gradient clipping, so a
leave-one-out rerun is an exact counterfactual. The reference run is wrapped
in a `HookManager` with an offload callback and the optimizer-state callback,
caching the per-sample gradients and the AdamW moments around every update;
the validation points' gradients are captured at the final model; both
methods score from those caches. The ground truth (TSLOO) reruns the
trajectory once per selected training example with that example removed
from the batch of its last occurrence (`--tsloo-at last`; the MLP has one
epoch, so the conventions coincide) and records the change of every
validation loss. Fidelity is the Spearman correlation across the selected
examples, averaged over validation points, then over seeds.

| | `mlp` | `gpt2` |
|---|---|---|
| model | 784-16-16-10 MLP, ReLU | GPT-2 124M, dropout off |
| data | 6,000 random MNIST images, 1 epoch, batch 64 | 512 random 128-token WikiText-2 blocks, 3 epochs, batch 32 |
| optimizer | AdamW, betas (0.9, 0.95), constant lr | AdamW, betas (0.9, 0.999), linear schedule, 10% warmup |
| attributed | every parameter | 10 random masks of 512 coordinates per layer (`--n-masks 0`: every coordinate, via snapshots and replay) |
| ground truth | 200 training images x 500 test images | 50 training blocks x 256 validation blocks; the first 64 are the queries every method is scored on |
| seeds | 0, 1, 2 | 0, 1, 2 |

MAGIC replays the trajectory with Bergson's functional trainer (torchopt
AdamW, `eps_root=1e-16`, GPT-2's tied embedding kept tied) and backpropagates
each query loss through the whole run. SOURCE runs Bergson's pipeline on
checkpoints saved every 8 steps from the same loop (3 segments, EK-FAC
factors, all queries in one pass); `SOURCE_WORK_DIR` puts its intermediates
on local disk. Both are scored against the ground truth of the matching
`ours.py` run (`--truth-dir`).

## Running

Run from the copy of this directory under `experiments_exe/` (see
`experiments/README.md`): results, caches and figures are written next to
the launcher and stay out of the tree.

```bash
export PYTHONPATH=/path/to/dattri-llm        # the library runs from the working tree
python fidelity.py --experiment mlp --dry-run  # list the runs
python fidelity.py --experiment mlp --run      # 27 runs, one GPU
python fidelity.py --experiment gpt2 --run
python fidelity.py --experiment gpt2-full --run       # every coordinate (disk-heavy: ~125 GB of snapshots per run;
                                                      # FIDELITY_SNAPSHOT_DIR points them at a local disk)
python fidelity.py --experiment gpt2-protocols --run  # other ground-truth conventions
python fidelity.py --experiment gpt2-mask --run       # k = 1024
python fidelity.py --experiment mlp-ablations --run   # curvature term, coverage
python fidelity.py --experiment timing --run          # one exclusive A40 in the paper
python fidelity.py --collect --table
```

Runs write to `results/<setting>/lr<lr>_seed<seed>[_variant]/` (`result.json`,
`rows.pt`, `matrices.pt`, and the gradient caches) and are skipped when their
`result.json` exists. `--collect` reads every run directory into
`results/fidelity.jsonl`, the tracked rows behind the tables; `--table`
prints Table 2, Table 9 and the appendix numbers from them.
The timing experiment is meant for one exclusive GPU with nothing else
running; the paper's numbers are from one A40 on a cluster node, with
`SOURCE_WORK_DIR` and `FIDELITY_SNAPSHOT_DIR` on node-local disk.
MNIST, WikiText-2 and GPT-2 come from the Hugging Face hub; Bergson
(`bergson`, `torchopt`) is needed for MAGIC and SOURCE.
