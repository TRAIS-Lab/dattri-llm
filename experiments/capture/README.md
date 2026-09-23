# Ordinary versus invasive capture

How much of dattri-llm's attribution time the weight-gradient matmul accounts
for, and whether skipping it moves the scores: each of its methods runs through
the same pipeline with the ordinary hook family and with the invasive one, on
the workload of `experiments/benchmark`.

## Protocol

**Capture paths.** `invasive_linear_io` replaces each hooked `nn.Linear`
forward so its backward skips the weight-gradient matmul; it produces no
`weight.grad`, so it serves attribution only, never training. The adapter's
`hook_family` task key assigns one family to every hooked layer by name. Unset,
GradDot and full-dimension K-FAC/EK-FAC use `invasive_linear_io` and the
rank-64 K-FAC/EK-FAC store uses `linear_io` (the paths of `experiments/benchmark`); every row
records `hook_family`.

`capture.py` runs each method and projection regime with both families on the
workload of `experiments/benchmark` (1024 measured sequences after 8 warm-up ones, batch 8, fp32
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

Results are written next to the launcher. The launcher shares the cell
runner, the data, the model registry and the result logger of
`experiments/benchmark`, which must sit beside this directory.

```bash
export PYTHONPATH=/path/to/dattri-llm            # the repository root, so that dattri_llm is importable
python capture.py --experiment capture-query16 --dry-run
python capture.py --experiment capture-query16 --run
python capture.py --experiment capture-query1 --run
python capture.py --experiment capture-shared-fit --run    # only if scores differ
python capture.py --table                        # results/<experiment>/ or out/<experiment>/
```

Runs append to `out/<experiment>/results.jsonl` and write the score matrices
to `out/<experiment>/runs/`; `--table` reads `results/<experiment>/` when it
exists and `out/<experiment>/` otherwise.
