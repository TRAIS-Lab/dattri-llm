# Efficiency pair: dattri_llm vs dattri

**Setting** (dattri's own benchmark protocol): mnist/mlp Grad-Dot -- per-sample
training gradients dotted against per-sample test gradients at one pre-trained
checkpoint.  5000 train x 500 test, dattri's `load_benchmark` checkpoint, all
linear layers (the whole MLP), batch 500, fp32, single GPU.

## Sides

* **dattri (default, cpu-proj)**: `TracInAttributor` exactly as shipped.
  Note: dattri's TracIn *always* random-projects per-sample gradients
  (`DEFAULT_PROJECTOR_KWARGS`: proj_dim=512; `projector_kwargs=None` cannot
  disable it), and the default projector runs on **CPU**.  Its score is
  therefore a 512-dim JL sketch of Grad-Dot.
* **dattri (cuda-proj)**: same, but with `projector_kwargs={"device": "cuda"}`
  -- dattri's best-case configuration (CudaProjector).
* **dattri_llm (ours)**: on-the-fly `TracInAttributor.attribute` -- factorized
  ("ghost") per-sample gradients captured by hooks, cross-grammed without
  materializing weight gradients, **no projection: the exact score**.

## Fairness

Same checkpoint, same 5000/500 subsets, same batch size, same device/dtype.
Phases measured with the shared `Meter` (walltime, samples/s, peak allocated
GPU memory).  Because the two libraries compute different quantities by
construction (dattri cannot turn its sketch off), each side reports
`pearson_vs_exact` -- correlation against the exact Grad-Dot matrix.  Ours was
verified elementwise against plain per-sample autograd (worst entry matches to
5 decimals); dattri's sketch lands at ~0.89.

Run: `./run.sh [gpu]` -> `results.jsonl`
