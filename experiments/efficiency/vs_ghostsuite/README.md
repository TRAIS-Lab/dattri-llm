# Efficiency pair: dattri_llm vs GhostSuite (online GradDotProd)

**Setting** (GhostSuite's ``examples/lm/graddotprod_lm``, In-Run Data Shapley):
train GPT2-Small (124M, their model class, seq 1024) while computing, **every
step**, the dot product between each train sample's gradient and a fixed
val-batch gradient at the current weights, then updating on the full batch.
Synthetic seeded token stream (their built-in mode), batch 16, val batch 8,
fp32, AdamW, grad-clip 1.0, 30 steps, single GPU.  Both sides train on
identical batches (same seed, same draw sequence).

## Sides

* **ghostsuite(regular)** / **dattri_llm(baseline)**: plain training, no
  scoring -- the overhead reference for each loop.
* **ghostsuite(graddotprod)**: their default decoupled in-graph +
  torch.compile fast path (val gradient recovered inside one combined
  backward).
* **ghostsuite(graddotprod,eager)**: their eager per-layer-hook engine.
* **dattri_llm(ghost-callback)**: ``HookManager`` +
  ``DataSelectionCallback(threshold=0, score_mode="ghost",
  target="val_loader")`` -- scores every sample in the factorized domain and
  drops nothing, leaving the update untouched.

## Reading the numbers

The meaningful metric for online engines is **overhead over the plain step**
(``step_time_median`` vs the baseline's), plus peak memory.  Step 0 is
excluded from the median (torch.compile warmup on their fast path).  Score
agreement was verified separately; ours differs from theirs only
by loss normalization (scale) and the tied wte/lm-head cross term.

Run: `./run.sh [gpu]` -> `results.jsonl`
