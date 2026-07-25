# Efficiency pair: dattri_llm vs kronfluence

**Setting** (kronfluence's `examples/wikitext`): GPT-2 fine-tuned 3 epochs on
wikitext-2 by *their* `train.py` (one shared checkpoint, `./checkpoints`),
512-token blocks, EKFAC influence in the **full parameter space** over their
48 tracked modules (all attention + MLP linears, Conv1D replaced by Linear by
their own pipeline).  fp32, single GPU, their README batch sizes (train 64,
query 32).

Queries: the first **32 validation blocks** -- exactly one of their documented
`--query_batch_size 32` batches.  (Their full-flagship run scores the whole
validation split; a single query batch keeps the pair runnable on an A40
while giving both sides an identical workload.)  Train side: full train split.

## Sides

* **kronfluence(ekfac)**: `fit_all_factors` (covariance pass + eigendecomp +
  lambda pass, persisted to disk) then `compute_pairwise_scores`.
* **dattri_llm(ekfac,otf)**: one `EKFACAttributor.attribute` call -- K-FAC
  fit pass, Lambda fit pass, scoring pass, nothing persisted.

Pipeline shape matches: both stream the train set three times and neither
stores per-sample gradients.  Same estimator, same damping (their default
1e-8), same loss (sum-CE, shifted), so the two sides are numerically
comparable directly (up to each side's numerics).

Run: `python ../run.py kronfluence` -> `results.jsonl`
