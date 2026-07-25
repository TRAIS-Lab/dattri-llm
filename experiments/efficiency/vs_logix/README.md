# Efficiency pair: dattri_llm vs logix (LoGRA)

**Setting** (logix's `examples/language_modeling` defaults): GPT-2 (124M,
Conv1D layers replaced with Linear by logix's own utils), wikitext-103-raw-v1,
0.5% train sample grouped into 512-token blocks -> **1164 train blocks**,
first **16 test-split blocks** as queries.  All attention + MLP linears hooked
(48 layers), sum-CE loss, fp32 + TF32 matmuls, batch 8 (train) / 1 (query),
single GPU.

Projected gradient space is matched: logix uses rank-64 LoRA (64x64 = 4096
dims/layer); ours uses capture-time factor projection ``proj_dim=64`` per side
(same 4096 dims/layer), on the same 48 layers.

## Sides

* **logix(raw)** (their README flagship): extract stage logs projected
  per-sample gradients to disk + accumulates the dense 4096^2/layer gradient
  covariance; score stage computes queries' projected grads and
  ``compute_influence_all`` preconditions with the raw covariance.
* **logix(kfac)**: same pipeline, K-FAC covariance factors instead of the
  dense covariance (estimator-parity with our side).
* **dattri_llm disk** (matching pipeline shape): ``KFACAttributor.cache``
  writes projected factorized gradients to disk; ``attribute_from_cache``
  fits K-FAC in the projected space and scores.
* **dattri_llm otf**: single ``attribute`` call, nothing persisted (streams
  the train set twice: Fisher fit + scoring).

## Fairness notes

* Identical model/data/layers/loss/batch sizes; both sketch to 4096
  dims/layer at rank 64 (different random sketches, so agreement is checked
  by rank correlation, not equality).
* Storage artifacts differ by design: logix stores token-summed per-sample
  projected gradients (~0.9 GB); ours stores **per-token** projected factors
  (~14 GB), which is what enables per-token attribution downstream.  This is
  the main driver of ours-disk's extra cache/score time.
* torch.load in ``run_logix.py`` falls back to ``weights_only=False``
  (logix's saved state holds defaultdicts; files are this run's own output).

Run: `./run.sh [gpu]` -> `results.jsonl`
