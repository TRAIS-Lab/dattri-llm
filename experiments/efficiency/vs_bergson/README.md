# Efficiency pair: dattri_llm vs bergson

**Setting** (bergson's README quickstart): EleutherAI/pythia-14m,
NeelNanda/pile-10k (10,000 documents, truncated to 2048 tokens), build an
on-disk index of per-document gradients sketched with a **rank-16 per-module
double-sided random projection** (16x16 = 256 dims/module, the 24
base-transformer Linear modules -- bergson's own module set, which excludes
``embed_out``), sum-CE loss, token-packed batches of <= 2048
tokens, fp32, single GPU.  Then score 16 query documents against the whole
index (plain grad-dot).

## Sides

* **bergson**: ``bergson build`` (their CLI, one subprocess -- peak GPU memory
  for this phase is sampled device-wide via nvidia-smi, coarser than torch's
  allocator counter), then a programmatic ``Attributor.trace`` query pass.
* **dattri_llm**: workflow-2 store-then-attribute -- ``HookManager`` +
  ``OffloadCallback`` collection with capture-time factor projection
  (``style=logra_factorized, proj_dim=16``, the same per-module sketch structure), then
  ``TracInAttributor.attribute_from_cache``.

## Fairness notes

* Same model, documents, loss, projection rank, module set, and token budget
  per batch (ours packs length-sorted docs to <= 2048 tokens/batch, bergson
  packs natively).
* Each side draws its own random sketch, so scores agree in rank, not value.
  Previously measured agreement (pearson 0.43) sits exactly at
  the rank-16 sketch-noise ceiling: rerunning **our own** pipeline with a
  different projection seed gives pearson 0.44 against itself (recorded as
  the ``sketch-noise ceiling`` row in ``results.jsonl``) -- the two libraries
  agree as much as the sketch dimension permits.
* Storage artifacts: bergson stores one 256-dim vector per (doc, module);
  ours stores per-token projected factors (a 16 + g 16 per token), which is
  what enables per-token attribution -- expect a larger store.

Run: `./run.sh [gpu]` -> `results.jsonl`
