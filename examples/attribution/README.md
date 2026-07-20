# attribution/

The two attribution examples run TracIn on the same toy MLP and data, one per
workflow — and produce **identical score matrices**, demonstrating that the live
and cached paths are interchangeable.

## `attribution_from_disk.py` — store-then-attribute (workflow 2)

Stage 1 collects per-sample gradients to disk: a `HookManager` with factorized
(`linear_io`) hooks captures one full-batch backward (a **sum** loss, so each captured gradient is
that sample's own `dL_i/dW`) and an `OffloadCallback` persists per-sample records
via `GradientStorageManager`. Stage 2 attributes from the cache alone —
`TracInAttributor.attribute_from_cache(train_dir, test_dir)` needs no model and no
backward pass, so it can be re-run with different settings (layer subsets,
`normalized_grad=True` for GradCos) for free. Rows/columns are keyed by content
hash and realigned to sample order with `score.query(...)`.

```bash
python examples/attribution/attribution_from_disk.py
```

## `attribution_on_the_fly.py` — one-call live attribution (workflow 1)

The attribution target is described with a `dattri` `AttributionTask`
(functorch-style loss + checkpoint list); `TracInAttributor.attribute(train_ds,
test_ds)` then streams the gradients live and scores them in one call — nothing is
persisted. The full `(num_train, num_test)` matrix is read back with
`score.agnostic_matrix()`.

```bash
python examples/attribution/attribution_on_the_fly.py
```
