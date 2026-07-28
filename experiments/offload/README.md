# Offload write-path baseline

Unlike `experiments/efficiency`, this is not a head-to-head against another
library -- it measures our own store against itself, so later work on the write
path (async write-behind, pinned staging, memmap for factorized groups) has a
before/after number instead of an argument.  Self-contained: no dependency on
the `efficiency` harness.

**Setting**: a transformer-shaped stack defined in `run_offload.py` (4 blocks x
4 Linears + embedding + LayerNorm + head, d=256, vocab 512 -- no `transformers`
dependency), 30 steps of batch 4 x seq 64, SGD, fp32, single device, one
discarded warm-up step.  Each config trains under `HookManager` +
`OffloadCallback` into a fresh store, then reports the per-phase breakdown from
`GradientStorageManager.timing`, walltime, offload's share of it, bytes
written, and which serializer actually ran (`as`).  Stores are deleted between
configs, so disk use is peak, not cumulative.

## Configurations

Eight points, varying one axis at a time from `factorized/pickle/cpu=0/int=8/disk`:

* **`disk_format`** (pickle, memmap) -- pickle holds the GIL, the memmap write
  releases it; decides whether a writer thread can overlap.
* **representation** (factorized, materialized) -- factorized is the library
  default; materialized expands the outer product, so it is ~2.8x larger on
  disk.  Both take the memmap path.
* **`offload_to_cpu`** (False, True) -- moves the device->host copy between
  `to_cpu` and the capture hooks.
* **`offload_interval`** (1, 8, 32) -- stall frequency vs stall size.
* **`residency`** (disk, tiered) -- tiered pins a 64 MiB budget so groups are
  evicted and the `spill` phase is exercised.

## Notes

* **CPU numbers do not carry to GPU.**  `to_cpu` reads 0 here for lack of a
  device->host copy; on GPU with `offload_to_cpu=False` the whole flush window
  crosses PCIe there out of pageable memory (~6 GB/s vs ~26 GB/s pinned).  It
  also over-reports: the copy synchronizes inside an autograd hook, so in-flight
  backward kernels are charged to it.  Read it as an upper bound.
* **`to_cpu` ~0 under `offload_to_cpu=True` is not a free copy** -- it happened
  per-layer in the capture hooks, which are uninstrumented.  Compare total
  walltime instead.  On CPU that axis is a no-op entirely.
* The store is held on device until the flush, so a large `offload_interval`
  with a large batch can OOM before writing.  Scale GPU runs up (`--steps 200
  --batch-size 16 --seq-len 256`); the defaults are laptop-sized.
* Single-device only: per-rank layout and DDP/FSDP flush behaviour are
  unmeasured.  Laptop variance is ~20% on walltime -- ratios are stable,
  absolutes are not.
* A throwaway config runs before the recorded sweep, absorbing one-time
  process costs (filesystem cache, allocator, `torch.save` code paths) that
  would otherwise all land on config #1 and penalise it ~12%.

## Baseline -- CPU only (2026-07-27)

```
config                                  as    wall  offload  share   write_group  index_write     spill    store
factorized/pickle/cpu=0/int=8/disk      pt    1.22    0.814  66.5%        0.8119       0.0020    0.0000   480.9M
factorized/memmap/cpu=0/int=8/disk    mmap    0.59    0.128  21.6%        0.1260       0.0013    0.0000   480.7M
materialized/pickle/cpu=0/int=8/disk    pt    2.89    2.183  75.6%        2.1801       0.0023    0.0000  1364.4M
materialized/memmap/cpu=0/int=8/disk  mmap    1.06    0.323  30.5%        0.3202       0.0024    0.0000  1364.3M
factorized/pickle/cpu=1/int=8/disk      pt    1.14    0.688  60.6%        0.6854       0.0020    0.0000   480.9M
factorized/pickle/cpu=0/int=1/disk      pt    1.27    0.734  57.7%        0.7265       0.0067    0.0000   481.0M
factorized/pickle/cpu=0/int=32/disk     pt    1.38    0.872  63.1%        0.8715       0.0007    0.0000   480.9M
factorized/pickle/cpu=0/int=8/tiered    pt    1.28    0.787  61.5%        0.0004       0.0000    0.7860   480.9M
```

* **`write_group` dominates the pickle configs**: offloading is 58-76% of
  walltime there and essentially all of it is the serializer.  `index_write` at
  0.0007-0.007 s puts the append-only index and write-once meta sidecar well
  below the noise floor.
* **memmap is ~6.8x faster than pickle on factorized `write_group`.**  Median of
  5 runs each: 0.748 s -> 0.111 s, with memmap far tighter run-to-run
  (0.095-0.115 vs 0.707-0.835) since a memcpy is more predictable than pickling.
  Offload drops from 66% of walltime to 22%, and total training time roughly
  halves (1.22 s -> 0.59 s).  This is the library's *default* representation, so
  it applies to an ordinary capture, not a special case.
* **Store size is unchanged by format** (480.9 M vs 480.7 M): memmap changes
  speed, not bytes.  The 2.8x gap between factorized and materialized is the
  outer-product expansion, independent of serializer.
* **Tiered spends 61% of walltime spilling**, all in the `spill` phase that
  previously ran outside the timing blocks; the same run used to report ~0.1%
  offload overhead.  Note spill still uses `torch.save`, so it has not yet
  benefited from memmap.

Run: `python run_offload.py [--quick] [--list] [--device cuda:0]`
-> `results.jsonl`
