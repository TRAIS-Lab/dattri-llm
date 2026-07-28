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
  default and `_group_memmappable` rejects it, so only the materialized configs
  reach the memmap path.
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
* **The first config in a sweep is penalised ~12%.**  The per-config warm-up
  step does not cover one-time process costs (filesystem cache, allocator,
  `torch.save` code paths); running one config four times gives write_group
  0.805 / 0.731 / 0.693 / 0.719.  Only compare configs that are adjacent, or
  whose gap is far larger than that.  Fixing this needs a throwaway config
  before the sweep.

## Baseline -- CPU only (2026-07-27)

```
config                                  as    wall  offload  share   write_group  index_write     spill    store
factorized/pickle/cpu=0/int=8/disk      pt    1.73    1.052  60.8%        1.0493       0.0019    0.0000   480.9M
factorized/memmap/cpu=0/int=8/disk      pt    1.26    0.800  63.2%        0.7977       0.0016    0.0000   480.9M
materialized/pickle/cpu=0/int=8/disk    pt    3.37    2.581  76.6%        2.5765       0.0044    0.0000  1364.4M
materialized/memmap/cpu=0/int=8/disk  mmap    1.29    0.529  40.9%        0.5266       0.0025    0.0000  1364.3M
factorized/pickle/cpu=1/int=8/disk      pt    1.42    0.821  57.9%        0.8189       0.0020    0.0000   480.9M
factorized/pickle/cpu=0/int=1/disk      pt    1.79    1.094  61.1%        1.0857       0.0075    0.0000   481.0M
factorized/pickle/cpu=0/int=32/disk     pt    1.70    0.999  58.9%        0.9979       0.0007    0.0000   480.9M
factorized/pickle/cpu=0/int=8/tiered    pt    1.34    0.834  62.4%        0.0003       0.0000    0.8339   480.9M
```

* **`write_group` is the whole problem**: offloading is 58-77% of walltime and
  essentially all of it is the serializer.  `index_write` at 0.0007-0.008 s
  confirms the append-only index and write-once meta sidecar are no longer a
  factor.
* **`factorized/memmap` writes as `pt`.**  Both factorized rows did identical
  work, so their 0.80-vs-1.05 gap is the first-config penalty above, not the
  format -- the `as` column is the only reliable signal here.  Where memmap is
  genuinely reachable it is **4.9x faster** on `write_group` (2.58 s ->
  0.53 s, adjacent configs) -- the gap extending it to factorized groups is
  chasing.
* **Tiered spends 62% of walltime spilling**, all in the `spill` phase that
  previously ran outside the timing blocks; the same run used to report ~0.1%
  offload overhead.

Run: `python run_offload.py [--quick] [--list] [--device cuda:0]`
-> `results.jsonl`
