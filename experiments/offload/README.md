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

Nine points, all at `proj_dim=64` (`--proj-dim`) except the projection axis
itself.  Every one differs from the `factorized/pickle/proj=64` baseline along
exactly one axis, so any pair including the baseline isolates that axis.

`proj_dim=64` is the knee of the `--proj-sweep` curve for this model: 128->64
still buys 1.63x walltime, 64->32 only 1.07x, so below it you trade sketch
fidelity for almost no speed.  d=256 here, so it is d/4 -- 64*64 = 4096
dims/layer, the rank LoGRA and logix use.  **It does not transfer to another
model**; re-run `--proj-sweep` when the hidden size changes.

Pass `--proj-dim raw` when comparing serializers: at the default the write is
small enough that format differences shrink toward the noise floor.

* **`proj_dim`** (raw, 64, 16) -- capture-time random projection, i.e. how many
  bytes reach the store at all.
* **`disk_format`** (pickle, memmap) -- serializer cost per byte; pickle holds
  the GIL, the memmap write releases it.
* **representation** (factorized, materialized) -- materialized expands the
  outer product, ~2.8x the bytes.
* **`offload_to_cpu`** (False, True) -- whether the device->host copy lands in
  `to_cpu` or in the capture hooks.  Note this is **not** the compute device;
  the config label is `o2c=`, and the device is reported separately.
* **`offload_interval`** (1, 8, 32) -- stall frequency vs stall size.
* **`residency`** (disk, tiered) -- tiered pins a zero budget so every group is
  evicted and `spill` is exercised at any payload size.  A fixed budget stops
  firing once projection shrinks the store under it.

Only linear layers are hooked (`blocks.N.{attn,proj,fc_in,fc_out}`, `head`):
`logra_factorized` is undefined for the embedding and LayerNorm, and leaving
them out keeps every config's layer set identical.

## Reading the output

Four columns answer "did it get faster, and what is the limit":

| column | meaning |
|---|---|
| `wall` | total training walltime |
| `offload` | seconds inside `save_bulk` (the five phases summed) |
| `share` | `offload / wall` -- **the compute/disk split** |
| `compute` | `wall - offload` -- everything that is not the store |
| `async` | ceiling for async write-behind, `(C+W)/max(C,W)` |
| `MB/s` | `store / write_group` -- achieved write throughput |

`share` *is* the split: at 97% the GPU sat idle 97% of the time waiting on the
store, so compute is the remaining 3%.  Reading it:

* **`share` > 90%** -- storage-bound.  Only writing fewer bytes helps
  (`proj_dim`); serializer and threading changes cannot move a saturated disk.
* **`share` 20-60%** -- mixed.  Both fewer bytes and faster writes pay, and
  there is now enough compute for async write-behind to hide writes behind.
* **`share` < 20%** -- compute-bound.  The store is no longer the problem.

`MB/s` says *why*.  When every config reports roughly the same number
regardless of format or representation, storage bandwidth is the ceiling and
that number is the disk.  When configs differ, the serializer is still in play.
A number far above the device's rated speed means the writes stayed in page
cache -- see the RAM warning below.

`async` is the decision column for a writer thread.  Overlapping writes with
compute leaves `max(C, W)`, so the best a thread can do is `(C+W)/max(C,W)` --
2x when the two are balanced, ~1x at either extreme.  A saturated disk cannot
be made faster by overlapping it; there is only `C` seconds of compute to hide
`W` seconds of writing behind.

To decide whether a change helped, compare `wall` between two configs that
differ on one axis, and check the `x` column first: a gap smaller than `x` is
noise.  `store` shows whether the win came from writing less or writing faster.

## Finding the projection setting

`--proj-sweep` varies only `proj_dim` (raw, 512 ... 8) with everything else at
the baseline:

```
 proj  store MB  compute   write  W share   async   wall
  raw     310.0    0.290   0.566    66.1%   1.51x   0.86
  512     357.2    1.554   0.578    27.1%   1.37x   2.13
  256     178.7    1.032   0.264    20.3%   1.26x   1.30
  128      89.5    0.863   0.149    14.7%   1.17x   1.01
   64      44.8    0.541   0.078    12.5%   1.14x   0.62
   32      22.5    0.529   0.047     8.1%   1.09x   0.58
   16      11.4    0.503   0.028     5.3%   1.06x   0.53
    8       5.8    0.464   0.021     4.3%   1.05x   0.48
```

Two different questions, two different columns:

* **Lowest `wall`** is the throughput answer.  Store scales linearly with
  `proj_dim`, but projection is itself compute, so the curve is U-shaped:
  `proj=512` writes *more* than raw (357 vs 310 MB) **and** pays 5x the compute,
  ending 2.5x slower than not projecting.  Below ~64 the gains flatten.
* **Highest `async`** is the writer-thread answer, and it points the other way:
  raw has the best ceiling here because its 66% write share is nearest the 50/50
  balance point.  Projecting improves walltime while *reducing* what async could
  add.

The crossover is model- and disk-specific.  This model is d=256, so 512 is
wider than most of its layers and is pure overhead; on a d=4096 LLM the same
setting is an 8x reduction.  Re-run the ladder per model and per cluster --
absolute `proj_dim` values do not transfer.

## Notes

* **CPU numbers do not carry to GPU.**  `to_cpu` reads 0 here for lack of a
  device->host copy; on GPU with `offload_to_cpu=False` the whole flush window
  crosses PCIe there out of pageable memory (~6 GB/s vs ~26 GB/s pinned).  It
  also over-reports: the copy synchronizes inside an autograd hook, so in-flight
  backward kernels are charged to it.  Read it as an upper bound.
* **`to_cpu` ~0 under `offload_to_cpu=True` is not a free copy** -- it happened
  per-layer in the capture hooks, which are uninstrumented.  Compare total
  walltime instead.  On CPU that axis is a no-op entirely.
* **`--store-dir` selects the filesystem.**  The default is `TMPDIR`, which is
  tmpfs on many clusters -- that measures RAM, not disk.  The header line
  reports the resolved mount and fs type, and warns on a RAM-backed mount.
* **Watch the page-cache warning.**  If the largest store fits inside available
  RAM, writes may never reach the platter and `write_group` is measuring cache.
  Scale up until the store clears RAM.
* The store is held on device until the flush, so a large `offload_interval`
  with a large batch can OOM before writing.  Scale GPU runs up (`--steps 200
  --batch-size 16 --seq-len 256`); the defaults are laptop-sized.
* Single-device only: per-rank layout and DDP/FSDP flush behaviour are
  unmeasured.  Laptop variance is ~20% on walltime -- ratios are stable,
  absolutes are not.
* A throwaway config runs before the recorded sweep, absorbing one-time
  process costs (filesystem cache, allocator, `torch.save` code paths) that
  would otherwise all land on config #1 and penalise it ~12%.

## Baseline -- CPU only (2026-07-27, medians of 3)

```
config                                          as    wall  offload  share     x   write_group     spill    store
factorized/pickle/proj=raw/o2c=0/int=8/disk     pt    1.11    0.699  62.7%  1.11        0.6968    0.0000   457.6M
factorized/memmap/proj=raw/o2c=0/int=8/disk   mmap    0.52    0.097  18.8%  1.07        0.0955    0.0000   457.4M
materialized/pickle/proj=raw/o2c=0/int=8/disk   pt    2.76    2.003  72.7%  1.04        1.9997    0.0000  1302.2M
factorized/pickle/proj=64/o2c=0/int=8/disk      pt    1.09    0.113  10.4%  1.06        0.1117    0.0000    66.2M
factorized/pickle/proj=16/o2c=0/int=8/disk      pt    1.03    0.044   4.3%  1.07        0.0424    0.0000    16.8M
factorized/pickle/proj=raw/o2c=1/int=8/disk     pt    1.15    0.690  60.0%  1.03        0.6878    0.0000   457.6M
factorized/pickle/proj=raw/o2c=0/int=1/disk     pt    1.15    0.674  58.6%  1.10        0.6667    0.0000   457.6M
factorized/pickle/proj=raw/o2c=0/int=32/disk    pt    1.19    0.719  60.6%  1.07        0.7180    0.0000   457.6M
factorized/pickle/proj=raw/o2c=0/int=8/tiered   pt    1.23    0.711  57.7%  1.11        0.0003    0.7101   457.6M
```

* **Projection is the dominant lever.**  `proj=16` cuts the store 27x (457.6 ->
  16.8 MB) and offload from 62.7% of walltime to 4.3% -- a bigger effect than
  any serializer choice, because it removes the bytes instead of writing them
  faster.  `proj=64` sits between at 6.9x.
* **memmap is 7.3x faster than pickle** on the same factorized payload (0.697 ->
  0.096 s), taking offload from 62.7% to 18.8%.  Store size is unchanged
  (457.6 vs 457.4 MB): the format changes speed, not bytes.
* **The two levers are independent** and untested in combination; a projected
  memmap store should compound, but no config measures it.
* **`offload_interval` barely matters** (0.667 / 0.697 / 0.718 s for 1 / 8 / 32).
  Total bytes are identical and the disk absorbs them either way.
* **Tiered spends 58% of walltime spilling**, all in `spill`.  That path still
  uses `torch.save`, so it has not benefited from memmap.
* `index_update` and `index_write` are 0.0002-0.007 s throughout -- well below
  the noise floor.

**On a disk-bound machine these ratios collapse.**  A Colab run measured
40-160 MB/s with 1.7x variance between identical configs; there the serializer
is nearly irrelevant and only `proj_dim` moves the needle.  Always check the
`x` column before comparing anything.

Run: `python run_offload.py [--quick] [--list] [--device cuda:0]`
-> `results.jsonl`
