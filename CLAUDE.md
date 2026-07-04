# CLAUDE.md

This file orients coding agents working in this repository. It describes what the
library is for, how it is organized, and the conventions to follow. API-level
details are intentionally kept high-level, since specific signatures and internal
designs may change between versions—prefer reading the current source over relying
on the descriptions here when implementing against a concrete interface.

## Project Overview

`dattri_llm` is a library for **Training Data Attribution (TDA)** at LLM scale.
Given a trained (or training) model, it attributes model behavior back to individual
training examples by computing and comparing per-sample gradients.

The primary goal is **not** to maximize the number of supported TDA algorithms. Instead, the
library provides a **unified, efficient, and flexible infrastructure** for training
data attribution. Three principles drive the design:

1. **Efficiency** — operate in the *factorized gradient domain* and avoid
   materializing full weight gradients whenever possible. Gradient operations
   (dot products, K-FAC / second-order quantities) should adaptively choose between
   reconstructing full gradients and working directly on factorized factors,
   depending on which is cheaper.
2. **Compatibility** — integrate with popular training frameworks (Hugging Face
   Transformers, VerL, OLMo, and others) without forcing users to rewrite their
   training loop or change their training configuration. As long as the training
   procedure calls `.backward()`, the library can perform TDA—covering pretraining,
   SFT, and RL pipelines such as PPO/GRPO.
3. **Flexibility** — support multiple usage modes (a high-level attributor interface
   *or* simply wrapping a training context), and multiple downstream applications
   (data selection, token-level attribution, influence analyses).

## Repository Structure

The library is three layers; **imports flow strictly downward**:

```
dattri_llm/
├── utils/            # Bottom layer: generic, gradient-agnostic helpers
│   ├── hashing.py    #   content hashing of model inputs (sample identity)
│   └── distributed.py#   guarded torch.distributed helpers
├── gradient/         # Middle layer: the gradient system
│   ├── gradient.py   #   data model: Factorized, Gradient, GradientRecord
│   ├── ops/          #   factorized-gradient math (types, preprocess, materialize,
│   │                 #   dot, projection, kronecker) behind a re-exporting façade
│   ├── hooks/        #   capture: low-level hooks, HookManagerConfig, HookManager
│   ├── callbacks/    #   extension point: one module per callback
│   ├── file_manager.py #  on-disk storage/retrieval of gradient records
│   ├── datasets.py   #   file-level reading of stored records (block streaming)
│   └── streaming.py  #   GradientSource: DiskGradientSource + live GradientStreamer
└── attribution/      # Top layer: the TDA methods
    ├── arguments.py  #   AttributionArguments (HF TrainingArguments-style config)
    ├── base.py       #   attributor ABCs
    ├── score.py      #   AttributionScore result container
    ├── utils.py      #   shared scoring/orchestration helpers
    └── algorithm/    #   one module per attributor (tracin, kronecker, dvemb)
```

The top-level `dattri_llm/__init__.py` exports the advertised API. The capture
core (gradient layer) is importable with **torch alone**; the attribution layer
and streamer resolve lazily so optional dependencies (`dattri`, `transformers`,
`tqdm`) are only required when actually used. Preserve this when adding exports.

Roughly, the moving parts are:

- **Gradient** — a formal representation of gradients.
- **HookManager** — captures gradients during forward/backward via PyTorch hooks.
- **Callbacks** — pluggable interventions in the training loop (offloading,
  online data selection, etc.).
- **GradientFileManager** — storage/retrieval layer for offloaded gradients.
- **GradientSource** — the streaming contract attributors consume: per-step
  `(step, Gradient, hashes)` blocks, read from disk or computed live.
- **Attributors** — the actual TDA methods (TracIn, GradCos, KFAC/EKFAC, etc.).

## Core Components (high-level)

### Gradient

A common container for per-sample gradients in different formats. It carries both
gradient **contents** and **metadata** (layer names/types, indexing scheme, batch and
token dimensions, projection dimension, device/dtype, etc.).

Two representations are supported:

- **Factorized** — store the per-layer input activation and output pre-activation
  gradient separately, so the full weight gradient is never materialized. Memory
  efficient; the preferred default.
- **Materialized** — the explicit gradient tensor (outer product or raw `.grad`).

The class should support operations including: `materialize` (factorized → explicit),
`aggregate` and `slice` along batch/token dimensions, `select_layers`, `clone`,
`project` (random projection for dimensionality reduction), `concatenate` across the
batch dimension, and `similarity` (e.g. per-layer cosine). A `validate` method should
check internal consistency, and a `to(device)` method should move payloads.

The same indexing scheme also enables LLM attribution at multiple granularities
(per-sample, per-token-position, per-token); the factorized ("ghost") representation
is what makes fine-grained per-token-position attribution practical.

### HookManager

Hooks into any PyTorch model **without modifying the training loop**. Two key design
choices:

- Interventions happen only **after each forward/backward step**, not mid-step.
- Collected gradients are stored or processed via **Callbacks**, keeping the manager
  itself generic and the behavior pluggable.

Typical usage wraps training in a collection context:

```python
hookmanager = HookManager(model, callbacks=[...])
with hookmanager.collect():
    trainer.train()
```

Internally it buffers per-layer activations and gradient outputs, detects when a
full step has completed (all hooked layers fired across all replicas), assembles a
batch-level `Gradient`, and emits per-sample records to its callbacks. It exposes
`remove` (deregister hooks) and `pause` (temporarily suspend collection, e.g. during
a secondary validation backward). Which layers are hooked, and with which hook
family (per-sample factorized `linear_io` vs. batch-level `param_grad`), is decided
by `HookManagerConfig`; capture-time random projection is configured there too.

### Callbacks

Callbacks intervene at events such as layer forward, layer backward, and
collection-complete. Two representative implementations:

- **OffloadCallback** — accumulates gradient records and periodically offloads them
  via the `GradientFileManager` (with a guaranteed flush when the context closes).
- **DataSelectionCallback** — performs **online data selection**: scores each sample
  by gradient alignment with a target gradient and removes low-influence samples'
  contributions from `param.grad` before the optimizer step, as if they were never in
  the batch. Scoring can run in ghost (factorized) or materialized mode—both should
  produce identical scores; the target can be the current batch, a fixed precomputed
  gradient, or a fresh gradient drawn from a validation loader.

### GradientFileManager

The storage layer responsible for file naming, index management, and retrieval.
Samples are identified by a **content hash** of their model inputs (position- and
shuffling-independent; see `utils/hashing.py`), and the on-disk index maps
`input_hash → (step, sample_idx) → file`, so a per-sample gradient is retrieved by
direct slicing—never by scanning a batch record. Design goals: O(1) lookups via the
in-memory index (no directory scans), crash-safety via write-then-index ordering and
frequent index persistence, and rank-transparent loading under DDP (each rank writes
to its own subdirectory; a fresh manager merges all per-rank indexes).

### GradientSource (streaming)

Attributors consume gradients through one contract: a `GradientSource` yielding
per-step `(step, Gradient, hashes)` blocks. Two implementations:

- **DiskGradientSource** — re-iterable reads of pre-collected gradients
  (built on `gradient/datasets.py`, which loads one *file* per item so a standard
  DataLoader can prefetch whole batch blocks).
- **GradientStreamer** — computes blocks live from a forward+backward pass. Its
  `enable_update=False` mode is a frozen probe at a fixed checkpoint (re-iterable);
  `enable_update=True` advances the optimizer so each step is a real checkpoint
  along the training trajectory (single-shot). The updating mode deliberately
  mirrors the HF Trainer's inner loop (optimizer/scheduler construction, mixed
  precision, clipping, DDP/FSDP wrapping) so that under the same
  `AttributionArguments` the trajectory matches the Trainer bit-for-bit.

## Supported Layers

Factorized gradients are (or should be) supported across a broad range of layer types:
linear (`nn.Linear`, `nn.Bilinear`), convolutions (`Conv{1,2,3}d` and their transposes),
embeddings (`nn.Embedding`, `nn.EmbeddingBag`), and normalization layers (`LayerNorm`,
`RMSNorm`, `GroupNorm`, `InstanceNorm{1,2,3}d`).

## Parallelism

Attribution must work under distributed training. A key correctness invariant is that
**gradients computed with parallelism (especially FSDP) match those computed without
it**. Support priority:

```
DDP = FSDP  >  Megatron  >  DeepSpeed
```

Target integrations include Transformers (DDP, FSDP), VerL (FSDP), and OLMo
(DDP, FSDP2). Verification scripts comparing sharded vs. replicated vs. single-device
gradients are part of the expected workflow.

## Attributors

Methods expected to run on top of the HookManager infrastructure include TracIn and
GradCos (incl. online variants), the LoGRA family (KFAC, EKFAC), and DVEmb (with GGN
approximation). Attributors branch on one axis of the source contract —
`GradientSource.reusable` (methods with a pre-pass, e.g. K-FAC's Fisher fit, need a
re-iterable source); everything method-specific stays inside the attributor.

## Workflows

Two intended usage paths:

1. **On-the-fly attribution** — call an `Attributor` directly; it drives a
   `GradientStreamer` over the model and datasets, collecting and scoring in one
   run (`attribute` == *cache + attribute_from_cache*).
2. **Store-then-attribute** — wrap the model, run training while caching gradients
   (HookManager + OffloadCallback), then run an attributor over the stored
   gradients via `attribute_from_cache`.

## Development Notes

- **Try not to modify user training loops.** The whole point of the hook-based design is
  that TDA is added by wrapping a context, not by editing the loop or the trainer
  config. Preserve this property in any new code.
- **Respect the layering.** Runtime imports flow `utils → gradient → attribution`
  only. The gradient layer must stay importable with torch alone (soft-import
  `transformers`; keep `dattri`/`tqdm` out of its eager paths); type-only references
  upward go under `TYPE_CHECKING`.
- **Stay in the factorized domain by default.** Only materialize gradients when
  necessary (verification, or when it is genuinely cheaper). New gradient operations
  should offer a factorized path.
- **Parallelism correctness is a hard requirement.** Any change touching gradient
  capture should be checked against the no-parallelism reference for both DDP and FSDP.
- **Callbacks are the extension point.** Prefer adding behavior as a callback over
  baking it into the HookManager.
- **Keep storage crash-safe.** Maintain the write-then-index ordering and index
  persistence guarantees in the file manager.

## Evaluation / Benchmarks

The library is benchmarked on attribution quality (LDS, LOO, and TSLOO settings) across
attributors and training-epoch budgets, and on runtime against other libraries (e.g.
Dattri, logix/kronfluence). When adding or changing a method, validate against the
existing benchmark suite rather than relying on unit correctness alone.
