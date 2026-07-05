# Examples

Example scripts demonstrating each part of `dattri-llm`, from
layer selection to full attribution workflows and training-framework integration.

```bash
python examples/<group>/<script>.py
```

All examples run on CPU in seconds to a few minutes; `hooks/multi_gpu_collect.py`
additionally supports multi-process (and multi-GPU) launches via `torchrun`.

## Overview

| Example | Shows | Extra requirements |
|---|---|---|
| [`hooks/modular_hooks.py`](hooks/modular_hooks.py) | selecting which layers to hook, and how | — |
| [`hooks/multi_gpu_collect.py`](hooks/multi_gpu_collect.py) | per-sample collection under DDP/FSDP | `transformers` |
| [`attribution/attribution_from_disk.py`](attribution/attribution_from_disk.py) | store-then-attribute workflow | — |
| [`attribution/attribution_on_the_fly.py`](attribution/attribution_on_the_fly.py) | one-call live attribution | `dattri`, `transformers` |
| [`data_selection/gpt2_data_selection.py`](data_selection/gpt2_data_selection.py) | online data selection on GPT-2 | `transformers` |
| [`trainers/transformers_trainer.py`](trainers/transformers_trainer.py) | wrapping the Hugging Face `Trainer` | `transformers`, `accelerate` |
| [`trainers/olmo_trainer.py`](trainers/olmo_trainer.py) | wrapping the OLMo `Trainer` | `ai2-olmo` |

("—" means the capture core's only dependency, `torch`, suffices.)

---

## hooks/

### `modular_hooks.py` — configuring which layers are captured, and how

Two hook families exist: `linear_io` registers factorized **per-sample**
hooks on linear-family layers, while `param_grad` registers materialized
`param.grad` hooks (batch-level) on any trainable layer. This example builds a
small model with embedding / MLP / norm / head sub-blocks and prints the resulting
hook assignment under five `HookManagerConfig` styles:

1. explicit per-layer assignment (`hook_types={...}`),
2. regex selectors (`linear_io=[r"mlp\."]`),
3. `REGISTER_ALL` (every `linear_io`-capable layer),
4. the zero-argument default (`linear_io` where possible, `param_grad` fallback),
5. mixing explicit assignment with regex selectors.

```bash
python examples/hooks/modular_hooks.py
```

### `multi_gpu_collect.py` — per-sample collection across ranks (DDP / FSDP)

Streams one frozen pass of per-sample gradients with
`GradientStreamer` under `torchrun`. Each rank collects its own
`DistributedSampler` shard; because rows are keyed by **content hash**, the shards
recombine into the full dataset with no duplicates. The *unwrapped* model is
passed in — the streamer registers the hooks, places the model, and applies the
DDP/FSDP wrapping itself (uncomment `fsdp="full_shard"` in the script to collect
under FSDP on GPUs instead).

```bash
torchrun --nproc_per_node=2 examples/hooks/multi_gpu_collect.py --cpu   # gloo, no GPU needed
torchrun --nproc_per_node=2 examples/hooks/multi_gpu_collect.py         # NCCL on GPUs
python examples/hooks/multi_gpu_collect.py                              # single-process fallback
```

---

## attribution/

The two attribution examples run TracIn on the same toy MLP and data, one per
workflow — and produce **identical score matrices**, demonstrating that the live
and cached paths are interchangeable.

### `attribution_from_disk.py` — store-then-attribute (workflow 2)

Stage 1 collects per-sample gradients to disk: a `HookManager` with factorized
(`linear_io`) hooks captures one full-batch backward (a **sum** loss, so each captured gradient is
that sample's own `dL_i/dW`) and an `OffloadCallback` persists per-sample records
via `GradientFileManager`. Stage 2 attributes from the cache alone —
`TracInAttributor.attribute_from_cache(train_dir, test_dir)` needs no model and no
backward pass, so it can be re-run with different settings (layer subsets,
`normalized_grad=True` for GradCos) for free. Rows/columns are keyed by content
hash and realigned to sample order with `score.query(...)`.

```bash
python examples/attribution/attribution_from_disk.py
```

### `attribution_on_the_fly.py` — one-call live attribution (workflow 1)

The attribution target is described with a `dattri` `AttributionTask`
(functorch-style loss + checkpoint list); `TracInAttributor.attribute(train_ds,
test_ds)` then streams the gradients live and scores them in one call — nothing is
persisted. The full `(num_train, num_test)` matrix is read back with
`score.agnostic_matrix()`.

```bash
python examples/attribution/attribution_on_the_fly.py
```

---

## data_selection/

### `gpt2_data_selection.py` — online data selection on GPT-2 (124M)

A validation gradient over NLP/ML prose is the selection target. One **batched**
forward+backward over a deliberately mixed-domain training batch (NLP/ML, code,
French, math, cooking) lets `DataSelectionCallback` score every sample by the
**ghost inner product** with that target — computed directly from the factorized
gradients — and drop the bottom fraction, exactly what it would do inside a real
training loop before the optimizer step.

```bash
python examples/data_selection/gpt2_data_selection.py                       # ~60 s on CPU
python examples/data_selection/gpt2_data_selection.py --drop_fraction 0.5
```

Downloads the `gpt2` checkpoint from the Hugging Face Hub on first run.

---

## trainers/

Both trainer examples make the same point: **the training loop is never
modified** — TDA is added by wrapping the trainer's fit/train call in a
`HookManager` collection context.

### `transformers_trainer.py` — Hugging Face `Trainer`

Trains a tiny GPT-2 (`sshleifer/tiny-gpt2`) for two epochs with real
`trainer.train()` calls, collecting batch-level gradients through two equivalent
integration patterns:

- **Pattern A (with-context, preferred):** `with collector.collect(): trainer.train()`
- **Pattern B (TrainerCallback):** open/close the context from
  `on_train_begin`/`on_train_end`, for pipelines where `trainer.train()` is buried
  inside a library you don't control.

It then demonstrates hash-based retrieval: `hash_sample(dataset[0])` identifies
*what* the sample is (independent of shuffling), `lookup_by_hash` reveals *where*
it was recorded (every `(step, sample_idx)` pair across epochs), and
`load_sample_by_hash` slices its gradient straight out of the stored record. The
cosine similarity between the first- and last-epoch gradients of the same sample
shows its gradient drifting as the model trains.

```bash
python examples/trainers/transformers_trainer.py             # 2 epochs
python examples/trainers/transformers_trainer.py --epochs 3
```

### `olmo_trainer.py` — OLMo `Trainer`

Builds a tiny OLMo model entirely in Python (no config YAML), trains it with the
**real** OLMo `Trainer` on CPU, and collects gradients from the feed-forward
projections (`transformer.blocks.<i>.ff_proj` / `ff_out`) selected by regex. Hooks
go on the *unwrapped* model, so they see the real `nn.Linear` modules regardless
of the DDP wrapping OLMo applies. Retrieval works exactly as in the Transformers
example: hash → `(step, sample_idx)` pairs → per-sample gradient slices → drift
between first and last occurrence.

```bash
pip install ai2-olmo
python examples/trainers/olmo_trainer.py
```