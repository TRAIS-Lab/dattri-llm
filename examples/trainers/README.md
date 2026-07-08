# trainers/

The trainer examples all make the same point: **the training loop is never
modified** — TDA is added by wrapping the trainer's fit/train call in a
`HookManager` collection context.

## `transformers_trainer.py` — Hugging Face `Trainer`

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

## `trl_trainer.py` — TRL `SFTTrainer`

Fine-tunes a tiny GPT-2 with TRL's `SFTTrainer` on **raw text** — TRL does its
own tokenization, collation, and label masking, and the wrapped
`trainer.train()` captures per-sample gradients below all of it. Retrieval
works by the content hash of the model inputs TRL actually produced: the same
sample lands at different `(step, sample_idx)` positions across shuffled
epochs, and the hash ties its occurrences together.

Three TRL-specific settings in the script are deliberate:

- **gradient checkpointing is disabled** — TRL enables it by default, and its
  non-reentrant recomputation runs each block forward twice with grad enabled,
  which would double-capture activations;
- **`lm_head` is not hooked** — TRL's SFT loss applies the tied output weight
  functionally (fused linear + cross-entropy), so the module's hooks would
  never fire and step completion would stall;
- **one fixed-length batch per epoch** — TRL feeds the batch-dependent
  `num_items_in_batch` count into the model forward, so a sample's content
  hash only stays epoch-stable when its batch context is stable.

```bash
pip install trl
python examples/trainers/trl_trainer.py
```

## `olmo_trainer.py` — OLMo `Trainer`

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
