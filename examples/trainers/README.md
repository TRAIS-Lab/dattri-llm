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
epochs, and the hash ties its occurrences together. The hash is taken over a
sample's unpadded tokens, so it does not depend on the batch the sample was
padded with, and batch-level inputs such as TRL's `num_items_in_batch` count
do not enter it.

Two things about the hook selection are worth knowing:

- **`lm_head` is not hooked** — TRL's SFT loss applies the tied output weight
  functionally (fused linear + cross-entropy), so the module is never invoked
  and has no captured gradient; a selected layer that does not run in a step
  is simply not part of that step's record;
- **`wpe` is not hooked** — GPT-2's position embedding takes one position
  tensor shared by the batch, so its gradient is not per-sample.

TRL's default gradient checkpointing (either `use_reentrant` variant) is
supported: the recomputed forward is matched to its backward and each step is
captured once. The script keeps it off only to keep the tiny run fast.

```bash
pip install trl
python examples/trainers/trl_trainer.py
```

## `trl_grpo_trainer.py` — TRL `GRPOTrainer`

Trains a tiny GPT-2 policy with TRL's `GRPOTrainer` against a toy reward
function, with the same wrapped `trainer.train()` call. Rollout generation and
the reward-model pass run without gradients, so the hooks skip them and each
captured step is one policy update. Every sample is a prompt together with one
sampled completion, identified by the content hash of the policy's inputs, so
the completions drawn for the same prompt are separate samples. After
training, the script pairs every stored gradient with its prompt, completion,
and reward: a callback reads, at the end of each captured step, the model
inputs the manager hashed into that step's sample identities, and the reward
function records the reward of each sequence it scores. It prints the samples
of the first update with their reward relative to the other completions of
the same prompt, which is the weight GRPO puts on each completion's
log-likelihood gradient.

TRL trains with bf16 autocast by default, so the captured gradients carry
bf16 rounding, as the parameter gradients of the same step do.

```bash
pip install trl
python examples/trainers/trl_grpo_trainer.py
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
