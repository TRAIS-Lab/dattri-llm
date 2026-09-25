"""This example shows gradient collection around TRL's GRPOTrainer.

The training loop is NOT modified -- TDA is added by wrapping trainer.train()
in a HookManager collection context, exactly as with the SFT and plain HF
trainers.  Reinforcement learning adds passes that do not train the policy:
rollout generation, and the reward (and, with a KL penalty, reference) model
forwards.  They run without gradients, so the hooks skip them and each
captured step is one policy update.

Each training sample is a prompt together with one sampled completion.  A
sample is identified by the content hash of the model inputs the trainer fed
the policy, so the completions GRPO draws for the same prompt are distinct
samples with their own gradients.  To read the records, the script pairs each
stored gradient with its prompt, completion, and reward: a callback reads, at
the end of every captured step, the model inputs the manager hashed into the
step's sample identities, and the reward function records the reward of every
sequence it scores.

Run (with trl installed):
    python examples/trainers/trl_grpo_trainer.py
"""

from __future__ import annotations

import operator
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:
    raise SystemExit(
        "trl (and datasets) are required for this example.\n"
        "Install with:  pip install trl",
    ) from exc

from dattri_llm.gradient.callbacks import HookManagerCallback, OffloadCallback
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.storage_manager import GradientStorageManager

MODEL_ID = "sshleifer/tiny-gpt2"  # 2-layer GPT-2, runs on CPU
PROMPTS = [
    "The weather today is",
    "My favourite food is",
    "Once upon a time",
    "The capital of France is",
]


class SequenceLog(HookManagerCallback):
    """Map each captured sample's content hash to its unpadded token ids.

    At the end of a step the manager still holds the model inputs it hashed
    into the record's sample identities (``_last_inputs``, overwritten by the
    next forward).  The policy update passes the prompt and completion tokens
    as ``input_ids``, with ``attention_mask`` marking the padding.
    """

    def __init__(self) -> None:
        self.manager: HookManager | None = None
        self.tokens: dict[str, tuple[int, ...]] = {}

    def on_step_end(self, record) -> None:  # noqa: ANN001
        inputs = self.manager._last_inputs  # noqa: SLF001
        ids, mask = inputs["input_ids"], inputs["attention_mask"].bool()
        for h, row, m in zip(record.input_hash, ids, mask, strict=True):
            self.tokens[h] = tuple(row[m].tolist())


if __name__ == "__main__":
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    dataset = Dataset.from_dict({"prompt": PROMPTS * 2})

    # Every scored sequence (prompt tokens followed by completion tokens) ->
    # (prompt, completion text, reward).
    scored: dict[tuple[int, ...], tuple[str, str, float]] = {}

    def reward_length(
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        **_kwargs: object,
    ) -> list[float]:
        """Toy reward: the character length of each completion."""
        rewards = [float(len(c)) for c in completions]
        for prompt, text, ids, r in zip(
            prompts, completions, completion_ids, rewards, strict=True
        ):
            seq = tuple(tokenizer(prompt)["input_ids"]) + tuple(ids)
            scored[seq] = (prompt, text, r)
        return rewards

    # Hook the linear and norm layers of the transformer blocks.
    hook_cfg = HookManagerConfig(linear_io=[r"transformer\.h\."])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Two completions per prompt, so a batch of four holds two groups.
        grpo_config = GRPOConfig(
            output_dir=tmpdir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            num_generations=2,
            max_completion_length=8,
            # TRL turns gradient checkpointing on by default; capture handles
            # it, it is off here only to keep the tiny run fast.
            gradient_checkpointing=False,
            use_cpu=True,
            logging_steps=100,
            save_strategy="no",
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_length,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        fm = GradientStorageManager(str(pathlib.Path(tmpdir) / "gradients"))
        log = SequenceLog()
        collector = HookManager(
            model,
            config=hook_cfg,
            callbacks=[
                OffloadCallback(
                    offload_interval=1,
                    file_manager=fm,
                    recording_type="per_batch",
                ),
                log,
            ],
        )
        log.manager = collector

        # the RL loop itself is untouched: wrap trainer.train() and every
        # policy-update backward pass inside is captured automatically
        print("\nRunning GRPOTrainer.train() with gradient collection ...")
        with collector.collect():
            trainer.train()
        collector.remove()

        # Pair every stored gradient with its prompt, completion, and reward.
        # GRPO scales each completion's log-likelihood gradient by its reward
        # relative to the other completions of the same prompt: a positive
        # relative reward makes the update raise that completion's
        # likelihood, a negative one lowers it.
        paired = sum(log.tokens.get(h) in scored for h in fm.index)
        print(f"\n{'Policy updates':<22}{trainer.state.global_step}")
        print(f"{'Steps collected':<22}{collector.steps_collected}")
        print(f"{'Sample records':<22}{len(fm.index)}")
        print(f"{'Records with reward':<22}{paired}")
        rows = []
        for h in fm.index:
            for step, idx in fm.lookup_by_hash(h):
                if step != 0:
                    continue
                prompt, text, reward = scored[log.tokens[h]]
                g = fm.load_sample_by_hash(h, step, idx)
                norm = g.similarity(g, metric="dot", reduce="all").sqrt().item()
                rows.append((prompt, text, reward, norm))
        group_mean = {
            p: sum(r[2] for r in rows if r[0] == p) / sum(r[0] == p for r in rows)
            for p, *_ in rows
        }
        rows.sort(key=operator.itemgetter(0, 2))
        print(f"\nFirst update ({len(rows)} samples):")
        print(f"{'prompt':<26}{'completion':<36}{'reward':>7}{'rel.':>7}{'|grad|':>11}")
        for prompt, text, reward, norm in rows:
            shown = repr(text.replace(chr(10), " ")[:32])
            rel = reward - group_mean[prompt]
            print(f"{prompt:<26}{shown:<36}{reward:>7.1f}{rel:>+7.1f}{norm:>11.3e}")
