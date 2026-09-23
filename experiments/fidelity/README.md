# Fidelity against cost

The fidelity experiment: how faithfully each attribution method predicts the
effect of removing a training example, and what it costs, on GPT-2 (124M),
Qwen2.5 at 0.5B, 1.5B and 3B, and OLMo-2-1B and OLMo-3-7B. GPT-2 and Qwen2.5
are trained by the protocol's own training loop; the OLMo models by
OLMo-core's trainer, with dattri-llm's capture wrapped around it.

## Protocol

**Trajectory.** The pretrained model is trained for one epoch on 512 random
blocks of 128 WikiText-2 tokens: 16 steps at batch 32, AdamW with betas
(0.9, 0.999), eps 1e-8, no weight decay, peak learning rate 1e-5 with linear
warmup over the first 10% of the steps and linear decay. Every parameter is
trained. The blocks are drawn with the model's own tokenizer, so the text of
a block differs between model families.

**Ground truth.** Trajectory-specific leave-one-out retraining (TSLOO): each
of 50 random training blocks is removed from its batch, the run is repeated,
and the change in loss on the validation blocks is recorded. Fidelity is the
Spearman correlation between a method's scores and those changes over the 50
blocks, computed per validation block and averaged over the 64 blocks every
method scores.

**Numerics.** Training and the reruns are fp32 with TF32 off, eager
attention, deterministic kernels, no dropout and no gradient clipping. For
GPT-2 and Qwen2.5 the pretrained weights are loaded once and the same model
is reset from a CPU copy for each of the 51 runs. The seed is 0.

**OLMo under OLMo-core.** The OLMo models are trained by OLMo-core's
`Trainer` (its transformer train module, numpy data loader and checkpoints)
under the same protocol: the released weights are converted once into an
OLMo-core checkpoint, the learning-rate schedule is passed as a scheduler
config, and a block is removed through the data loader's label mask. Each
run is its own process group, as the trainer holds one train module per
process. OLMo-3-7B trains on one GPU; OLMo-2-1B with FSDP over four GPUs,
where the loss of a process is the mean over its unmasked tokens and two
identical runs differ by about 1e-6 in validation loss (the leave-one-out
effect is about 3e-4). dattri-llm's methods capture around `trainer.fit()`
(AdamW-influence, with the optimizer's moments) or on the trained model
(EK-FAC); Bergson's take the trained model through its Hugging Face export.
GPT-2's position embedding is trained but not hooked (its gradient is shared
by the batch); under FSDP the AdamW-influence capture hooks the linear layers
of the transformer blocks.

**Methods.** dattri-llm: AdamW-influence with ten disjoint random masks of
512, 2048 or 8192 coordinates per layer (scored independently, averaged) and over
every coordinate (the trajectory is snapshotted and replayed); EK-FAC with a
rank-64 factor projection and at full dimension. Bergson 0.26.1: MAGIC,
SOURCE (four segments of four steps, two checkpoints each), TrackStar
(per-module projection of 64) and EK-FAC. Both EK-FACs damp with 0.1 of each
layer's mean eigenvalue. Each method has a run of its own.

**Batch sizes.** Every method captures the training gradients in batches of
8 blocks (for Bergson a token batch of 1024, i.e. 8 blocks of 128 tokens).
The test batch is, for each library, the largest power of two that fits the
budget, as in the throughput benchmark: the full-dimension methods score the
64 queries in chunks of that size (`EKFAC_CHUNK` and `BERGSON_CHUNK` in
`fidelity.py`), each chunk of dattri-llm's EK-FAC computed and preconditioned
once; from 3B on its factors are kept in host memory
(`--factor-cache-residency memory`).

**Budget and measurement.** One GPU, 16 CPU cores, 256 GB of host memory and
1 TB of local disk per run: a B200 for GPT-2, Qwen2.5-1.5B and OLMo-3-7B, an
H200 for Qwen2.5-0.5B and 3B. The OLMo-2-1B runs use four A40s. The reported
time is the attribution time of all 64 queries: training the trajectory and
loading the model and data are excluded. Memory is the peak GPU allocation
over the run. A run that does not complete within the budget is recorded as
infeasible: MAGIC and full-coordinate AdamW-influence at Qwen2.5-1.5B and
3B, and SOURCE at 3B. On OLMo-3-7B the methods that are infeasible at
Qwen2.5-1.5B or 3B (MAGIC, SOURCE, full-coordinate AdamW-influence, Bergson's
EK-FAC) are not run; SOURCE and MAGIC also train the trajectory themselves
and so cannot run under OLMo-core's trainer.

## Running

Results are written to `results/` next to the launcher.

```bash
export PYTHONPATH=/path/to/dattri-llm            # the repository root, so that dattri_llm is importable
python fidelity.py --experiment truth --dry-run   # list the runs
python fidelity.py --experiment truth --run       # the ground truth of GPT-2 and Qwen2.5
python fidelity.py --experiment attr --run        # every method against it
python fidelity.py --experiment olmo-truth --run  # the ground truth of the OLMo models (OLMo-core)
python fidelity.py --experiment olmo-attr --run   # the methods on the OLMo models
python fidelity.py --collect                      # the run directories -> results/fidelity.jsonl
```

Runs write to `results/<setting>/lr1e-05_seed0[_<run>]/` (`result.json`,
`matrices.pt`, `log.txt`) and are skipped when their `result.json` exists;
the OLMo training runs write to `$OLMOCORE_WORK_DIR/<scale>/` (default
`results/work/<scale>/`) and are skipped when their output exists. The
Bergson runs need `bergson==0.26.1`; the drivers refuse any other Bergson
version. The OLMo runs need `ai2-olmo-core` (2.6.0) and, for OLMo-2-1B,
`torchrun` with four GPUs; the launcher starts them with the right number of
processes.
