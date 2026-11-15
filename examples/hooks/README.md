# hooks/

## `modular_hooks.py` — configuring which layers are captured, and how

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

## `multi_gpu_collect.py` — per-sample collection across ranks (DDP / FSDP)

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

## `moe_capture.py` — Mixture-of-Experts blocks

An MoE block hands its router the block input flattened to `(B*T, d)` and each
expert only the `(n_e, d)` rows routed to it, so these layers do not carry the
batch axis. Without a declaration the first backward fails and names the layer;
with `HookManager(model, moe_routed_layers=[...])` naming the MoE blocks, every
row is located in the block input and returned to its sample and position. The
example runs a tiny Mixtral, prints the records of one block (each expert at
the full `(B, T, d)` layout, zero where a token was not routed to it), checks
every expert and router gradient against one-sample autograd, and scores the
batch with the experts included. The same script covers both layouts:
per-expert `nn.Linear` experts (Transformers 4) and fused 3-D expert parameters
(Transformers 5), for which parameter-free per-expert proxies are installed
while collecting.

```bash
python examples/hooks/moe_capture.py
```

## `sequence_mixer_capture.py` — state-space and linear-attention mixers

A Mamba mixer applies some of its weights functionally (`conv1d`, `dt_proj`) or
through a fused kernel, so module hooks would never see them. The manager
recognises these mixers and, while collecting, routes their forward through
the modules. The example shows that nothing needs to be declared, that the
model's output is unchanged, which layers are captured per sample and which at
batch level (`A_log`, `D`), and that every projection matches one-sample
autograd. The same mechanism serves the Mamba-2 family and its hybrids, Jamba,
and the gated delta-net mixers of Qwen3-Next and Kimi-Linear.

```bash
python examples/hooks/sequence_mixer_capture.py
```
