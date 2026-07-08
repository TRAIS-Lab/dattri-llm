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
