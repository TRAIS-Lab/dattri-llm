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
| [`projection/gradient_projection.py`](projection/gradient_projection.py) | random projection (LoGRA / TRAK), per-layer config, custom layers | `dattri` |
| [`attribution/attribution_from_disk.py`](attribution/attribution_from_disk.py) | store-then-attribute workflow | — |
| [`attribution/attribution_on_the_fly.py`](attribution/attribution_on_the_fly.py) | one-call live attribution | `dattri`, `transformers` |
| [`data_selection/gpt2_data_selection.py`](data_selection/gpt2_data_selection.py) | online data selection on GPT-2 | `transformers` |
| [`trainers/transformers_trainer.py`](trainers/transformers_trainer.py) | wrapping the Hugging Face `Trainer` | `transformers`, `accelerate` |
| [`trainers/trl_trainer.py`](trainers/trl_trainer.py) | wrapping TRL's `SFTTrainer` | `trl` |
| [`trainers/olmo_trainer.py`](trainers/olmo_trainer.py) | wrapping the OLMo `Trainer` | `ai2-olmo` |

("—" means the capture core's only dependency, `torch`, suffices.)

Each subfolder has its own README describing its examples in detail:

- [`hooks/`](hooks/README.md) — hook configuration and multi-GPU collection
- [`projection/`](projection/README.md) — random projection of per-sample gradients
- [`attribution/`](attribution/README.md) — the two attribution workflows
- [`data_selection/`](data_selection/README.md) — online data selection
- [`trainers/`](trainers/README.md) — Transformers, TRL, and OLMo integration
