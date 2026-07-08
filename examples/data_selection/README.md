# data_selection/

## `gpt2_data_selection.py` — online data selection on GPT-2 (124M)

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
