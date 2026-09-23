# Experiments

This directory contains experiments evaluating the efficiency and attribution fidelity of `dattri-LLM`. The experiments cover runtime, memory usage, and scalability across attribution libraries, as well as attribution accuracy against leave-one-out retraining.

| directory | measures |
|---|---|
| `benchmark/` | efficiency of four attribution libraries: time and memory on one model, scaling up a model ladder, throughput at the largest batch, cost against the sequence length |
| `fidelity/` | fidelity of attribution methods against leave-one-out retraining, and their cost |
| `capture/` | dattri-llm's ordinary versus invasive gradient capture: time saved and score agreement, on the workload of `benchmark/` |
