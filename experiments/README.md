# Experiments

| directory | paper |
|---|---|
| `benchmark/` | efficiency: the cross-library tables (one and sixteen queries) and the scaling figure |
| `fidelity/` | fidelity of optimizer-aware attribution against leave-one-out retraining |

Each is self-contained, with one launcher and a README. This tree holds the
code only. Runs happen in `experiments_exe/` at the repository root, a copy
of this tree that also holds every result, cache and figure and is not
tracked:

```bash
rsync -a experiments/ experiments_exe/      # refresh the code there
cd experiments_exe/benchmark && python benchmark.py --experiment query1 --run
```
