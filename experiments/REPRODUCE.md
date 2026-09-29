# Reproduction runbook

One page from each reported measurement to the command that produces it, the
hardware it was measured on, and the result files it ends in. The directory
READMEs ([`benchmark/`](benchmark/README.md), [`fidelity/`](fidelity/README.md),
[`capture/`](capture/README.md)) hold the full protocols; this page only
indexes them. Everything here is read from the launchers in this directory.

## Environment

```bash
pip install -e ".[transformers,attribution]"    # from the repository root
export PYTHONPATH=/path/to/dattri-llm            # the launchers import dattri_llm from the working tree
```

- Baselines are pinned in `benchmark/utils/versions.py` (Bergson 0.26.1,
  Kronfluence 1.0.1, LogIX 0.1.1); the adapters refuse any other version.
  LogIX needs Python < 3.11.
- The OLMo fidelity runs need OLMo-core: `pip install -e ".[olmo-core]"`
  (`ai2-olmo-core==2.6.0`).
- `benchmark/` and `capture/` read `BENCH_CACHE` (tokenized-block cache),
  `HF_HOME` (model weights) and `BERGSON_STORE` (Bergson's stores); see the
  benchmark README for the timed-region settings (local disk, warm page cache,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Every launcher takes `--dry-run` to list its cells or runs before `--run`.

## Measurements

| measurement | launcher and command | hardware (as the launcher states it) | result files |
|---|---|---|---|
| Time and memory of four libraries on one model, 1 and 16 queries, rank-64 and full dimension | `python benchmark/benchmark.py --experiment query1 --run`, then `--experiment query16` | one A40 | `benchmark/out/<experiment>/results.jsonl` |
| Scaling up the Qwen ladder, 0.5B to 32B | `python benchmark/scaling.py --experiment scaling-<method> --run`, `<method>` in `graddot`, `kfac`, `ekfac` | one H200 | `benchmark/out/<experiment>/results.jsonl` |
| Scaling at 72B and 110B (sharded) | `python benchmark/scaling.py --experiment scaling-<method>-fsdp4 --run` | four H200s | `benchmark/out/<experiment>/results.jsonl` |
| Throughput at each library's largest batch | `python benchmark/throughput.py --run 0.5b,1b,3b,7b,14b,32b,72b,110b` | four H200s (`THROUGHPUT_N_GPUS`) | `benchmark/out/throughput/<scale>/results.jsonl` |
| Routing: cost model against each pinned route, per step, over the sequence length | `python benchmark/routing.py --experiment routing-steps-b8 --run` (also `routing-steps-b{1,2,4,16}`) | one A40 | `benchmark/out/<experiment>/results.jsonl` |
| Routing: dattri-llm's three routes at batch 8 on to T = 8192 | `python benchmark/routing.py --experiment routing-steps-b8-long --run` | one H200 | `benchmark/out/<experiment>/results.jsonl` |
| Routing: fixed-token protocol | `python benchmark/routing.py --experiment routing --run` (also `routing-batch1`) | one A40 | `benchmark/out/<experiment>/results.jsonl` |
| Fidelity ground truth (leave-one-out), GPT-2 and Qwen2.5 | `python fidelity/fidelity.py --experiment truth --run` | one B200 (GPT-2, Qwen2.5-1.5B) or one H200 (Qwen2.5-0.5B, 3B) | `fidelity/results/<setting>/lr1e-05_seed0[_<run>]/` |
| Fidelity and cost of every method, GPT-2 and Qwen2.5 | `python fidelity/fidelity.py --experiment attr --run` | as for `truth` | as for `truth` |
| Fidelity ground truth, OLMo (OLMo-core trainer) | `python fidelity/fidelity.py --experiment olmo-truth --run` | one B200 (OLMo-3-7B); four A40s with FSDP (OLMo-2-1B) | `$OLMOCORE_WORK_DIR/<scale>/` (default `fidelity/results/work/<scale>/`) |
| Fidelity and cost of the methods, OLMo | `python fidelity/fidelity.py --experiment olmo-attr --run` | as for `olmo-truth` | `fidelity/results/<setting>/...` |
| Fidelity table | `python fidelity/fidelity.py --collect` | — | `fidelity/results/fidelity.jsonl` |
| Ordinary versus invasive capture | `python capture/capture.py --experiment capture-query16 --run` (also `capture-query1`, and `capture-shared-fit` if scores differ), or on Modal `modal run --detach capture/capture.py::bench --all` then `modal run capture/capture.py::fetch --all` | one A40 locally; one L40S (48 GB) on Modal, which offers no A40 | `capture/out/<experiment>/` locally, `capture/results/<experiment>/` from Modal; `python capture/capture.py --table` |

Commands are given from `experiments/`; each launcher writes its results next
to itself, whatever the working directory (`--out_dir` overrides it for the
benchmark launchers).

## Fidelity model ladder

The fidelity launcher covers six models (`SCALES` and `OLMO` in
`fidelity/fidelity.py`). The main ladder is GPT-2 (124M), Qwen2.5-1.5B and
OLMo-3-7B, the three settings budgeted on one B200. Qwen2.5-0.5B, Qwen2.5-3B
and OLMo-2-1B are **extended** scales that fill in the ladder; they run under
the same protocol, on the hardware listed above.

## Provenance of a result

- Every row written by the benchmark and capture launchers carries a `device`
  record (`benchmark/utils/log.py`): the GPU model (`device.gpu_name`, and
  `device.gpus` for every card), the CUDA, cuDNN, torch and transformers
  versions, the host's CPU count and RAM, and the installed version of every
  baseline library (`device.baseline_versions`). Check it against the
  hardware column above before comparing numbers.
- Each benchmark row also records its task (`task`), its per-phase times and
  memory (`phases`), and its status (`oom`, `timeout` or `error` for a failed
  cell). Throughput cells stop at `TIME_LIMIT_S = 3600` seconds
  (`benchmark/throughput.py`); Kronfluence scaling cells at 1800 seconds.
- Fidelity runs write `result.json`, `matrices.pt` and `log.txt` per run; a
  run that does not complete within its budget is recorded with a `status`.
- Token blocks are drawn with a fixed seed (`benchmark/utils/data.py`); the
  fidelity protocol uses seed 0 and learning rate 1e-5 (`LR`, `SEED` in
  `fidelity/fidelity.py`).
