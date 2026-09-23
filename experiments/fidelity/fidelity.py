"""Fidelity against cost: GPT-2, Qwen2.5 at three scales, OLMo at two.

A pretrained model is trained for one epoch on WikiText-2; every method
scores the same 64 validation blocks against that trajectory and is compared
with trajectory-specific leave-one-out retraining (TSLOO) by the Spearman
correlation.  ``utils/`` holds the setting (settings.py), the protocol
(protocol.py), the OLMo-core training and capture (olmocore.py) and the
helpers of the scripts; ``utils/truth/`` the ground-truth scripts and
``utils/attribution/`` one script per method.  GPT-2 and Qwen2.5 are trained by
the protocol's own loop; the OLMo models by OLMo-core's trainer, OLMo-2-1B
with FSDP over four GPUs and OLMo-3-7B on one.

    truth        the ground truth of GPT-2 and Qwen2.5: the reference run and
                 50 leave-one-out reruns
    attr         every method on GPT-2 and Qwen2.5, against that ground truth:
                 dattri-llm's AdamW-influence (masks of 512, 2048 and 8192
                 coordinates per layer, and every coordinate) and EK-FAC
                 (rank-64 projection, and full dimension); Bergson's MAGIC,
                 SOURCE, TrackStar and EK-FAC
    olmo-truth   the ground truth of the OLMo models under OLMo-core's trainer
    olmo-attr    the methods on the OLMo models: dattri-llm's AdamW-influence
                 and EK-FAC on the trained model, and Bergson's TrackStar and
                 EK-FAC on its Hugging Face export

    python fidelity.py --experiment truth --dry-run   # list the runs
    python fidelity.py --experiment truth --run       # run them in order
    python fidelity.py --experiment attr --run        # every method, after truth
    python fidelity.py --collect                      # the run directories -> results/fidelity.jsonl

Every run records its attribution time (the training of the trajectory and
the loading of the model and data excluded) and its peak GPU memory.  The
budget of a run is one GPU (a B200 for GPT-2, Qwen2.5-1.5B and OLMo-3-7B; an
H200 for Qwen2.5-0.5B and 3B), 16 CPU cores, 256 GB of host memory and 1 TB
of local disk; the OLMo-2-1B runs use four A40s.  A run that does not
complete within the budget is recorded as infeasible (``result.json`` with a
``status``).  Runs write to results/<setting>/lr1e-05_seed0[_<run>]/ and are
skipped when their result.json exists.  The Bergson runs need
``bergson==0.26.1``; the OLMo runs need ``ai2-olmo-core``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UTILS = HERE / "utils"
RESULTS = HERE / "results"
sys.path.insert(0, str(UTILS))

SCALES = ("gpt2", "0.5b", "1.5b", "3b")
OLMO = {"olmo2-1b": 4, "olmo3-7b": 1}  # scale -> processes (GPUs) of a training run
LR, SEED, BASE = "1e-5", "0", "lr1e-05_seed0"
MASKS = (512, 2048, 8192)  # coordinates per layer per mask of the masked AdamW-influence
# Every method captures the training gradients in batches of 8 blocks.  The
# test batch -- how many of the 64 queries a full-dimension method holds at
# once -- is, for each library, the largest power of two that fits the budget.
EKFAC_CHUNK = {"gpt2": "64", "0.5b": "64", "1.5b": "8", "3b": "8", "olmo2-1b": "16", "olmo3-7b": "4"}
BERGSON_CHUNK = {"gpt2": "16", "0.5b": "4", "1.5b": "1", "3b": "1", "olmo2-1b": "16"}
# Blocks per forward/backward of an OLMo-core training run (the step is the
# same sum of gradients); the optimizer-aware capture needs one per step.
OLMO_MICROBATCH = {"olmo2-1b": "8", "olmo3-7b": "32"}


def setting_name(scale: str) -> str:
    """The results sub-directory of a scale (as ``settings.setting_name``)."""
    return f"qwen{scale}" if scale in ("0.5b", "1.5b", "3b") else scale


def cell(scale: str, run: str, script: str, *args: str, nproc: int = 1, done: str | None = None) -> dict:
    """One run: its result directory (for skipping), its command line, the
    number of processes it is launched with, and the file that marks it done
    (default: the result.json of its result directory)."""
    return {"dir": f"{setting_name(scale)}/{BASE}{run}", "nproc": nproc, "done": done,
            "cmd": [script, "--scale", scale, "--lr", LR, "--seed", SEED, *args]}


def attribution(scale: str) -> list[dict]:
    truth = f"results/{setting_name(scale)}/{BASE}"
    queries = ["--n-queries", "64", "--truth-dir", truth]
    masked = [cell(scale, f"_masked{'' if k == 512 else k}", "attribution/adamw_influence.py", "--tag", f"_masked{'' if k == 512 else k}",
                   "--mask-dim", str(k), "--n-val", "64", "--tsloo-from", truth) for k in MASKS]
    return [
        *masked,
        cell(scale, "_full", "attribution/adamw_influence.py", "--tag", "_full", "--n-masks", "0", "--recompute",
             "--n-val", "64", "--val-batch-size", "2", "--tsloo-from", truth),
        cell(scale, "_ekfac_k64", "attribution/ekfac.py", "--projection", "64", *queries),
        # Full dimension: each chunk of queries is captured as one block and
        # preconditioned once; at 3B the factors are kept in host memory.
        cell(scale, "_ekfac", "attribution/ekfac.py", "--query-chunk", EKFAC_CHUNK[scale],
             "--eval-batch", EKFAC_CHUNK[scale], *(["--factor-cache-residency", "memory"] if scale == "3b" else []), *queries),
        cell(scale, "_magic", "attribution/magic.py", *queries),
        cell(scale, "_source", "attribution/source.py", "--query-chunk", BERGSON_CHUNK[scale], *queries),
        cell(scale, "_trackstar", "attribution/bergson.py", "--method", "trackstar", *queries),
        cell(scale, "_bergson_ekfac", "attribution/bergson.py", "--method", "ekfac",
             "--query-chunk", BERGSON_CHUNK[scale], *queries),
    ]


def olmo_work(scale: str) -> Path:
    return Path(os.environ.get("OLMOCORE_WORK_DIR", RESULTS / "work")) / scale


def selected_blocks(seed: int) -> list[int]:
    """The 50 blocks with a ground truth (as ``settings.build``)."""
    import torch

    gen = torch.Generator().manual_seed(seed + 7)
    return torch.randperm(512, generator=gen)[:50].tolist()


def olmo_truth(scale: str) -> list[dict]:
    """The token files and initial checkpoint, the reference run, one run per
    removed block (each its own process group), and the ground truth."""
    n, work = OLMO[scale], olmo_work(scale)

    def run(left_out: str, tag: str) -> dict:
        return cell(scale, "", "truth/olmo.py", "train", "--microbatch", OLMO_MICROBATCH[scale],
                    "--left-out", left_out, "--tag", tag, nproc=n, done=str(work / "runs" / tag / "run.pt"))

    return [
        cell(scale, "", "truth/olmo.py", "prepare", nproc=n, done=str(work / "init" / "model_and_optim")),
        run("none", "ref"),
        *[run(str(i), str(i)) for i in selected_blocks(int(SEED))],
        cell(scale, "", "truth/olmo.py", "truth"),
    ]


def olmo_attribution(scale: str) -> list[dict]:
    n, work = OLMO[scale], olmo_work(scale)
    truth = f"results/{scale}/{BASE}"
    final = str(work / "runs" / "final" / "final_hf")
    queries = ["--n-queries", "64", "--truth-dir", truth, "--final-model", final]
    mb = ["--microbatch", OLMO_MICROBATCH[scale]]
    # The optimizer-aware capture hooks every layer on one process; under
    # FSDP the linear layers of the transformer blocks.
    layers = ["--hook-layers", "all" if n == 1 else "blocks"]
    cells = [cell(scale, "", "attribution/olmo.py", "export", *mb, nproc=n, done=final)]
    for k in MASKS:
        tag = "" if k == 512 else str(k)
        mask = ["--mask-dim", str(k), *(["--mask-tag", tag] if tag else [])]
        cells += [
            cell(scale, f"_masked{tag}", "attribution/olmo.py", "adamw", *mb, *layers, *mask,
                 nproc=n, done=str(work / "runs" / f"adamw{tag}" / "capture.json")),
            cell(scale, f"_masked{tag}", "attribution/olmo.py", "adamw-score", *mask),
        ]
    cells += [
        cell(scale, "_ekfac_k64", "attribution/ekfac.py", "--projection", "64", *queries),
        cell(scale, "_ekfac", "attribution/ekfac.py", "--query-chunk", EKFAC_CHUNK[scale], "--eval-batch", EKFAC_CHUNK[scale],
             "--factor-cache-residency", "memory", *queries),
        cell(scale, "_trackstar", "attribution/bergson.py", "--method", "trackstar", *queries),
    ]
    if scale in BERGSON_CHUNK:
        cells.append(cell(scale, "_bergson_ekfac", "attribution/bergson.py", "--method", "ekfac",
                          "--query-chunk", BERGSON_CHUNK[scale], *queries))
    return cells


EXPERIMENTS = {
    "truth": lambda: [cell(scale, "", "truth/tsloo.py") for scale in SCALES],
    "attr": lambda: [c for scale in SCALES for c in attribution(scale)],
    "olmo-truth": lambda: [c for scale in OLMO for c in olmo_truth(scale)],
    "olmo-attr": lambda: [c for scale in OLMO for c in olmo_attribution(scale)],
}


def command(c: dict) -> list[str]:
    script = str(UTILS / c["cmd"][0])
    if c["nproc"] > 1:
        return ["torchrun", "--standalone", f"--nproc-per-node={c['nproc']}", script, *c["cmd"][1:]]
    return [sys.executable, "-u", script, *c["cmd"][1:]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--run", action="store_true", help="execute the runs in order")
    ap.add_argument("--dry-run", action="store_true", help="list the runs only")
    ap.add_argument("--collect", action="store_true", help="run directories -> results/fidelity.jsonl")
    a = ap.parse_args()
    if a.collect:
        import collect

        path = RESULTS / "fidelity.jsonl"
        rows = collect.collect(RESULTS)
        collect.write_rows(rows, path)
        print(f"{len(rows)} rows -> {path}")
        return
    if a.experiment is None:
        ap.error("--experiment is required (or --collect)")
    cells = EXPERIMENTS[a.experiment]()
    for c in cells:
        c["done_path"] = Path(c["done"]) if c["done"] else RESULTS / c["dir"] / "result.json"
    for i, c in enumerate(cells):
        launcher = f"torchrun x{c['nproc']}" if c["nproc"] > 1 else "python"
        print(f"{i:3d}  {'done' if c['done_path'].exists() else '    '}  {c['dir']:32} "
              f"{launcher} utils/{' '.join(c['cmd'])}")
    if not a.run:
        print("\n(add --run to execute; runs whose marker exists are skipped)")
        return
    for c in cells:
        if c["done_path"].exists():
            continue
        print(f"\n##### {c['dir']}: {' '.join(c['cmd'][:2])}", flush=True)
        tail: collections.deque = collections.deque(maxlen=40)
        proc = subprocess.Popen(command(c), cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        result = RESULTS / c["dir"] / "result.json"
        if proc.wait() != 0 and not c["done_path"].exists():
            if c["done"] is not None:  # a step of a multi-run cell: stop, the later steps depend on it
                raise SystemExit(f"{c['dir']}: {' '.join(c['cmd'])} failed")
            # record the run as not completed and continue with the next one
            text = "".join(tail)
            status = "oom" if "OutOfMemoryError" in text or "out of memory" in text.lower() else "failed"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"status": status, "cmd": c["cmd"], "error": text[-3000:]}, indent=2))
            print(f"##### {c['dir']}: {status}", flush=True)


if __name__ == "__main__":
    main()
