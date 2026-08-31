"""The universal task matrix + per-library coverage.

A *task* is (family, scale, dataset, method, parallelism).  Every library that
covers the method AND the parallelism runs the SAME task; the library may pick
any *strategy* (store-then-attribute vs on-the-fly, cached vs recompute) -- the
benchmark records the end-to-end TOTAL (runtime, peak GPU memory, peak disk), so
different strategies stay comparable on identical inputs.

Parallelism is a first-class axis, coupled to scale: a model runs on one card
while it fits (<=7b) and MUST be sharded with FSDP beyond that.  Small models may
also be run under DDP/FSDP deliberately -- that arm doubles as the parallelism
study (throughput + the hard correctness invariant that FSDP scores == single-
device scores, checked by correctness.py).  A library that lacks a parallelism is
simply dropped from those tasks, so e.g. a 32b+FSDP task excludes the single-
device-only backends (dattri) and DDP-only ones (logix).

    python tasks.py                 # print the grid + coverage matrix
"""

from __future__ import annotations

import models
from models import MODELS, resolve

# Attribution methods (library-agnostic names).
METHODS = ["graddot", "tracin", "gradcos", "kfac", "ekfac", "efim", "dvemb"]

# Which method each library implements on HF causal-LM models.
METHOD_COVERAGE: dict[str, set[str]] = {
    "dattri_llm": {"graddot", "tracin", "gradcos", "kfac", "ekfac", "efim", "dvemb"},
    "grass": {"efim", "graddot"},          # eFIM (dense Fisher) + Identity (raw)
    "logix": {"kfac", "graddot", "ekfac"}, # LoGRA raw/kfac/ekfac hessians
    "kronfluence": {"graddot", "kfac", "ekfac"},  # identity/kfac/ekfac strategies
    "bergson": {"kfac", "ekfac", "graddot"},
    "dattri": {"graddot", "gradcos", "tracin"},  # torch.func -- only feasible <=1b
}

# Which parallelism each library supports.  This is what makes libraries "pass"
# at scale: no-parallelism backends cannot run the sharded (FSDP) tasks.
#   single -- one device        ddp -- replicated        fsdp -- sharded
PARALLELISM_COVERAGE: dict[str, set[str]] = {
    "dattri_llm": {"single", "ddp", "fsdp"},   # DDP=FSDP is a hard requirement
    "kronfluence": {"single", "ddp", "fsdp"},  # designed for FSDP / large models
    "bergson": {"single", "fsdp"},             # distributed index config
    "grass": {"single", "fsdp"},               # Llama-3-8B distributed recipe
    "logix": {"single", "ddp"},                # DDP examples; no FSDP path
    "dattri": {"single"},                      # torch.func -- single device only
}

# ghostsuite is intentionally absent: its model is a bespoke nanoGPT (not HF), so
# it cannot run the model ladder -- it stays a separate online-selection arm.

DATASETS = ["wikitext103", "pile", "c4"]


def libs_for(method: str, parallelism: str, params_b: float) -> list[str]:
    """Libraries that can run (method, parallelism) at this scale.

    A library qualifies only if it covers BOTH the method and the parallelism;
    torch.func-only dattri is additionally dropped above ~1B as infeasible.
    """
    out = [
        lib
        for lib in METHOD_COVERAGE
        if method in METHOD_COVERAGE[lib]
        and parallelism in PARALLELISM_COVERAGE[lib]
    ]
    if params_b > 1.0 and "dattri" in out:
        out.remove("dattri")
    return out


def grid(families=("qwen", "pythia"), scales=("0.5b", "1b", "3b", "7b"),
         datasets=("wikitext103", "c4"), methods=tuple(METHODS),
         parallelisms=None):
    """Yield the task dicts of the benchmark grid.

    ``parallelisms=None`` uses each scale's *natural* parallelism (single while it
    fits, FSDP beyond) -- the default, comparable sweep.  Pass an explicit list
    (e.g. ``["single", "ddp", "fsdp"]``) to expand into the parallelism study; a
    strategy infeasible at a scale (e.g. ``single`` at 32b) is skipped.
    """
    for fam in families:
        for sc in scales:
            if sc not in MODELS[fam]:
                continue
            hf_id, params_b = resolve(fam, sc)
            # None -> the A40-based natural choice; an explicit list is trusted
            # as-is (the caller knows the target hardware, e.g. 32b single-GPU on
            # a 141 GB H200, which the A40 feasibility gate would forbid).
            pars = [models.natural_parallelism(params_b)] if parallelisms is None \
                else list(parallelisms)
            for par in pars:
                for ds in datasets:
                    for m in methods:
                        yield {
                            "family": fam, "scale": sc, "model": hf_id,
                            "params_b": params_b, "dataset": ds, "method": m,
                            "parallelism": par,
                            "n_gpus": models.n_gpus(params_b, par),
                            "libs": libs_for(m, par, params_b),
                        }


def main() -> None:
    print("=== model ladder (family x scale -> hf id, params) ===")
    for fam, scales in MODELS.items():
        row = "  ".join(f"{s}:{hf.split('/')[-1]}({p}B)" for s, (hf, p) in scales.items())
        print(f"  {fam:8s} {row}")
    print("\n=== coverage (library -> methods | parallelism) ===")
    for lib in METHOD_COVERAGE:
        print(f"  {lib:12s} {sorted(METHOD_COVERAGE[lib])}"
              f"  |  {sorted(PARALLELISM_COVERAGE[lib])}")
    print("\n=== parallelism coupled to scale (natural / feasible / gpus) ===")
    for sc in ("0.5b", "3b", "7b", "14b", "32b"):
        if sc not in MODELS["qwen"]:
            continue
        _, pb = resolve("qwen", sc)
        nat = models.natural_parallelism(pb)
        opts = models.parallelism_options(pb)
        gpus = {p: models.n_gpus(pb, p) for p in opts}
        print(f"  {sc:4s} ({pb:5.1f}B)  natural={nat:6s}  feasible={opts}  gpus={gpus}")
    print("\n=== default grid (natural parallelism) ===")
    tasks = list(grid())
    runs = sum(len(t["libs"]) for t in tasks)
    print(f"  {len(tasks)} tasks x their libraries = {runs} (task, library) runs")
    print("\n=== example: 32b + FSDP drops the no-FSDP backends ===")
    for t in grid(families=("qwen",), scales=("32b",), datasets=("pile",),
                  methods=("efim", "kfac", "ekfac", "graddot")):
        print(f"    {t['scale']} {t['parallelism']}({t['n_gpus']}gpu) "
              f"{t['dataset']:11s} {t['method']:7s} -> {t['libs']}")
    print("\n=== example: pythia-7b + pile + FSDP study ===")
    for t in grid(families=("pythia",), scales=("7b",), datasets=("pile",),
                  methods=("kfac",), parallelisms=["single", "ddp", "fsdp"]):
        print(f"    {t['scale']} {t['parallelism']:6s}({t['n_gpus']}gpu) "
              f"{t['dataset']:11s} {t['method']:7s} -> {t['libs']}")


if __name__ == "__main__":
    main()
