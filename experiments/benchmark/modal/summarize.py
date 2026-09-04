"""Read results.jsonl and print the GradDot / K-FAC / EK-FAC comparison.

    python summarize.py /results/scaling-pythia-A40
    python summarize.py /results/*                        # several at once
    python summarize.py /results/scaling-families-H200 --by-family=kfac
    python summarize.py /results/* --memory                # panel (b)
    python summarize.py /results/* --memory --methods=graddot,kfac

One row per (family, scale), one column group per method, showing the
``attribute`` phase only -- ``build_model`` and ``load_data`` are separate
phases and are not part of attribution cost.  Reports wall-clock, seconds per
training sample, and peak allocated GPU memory (the README is explicit that the
paper reports allocated, not reserved).

It also refuses to compare rows quietly across a confound.  Every table checks
that its rows share one GPU and one dtype, because both have already produced
misleading curves here: the size-based dtype rule put the 0.5B rung of every
scaling ladder in fp32 and the rest in bf16, and Modal has no A40 so the `-A40`
experiment runs on an L40S.  A mixed table is still printed -- with a MIXED
banner naming what differs -- rather than suppressed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

METHOD_ORDER = ["graddot", "tracin", "gradcos", "kfac", "ekfac", "efim", "dvemb"]


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        f = Path(p)
        f = f / "results.jsonl" if f.is_dir() else f
        if not f.exists():
            print(f"  (no results.jsonl under {p})", file=sys.stderr)
            continue
        rows += [json.loads(x) for x in f.open() if x.strip()]
    return rows


def dtype_provenance(recs: list[dict]) -> None:
    """Warn when a row's dtype came from the task rather than from the adapter.

    `task["dtype"]` is what the experiment ASKED for; a top-level `dtype` is
    what the adapter actually built. When an adapter ignores the override the
    two diverge silently and every row still reads the same, so a mixed-dtype
    check on the task field sees nothing wrong. That is exactly how a
    cross-library ladder ended up comparing fp32 against bf16 at three of four
    scales, undetected.
    """
    unstated = sorted({r["lib"] for r in recs if not r.get("dtype")})
    if unstated:
        print(f"  !! {', '.join(unstated)}: no adapter-reported dtype -- the row "
              "shows what the experiment asked for, not what ran")
    built = {}
    for r in recs:
        if r.get("dtype"):
            built.setdefault(r["lib"], set()).add(r["dtype"])
    if len({d for ds in built.values() for d in ds}) > 1:
        print(f"  !! libraries did not share one dtype: "
              f"{ {k: sorted(v) for k, v in built.items()} }")


# Phases that are setup, not attribution.  Every other phase a library records
# is attribution work, and the names differ per library because each was written
# against its own pipeline:
#     dattri_llm   attribute            (single)   /  cache + score  (FSDP)
#     bergson      fit + score
#     kronfluence  fit_factors + pairwise_scores
# Reading one hard-coded name understated kronfluence to ~0 s and would have
# drawn its curve along the axis.  Summing the non-setup phases is the only rule
# that is correct for all three, and it leaves single-device dattri_llm rows
# unchanged (their sole non-setup phase IS `attribute`).
SETUP_PHASES = frozenset({"build_model", "load_data"})


def attribution_phases(rec: dict) -> list[dict]:
    return [p for p in rec.get("phases", []) if p["phase"] not in SETUP_PHASES]


def attribution_time(rec: dict) -> float | None:
    ps = attribution_phases(rec)
    return sum(p["wall_s"] for p in ps) if ps else None


def attribution_units(rec: dict) -> int | None:
    for p in attribution_phases(rec):
        if p.get("work_units"):
            return p["work_units"]
    return None


def attribution_peak(rec: dict) -> float:
    return max((g["alloc_gb"] for p in attribution_phases(rec)
                for g in p.get("gpu_peak", [])), default=0.0)


def mem_basis(rec: dict) -> str:
    """How this row's GPU peak was measured.

    Not every library can be instrumented the same way.  Adapters that run the
    model in-process report `torch.cuda.max_memory_allocated` -- allocator bytes
    only.  bergson runs through its own CLI in a SUBPROCESS, where those
    counters see nothing, so its adapter polls `nvidia-smi memory.used`, which
    is the whole process: allocator bytes PLUS reserved-but-unallocated PLUS the
    CUDA context.  On one measured row reserved alone was 1.14x allocated before
    context, so comparing smi-used against allocated overstates bergson by
    roughly 20%.  The tell is structural: BenchRun emits `reserved_gb`, the
    nvidia-smi path cannot.
    """
    for p in attribution_phases(rec):
        for g in p.get("gpu_peak", []):
            return "torch_allocated" if "reserved_gb" in g else "nvidia_smi_used"
    return "unknown"


def comparable_peak(rec: dict) -> float:
    """Peak on the SAME basis for every library: whole-process device memory.

    For torch-instrumented rows that means `reserved_gb`, the closest analogue
    to what nvidia-smi reports; for bergson it is the smi reading as recorded.
    A residual bias remains -- reserved still excludes the CUDA context that
    smi-used includes, a few hundred MB -- and it favours the torch-measured
    libraries, so this is the conservative direction for a claim that ours uses
    less.  Use `run_peak` for single-library panels, where the paper's
    allocated-bytes convention applies and nothing is being compared across
    measurement methods.
    """
    key = "reserved_gb" if mem_basis(rec) == "torch_allocated" else "alloc_gb"
    return max((g.get(key, g["alloc_gb"]) for p in attribution_phases(rec)
                for g in p.get("gpu_peak", [])), default=0.0)


def peak_alloc(phase: dict) -> float:
    """Max allocated GB across visible devices -- under FSDP each rank holds a
    shard, so the per-device peak is what a card must supply."""
    return max((g["alloc_gb"] for g in phase.get("gpu_peak", [])), default=0.0)


def by_family(rows: list[dict], method: str) -> None:
    """Overlay the families on one parameter axis for a single method.

    This is the panel-(b) view.  It is a separate mode because a cross-family
    comparison is only meaningful with the device held fixed, and the per-family
    tables above cannot enforce that -- `scaling-pythia-A40` and
    `scaling-qwen-H200` differ in family AND device at once.  Run
    `scaling-families-H200`, which puts both ladders on one card, or pass two
    result dirs from the same GPU.
    """
    sel = [r for r in rows if r["task"]["method"] == method and attribution_time(r)]
    if not sel:
        print(f"\nno rows for method={method!r}")
        return
    gpus = {r["device"]["gpu_name"] for r in sel}
    dtypes = {r.get("dtype") or r["task"].get("dtype") or "?" for r in sel}
    fams = sorted({r["task"]["family"] for r in sel})

    print(f"\n=== families x scale, method={method} "
          f"[{', '.join(sorted(gpus))} | {', '.join(sorted(dtypes))}] ===")
    if len(gpus) > 1:
        print(f"  !! MIXED DEVICE {sorted(gpus)} -- family and device vary "
              "together, so this is not a family comparison")
    if len(dtypes) > 1:
        print(f"  !! MIXED DTYPE {sorted(dtypes)}")
    if len(fams) < 2:
        print(f"  (only one family present: {fams[0]}; nothing to overlay)")

    print(f"  {'family':>7} {'scale':>6} {'params':>8} {'wall_s':>8} "
          f"{'s/smp':>8} {'peakGB':>8}")
    recs = []
    for r in sel:
        t = r["task"]
        n = attribution_units(r)
        wall = attribution_time(r)
        recs.append((t["params_b"], t["family"], t["scale"], wall,
                     (wall / n) if n else None, attribution_peak(r)))
    for pb, fam, sc, wall, sps, gb in sorted(recs):
        sps_s = f"{sps:.3f}" if sps else "-"
        print(f"  {fam:>7} {sc:>6} {pb:>7.2f}B {wall:>8.1f} {sps_s:>8} {gb:>8.2f}")


def run_peak(rec: dict) -> float:
    """Peak allocated GB over EVERY phase -- what the card must supply.

    Not the attribution phase alone: the FSDP adapter tears the model down
    before scoring and runs `attribute_from_cache` against a
    `torch.nn.Linear(1, 1)` stand-in, so its `score` phase peaks at almost
    nothing while the real per-shard peak sits in `cache`.  Reading one named
    phase would draw the sharded points near zero.  Taking the max over all
    phases is both correct for FSDP and unchanged for single-device rows, where
    `attribute` dominates `build_model` in all 27 measured cells.
    """
    return max((g["alloc_gb"] for p in rec.get("phases", [])
                for g in p.get("gpu_peak", [])), default=0.0)


def memory_curve(rows: list[dict], methods: set[str] | None = None) -> None:
    """Panel (b): peak per-device memory against model scale, per family.

    Two rules this enforces, because both are easy to get wrong:

    1. **Single-device and FSDP peaks are different quantities** and are never
       merged into one series.  A single-device peak is what one card must
       supply for the whole model; an FSDP peak is what one card supplies for
       its *shard*.  Plotting a 4-way-sharded 32B point as the continuation of
       a single-device curve would show memory falling as the model grows.

    2. **Peak is taken as the max across methods**, not one method's value.
       The methods do different amounts of work -- K-FAC and EK-FAC carry
       factors GradDot does not -- so the honest answer to "what must this card
       supply" is the worst case over what was run.  The method that set the
       peak is named so the choice stays visible.
    """
    for par in ("single", "fsdp"):
        sel = [r for r in rows if r["task"]["parallelism"] == par]
        if methods:
            sel = [r for r in sel if r["task"]["method"] in methods]
        if not sel:
            continue
        gpus = {r["device"]["gpu_name"] for r in sel}
        dtypes = {r.get("dtype") or r["task"].get("dtype") or "?" for r in sel}
        label = ("peak one card must supply" if par == "single"
                 else "peak PER SHARD -- not comparable to the single-device curve")
        print(f"\n=== peak memory, {par}  [{', '.join(sorted(gpus))} | "
              f"{', '.join(sorted(dtypes))}] ===")
        print(f"  {label}")
        if len(dtypes) > 1:
            print(f"  !! MIXED DTYPE {sorted(dtypes)} -- a precision change moves "
                  "peak memory and will read as a scaling effect")

        print(f"  methods: {', '.join(sorted({r['task']['method'] for r in sel}))}")
        best: dict[tuple, tuple] = {}
        for r in sel:
            t = r["task"]
            key = (t["family"], t["scale"], t["params_b"], t.get("n_gpus", 1))
            gb = run_peak(r)
            if key not in best or gb > best[key][0]:
                best[key] = (gb, t["method"])
        print(f"  {'family':>7} {'scale':>6} {'params':>8} {'gpus':>5} "
              f"{'peakGB':>8}  set by")
        for (fam, sc, pb, ng), (gb, m) in sorted(best.items(),
                                                 key=lambda kv: (kv[0][0], kv[0][2])):
            print(f"  {fam:>7} {sc:>6} {pb:>7.2f}B {ng:>5} {gb:>8.2f}  {m}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    family_mode = next((f.split("=", 1)[1] if "=" in f else "graddot"
                        for f in flags if f.startswith("--by-family")), None)
    rows = load(args or ["out"])
    if not rows:
        print("no rows found")
        return
    dtype_provenance(rows)
    if family_mode:
        by_family(rows, family_mode)
        return
    if any(f.startswith("--memory") for f in flags):
        sel = next((f.split("=", 1)[1] for f in flags if f.startswith("--methods=")), None)
        memory_curve(rows, set(sel.split(",")) if sel else None)
        return

    # `lib` is part of the key.  Without it a cross-library file collapses every
    # library into one grid and each cell keeps whichever row was written last,
    # so a table can read as one coherent curve while its points come from
    # different libraries.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        t = r["task"]
        groups[(t["family"], t.get("proj_mode", "?"), t["parallelism"],
                r["lib"])].append(r)

    for (family, proj, par, lib), recs in sorted(groups.items()):
        gpus = {r["device"]["gpu_name"] for r in recs}
        dtypes = {r.get("dtype") or r["task"].get("dtype") or "?" for r in recs}
        methods = [m for m in METHOD_ORDER
                   if m in {r["task"]["method"] for r in recs}]

        print(f"\n=== {lib}  {family}  proj={proj}  {par} "
              f"[{', '.join(sorted(gpus))} | {', '.join(sorted(dtypes))}] ===")
        if len(gpus) > 1 or len(dtypes) > 1:
            print("  !! MIXED: rows below are NOT directly comparable")
            if len(gpus) > 1:
                print(f"     GPUs  : {sorted(gpus)}")
            if len(dtypes) > 1:
                print(f"     dtypes: {sorted(dtypes)}  <- a dtype change reads as "
                      "a scaling effect; see README")

        cells: dict[tuple, dict] = {}
        scales: dict[str, float] = {}
        for r in recs:
            t = r["task"]
            wall = attribution_time(r)
            if wall is None:
                continue
            scales[t["scale"]] = t["params_b"]
            n = attribution_units(r)
            cells[t["scale"], t["method"]] = {
                "s": wall,
                "sps": (wall / n) if n else None,
                "gb": attribution_peak(r),
                "n": n,
                "disk": r.get("disk_total_gb", 0.0),
            }

        head = f"  {'scale':>6} {'params':>7} " + "".join(
            f"| {m:^24} " for m in methods)
        print(head)
        print(f"  {'':>6} {'':>7} " + "".join(
            f"| {'wall_s':>7} {'s/smp':>7} {'GB':>6} " for _ in methods))
        for sc, pb in sorted(scales.items(), key=lambda kv: kv[1]):
            line = f"  {sc:>6} {pb:>6.2f}B "
            for m in methods:
                c = cells.get((sc, m))
                if c is None:
                    line += f"| {'-':>7} {'-':>7} {'-':>6} "
                else:
                    sps = f"{c['sps']:.3f}" if c["sps"] else "-"
                    line += f"| {c['s']:>7.1f} {sps:>7} {c['gb']:>6.2f} "
            print(line)

        ns = {c["n"] for c in cells.values() if c["n"]}
        if len(ns) > 1:
            print(f"  !! work_units differ across cells {sorted(ns)} -- compare "
                  "s/smp, not wall_s")
        else:
            print(f"  (measured on {ns.pop() if ns else '?'} training samples "
                  "after an untimed warm-up)")


if __name__ == "__main__":
    main()
