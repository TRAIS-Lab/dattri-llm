"""Draw the paper's scaling figures straight from raw benchmark records.

    python plot.py                       # panel (b), memory scaling
    python plot.py --panel a             # compute scaling
    python plot.py --panel both          # both
    python plot.py --results other.jsonl # a different run

Reads ``results/aggregate.jsonl`` -- the records `log.BenchRun` appends, one
JSON line per (task, library) cell -- and writes ``results/panel_b.png``, or
``results/panel_a.png`` on request. There is no intermediate data file: every
reduction happens here, so the figures cannot drift from the runs they claim to
show. Panel (b) is the default because it is the only one committed; panel (a)
is written on demand and is not tracked.

Needs matplotlib, which the repo does not depend on: ``pip install matplotlib``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "aggregate.jsonl"

# Fixed order, so a re-run cannot repaint a series and a tie always resolves the
# same way.
METHODS = [
    ("graddot", "GradDot", "#2a78d6", "o"),
    ("kfac", "K-FAC", "#1baf7a", "s"),
    ("ekfac", "EK-FAC", "#eb6834", "^"),
]
FAMILIES = [("pythia", "Pythia"), ("qwen", "Qwen")]
SERIES_B = [
    ("pythia", "single", "Pythia — 1 GPU", "#2a78d6", "o", "-"),
    ("qwen", "single", "Qwen — 1 GPU", "#1baf7a", "s", "-"),
    ("qwen", "fsdp", "Qwen — 4× FSDP (per card)", "#eb6834", "^", "--"),
]
CEILING_GB = 141.0
GRID = dict(color="#e6e5df", lw=0.7)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_peak(rec: dict) -> float:
    """Peak allocated GB over EVERY phase -- what the card must supply.

    Not the attribution phase alone: the FSDP adapter tears the model down
    before scoring and runs against a torch.nn.Linear(1, 1) stand-in, so its
    ``score`` phase peaks at almost nothing while the real per-shard peak sits
    in ``cache``. Reading one named phase draws the sharded points near zero.
    """
    return max((g["alloc_gb"] for p in rec.get("phases", [])
                for g in p.get("gpu_peak", [])), default=0.0)


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


def setup(ax) -> None:
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_a(recs: list[dict], out: Path) -> None:
    """Attribution wall-clock against model scale, per method, log-log.

    Single-device rows ONLY. `run_ours_fsdp.py` has no warm-up and times all
    `n_train + n_test` samples, where `run_ours.py` warms up and times `n_meas`
    -- 65 samples against 32. Those timings are not comparable and must not
    share an axis; the memory panel is unaffected because peak is per-batch.
    """
    import matplotlib.pyplot as plt

    sel = [r for r in recs
           if r["task"]["parallelism"] == "single" and attribution_time(r)]
    units = {attribution_units(r) for r in sel}
    if len(units) > 1:
        print(f"  !! work_units differ across cells {sorted(units)} -- "
              "wall-clock is not comparable; plot s_per_sample instead")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), dpi=170, sharey=True)
    for ax, (fam, fam_label) in zip(axes, FAMILIES):
        for key, label, color, marker in METHODS:
            pts = sorted((r["task"]["params_b"], attribution_time(r))
                         for r in sel
                         if r["task"]["family"] == fam and r["task"]["method"] == key)
            if not pts:
                continue
            ax.plot([p for p, _ in pts], [t for _, t in pts], "-", marker=marker,
                    color=color, lw=2, ms=6, mec="white", mew=1.4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model size (B params, log)")
        ax.set_title(fam_label, fontsize=11)
        setup(ax)
    n = units.pop() if len(units) == 1 else "?"
    axes[0].set_ylabel(f"Attribution time (s, {n} samples)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


LIBS = [
    ("dattri_llm", "dattri-llm", "#2a78d6", "o"),
    ("bergson", "bergson", "#1baf7a", "s"),
    ("kronfluence", "kronfluence", "#eb6834", "^"),
    ("logix", "logix", "#eda100", "D"),
]


def panel_c(recs: list[dict], out: Path, metric: str = "memory") -> None:
    """Peak memory (default) or attribution time against scale, per library.

    Memory is the default because it is the only axis these rows can honestly
    share. Peak is per-batch and the workload is batch 1, so it does not depend
    on how many samples each adapter timed -- and they differ: run_ours.py warms
    up and times `n_meas` (32), the baseline adapters time the full `n_train`
    (64-65) with no warm-up, and bergson records no work_units at all. On raw
    wall-clock that difference alone flatters us ~2x and leaves bergson
    unplaceable, so `--metric time` is available but not the default.

    A library that stops partway is the result, not a gap to apologise for:
    run.py records nothing for a failed cell, so a curve ends where that library
    stopped reaching. These cells are fp32 and therefore a separate ladder from
    panels (a) and (b), which are bf16.
    """
    import matplotlib.pyplot as plt

    sel = [r for r in recs
           if attribution_time(r) and r["task"]["parallelism"] == "single"]
    bases = {mem_basis(r) for r in sel}
    if metric == "memory" and len(bases) > 1:
        print(f"  note: mixed measurement bases {sorted(bases)}; plotting "
              "whole-process memory (reserved for torch rows, smi-used for the "
              "rest) so the libraries share one quantity")
    if not sel:
        print("  no single-device rows to draw")
        return
    methods = [m for m, *_ in METHODS if any(r["task"]["method"] == m for r in sel)]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.3),
                             dpi=170, sharey=True, squeeze=False)
    for ax, meth in zip(axes[0], methods):
        for key, label, color, marker in LIBS:
            val = comparable_peak if metric == "memory" else attribution_time
            pts = sorted((r["task"]["params_b"], val(r))
                         for r in sel
                         if r["lib"] == key and r["task"]["method"] == meth)
            if not pts:
                continue
            ax.plot([p for p, _ in pts], [t for _, t in pts], "-", marker=marker,
                    color=color, lw=2, ms=6, mec="white", mew=1.4, label=label)
        ax.set_xscale("log")
        if metric == "time":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, CEILING_GB * 1.18)
        ax.set_xlabel("Model size (B params, log)")
        ax.set_title(dict((m, l) for m, l, *_ in METHODS)[meth], fontsize=11)
        setup(ax)
    if metric == "memory":
        # The ceiling is the panel's argument: a library whose curve is heading
        # through it is a library that stops running, and the scales where that
        # happened are marked rather than left as a blank gap.
        for ax, meth in zip(axes[0], methods):
            ax.axhspan(CEILING_GB, 1e4, color="#c0392b", alpha=0.05, lw=0)
            ax.axhline(CEILING_GB, color="#8a8a85", ls=(0, (7, 4)), lw=1.1)
            attempted = sorted({r["task"]["params_b"] for r in sel
                                if r["task"]["method"] == meth})
            # Several libraries usually drop out at the SAME scale, so the
            # markers are staggered on the log axis; stacked exactly they read
            # as one failure instead of two.
            dropped = [(k, c) for k, _l, c, _m in LIBS
                       if (g := {r["task"]["params_b"] for r in sel
                                 if r["lib"] == k and r["task"]["method"] == meth})
                       and any(pb > max(g) for pb in attempted)]
            for i, (key, color) in enumerate(dropped):
                got = {r["task"]["params_b"] for r in sel
                       if r["lib"] == key and r["task"]["method"] == meth}
                for pb in (x for x in attempted if x > max(got)):
                    off = 1.0 + 0.055 * (i - (len(dropped) - 1) / 2)
                    ax.plot([pb * off], [CEILING_GB], marker="x", color=color,
                            ms=10, mew=2.4, ls="none", clip_on=False)
            if dropped:
                names = " and ".join(k for k, _ in dropped)
                ax.annotate(f"{names}: exceeded one H200",
                            xy=(0.5, CEILING_GB), xytext=(0, 13),
                            textcoords="offset points", ha="center", fontsize=8,
                            color="#8a3a2a",
                            xycoords=ax.get_yaxis_transform())
        # Ceiling label on the FIRST subplot, where no dropout note sits.
        axes[0][0].text(0.02, CEILING_GB - 9, "one H200 — 141 GB", ha="left",
                        fontsize=8.5, color="#6c6c68",
                        transform=axes[0][0].get_yaxis_transform())
    axes[0][0].set_ylabel("Peak device memory (GB)" if metric == "memory"
                          else "Attribution time (s)")
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    reached = {k: max((r["task"]["params_b"] for r in sel if r["lib"] == k), default=None)
               for k, *_ in LIBS}
    for k, v in reached.items():
        if v is not None:
            print(f"    {k:12s} reached {v:6.2f}B")


def panel_b(recs: list[dict], out: Path) -> None:
    """Peak per-device memory against model scale.

    Single-device and FSDP are separate series and are never joined. A
    single-device peak is what one card must supply for the whole model; an
    FSDP peak is what one card supplies for its shard -- 72.7B sharded costs
    less per card than 14.8B unsharded, so one connected line would show memory
    falling as the model grows. Peak is the max across methods (the worst case
    over what was run), compared on the raw value and rounded only for display.
    """
    import matplotlib.pyplot as plt

    best: dict[tuple, tuple] = {}
    for r in recs:
        t = r["task"]
        key = (t["family"], t["parallelism"], t["params_b"])
        gb = run_peak(r)
        if key not in best or gb > best[key][0]:
            best[key] = (gb, t.get("n_gpus", 1))

    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=170)
    for fam, par, label, color, marker, ls in SERIES_B:
        pts = sorted((pb, v) for (f, p, pb), v in best.items() if f == fam and p == par)
        if not pts:
            continue
        sharded = par != "single"
        ax.plot([p for p, _ in pts], [v[0] for _, v in pts], ls, marker=marker,
                color=color, lw=2, ms=9 if sharded else 6,
                mfc="white" if sharded else color,
                mec=color if sharded else "white",
                mew=2 if sharded else 1.4, label=label)
        if sharded:
            # Mark the shard count: a per-shard point must not be read as a
            # whole-model one.
            for pb, (gb, ng) in pts:
                ax.annotate(f"{ng}×", xy=(pb, gb), xytext=(0, -16),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=color)

    ax.axhline(CEILING_GB, color="#8a8a85", ls=(0, (7, 4)), lw=1.1)
    ax.text(105, CEILING_GB + 4, "H200 ceiling — 141 GB", ha="right",
            fontsize=9, color="#6c6c68")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 110)
    ax.set_ylim(0, 160)
    ticks = [0.5, 1, 3, 7, 14, 32, 72]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_ylabel("Peak allocated memory per device (GB)")
    setup(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["a", "b", "c", "both"], default="b")
    ap.add_argument("--results", nargs="+", default=[str(RESULTS)],
                    help="one or more results.jsonl; several are concatenated, "
                         "which is how a panel spanning experiments is drawn")
    ap.add_argument("--metric", choices=["memory", "time"], default="memory",
                    help="panel (c) axis; memory is the only one these rows share")
    a = ap.parse_args()

    paths = [Path(x) for x in a.results]
    recs = [r for p in paths for r in load(p)]
    order = {m: i for i, (m, *_) in enumerate(METHODS)}
    recs.sort(key=lambda r: order.get(r["task"]["method"], 99))
    out_dir = paths[0].parent
    devices = sorted({r["device"]["gpu_name"] for r in recs})
    dtypes = sorted({r.get("dtype") or r["task"].get("dtype") or "?" for r in recs})
    src = ", ".join(p.name for p in paths)
    print(f"{len(recs)} rows from {src}  |  {', '.join(devices)}  |  "
          f"{', '.join(dtypes)}")
    if len(devices) > 1 or len(dtypes) > 1:
        print("  !! MIXED device or dtype -- a change in either moves both panels "
              "and will read as a scaling effect")

    import matplotlib
    matplotlib.use("Agg")
    if a.panel in ("a", "both"):
        panel_a(recs, out_dir / "panel_a.png")
    if a.panel in ("b", "both"):
        panel_b(recs, out_dir / "panel_b.png")
    if a.panel == "c":
        panel_c(recs, out_dir / f"panel_c_{a.metric}.png", a.metric)


if __name__ == "__main__":
    main()
