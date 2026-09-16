"""Draw the cross-library scaling figure (the paper's) from results/scaling-*.jsonl.

One row per method: attribution time (T) on the left, total peak GPU memory
over the cards a run used (M) on the right, against model size.  Hollow markers
are sharded runs; a cross marks a cell that ran out of memory, drawn on the
ceiling of the hardware it ran on (1x or 4x H200).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

SETUP = {"build_model", "load_data"}
H200_GB = 141.0
# The paper's palette (the data-selection figure), in legend order: black,
# green, red (ours).
LIBS = [("logix", "LogIX", "#000000"), ("bergson", "Bergson", "#2ca02c"),
        ("kronfluence", "Kronfluence", "#1f77b4"), ("dattri_llm", "dattri-llm", "#d62728")]
METHODS = [("graddot", "GradDot"), ("kfac", "K-FAC"), ("ekfac", "EK-FAC")]
DODGE = {"dattri_llm": 0.88, "bergson": 0.96, "kronfluence": 1.04, "logix": 1.13}   # x-offset of OOM marks
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Cells that cannot run on the hardware ladder at all and were therefore never
# launched, drawn as a cross on the ceiling of the hardware they would need:
# LogIX has no sharded path and neither 72B nor 110B fits one H200; Bergson's
# full-dimension K-FAC / EK-FAC factors already exhaust four H200s at 32B, so
# 72B and 110B sit on the four-card ceiling.  Every other cross comes from a
# recorded failure row.
CANNOT_RUN = {("logix", m): [(72.7, 1), (111.2, 1)] for m, _ in METHODS}
CANNOT_RUN.update({("bergson", m): [(72.7, 4), (111.2, 4)] for m in ("kfac", "ekfac")})
CANNOT_RUN.update({("kronfluence", m): [(72.7, 1), (111.2, 1)] for m, _ in METHODS})
SCALES_B = (0.49, 1.54, 3.09, 7.62, 14.77, 32.5, 72.7, 111.2)


def load(results: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(results / "scaling-*.jsonl"))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return rows


def attribution_time(rec: dict) -> float | None:
    secs = [p.get("wall_s") or 0.0 for p in rec.get("phases", []) if p["phase"] not in SETUP]
    return sum(secs) if secs else None


def peak_gb(rec: dict) -> float:
    vals = [d.get("alloc_gb") or 0.0 for p in rec.get("phases", [])
            for d in p.get("gpu_peak", []) if isinstance(d, dict)]
    return max(vals) if vals else 0.0


def cells(rows: list[dict], *, method: str, lib: str):
    """Sorted (params, secs, total_gb, n_gpus) of ok cells, and {params: n_gpus}
    of cells that failed, at the largest configuration tried.  Repeated cells
    reduce to the median-time run.  A failure is superseded by a success at
    the same scale (e.g. a single-card OOM followed by a sharded run)."""
    ok: dict[float, list[dict]] = {}
    failed: dict[float, int] = {}
    for r in rows:
        t = r["task"]
        if t.get("method") != method or r.get("lib") != lib:
            continue
        n_gpus = int(t.get("n_gpus") or 1)
        if r.get("status") == "ok" and attribution_time(r) is not None:
            ok.setdefault(t["params_b"], []).append(r)
        elif r.get("status") in ("oom", "error", "timeout"):
            failed[t["params_b"]] = max(failed.get(t["params_b"], 0), n_gpus)
    pts = []
    for params, rs in ok.items():
        rs = sorted(rs, key=attribution_time)
        r = rs[(len(rs) - 1) // 2]
        n_gpus = int(r["task"].get("n_gpus") or 1)
        pts.append((params, attribution_time(r), peak_gb(r) * n_gpus, n_gpus))
    for params, n_gpus in CANNOT_RUN.get((lib, method), ()):
        failed.setdefault(params, n_gpus)
    failed = {p: n for p, n in failed.items() if p not in ok}
    return sorted(pts), failed


def draw(rows: list[dict], out: Path) -> None:
    """The paper's figure: one column per method, time on the top row and
    memory below, y shared within each row.  It follows the data-selection
    figure -- a serif face at 10 pt on a canvas the width of the text column,
    all four spines, a light grid, lettered panel titles -- so both print at
    the same type size.  *out* is written as given and again as PNG.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    base = 10
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": base,
        "font.family": "serif",
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": base - 2, "ytick.labelsize": base - 2,
        "legend.fontsize": base - 1,
    })
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.1), sharex=True, sharey="row")
    for k, (method, m_label) in enumerate(METHODS):
        ax_t, ax_m = axes[0][k], axes[1][k]
        for lib, _, colour in LIBS:
            pts, failed = cells(rows, method=method, lib=lib)
            covered = {p for p, *_ in pts} | set(failed)
            missing = [p for p in SCALES_B if p not in covered]
            if missing:
                raise ValueError(
                    f"{lib}/{method}: no dot or cross at {missing} B parameters"
                )
            if pts:
                x = [p for p, *_ in pts]
                ax_t.plot(x, [s for _, s, _, _ in pts], "-", color=colour, lw=1.5, zorder=3)
                ax_m.plot(x, [g for _, _, g, _ in pts], "-", color=colour, lw=1.5, zorder=3)
            for p, s, g, n_gpus in pts:
                for ax, v in ((ax_t, s), (ax_m, g)):
                    ax.plot(p, v, marker="o", ms=4.5, ls="none", zorder=4,
                            mfc="white" if n_gpus > 1 else colour, mec=colour, mew=1.3)
            for p, n_gpus in failed.items():
                ax_t.plot(p * DODGE[lib], 0.985, marker="x", ms=6, mew=1.8, color=colour,
                          ls="none", zorder=5, clip_on=False,
                          transform=ax_t.get_xaxis_transform())
                ax_m.plot(p * DODGE[lib], H200_GB * n_gpus, marker="x", ms=6, mew=1.8,
                          color=colour, ls="none", zorder=5, clip_on=False)
        for cards in (1, 4):
            ax_m.axhline(H200_GB * cards, color=INK_2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            # Inside the axes, just under the line.
            ax_m.text(0.42, H200_GB * cards * 0.86, f"{cards}x H200",
                      fontsize=base - 3, color=INK_2, ha="left", va="top")
        for ax in (ax_t, ax_m):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.5, 1, 3, 7, 14, 32, 72, 110])
            ax.set_xticklabels(["0.5", "1", "3", "7", "14", "32", "72", "110"])
            # Narrow panels: 72 and 110 sit close on the log axis.
            ax.tick_params(axis="x", labelrotation=90, pad=2)
            ax.tick_params(axis="y", labelrotation=90, pad=2)
            for lab in ax.get_yticklabels():
                lab.set_va("center")
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.margins(x=0.08)
        if k == 0:
            ax_t.set_ylabel("Time (s)")
            ax_m.set_ylabel("Peak memory (GB)")
        ax_t.set_title(f"({'abc'[k]}) {m_label}", loc="center", color=INK)
    handles = [Line2D([], [], color=c, lw=1.6, label=lab) for _, lab, c in LIBS]
    handles += [Line2D([], [], color=INK_2, lw=0, marker="o", ms=5, mfc="white",
                       mec=INK_2, mew=1.3, label="sharded (4x H200)"),
                Line2D([], [], color=INK_2, lw=0, marker="x", ms=6, mew=1.8, label="failed")]
    # Legend above the panels, as in the data-selection figure.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, handlelength=1.4,
               columnspacing=0.9, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), h_pad=0.6, w_pad=0.6)
    # One x label for the row, placed just under the (vertical) tick labels.
    fig.canvas.draw()
    low = min(
        lab.get_window_extent().y0
        for ax in axes[-1]
        for lab in ax.get_xticklabels()
        if lab.get_text()
    )
    fig.supxlabel("Model size (B parameters)", fontsize=base,
                  y=low / fig.bbox.height - 0.015, va="top")
    for path in (out, out.with_suffix(".png")):
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("wrote", path)


def print_table(rows: list[dict]) -> None:
    print("attribution s (total peak GB over the cards used); F = sharded over 4 cards")
    print(f"{'method':8} {'scale':6}" + "".join(f"{lab:>22}" for _, lab, _ in LIBS))
    for method, _ in METHODS:
        for scale in ("0.5b", "1b", "3b", "7b", "14b", "32b", "72b", "110b"):
            line = f"{method:8} {scale:6}"
            for lib, _, _ in LIBS:
                cands = [r for r in rows if r["lib"] == lib and r["task"]["method"] == method
                         and r["task"]["scale"] == scale]
                ok = sorted((r for r in cands if r["status"] == "ok"), key=attribution_time)
                if ok:
                    r = ok[(len(ok) - 1) // 2]
                    n_gpus = int(r["task"].get("n_gpus") or 1)
                    tag = " F" if r["task"].get("parallelism") == "fsdp" else ""
                    line += f"{attribution_time(r):>11.1f} ({peak_gb(r) * n_gpus:>5.1f}){tag:<3}"[:22].rjust(22)
                elif cands:
                    r = max(cands, key=lambda r: int(r["task"].get("n_gpus") or 1))
                    tag = " F" if r["task"].get("parallelism") == "fsdp" else ""
                    line += f"{r['status'].upper() + tag:>22}"
                else:
                    line += f"{'':>22}"
            print(line)


def main(results: Path) -> None:
    rows = load(results)
    print(f"{len(rows)} rows from {results}/scaling-*.jsonl")
    print_table(rows)
    draw(rows, results / "scaling.pdf")
