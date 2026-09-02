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
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    order = {m: i for i, (m, *_) in enumerate(METHODS)}
    # Deterministic tie-break: the FSDP adapter captures identically for every
    # method, so its peaks tie to the byte.
    recs.sort(key=lambda r: order.get(r["task"]["method"], 99))
    return recs


def run_peak(rec: dict) -> float:
    """Peak allocated GB over EVERY phase -- what the card must supply.

    Not the attribution phase alone: the FSDP adapter tears the model down
    before scoring and runs against a torch.nn.Linear(1, 1) stand-in, so its
    ``score`` phase peaks at almost nothing while the real per-shard peak sits
    in ``cache``. Reading one named phase draws the sharded points near zero.
    """
    return max((g["alloc_gb"] for p in rec.get("phases", [])
                for g in p.get("gpu_peak", [])), default=0.0)


def attribute_phase(rec: dict) -> dict | None:
    by = {p["phase"]: p for p in rec.get("phases", [])}
    return by.get("attribute") or by.get("score")


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

    sel = [r for r in recs if r["task"]["parallelism"] == "single" and attribute_phase(r)]
    units = {attribute_phase(r).get("work_units") for r in sel}
    if len(units) > 1:
        print(f"  !! work_units differ across cells {sorted(units)} -- "
              "wall-clock is not comparable; plot s_per_sample instead")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), dpi=170, sharey=True)
    for ax, (fam, fam_label) in zip(axes, FAMILIES):
        for key, label, color, marker in METHODS:
            pts = sorted((r["task"]["params_b"], attribute_phase(r)["wall_s"])
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
    ap.add_argument("--panel", choices=["a", "b", "both"], default="b")
    ap.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args()

    path = Path(a.results)
    recs = load(path)
    devices = sorted({r["device"]["gpu_name"] for r in recs})
    dtypes = sorted({r.get("dtype") or r["task"].get("dtype") or "?" for r in recs})
    print(f"{len(recs)} rows from {path}  |  {', '.join(devices)}  |  {', '.join(dtypes)}")
    if len(devices) > 1 or len(dtypes) > 1:
        print("  !! MIXED device or dtype -- a change in either moves both panels "
              "and will read as a scaling effect")

    import matplotlib
    matplotlib.use("Agg")
    if a.panel in ("a", "both"):
        panel_a(recs, path.parent / "panel_a.png")
    if a.panel in ("b", "both"):
        panel_b(recs, path.parent / "panel_b.png")


if __name__ == "__main__":
    main()
