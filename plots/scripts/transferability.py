"""Does one heuristic pool help models it was not built with?

    python plots/scripts/transferability.py \
        --benchmark appworld --split test_normal \
        --pool outputs/daedalus/appworld/consolidated_pool.json

One ARROW per model: it starts at that model's memory-off baseline and ends at the same model
run with the WHOLE pool injected at the start of every task (`memory.enabled: true` +
`heuristics_at_start: true`, the same pool path). x is the mean number of turns per task and y
the mean success rate, so the arrow shows both what the pool bought and what it cost in turns —
a pool that lifts the score by making every episode longer is a different result from one that
lifts it for free.

    <benchmark>_<split>_<pool>_transferability_success.pdf

A model only appears when BOTH runs exist, are scored, and sit on the same split; the pool is
matched on the `memory.pool_path` recorded in each run's own `config.yaml`, not on the run name.
Only `daedalus` runs are eligible on both ends, for the same reason `testset_correlation.py`
restricts its x axis: every baseline under `references/` sets `memory.enabled: false` while doing
its own retrieval, so it is neither a bare baseline nor an at-start pool run.

`EXCLUDED_MODELS` drops models from the figure whatever runs they have — the figure is a claim
about transfer, not a census, and it only has eight validated marks to spend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _family import (FAMILY_GLYPHS, FAMILY_HUES, finish_plain,  # noqa: E402
                     hollow_kw, mark_kw)
from _runs import (is_original_prompt, load_json, memory_block,  # noqa: E402
                   mean_turns, same_pool, slug)
from _style import INK, MUTED, SURFACE, figsize, plt  # noqa: E402

from daedalus.core.config import DEFAULT_SETUP, run_dirs  # noqa: E402

# Models kept off the figure whatever runs exist for them. Matched EXACTLY against the last
# path segment of the model string, never as a prefix: `deepseek-v4-flash` and
# `deepseek-v4-flash-0731` are two different models with two different arrows, and a prefix
# rule would silently take both.
EXCLUDED_MODELS = {
    "gpt-5.6-luna",
    "gemma-4-31b-it",
    "deepseek-v4-flash",
    "gpt-oss-120b",
    "mimo-v2.5",
}


def collect(benchmark: str, split: str, pool: str) -> dict[str, dict]:
    """{model: {"base": row, "pool": row}} for every scored daedalus run on this split.

    Both ends are keyed on the FULL model string from `run_meta.json`, so a model is only
    ever paired with itself.
    """
    found: dict[str, dict] = {}
    for setup, d in run_dirs("inference"):
        if setup != DEFAULT_SETUP:
            continue  # see the module docstring: references/ runs are neither end
        if d.name.startswith("old_"):
            # A superseded run renamed in place. It keeps its config, its traces and its
            # repeat count, so the "most repeats, then newest" tie-break cannot tell it from
            # the run that replaced it — the prefix is the only signal, and it is deliberate.
            continue
        meta, ev = load_json(d / "run_meta.json"), load_json(d / "evaluation.json")
        if not meta or not ev:
            continue
        if meta.get("benchmark") != benchmark or str(meta.get("domain")) != split:
            continue
        if str(meta.get("model") or "").split("/")[-1] in EXCLUDED_MODELS:
            continue
        agg = ev.get("aggregate") or {}
        if agg.get("success_rate_mean") is None:
            continue
        mem = memory_block(d)
        at_start = bool(mem.get("enabled")) and bool(mem.get("heuristics_at_start"))
        if at_start:
            side = "pool" if same_pool(mem.get("pool_path"), pool) else None
        elif not mem.get("enabled") and not mem.get("heuristics_at_start"):
            # A run on the benchmark's OWN solver prompt is a prompt control, not this
            # figure's floor: it is memory-off and otherwise indistinguishable from the
            # baseline, which is how it once won the tie-break and understated the gain
            # (+4.3 pp reported where the real number was +14.8 pp).
            side = None if is_original_prompt(d) else "base"
        else:
            side = None  # per-turn retrieval, or a different pool — not this figure
        if side is None:
            continue
        turns = mean_turns(d)
        row = {
            "name": d.name, "model": str(meta.get("model") or "?"),
            "y": 100 * agg["success_rate_mean"],
            "turns": turns, "n_runs": agg.get("num_runs") or 0,
            "timestamp": meta.get("timestamp", ""),
        }
        slot = found.setdefault(row["model"], {})
        # Most repeats wins, then newest — the same rule the correlation figure uses.
        prev = slot.get(side)
        if prev is None or (row["n_runs"], row["timestamp"]) > (prev["n_runs"],
                                                               prev["timestamp"]):
            slot[side] = row
    return found


# The turns axis's window, chosen rather than derived: no model comes near 0 turns.
X_MIN, X_MAX = 6, 22
# Per-GLYPH size corrections, local to this figure. `ms` is a nominal size, not a perceived
# one: at one `ms` matplotlib's star reads smaller than the other shapes and its diamond
# reads larger, because the ink is distributed differently inside the same bounding box.
# Kept here rather than in `_family.mark_kw`, which the other figures share.
BASE_MS = 7.6
GLYPH_MS = {"*": 9.2, "D": 6.0}


def _ms(slot: int) -> float:
    return GLYPH_MS.get(FAMILY_GLYPHS[slot % len(FAMILY_GLYPHS)], BASE_MS)


# The arrow takes the same grey family as the model names: it is chrome, not a series — the
# two marks at its ends carry the model's identity. Midway between MUTED (#6f6e6a) and #b3b2af.
ARROW = "#91908d"


def plot(pairs: list[dict], path: Path) -> Path:
    # 5.4/1.2 square. `_style` fixes the point sizes, so the canvas size is what sets the
    # labels' size RELATIVE to the marks.
    fig, ax = plt.subplots(figsize=figsize(5.4 / 1.2, 5.4 / 1.2))
    # The AXES BOX is a square, not the page. `set_box_aspect(1)` fixes the ratio of the
    # plotting rectangle in display units, independently of the data ranges — which is what
    # is wanted here, because x is turns and y is percent, so a data-space aspect would be
    # meaningless. The saved PDF is still wider than tall: `savefig.bbox="tight"` crops to the
    # ink, and the axis names and the model labels live outside the box.
    ax.set_box_aspect(1)
    # One slot per MODEL, not per family. The family map collapsed gpt-5.4-nano and
    # gpt-5.4-mini onto one mark — the two ends of the figure's biggest arrow — and pushed
    # z-ai past the eight validated slots, where it repeated an earlier family's hue.
    # Assignment is by sorted model name, never by score, so a model keeps its identity when
    # its position moves or a sibling drops out.
    names = sorted({p["model"] for p in pairs})
    if len(names) > len(FAMILY_HUES):
        print(f"  ! {len(names)} models past the {len(FAMILY_HUES)} validated marks — "
              f"some will repeat: {names[len(FAMILY_HUES):]}")
    slots = {m: i for i, m in enumerate(names)}

    for p in pairs:
        i = slots[p["model"]]
        b, q = p["base"], p["pool"]
        # The arrow first, so the marks sit on top of its ends.
        ax.annotate("", xy=(q["turns"], q["y"]), xytext=(b["turns"], b["y"]),
                    # `shrink` is the gap left at each end, in points. Kept just big enough
                    # to clear the ink outline of a 7.6pt mark — any more and the arrow reads
                    # as floating between the two rather than joining them.
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0,
                                    shrinkA=5, shrinkB=6, mutation_scale=11))
        # Hollow = no memory, filled = the whole pool at the start of every task.
        ax.errorbar([b["turns"]], [b["y"]], **hollow_kw(i, ms=_ms(i)), zorder=3)
        ax.errorbar([q["turns"]], [q["y"]], **mark_kw(i, ms=_ms(i)), zorder=4)

        d = q["y"] - b["y"]
        # Offset PERPENDICULAR to the arrow, then forced to the UPPER-RIGHT side.
        # Perpendicular alone keeps the text off the line but its sign follows the arrow's
        # direction, so flipping a downward perpendicular keeps that clearance and puts every
        # delta on the same side of every arrow.
        vx, vy = q["turns"] - b["turns"], q["y"] - b["y"]
        n = (vx * vx + vy * vy) ** 0.5 or 1.0
        ox, oy = -vy / n * 5, vx / n * 5          # rotate the unit vector 90 degrees
        if oy < 0:
            ox, oy = -ox, -oy
        if abs(oy) < 1e-6:     # vertical arrow: the perpendicular is horizontal, so it
            ox, oy = 5.0, 0.0  # would sit beside the line rather than above it
        ax.annotate(f"{d:+.1f}%",
                    ((b["turns"] + q["turns"]) / 2, (b["y"] + q["y"]) / 2),
                    xytext=(ox, oy), textcoords="offset points",
                    ha="left", va="bottom", fontsize=5.8, style="italic", color=MUTED)
        # The model name sits at the END of the arrow, the delta at its MIDPOINT, so the two
        # clear each other even though both now take the upper side.
        ax.annotate(p["model"].split("/")[-1], (q["turns"], q["y"]),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color=MUTED, clip_on=False)

    ax.set_xlim(X_MIN, X_MAX)
    # The FULL percentage range, not a window around the data: every arrow is a gain in
    # points, and a data-driven window would silently rescale how big the arrows look.
    ax.set_ylim(0, 100)
    ax.set_xlabel("mean turns per task")
    ax.set_ylabel("mean success rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(10))

    # Fill is the only channel the reader has to tell the two ends of an arrow apart, and the
    # arrowhead alone is easy to miss at this size. Both keys are drawn in a NEUTRAL grey and
    # a circle: hue and shape belong to the model, so the key may claim neither.
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=BASE_MS, mew=1.0, mec=INK, mfc=SURFACE,
               label="baseline"),
        Line2D([], [], ls="none", marker="o", ms=BASE_MS, mew=1.0, mec=INK, mfc=MUTED,
               label="using Daedalus"),
    ], loc="lower left", frameon=False, fontsize=7, handletextpad=0.6, borderpad=0.4,
        labelspacing=0.6)
    return finish_plain(ax, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, help="appworld | tau2 | automationbench")
    ap.add_argument("--split", required=True, help="the split both runs must share")
    ap.add_argument("--pool", required=True,
                    help="the heuristic pool, as it appears in memory.pool_path "
                         "(e.g. outputs/daedalus/<run>/consolidated_pool.json)")
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/cross-family"))
    args = ap.parse_args()

    print(f"excluded models: {', '.join(sorted(EXCLUDED_MODELS))}")
    found = collect(args.benchmark, args.split, args.pool)
    pairs, half = [], []
    for model, sides in sorted(found.items()):
        if "base" in sides and "pool" in sides and sides["pool"]["turns"] is not None \
                and sides["base"]["turns"] is not None:
            pairs.append({"model": model, "base": sides["base"], "pool": sides["pool"]})
        else:
            half.append((model, sorted(sides)))
    pool_name = Path(args.pool).parent.name or Path(args.pool).stem
    print(f"pool: {args.pool}")
    for model, have in half:
        missing = "at-start run with this pool" if have == ["base"] else "memory-off baseline"
        print(f"  {model.split('/')[-1]}: only {', '.join(have)} — no {missing}, skipped")
    if not pairs:
        raise SystemExit(f"No model has BOTH a memory-off baseline and an at-start run of "
                         f"{args.pool} on {args.benchmark}/{args.split}.")

    print(f"\n{args.benchmark} · {args.split} · {len(pairs)} model(s) with both runs")
    for p in sorted(pairs, key=lambda p: -(p["pool"]["y"] - p["base"]["y"])):
        b, q = p["base"], p["pool"]
        print(f"  {p['model'].split('/')[-1]:24} "
              f"{b['y']:5.1f}% @ {b['turns']:5.2f} turns  ->  "
              f"{q['y']:5.1f}% @ {q['turns']:5.2f} turns   "
              f"{q['y'] - b['y']:+5.1f}%, {q['turns'] - b['turns']:+5.2f} turns")
        print(f"    {'':24} {b['name']}  ->  {q['name']}")
    tag = "_".join(filter(None, (args.benchmark, slug(args.split), slug(pool_name),
                                 "transferability", "success")))
    plot(pairs, args.out / f"{tag}.pdf")


if __name__ == "__main__":
    main()
