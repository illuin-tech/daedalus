"""Shared look for the figure scripts: one palette, one set of rcParams, one save helper.

The house style is **ASCII-plot**: Menlo at every size, a closed hairline box in near-black
ink, ticks pointing out of it, a dotted grid that reads like character cells, and warm paper.
Marks are outlined in ink and filled with a pale tint of their hue.

The eight hues are ColorBrewer **Set3**, as shipped in matplotlib (`mpl.colormaps["Set3"]`),
minus the two of its twelve that carry no identity: `#ffffb3` (near-white yellow) and
`#d9d9d9` (grey), both of which vanish as a line on this paper. They were chosen for looks,
on an explicit instruction not to worry about contrast, and are NOT validated against the
data-viz colour gates — the previous saturated eight were (worst adjacent CVD ΔE 9.1), these
have not been measured and, being pale, certainly separate less. So do not remove the glyph,
the dash pattern, the legend or the direct labels to save space: with a pale palette those
are the channels actually carrying identity, not decoration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ink and paper. INK is near-black on purpose: pure #000 vibrates at hairline widths.
SURFACE, INK, MUTED = "#fdfdfc", "#111111", "#6f6e6a"
AXIS, GRID = INK, "#e4e3df"          # the box is ink; the grid stays a whisper
MONO = ["Menlo", "DejaVu Sans Mono", "Courier New"]

# One knob for the size of every figure in the set. The point sizes below are deliberately
# NOT scaled with it: these land in a document at a fixed column width, so shrinking the
# canvas while holding the type is exactly what makes the labels legible there.
FIG_SCALE = 0.78


def figsize(w: float = 6.2, h: float = 3.8) -> tuple[float, float]:
    """`FIG_SCALE` applied to a figure's nominal inches — every script sizes through this."""
    return (w * FIG_SCALE, h * FIG_SCALE)


# The validated categorical order: assign from slot 0 onward, never cycle. Past eight,
# fold the rest into "other" or facet rather than inventing a hue.
SERIES = ("#8dd3c7", "#bebada", "#fb8072", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5", "#bc80bd")
# The same eight mixed 45% into the paper: mark fills, bar fills, context lines. Far too
# close together to carry identity on their own, which is why they only ever appear behind an
# ink outline and a glyph.
TINTS = ("#bfe6df", "#dad8e9", "#fcb8b0", "#b8d3e5",
         "#fdd5a7", "#d4ecab", "#fce3ef", "#d9b8d9")

# The hue for a figure that carries ONE colour — a single curve, a histogram, a scatter of one
# model. Deliberately outside `SERIES`: those eight are a categorical set, chosen pale because
# a glyph and a dash pattern carry identity beside them, and slot 0's mint is far too faint to
# hold a line on its own. Nothing here needs telling apart from a sibling, so it can be the
# saturated colour the pale set cannot be. `TEAL_TINT` is the same 55/45 mix into the paper
# that `TINTS` is of `SERIES`, for mark and bar fills behind an ink outline.
TEAL, TEAL_TINT = "#17a99a", "#7dcec6"

# Dash patterns and hatches, same fixed order as the hues.
DASHES = ((0, ()), (0, (4, 2)), (0, (1, 1.6)), (0, (7, 2, 1, 2)),
          (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 3)), (0, (6, 2, 1, 2, 1, 2)))
# Single characters: a repeated one ("///") is a wall of ink at these bar heights, and "-"
# reads as an artifact rather than a fill, so it is not in the set.
HATCHES = ("/", ".", "\\", "x", "|", "+", "o", "*")

mpl.rcParams.update({
    "figure.figsize": figsize(), "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    # Warm paper on screen, NOTHING in the file: `savefig.transparent` drops the figure and
    # axes patches at save time (and overrides `savefig.facecolor`), so a figure takes the
    # background of whatever document embeds it instead of stamping a near-white rectangle
    # onto it. Everything else on the page is opaque and unchanged.
    "savefig.transparent": True,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "font.family": "monospace", "font.monospace": MONO,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "axes.spines.top": True, "axes.spines.right": True,     # a closed box
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": (0, (1, 4)),
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "hatch.linewidth": 0.55, "hatch.color": INK,
    "legend.frameon": False, "legend.fontsize": 7.5, "legend.labelcolor": INK,
    "legend.borderpad": 0.55, "legend.handletextpad": 0.6, "legend.columnspacing": 1.6,
    "lines.solid_capstyle": "butt",
})


def bar_kw(slot: int, **over: Any) -> dict:
    """Bar/patch kwargs: tint fill, ink outline, one sparse hatch."""
    i = slot % len(SERIES)
    kw = dict(color=TINTS[i], edgecolor=INK, linewidth=0.8, hatch=HATCHES[i])
    return {**kw, **over}


def ylabel(ax, text: str) -> None:
    """Name the y axis. Every figure in the set names BOTH axes, without exception."""
    ax.set_ylabel(text)


def finish(ax, path: Path, xgrid: bool = False, named_x: bool = True) -> Path:
    """Save and close. Nothing above the box — no title, no subtitle, no rule.

    The document that embeds a figure supplies its caption, and anything restating that
    caption on the figure says the same sentence twice. What the run selection was
    (benchmark, split, model, repeat count) still goes to stdout with every figure, where
    it belongs: it is provenance for whoever regenerates the figure, not a reading for
    whoever looks at it. The two axis names carry the reading.

    `xgrid` off for time series, on where x carries a value.

    Refuses to save a figure with an unnamed axis. With no title and no subtitle the two axis
    names are the only thing on the page saying what is being measured, so a missing one is a
    broken figure, not a cosmetic slip — and it is invisible in review precisely because the
    axis still LOOKS fine, just with tick numbers and nothing else.

    `named_x=False` is the one exemption, and it has to be asked for: a CATEGORICAL x whose
    tick labels are already the names ("Daedalus", "No survey", ...) is named, and adding
    "generation run" under it only repeats what the reader just read. It is opt-in so that
    forgetting a label still fails, which is the whole point of the check.
    """
    unnamed = [name for name, text in ((("x", ax.get_xlabel()) if named_x else ("", "x")),
                                       ("y", ax.get_ylabel())) if name and not text.strip()]
    if unnamed:
        raise ValueError(f"{path.name}: unnamed {' and '.join(unnamed)} axis — "
                         f"every figure in this set names both")
    ax.grid(axis="x", visible=xgrid)
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path)
    plt.close(ax.figure)
    print(f"  {path}")
    return path
