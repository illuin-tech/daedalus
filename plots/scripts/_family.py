"""Marks and chrome for the transferability and "generated vs real" figures.

`finish_plain` is the house `_style.finish` with the x grid on, which every figure whose x
axis carries a value wants. `FAMILY_HUES` / `FAMILY_GLYPHS` / `mark_kw` / `hollow_kw` are
the fillable, ink-outlined marks `transferability.py` gives each model.
"""

from __future__ import annotations

from pathlib import Path

from _style import INK, SURFACE, finish

# Eight pale hues, the same eight as `_style.SERIES` (ColorBrewer Set3 minus its near-white
# yellow and its grey). Being pale they separate poorly on their own, so identity also rides
# on a distinct FILLABLE shape per slot and a dark ink outline on every mark ("+" and "x"
# take no fill, so they cannot carry a hue).
FAMILY_HUES = ("#8dd3c7", "#bebada", "#fb8072", "#80b1d3",
               "#fdb462", "#b3de69", "#fccde5", "#bc80bd")
FAMILY_GLYPHS = ("v", "s", "D", "^", "o", "p", "h", "*")


def mark_kw(slot: int, **over) -> dict:
    """`errorbar` kwargs for one slot: solid hue, thin ink edge, no join line.

    The hue fills the mark rather than tinting it — with no error bars drawn there is
    nothing else on the mark to carry the colour.
    """
    i = slot % len(FAMILY_HUES)
    kw = dict(fmt=FAMILY_GLYPHS[i], ms=7.6, mew=1.0, mec=INK, mfc=FAMILY_HUES[i],
              ecolor=INK, elinewidth=0.9, capsize=2.0, linestyle="none")
    return {**kw, **over}


def hollow_kw(slot: int, **over) -> dict:
    """Same mark, unfilled — for the "before" end of a pair (e.g. memory off)."""
    return mark_kw(slot, mfc=SURFACE, **over)


def finish_plain(ax, path: Path) -> Path:
    """The house `finish` with the x grid on."""
    return finish(ax, path, xgrid=True)
