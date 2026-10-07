"""Histograms for one self-play generation run, over its BANKED sessions.

    python plots/scripts/generation.py outputs/daedalus/<run> [--out plots/pdf/generation-dynamics]

Writes two PDFs, named after the run:
    <run>_refinements_hist.pdf                 distribution of refinements per session
    <run>_failures_before_banking_hist.pdf     failed solver attempts before banking

Reads only `<run>/sessions/session_*.json`, so it is safe to run on a live run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import DASHES, INK, TEAL_TINT, bar_kw, finish, plt, ylabel  # noqa: E402


def load_sessions(run: Path) -> list[dict]:
    """Real solver sessions (skips survey/discover/error records)."""
    out = []
    for f in sorted((run / "sessions").glob("session_*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(rec, dict) and isinstance(rec.get("session_index"), int) and rec.get("explorer"):
            out.append(rec)
    return out


def failures_before_banking(rec: dict) -> int | None:
    """Failed solver attempts in the deciding solve of a banked session; None if not banked.

    Every failure precedes the closing success streak (the loop ends on
    `num_success_to_continue` consecutive successes), so the total failure count IS the
    count before banking. Counting only the failures that lead the sequence would report 0
    for a task that started well, failed, then recovered — which is most of them."""
    if rec.get("result") != "banked":
        return None
    attempts = (rec.get("accumulation_summary") or {}).get("attempts") or []
    return sum(1 for a in attempts if a.get("outcome") == "failure") if attempts else None


def plot_hist(values, xlabel, path: Path) -> Path:
    """A bar per observed integer value. The axis spans the DATA, not [0, max].

    Anchoring at 0 drew an empty leading bin for a count that cannot occur: a banked session
    defeated the solver by construction, so "0 failures before banking" is not a low outcome,
    it is an impossible one. The refinements histogram genuinely reaches 0 and keeps it,
    because `lo` comes from the values.
    """
    fig, ax = plt.subplots()
    lo, hi = min(values), max(values)
    # One-hue figure, so the bar carries no categorical identity and needs no texture.
    ax.hist(values, bins=np.arange(lo - 0.5, hi + 1.5), rwidth=0.9,
            **bar_kw(0, color=TEAL_TINT, hatch=None))
    mean = float(np.mean(values))
    ax.axvline(mean, color=INK, lw=1.2, ls=DASHES[1])
    ax.annotate(f"mean {mean:.1f}", (mean, 0.96), xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points", color=INK, fontsize=7.5, va="top")
    ax.set_xlabel(xlabel)
    ylabel(ax, "sessions")
    ax.set_xticks(range(lo, hi + 1))
    ax.margins(y=0.16)
    return finish(ax, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Histograms for one generation run")
    ap.add_argument("run", type=Path, help="a generation run folder (outputs/daedalus/<name>)")
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/generation-dynamics"),
                    help="output directory")
    args = ap.parse_args()

    sessions = load_sessions(args.run)
    dropped = Counter(s.get("result") for s in sessions if s.get("result") != "banked")
    sessions = [s for s in sessions if s.get("result") == "banked"]
    if dropped:
        print(f"  dropped {sum(dropped.values())}: "
              + ", ".join(f"{k} {v}" for k, v in dropped.most_common()))
    if not sessions:
        raise SystemExit(f"No banked sessions under {args.run}/sessions")
    name = args.run.name
    refines = [len(s.get("refinements") or []) for s in sessions]
    fails = [n for n in (failures_before_banking(s) for s in sessions) if n is not None]
    print(f"{name} · {len(sessions)} banked sessions · mean refinements "
          f"{np.mean(refines):.2f} · mean failures before banking "
          f"{np.mean(fails) if fails else float('nan'):.2f}")
    plot_hist(refines, "refinements in a session", args.out / f"{name}_refinements_hist.pdf")
    if fails:
        plot_hist(fails, "failed solver attempts before banking",
                  args.out / f"{name}_failures_before_banking_hist.pdf")
    else:
        print("  (no banked session with attempt records — skipped the failures histogram)")


if __name__ == "__main__":
    main()
