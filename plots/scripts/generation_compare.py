"""Generation cost of several self-play runs, one stacked bar per run split by agent.

    python plots/scripts/generation_compare.py <run> <run> ... [--out plots/pdf/generation-dynamics]

Writes `generation_cost_by_role.pdf`.

Every bar's TOTAL is the run browser's raw $/run (`full_cache_cost` at standard tier; see
`raw_run_cost`). The split by agent comes from the run's per-call ledger, scaled onto that
total by one raw/actual factor per run.

A pre-ledger run (`appworld`, the Table 1 bank, migrated with `migration_basis: "solver-only"`) has
no per-call ledger. Its solver cost is recovered exactly from the solver traces, and the
remaining explorer+judge+extraction bucket is exact as a group (session costs minus traced
solver); only its three-way split is ESTIMATED, from constants calibrated on the ledgered runs
on the same figure (see `_agent_cost`).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).parent))
from _runs import load_json  # noqa: E402
from _style import INK, MUTED, SURFACE, bar_kw, figsize, finish, plt, ylabel  # noqa: E402


def raw_run_cost(run: Path) -> tuple[float | None, str]:
    """The paper's generation cost: every call at standard tier, full caching (paper_cost)."""
    from daedalus.core.logging.paper_cost import cost_per_run

    try:
        total, info = cost_per_run(run)
    except Exception as exc:  # noqa: BLE001 — a figure must not die on the cost audit
        return None, f"cost unavailable ({type(exc).__name__})"
    if total is None:
        return None, "cost unavailable (no priceable calls)"
    return total, (f"{info.get('num_priced_calls')} calls @ standard"
                   f"{'' if info.get('complete') else ', INCOMPLETE'}")


# Fixed order, so a role keeps its hue and hatch from one figure to the next — the house
# rule that colour follows the entity, never its rank. Roughly the order a session runs in.
# `unattributed` is last and deliberately NOT a palette slot: it is the absence of a
# measurement, not another agent, and must not read as one.
ROLES = ("explorer", "solver", "judge", "extraction")
UNATTRIBUTED = "unattributed"
# Drawn nowhere: one post-hoc call per run, ~$0.3 of a $110-210 bill, and it is not an agent
# that ran during generation. Keeping it added a legend entry for an invisible sliver.
OMITTED = ("consolidation",)
# The bucket a pre-ledger run collapses its three non-solver agents into; see `rebuild_roles`.
COMBINED = "explorer+judge+extraction"


def split_combined(run: Path, roles: dict[str, float], cals: list,
                   rest_cals: list) -> dict[str, float]:
    """Split a pre-ledger run's combined bucket three ways: explorer, judge, extraction.

    The bucket's TOTAL stays exactly what the artifacts say (session costs minus the traced
    solver, an identity checked against both ledgered runs). Only its internal division is
    modelled — see `_agent_cost` for each predictor and the disagreement it was chosen on.

    The three predictions enter as SHARES, not as dollars, so the bar total is untouched by
    the model and stays the run browser's raw figure. That also absorbs the ~14% by which the
    three over-predict the bucket when summed: they are normalised onto it rather than left to
    contradict a total that is measured.
    """
    from _agent_cost import estimated_cost, failed_attempts, trace_tokens

    bucket = roles.get(COMBINED)
    if not bucket or not cals or not rest_cals:
        return roles
    cal = tuple(sum(c[i] for c in cals) / len(cals) for i in range(3))
    jpt = sum(c[0] for c in rest_cals) / len(rest_cals)
    xpf = sum(c[1] for c in rest_cals) / len(rest_cals)
    parts = {
        "explorer": estimated_cost(run, cal, "flex") or 0.0,
        "judge": jpt * trace_tokens(run),
        "extraction": xpf * failed_attempts(run),
    }
    total = sum(parts.values())
    if total <= 0:
        return roles
    out = {k: v for k, v in roles.items() if k != COMBINED}
    for role, value in parts.items():
        out[role] = out.get(role, 0.0) + bucket * value / total
    return out


def rebuild_roles(run: Path) -> dict[str, float]:
    """Per-agent cost for a run whose ledger cannot supply it, from the artifacts it kept.

    A pre-ledger run recorded no per-call usage, but it did keep a solver trace per attempt
    (each with its own `total_cost_usd`) and a `cost_usd` per session. Two identities recover
    most of the split, and BOTH were checked against the two runs that do have a ledger,
    where the answer is known:

        Σ trace.total_cost_usd            == ledger[solver]                    (to the cent)
        Σ session.cost_usd  −  that sum   == ledger[explorer+judge+extraction] (to <$0.001)

    So the solver is a real measurement, not an estimate, and the other three are exact as a
    GROUP but cannot be separated — nothing in a pre-ledger run distinguishes an explorer call
    from a judge call. They are reported as one `explorer+judge+extraction` bucket rather than
    split by a ratio borrowed from another run, which would be modelling dressed as data.

    `accumulation_summary.cost_usd` is deliberately unused: it looks like it should be
    solver+extraction and is neither (measured $6.99 against a $18.46 solver and a $10.92
    extraction), so nothing here is built on it.
    """
    from daedalus.core.logging.paper_cost import _read_traces

    recs = [load_json(f) or {} for f in sorted((run / "sessions").glob("session_*.json"))]
    recs = [r for r in recs if r]
    known = {str(k) for r in recs
             for k in (r.get("sandbox_id"), (r.get("accumulation_summary") or {}).get("task_id"))
             if k}
    base = lambda t: re.sub(r"_r\d+$", "", re.sub(r"_attempt\d+$", "", str(t)))  # noqa: E731
    solver = sum(float(t.get("total_cost_usd") or 0.0) for t in _read_traces(run)
                 if base(t.get("task_id")) in known)
    total = sum(float(r.get("cost_usd") or 0.0) for r in recs)
    out = {}
    if solver > 0:
        out["solver"] = solver
    if total - solver > 0.005:
        out[COMBINED] = total - solver
    return out


def cost_by_role(run: Path, cals: list | None = None,
                 rest_cals: list | None = None) -> tuple[dict[str, float], float, str]:
    """({role: USD}, bar total, note) on the RAW basis the run browser quotes.

    The ledger's `by_role` is on the ACTUAL basis (the tier really used). The bar total has to
    be the raw $/run the UI shows, so every role is scaled by that run's raw/actual ratio: the
    split is measured, the level is the UI's, and the ratio is assumed uniform across roles.

    A run whose ledger does not cover it is rebuilt from its artifacts (`rebuild_roles`, then
    `split_combined`); anything still unattributed is drawn as an explicit `unattributed`
    remainder rather than a redistributed guess.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    info = resolve_run_cost(run, legacy_total=None)
    actual = float(info.get("total_usd") or 0.0)
    by_role = {k: float(v or 0.0) for k, v in (info.get("by_role") or {}).items()}
    raw, note = raw_run_cost(run)
    if raw is None or not actual:
        return {}, 0.0, note
    k = raw / actual
    # A ledger that only ever saw the post-hoc consolidation call cannot split the run; fall
    # back to what the artifacts themselves support.
    if not {r for r in by_role if by_role[r]} - {"consolidation"}:
        by_role = {**by_role, **rebuild_roles(run)}
        note += "; roles rebuilt from traces + session costs"
        split = split_combined(run, by_role, cals or [], rest_cals or [])
        if split is not by_role:
            by_role = split
            note += "; agent split estimated (see _agent_cost)"
    out = {r: by_role.get(r, 0.0) * k for r in ROLES if by_role.get(r)}
    # Anything the ledger did not attribute, minus the roles deliberately not drawn. Never
    # negative: a by_role that somehow exceeded the total would draw a segment pointing down.
    omitted = sum(by_role.get(r, 0.0) for r in OMITTED) * k
    rest = max(raw - sum(out.values()) - omitted, 0.0)
    if rest > 0.005:
        out[UNATTRIBUTED] = rest
    raw -= omitted
    extra = load_json(run / "usage_migrated.json") or {}
    if extra.get("migration_basis"):
        note += f"; migrated {extra['migration_basis']}"
    return out, raw, note


# The order the BARS stand in, and the panel letter each one carries in the paper. Fixed
# here rather than taken from the command line, so the figure reads the same whatever order
# the runs are passed in. Anything unlisted keeps its given position, after these.
BAR_ORDER = ("No survey, no guidelines", "No survey", "Daedalus")
BAR_TAGS = {"No survey, no guidelines": "(D)", "No survey": "(E)", "Daedalus": "(F)"}
# Bars sit one unit apart, so the gap BETWEEN two of them is 1 - BAR_W. The x limits below
# give the two end gaps that same width, so the bars are evenly spaced across the whole box
# instead of being pushed against the spines by matplotlib's default margins.
BAR_W = 0.62


def plot_cost_by_role(bars: list[dict], path: Path) -> Path:
    """One stacked bar per run, split by the agent that spent it."""
    rank = {lab: i for i, lab in enumerate(BAR_ORDER)}
    bars = sorted(bars, key=lambda b: rank.get(b["label"], len(rank)))
    # 6.2x4.0 / 1.1, local to this figure. `_style` fixes the point sizes, so shrinking the
    # canvas is what makes the labels and the legend larger RELATIVE to the bars.
    fig, ax = plt.subplots(figsize=figsize(6.2 / 1.1, 4.0 / 1.1))
    order = [r for r in ROLES if any(b["roles"].get(r) for b in bars)]
    order += [UNATTRIBUTED] if any(b["roles"].get(UNATTRIBUTED) for b in bars) else []
    xs = range(len(bars))
    bottom = [0.0] * len(bars)
    for i, role in enumerate(order):
        vals = [b["roles"].get(role, 0.0) for b in bars]
        kw = (dict(color="#e8e7e3", edgecolor=INK, linewidth=0.8, hatch="")
              if role == UNATTRIBUTED else bar_kw(i))
        ax.bar(xs, vals, bottom=bottom, width=BAR_W, label=role, **kw)
        # Name a segment in place only when it is tall enough to hold the text.
        for x, v, b0 in zip(xs, vals, bottom):
            if v > 0.06 * max(bb["total"] for bb in bars):
                # On a paper ground, not bare: every fill carries a hatch and the digits
                # disappear into it otherwise.
                ax.text(x, b0 + v / 2, f"${v:,.0f}", ha="center", va="center",
                        fontsize=6.5, color=INK, zorder=5,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4,
                                  alpha=0.90))
        bottom = [b0 + v for b0, v in zip(bottom, vals)]
    for x, b in zip(xs, bars):
        ax.text(x, b["total"], f"${b['total']:,.0f}", ha="center", va="bottom",
                fontsize=7.5, color=INK)
    ax.set_xticks(list(xs))
    # The run's name, then the panel letter it carries in the paper. The name wraps at its
    # comma — at this width "No survey, no guidelines" ran into its neighbour. Every label is
    # TOP-aligned under the axis, so the names line up on their first row and a letter under
    # a two-line name simply sits a row lower.
    ax.set_xticklabels(
        [b["label"].replace(", ", ",\n") + f"\n{BAR_TAGS.get(b['label'], '')}".rstrip()
         for b in bars],
        fontsize=7.5, color=MUTED, va="top")
    # No x name: the tick labels ARE the run names, so naming the axis repeats them.
    ylabel(ax, "Cost")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0f"))
    ax.set_ylim(0, max(b["total"] for b in bars) * 1.14)
    gap = 1.0 - BAR_W
    ax.set_xlim(-(BAR_W / 2 + gap), (len(bars) - 1) + BAR_W / 2 + gap)
    # One row, however many agents there are: a wrapped legend reads as two groupings.
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=len(order),
                    borderaxespad=0)
    # The plotting box takes the LEGEND's width. The legend is one row of fixed text, so its
    # width is set by its content and does not move when the axes does; the axes is resized
    # about its own centre, which is the point the legend is anchored to, so the two stay
    # concentric. Done after everything else is on the figure, because it needs a real draw
    # to measure — and before `finish`, which saves.
    fig.canvas.draw()
    lw = leg.get_window_extent(fig.canvas.get_renderer()).width / (
        fig.get_figwidth() * fig.dpi)
    pos = ax.get_position()
    ax.set_position([pos.x0 + pos.width / 2 - lw / 2, pos.y0, lw, pos.height])
    return finish(ax, path, named_x=False)


# How each run is named on a figure. The folder name is an experiment id, not a label a
# reader should have to decode; anything unlisted keeps its folder name.
DISPLAY = {
    "appworld": "Daedalus",
    "appworld_ablation-E": "No survey",
    "appworld_ablation-D": "No survey, no guidelines",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generation cost by agent across generation runs")
    ap.add_argument("runs", type=Path, nargs="+",
                    help="generation run folders (outputs/daedalus/<name>)")
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/generation-dynamics"),
                    help="output directory")
    args = ap.parse_args()

    # Calibrate the explorer-cost model on whichever runs on this figure DO have a ledger,
    # then use it to split the bucket of any run that does not. Same benchmark, same solver,
    # same prompts, so the constants transfer.
    cals, rest_cals = [], []
    for run in args.runs:
        try:
            from _agent_cost import calibrate, calibrate_rest
            from daedalus.core.logging.paper_cost import UsageAggregator
            done = list(UsageAggregator.from_experiment_dir(run).completed)
            cal, rest = calibrate(run, done), calibrate_rest(run, done)
        except Exception:  # noqa: BLE001 — a missing/unreadable ledger is the normal case
            cal = rest = None
        if cal:
            cals.append(cal)
            print(f"{run.name}: explorer calibration base={cal[0]:,.0f} tok/turn "
                  f"reasoning x{cal[1]:.2f} cache {cal[2]:.0%}")
        if rest:
            rest_cals.append(rest)
            print(f"{' ' * len(run.name)}  judge ${1000*rest[0]:.4f}/k trace-token | "
                  f"extraction ${rest[1]:.4f}/failed attempt")
    byrole = []
    for run in args.runs:
        roles, total, rnote = cost_by_role(run, cals, rest_cals)
        byrole.append({"label": DISPLAY.get(run.name, run.name), "roles": roles,
                       "total": total})
        print(f"{run.name}: ${total:,.2f} raw | by agent: "
              + (", ".join(f"{k} ${v:,.2f}" for k, v in roles.items()) or "none attributable")
              + f"  [{rnote}]")
    if not any(b["roles"] for b in byrole):
        raise SystemExit("No run has an attributable cost — nothing to draw.")
    plot_cost_by_role(byrole, args.out / "generation_cost_by_role.pdf")


if __name__ == "__main__":
    main()
