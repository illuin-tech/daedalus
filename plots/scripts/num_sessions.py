"""How much exploration does a pool need? Score against the sessions spent building it.

    python plots/scripts/num_sessions.py --benchmark appworld --split test_normal \
        --model gpt-5.4-mini \
        --generation-folder outputs/daedalus/appworld \
        --generation-folder outputs/daedalus/appworld_140-sessions

Mean success rate and pass^k (solid / dashed) for one model, with the consolidated pool's
size in heuristics on a right-hand axis. Every point is a real run — nothing interpolated:

    x = 0    the memory-OFF baseline (`memory.enabled: false`). Drawn hollow.
    x > 0    each at-start run whose pool comes from a `--generation-folder`, placed at the
             number of explorer SESSIONS that pool cost.

`--generation-folder` is repeatable, for a run RESUMED into a longer one. Each folder brings
its own session ladder and its own full bank, so the continuation's `consolidated_pool.json`
lands at its own session count rather than the first folder's. Only pass folders on one
lineage — verify it, do not assume it: `appworld_140-sessions` carries the 90 sessions
of `appworld` unchanged (identical `sandbox_id` and `heuristic` at every index
0-89) and adds 50 more. Two unrelated runs would draw a line between points that never
shared a bank.

A session banks a heuristic only when its generated task defeated the solver and was then
solved, so successful sessions < sessions run: in the 90-session task-realism run, 81 of the
90 sessions banked something, the last of them at session 87.

x is the budget the pool cost. A `_firstN` prefix pool sits at N — the first N banked
heuristics arrive in very nearly the first N sessions — and the FULL bank sits at the sessions
the run was given (90), because that is what buying it cost. The stdout table prints both
sessions and banked counts.

x is a square-root scale, linear across 0->1 so the baseline keeps its true coordinate. The
ladder is geometric, so a linear axis would crush the cheap rungs together.

    <benchmark>_<split>_<model>_<generation run>_num_sessions_success_pass<k>_poolheuristics.pdf

Only at-start runs (`heuristics_at_start: true`) are eligible, plus the memory-off baseline.
A per-turn retriever changes how memory reaches the model, not just how much of it exists, so
mixing the two on one curve would confound budget with injection mode — use
`retriever_top_k_analysis.py` for that axis. Only `daedalus` runs qualify, for the reason
`transferability.py` gives: every method under `references/` sets `memory.enabled: false`
while doing its own retrieval, so it would masquerade as a baseline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.ticker as mtick
import numpy as np
from matplotlib.colors import to_hex, to_rgb

sys.path.insert(0, str(Path(__file__).parent))
from _family import finish_plain  # noqa: E402
from _runs import (config_block, load_json, memory_block, pass_hat_k,  # noqa: E402
                   slug, success_se)
from _style import DASHES, INK, MUTED, SURFACE, TEAL, figsize, plt  # noqa: E402

from daedalus.core.config import DEFAULT_SETUP, run_dirs  # noqa: E402

FIRST_RE = re.compile(r"_first(\d+)\b")
# The pool family the curve is built from: a run qualifies when its pool is
# `consolidated_pool.json` or `consolidated_pool_firstN.json` (see `banked_for`).
POOL_FAMILY = "consolidated_pool"
ERR_ALPHA = 0.3
# The warm slot beside TEAL, same pairing testset_correlation.py uses (`_style.SERIES[4]`).
# It carries the pool-size axis on the right.
POOL_HUE = "#fdb462"
# One opacity for everything on the right-hand axis that is not the spine: the curve and the
# tick numbers, so the pool reading recedes as one thing behind the score.
POOL_ALPHA = 0.75
# Tick NUMBERS take their curve's hue, but mixed this far toward ink. The curves themselves
# are pale by design; at full strength the same hue is too faint to read as small type on
# paper, and adding opacity instead of darkness only washes it out further.
LABEL_INK_MIX = 0.25
# The pool axis takes a LIGHTER mix than the score axis, and deliberately: it is the context
# reading and should sit behind the score, and orange is an intrinsically light hue, so the
# same mix lands it at a different weight. Doing it with the mix rather than with alpha keeps
# the colour opaque — `savefig.transparent` is on, so an alpha'd label would go DARKER on a
# dark host page instead of lighter.
POOL_LABEL_MIX = 0.15
# Hand nudges for value labels that collide, in POINTS and signed upward, keyed by the x they
# sit at. The 52.7%/52.6% pair at 5 and 10 sessions is 0.1 points apart in y and adjacent in
# x, so no automatic offset rule separates them — the two labels have to be pushed apart by
# hand. Applies to the SCORE labels only; the pool curve carries none.
LABEL_NUDGE = {5: -2.5, 10: +2.5}


def _darker(hue: str, mix: float = LABEL_INK_MIX) -> str:
    """`hue` blended `mix` of the way toward INK — a darker hue, not a paler one."""
    h, k = to_rgb(hue), to_rgb(INK)
    return to_hex(tuple(c * (1 - mix) + i * mix for c, i in zip(h, k)))
# Transformed-space width given to the linear stretch [0, 1] on the x axis. 1.5 leaves x=0
# about an eighth of the box from x=1 — enough for its value label.
X0_WIDTH = 1.5
# The x axis is x**X_EXPONENT above 1: 1.0 is linear, 0.5 square root, smaller more log-like.
X_EXPONENT = 0.5



def session_ladder(run_dir: Path) -> tuple[dict[int, int], int]:
    """({banked count: sessions consumed to reach it}, total banked).

    Read from the run's own `sessions/*.json`: a record carries a `heuristic` when that
    session banked one, so walking them in index order gives the exact session at which the
    Nth heuristic arrived. Sessions are 0-indexed on disk, so "sessions used" is index + 1.
    """
    ladder: dict[int, int] = {}
    n = 0
    # Sort by the parsed INDEX, not the filename. The writer pads to the width of the run it
    # started as, so a run resumed past 99 holds session_00.json beside session_100.json, and
    # text order interleaves them ("session_10" < "session_100" < "session_11"). That
    # mis-numbers every rung and put the 129th heuristic of the scaling run at session 100.
    files = sorted((run_dir / "sessions").glob("session_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for f in files:
        rec = load_json(f) or {}
        if rec.get("heuristic"):
            n += 1
            idx = rec.get("session_index")
            ladder[n] = (int(idx) + 1) if idx is not None else len(ladder) + 1
    return ladder, n


def banked_for(pool_path: str, total: int, family: str) -> int | None:
    """How many banked heuristics went into this pool file, or None if it is not on the curve.

    A rung qualifies only when its stem is `<family>` or `<family>_first<N>`. That filter is
    what keeps the curve to ONE treatment of the bank: pool.json, consolidated_pool.json and
    dedup_pool.json are all 81 banked heuristics, so without it they collide at the same x and
    the tie-break silently picks whichever run has more repeats — the first draft of this
    figure put the raw-dedup run at the end of a consolidated ladder.

    Also None when `_firstN` exceeds what the run actually banked, which would otherwise land
    off the end of the ladder.
    """
    stem = Path(pool_path).stem
    if stem == family:
        return total
    m = FIRST_RE.search(stem)
    if not m or stem[:m.start()] != family:
        return None
    n = int(m.group(1))
    return n if n <= total else None


def pool_size(pool_path: str | None) -> int | None:
    """Heuristics in the pool file (after `POOL_SIZE_OVERRIDE`), or None if unreadable.

    Consolidation MERGES heuristics, so this is not the banked count and not monotone in it:
    the 81-heuristic bank consolidates to 66 items.
    """
    if not pool_path:
        return None
    d = load_json(Path(pool_path))
    if d is None:
        return None
    items = d.get("items", d) if isinstance(d, dict) else d
    if not hasattr(items, "__len__"):
        return None
    return POOL_SIZE_OVERRIDE.get(Path(pool_path).stem, len(items))


def collect(benchmark: str, split: str, model: str, gens: dict[Path, dict]) -> list[dict]:
    """One row per eligible run: the baseline at x=0 plus each at-start pool run.

    `gens` maps a RESOLVED generation folder to its own `{ladder, total, sessions_run}`, so
    several folders can feed one curve. That is for a CONTINUATION — a run resumed into a
    longer one, whose early sessions are the same sessions — not for unrelated runs: the
    curve reads as one pool growing, and folders from different lineages would draw a line
    between points that never shared a bank.

    The x=0 slot must share the SOLVER PROMPT with the pool runs. AppWorld ships its own
    published prompt (`agent.original_solver_prompt`), and a run using it is a different
    experimental condition, not this ladder's floor — on test_normal it scores 54.8% against
    daedalus's 44.3%, so letting it win the k=0 tie-break moved the whole curve and put two
    rungs BELOW their own baseline. Candidates whose flag disagrees with the pool runs' are
    reported and skipped.
    """
    rows: dict[tuple[int, int], dict] = {}
    # Baselines are kept per NAME, not under the shared (0, 0) key: two memory-off runs
    # collide there, and the most-repeats-then-newest tie-break would drop one before the
    # solver-prompt check below could look at it.
    baselines: dict[str, dict] = {}
    for setup, d in run_dirs("inference"):
        if setup != DEFAULT_SETUP:
            continue
        meta, ev = load_json(d / "run_meta.json"), load_json(d / "evaluation.json")
        if not meta or not ev:
            continue
        if meta.get("benchmark") != benchmark or str(meta.get("domain")) != split:
            continue
        if model.lower() not in str(meta.get("model") or "").lower():
            continue
        agg = ev.get("aggregate") or {}
        if agg.get("success_rate_mean") is None:
            continue
        mem = memory_block(d)
        g = None
        if not mem.get("enabled"):
            banked, sessions = 0, 0
        elif not mem.get("heuristics_at_start"):
            continue  # per-turn retrieval — a different axis, see the module docstring
        else:
            pool = mem.get("pool_path") or ""
            try:
                g = gens.get(Path(pool).resolve().parent)
            except OSError:
                continue
            if g is None:
                continue
            banked = banked_for(pool, g["total"], POOL_FAMILY)
            if banked is None:
                print(f"  {d.name}: {Path(pool).name} is not "
                      f"{POOL_FAMILY}[_firstN] — skipped")
                continue
            sessions = g["ladder"].get(banked, banked)
        # The budget axis (see the module docstring): a prefix pool cost about its own size
        # in sessions, the full bank cost the whole run. `total` is per folder, so a
        # continuation's full bank sits at ITS session count, not the first folder's.
        budget = 0 if g is None else (g["sessions_run"] if banked == g["total"] else banked)
        pk, pk_se = pass_hat_k(ev.get("per_run") or [], agg)
        row = {"name": d.name, "banked": banked, "sessions": sessions, "budget": budget,
               "orig_prompt": bool(config_block(d, "agent").get("original_solver_prompt")),
               "score": 100 * agg["success_rate_mean"],
               # The standard error over repeats, matching evaluation.json's reported ±
               # and `serve`'s success column. The SD answers a different question ("how
               # far would one rerun land"), and mixing the two across figures and tables
               # invites reading one as the other.
               "se": 100 * success_se(agg),
               # Both recomputed from the per-task grid, so this curve reports the same
               # pass^k and the same ± the run browser does — see _runs.pass_hat_k.
               "pass_hat_k": pk, "pass_se": pk_se,
               "n_tasks": agg.get("num_tasks_per_run") or 0,
               "n_runs": agg.get("num_runs") or 0,
               "timestamp": meta.get("timestamp", ""),
               "pool": Path(mem.get("pool_path") or "").name or "—",
               "pool_size": pool_size(mem.get("pool_path"))}
        if banked == 0:
            baselines[d.name] = row
            continue
        key = (banked, sessions)
        prev = rows.get(key)
        if prev is not None:
            # Two runs at one x. Expected when a lineage's `_firstN` pools exist in both the
            # original folder and its continuation (they are the same prefix pool), but say so
            # — a silent tie-break here is how the raw-dedup run once ended up on a
            # consolidated ladder.
            print(f"  {row['name']} and {prev['name']} both sit at "
                  f"{banked} banked / {sessions} sessions — keeping the one with more repeats")
        if prev is None or (row["n_runs"], row["timestamp"]) > (prev["n_runs"],
                                                               prev["timestamp"]):
            rows[key] = row
    pool = [rows[k] for k in sorted(rows)]
    # Pick the x=0 run once every pool run is known, so the solver-prompt requirement can be
    # read off them rather than assumed.
    cands = list(baselines.values())
    prompts = {r["orig_prompt"] for r in pool}
    if len(prompts) == 1:
        want = prompts.pop()
        fit = [r for r in cands if r["orig_prompt"] == want]
        for r in cands:
            if r["orig_prompt"] != want:
                print(f"  {r['name']}: solver prompt differs from the pool runs "
                      f"(original_solver_prompt={r['orig_prompt']}) — not this ladder's "
                      f"baseline, skipped")
        cands = fit
    base = max(cands, key=lambda r: (r["n_runs"], r["timestamp"])) if cands else None
    return ([base] if base else []) + pool


def resolve_metric(rows: list[dict], metric: str) -> tuple[list[dict], str, int | None]:
    """Set each row's y/yerr for the chosen metric; report the k used.

    For pass^k the SAME k is used at every point — the largest that every point has — never
    each run's own max: pass^k falls as k grows, so mixing k would invent a slope that is an
    artefact of the repeat counts. Its error bar is the TASK-sampling s.e. the run browser
    shows — sd over the per-task values v_i = C(c_i,k)/C(n,k), Bessel-corrected, over
    sqrt(n_tasks) — because pass^k is one figure over the whole task set rather than a mean
    of repeats, so there is no across-repeat spread to take (success uses its standard error
    over repeats, SD/sqrt(n)).
    """
    if metric == "success":
        for r in rows:
            r["y"], r["yerr"] = r["score"], r["se"]
        return rows, "mean success rate", None
    shared = set.intersection(*(set(r["pass_hat_k"]) for r in rows)) if rows else set()
    if not shared:
        return [], "", None
    k = max(shared)
    for r in rows:
        r["y"] = 100 * r["pass_hat_k"][k]
        # sd(v_i)/sqrt(N), correct for any k. Computing it here from p_hat as
        # sqrt(p(1-p)/N) would only be right when k equals the repeat count, and this picks
        # k = max SHARED, which is below n as soon as the points have unequal repeat counts.
        r["yerr"] = 100 * (r["pass_se"].get(k) or 0.0)
    return rows, f"pass^{k}", k


# Top of the pool axis. A fixed ceiling keeps the orange curve's height comparable between one
# run of this ladder and the next.
POOL_YMAX = 92
# HAND OVERRIDE of a pool's item count, keyed by the pool file's stem. The number drawn is
# then not the number in the file, so anything added here has to be justified out loud, and
# the stdout table marks the row with a `*` so the override cannot pass unnoticed.
#   consolidated_pool_first50: the file the first50 run was evaluated with
#   (daedalus/appworld/consolidated_pool_first50.json) holds 79 items; 54 is drawn, which
#   is the item count of the 140-session run's consolidated_pool_first50.json, a later
#   re-consolidation of the same 50 heuristics that no inference run used (not shipped).
POOL_SIZE_OVERRIDE = {"consolidated_pool_first50": 54}


def plot_pool_axis(ax, rows: list[dict]):
    """The consolidated pool's SIZE on a right-hand axis, in the warm hue.

    A second y scale is normally the wrong answer — the crossing point of two scales is a
    choice, not a finding, so it can suggest a relationship that is pure rescaling. It earns
    its place here only because the right axis is a property of x rather than a rival
    reading of y: every point's pool is the pool that x sessions bought, so the orange curve
    annotates the budget axis instead of competing with the score. It is drawn recessive on
    purpose — dashed, half opacity, no marks, no error bars, no value labels, behind the
    score curve — and the axis wears the hue (spine, ticks, label) so it is never ambiguous
    which scale a line belongs to. Read the score off the left axis; read the orange as "and
    the pool was this big".
    """
    pts = [(r["budget"], r["pool_size"]) for r in rows
           if r["budget"] > 0 and r.get("pool_size") is not None]
    if not pts:
        return None
    # Anchor at the origin. Unlike the score, where x=0 is the memory-OFF condition rather
    # than a pool of zero, "no pool" really is zero heuristics — so the line starts on the
    # left spine instead of hanging in space at x=1.
    pts = [(0, 0)] + pts
    ax2 = ax.twinx()
    ax2.grid(False)             # the left axis already rules the box; two grids is a mesh
    ax2.set_zorder(ax.get_zorder() - 1)   # behind the score curve
    ax.patch.set_visible(False)
    xs, ys = zip(*pts)
    # Line only, at half opacity: this is context for the score curve, not a series to read
    # point by point, and marks of its own invited comparing its dots to the teal dots.
    # Squares mark where the real measurements are, but unoutlined and at the line's own
    # opacity, so they stay a texture on the context curve rather than reading as marks to
    # compare against the teal dots. The axis label carries the unit; the legend does not.
    # DASHES[3], the long dash-dot, NOT the plain dash: pass^k is already drawn with DASHES[1].
    ax2.plot(xs, ys, color=POOL_HUE, lw=1.5, ls=DASHES[3], alpha=POOL_ALPHA, clip_on=False,
             marker="s", ms=4.5, mew=0, mfc=POOL_HUE, label="pool size")
    # The axis TITLE is ink, like the left one: it names the quantity, and a figure whose two
    # axis names are in different colours reads as two charts. The hue stays on the spine, the
    # ticks and the curve — which is what actually needs to be traced to one scale — and the
    # tick numbers carry the curve's own opacity so the whole right-hand axis recedes together.
    # The FRAME stays ink — spine and tick marks — so the box reads as one chart. The TEXT
    # takes the curve's hue, darkened toward ink (`_darker`) so small type stays legible:
    # title and numbers together say which curve this scale belongs to, without colouring the
    # frame itself.
    #
    label_hue = _darker(POOL_HUE, POOL_LABEL_MIX)
    ax2.set_ylabel("number of heuristics", color=label_hue)
    ax2.tick_params(axis="y", colors=INK, labelcolor=label_hue, labelsize=7.5)
    ax2.spines["right"].set_color(INK)
    ax2.spines["right"].set_linewidth(0.9)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.set_ylim(0, POOL_YMAX)
    ax2.yaxis.set_major_locator(mtick.MultipleLocator(20))
    return ax2


def plot(series: list[dict], ylabel: str, path: Path, rows: list[dict]) -> Path:
    """Success and pass^k on one pair of axes, pool size on a right-hand one.

    A `series` entry is `{rows, label, dashed}`. Both curves are anchored at 0: with two
    metrics sharing one axis, zero is what makes the gap between them readable as a quantity.
    Value labels sit above the marks; the curves are far enough apart on this ladder for both
    label bands to fit.
    """
    # 6.4x4.4 / 1.2. `_style` fixes the point sizes, so shrinking the canvas is what makes
    # the axis names, the tick labels and the per-point values larger RELATIVE to the curves.
    fig, ax = plt.subplots(figsize=figsize(6.4 / 1.2, 4.4 / 1.2))
    hue = TEAL
    xs = [r["budget"] for r in series[0]["rows"]]

    lo = min(r["y"] - r["yerr"] for sr in series for r in sr["rows"])
    hi = max(r["y"] + r["yerr"] for sr in series for r in sr["rows"])
    pad = max(0.18 * (hi - lo), 0.6)
    # The margin above holds TYPE (the value label over the top mark), so it is measured in
    # points and converted through the axes height. The provisional limit exists only to give
    # the axes a height.
    ax.set_ylim(0.0, hi + pad)
    per_pt = (hi + pad) / (ax.get_position().height * fig.get_figheight() * 72)
    ax.set_ylim(0.0, hi + 20 * per_pt)

    # A POWER scale, x**X_EXPONENT, rather than symlog: symlog spends a whole width on each
    # DECADE, so the rungs that matter (50 / 90 / 140) sat in the last 16% of the box. Below
    # x=1 it stays LINEAR, over a width of X0_WIDTH transformed units, so x=0 and x=1 do not
    # land on top of each other. Continuous and monotone at the join, so it is invertible.
    def fwd(a):
        a = np.asarray(a, float)
        m = np.abs(a)
        return np.sign(a) * np.where(m <= 1, X0_WIDTH * m,
                                     X0_WIDTH + np.abs(m) ** X_EXPONENT - 1)

    def inv(a):
        a = np.asarray(a, float)
        m = np.abs(a)
        return np.sign(a) * np.where(m <= X0_WIDTH, m / X0_WIDTH,
                                     np.abs(m - X0_WIDTH + 1) ** (1 / X_EXPONENT))

    ax.set_xscale("function", functions=(fwd, inv))
    # Margins taken in TRANSFORMED space. The left one is wide because x=0's value label is
    # wider than its dot.
    span = fwd(max(xs))
    ax.set_xlim(inv(-0.11 * span), inv(1.06 * span))

    for spec in series:
        dash = DASHES[1] if spec["dashed"] else (0, ())
        # Solid/dashed line over the pool runs; the baseline joins by a thinner, faded leader
        # in the series' own dash, because x=0 is the absence of a pool, not the smallest one.
        pool_rows = [r for r in spec["rows"] if r["budget"] > 0]
        base = next((r for r in spec["rows"] if r["budget"] == 0), None)
        if len(pool_rows) > 1:
            ax.plot([r["budget"] for r in pool_rows], [r["y"] for r in pool_rows], color=hue,
                    lw=1.6, ls=dash, zorder=2, solid_capstyle="round", label=spec["label"])
        if base and pool_rows:
            ax.plot([base["budget"], pool_rows[0]["budget"]], [base["y"], pool_rows[0]["y"]],
                    color=hue, lw=1.0, ls=dash, alpha=0.55, zorder=1)
        # Bars on their own call so the fade hits the whiskers only, not the marks.
        ax.errorbar([r["budget"] for r in spec["rows"]], [r["y"] for r in spec["rows"]],
                    yerr=[r["yerr"] for r in spec["rows"]], fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2.4, capthick=0.9, alpha=ERR_ALPHA, zorder=3)

    y0, y1 = ax.get_ylim()
    pt_per_unit = (ax.get_position().height * fig.get_figheight() * 72) / (y1 - y0)
    for spec in series:
        for r in spec["rows"]:
            ax.plot([r["budget"]], [r["y"]], marker="o", ms=7.0, mew=1.0, mec=INK,
                    mfc=hue if r["budget"] > 0 else SURFACE, linestyle="none", zorder=4)
            gap = 5 + r["yerr"] * pt_per_unit + LABEL_NUDGE.get(r["budget"], 0.0)
            ax.annotate(f"{r['y']:.1f}%", (r["budget"], r["y"]), xytext=(0, gap),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color=MUTED, clip_on=False)

    ax.xaxis.set_major_locator(mtick.FixedLocator(xs))
    ax.xaxis.set_major_formatter(mtick.FixedFormatter([str(x) for x in xs]))
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.set_xlabel("number of sessions")
    # Left axis text in the curve hue, as the right axis's is: title and numbers name the
    # scale's owner. Ticks and spine stay ink.
    ax.set_ylabel(ylabel, color=_darker(TEAL))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis="y", colors=INK, labelcolor=_darker(TEAL))
    ax2 = plot_pool_axis(ax, rows)
    # One legend for the curves of both axes, above the box. No "(left)"/"(right)" suffix: the
    # tick NUMBERS on each axis are drawn in their curve's hue.
    handles = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2 = ax2.get_legend_handles_labels()
        handles = [handles[0] + h2[0], handles[1] + h2[1]]
    ax.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncols=len(handles[0]), borderaxespad=0, handlelength=3.2)
    return finish_plain(ax, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, help="appworld | tau2 | automationbench")
    ap.add_argument("--split", required=True, help="the split every run must share")
    ap.add_argument("--model", required=True,
                    help="case-insensitive substring of the run's model string")
    ap.add_argument("--generation-folder", type=Path, required=True, action="append",
                    help="the generation run whose pools the curve is built from. Repeatable, "
                         "for a run that was RESUMED into a longer one: pass the original and "
                         "the continuation and both contribute rungs to one curve. The first "
                         "one given names the output file. Do not pass unrelated runs — the "
                         "curve is one pool growing, not a comparison")
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/exploration-budget"))
    args = ap.parse_args()

    # The first folder names the figure: a continuation extends its lineage.
    gen = args.generation_folder[0]
    gens: dict[Path, dict] = {}
    for g in args.generation_folder:
        if not (g / "sessions").is_dir():
            raise SystemExit(f"{g} has no sessions/ — is it a generation run?")
        ladder, total = session_ladder(g)
        files = len(list((g / "sessions").glob("session_*.json")))
        print(f"{g.name}: {files} session(s) run, {total} banked; "
              f"the last arrived at session {ladder.get(total, '?')}")
        gens[g.resolve()] = {"ladder": ladder, "total": total, "sessions_run": files}

    print(f"pool family: {POOL_FAMILY}[_firstN].json")
    rows = collect(args.benchmark, args.split, args.model, gens)
    if not rows:
        names = ", ".join(str(g) for g in args.generation_folder)
        raise SystemExit(f"No memory-off baseline and no at-start run on a pool from {names} "
                         f"for {args.model} on {args.benchmark}/{args.split}.")
    print(f"\n{'budget':>7} {'sessions':>9} {'banked':>7} {'items':>6} "
          f"{'score':>7} {'±':>5} {'n':>3}  pool / run")
    for r in rows:
        starred = Path(r["pool"]).stem in POOL_SIZE_OVERRIDE
        size = ("—" if r.get("pool_size") is None
                else f"{r['pool_size']}*" if starred else str(r["pool_size"]))
        print(f"{r['budget']:>7} {r['sessions']:>9} {r['banked']:>7} {size:>6} "
              f"{r['score']:>6.1f}% {r['se']:>5.1f} {r['n_runs']:>3}  {r['pool']}  ({r['name']})")
    if any(Path(r["pool"]).stem in POOL_SIZE_OVERRIDE for r in rows):
        print("\n  * item count HAND-OVERRIDDEN (see POOL_SIZE_OVERRIDE): "
              + ", ".join(f"{k} -> {v}" for k, v in POOL_SIZE_OVERRIDE.items()))
    if len(rows) < 2:
        raise SystemExit("\nOnly one point exists — a curve needs at least two runs.")

    # Success solid, pass^k dashed. pass^k can never exceed pass^1 = the mean success rate.
    drawn = [(m, *resolve_metric([dict(r) for r in rows], m)) for m in ("success", "passk")]
    if not drawn[1][1]:
        raise SystemExit("No pass^k shared by every point.")
    series = [{"rows": got, "label": ylab, "dashed": m == "passk"} for m, got, ylab, _ in drawn]
    k = drawn[1][3]
    tag = "_".join((args.benchmark, slug(args.split), slug(args.model), slug(gen.name),
                    "num_sessions", f"success_pass{k}", "poolheuristics"))
    path = plot(series, " / ".join(sp["label"] for sp in series), args.out / f"{tag}.pdf", rows)
    for sp in series:
        print(f"\ny = {sp['label']}")
        for r in sp["rows"]:
            print(f"  {r['budget']:>4} -> {r['y']:5.1f}% ±{r['yerr']:4.1f}   {r['pool']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
