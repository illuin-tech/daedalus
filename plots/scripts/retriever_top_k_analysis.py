"""How much retrieval does a pool actually need? Score against the retriever's top-k.

    python plots/scripts/retriever_top_k_analysis.py \
        --benchmark appworld --split test_normal --model gpt-5.4-mini --retriever bm25 \
        --pool outputs/daedalus/appworld/consolidated_pool.json

One curve for one model on one pool, sweeping the retriever's `top_k`. Two kinds of point,
both real runs — nothing is interpolated:

    k = 0          the memory-OFF baseline (`memory.enabled: false`). Drawn hollow.
    k = 1..        every scored run using `--retriever` on this pool, placed at its own
                   `memory.retriever.top_k`.

Deliberately NOT points on this curve: the whole-pool-every-turn arm (`all_every_turn`) and
the at-start arm (`heuristics_at_start`, drawn only as a reference line, see below). Both
would sit at k = pool size. The per-turn top-k retrievers PERSIST what they surface, so the model's
context grows as the task runs; `all_every_turn` is EPHEMERAL — the pool is re-emitted into
each turn's prompt and never kept — and at-start pins the pool into the system prompt once.
Plotting them as "k = N" would put three different injection mechanisms on one axis and
invite reading the gap as an effect of k.

x is symlog with linthresh=1: linear across 0->1 so the baseline sits at its TRUE coordinate,
logarithmic above it where the k values live. A pure log axis cannot hold k=0 at all, and
faking it (0.5, or a reserved slot) puts a decorative number on a metric axis.

    <benchmark>_<split>_<model>_<pool>_topk_<retriever>_success.pdf    mean success rate
    <benchmark>_<split>_<model>_<pool>_topk_<retriever>_pass<N>.pdf    pass^N, N = --runs

The whole-pool-at-start run on the same pool, when one exists, is drawn as a dotted
horizontal reference line.

Reading it: the per-turn top-k retrievers PERSIST what they surface (dedup by text, reset per
task), so by turn ~15 a large k has injected most of the pool anyway and the curve flattens
toward the k=N point. The sweep measures how fast memory accumulates, not how much is present
at any one moment — see the "Memory injection" section of the README.

Runs are matched on the `memory:` block of each run's own `config.yaml`, never on the run
name, and only `daedalus` runs are eligible: every method under `references/` sets
`memory.enabled: false` while doing its own retrieval, so it would masquerade as a baseline.

The k=0 point must also share the SOLVER with the swept points. `agent.original_solver_prompt`
swaps daedalus's prompt for AppWorld's own two-shot one and is worth ~10 points on its own, so
a memory-off run using it is a floor for a different curve. Both baselines classify as k=0 on
the memory block alone; the prompt style breaks the tie, before repeats or recency, or the
sweep gets a floor 10 points above its own points and every low k reads as a regression.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).parent))
from _family import finish_plain  # noqa: E402
from _runs import (is_original_prompt, load_json, memory_block, pass_hat_k,  # noqa: E402
                   same_pool, slug)
from _style import INK, MUTED, SERIES, SURFACE, TEAL, figsize, plt  # noqa: E402

from daedalus.core.config import DEFAULT_SETUP, run_dirs  # noqa: E402

SHORT: list[tuple[str, int]] = []   # runs with fewer repeats than the cap
ERR_ALPHA = 0.3                     # error bars sit behind the marks, deliberately faint


def pool_size(pool: str) -> int:
    """Number of heuristics in the pool — the x coordinate of the all-at-turn point."""
    data = load_json(Path(pool))
    if data is None:
        raise SystemExit(f"cannot read pool {pool}")
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise SystemExit(f"{pool}: expected a list of items (or {{'items': [...]}})")
    return len(items)


def metrics(per_run: list[dict], cap: int) -> dict | None:
    """Both metrics recomputed from the FIRST `cap` repeats, over the tasks all of them share.

    Why not read `aggregate`: the runs on one figure have different repeat counts (the
    baseline and the at-turn arm have 5, a fresh sweep point has 3), and both metrics move
    with that count. `success_rate_std` shrinks as repeats grow, and pass^k is
    `C(c,k)/C(n,k)` — pass^3 out of 5 repeats is a different, tighter estimator than pass^3
    out of 3. Comparing them across points would show an artefact of the repeat count. So
    every point is cut back to the same `cap`, which makes the k=3-repeat run the ceiling.

    Errors are deliberately different per metric, because the metrics are:
      success  spread ACROSS the repeats (sample stdev of the per-repeat rates) — the
               quantity `success_rate_std` reports, just restricted to `cap` repeats.
      pass^k   one figure over the whole task set, not a mean of repeats, so there is no
               across-repeat spread to take. The bar is the TASK-sampling s.e. the run
               browser reports — `pass_hat_k_with_se` over the capped repeats, i.e. the
               Bessel-corrected sd of the per-task values over sqrt(n_tasks).
    """
    succ = [r.get("task_success") or {} for r in per_run[:cap]]
    if len(succ) < cap or not all(succ):
        return None
    common = sorted(set.intersection(*(set(s) for s in succ)))
    if not common:
        return None
    n = len(succ)
    rates = [sum(1 for t in common if s[t]) / len(common) for s in succ]
    # pass^n over n repeats: C(c,n)/C(n,n) = 1 only when every repeat solved the task. Taken
    # from the scorer rather than reimplemented, so the value AND the ± are the ones the run
    # browser shows — its estimator is Bessel-corrected (N-1), which a local sd is easy to
    # get wrong by one term.
    got = pass_hat_k(per_run[:cap], None)
    return {
        "n_used": n, "n_tasks": len(common),
        "success": 100 * st.mean(rates),
        "success_err": 100 * (st.stdev(rates) if len(rates) > 1 else 0.0),
        "passk": 100 * got[0][n],
        "passk_err": 100 * (got[1].get(n) or 0.0),
    }


def classify(mem: dict, retriever: str, pool: str) -> int | None:
    """The k this run sits at, or None if it does not belong on this figure.

    Only two things qualify: memory switched off outright (k=0), and `--retriever` running
    per turn on this pool (k = its top_k). `heuristics_at_start` and `all_every_turn` are
    rejected here — see the module docstring for why they are not "k = pool size". So is a
    reason-then-retrieve run (`retrieval_mode: reason_then_retrieve`, R2R): same retriever and
    k, but a different query, so it would otherwise win the tie-break at its k.
    """
    if not mem.get("enabled"):
        return 0 if not mem.get("pool_path") else None
    if mem.get("heuristics_at_start"):
        return None
    if mem.get("retrieval_mode", "pre_generation") != "pre_generation":
        return None
    if not same_pool(mem.get("pool_path"), pool):
        return None
    r = mem.get("retriever") or {}
    if r.get("type") != retriever:
        return None
    k = r.get("top_k")
    return int(k) if k else None


def collect(benchmark: str, split: str, model: str, retriever: str, pool: str,
            cap: int) -> dict[int, dict]:
    """{k: row} for this model on this split, every point cut back to `cap` repeats.

    A run with fewer than `cap` repeats cannot be placed and is reported, not silently
    dropped. Ties on one k go to the run with more repeats, then the newest.
    """
    found: dict[int, dict] = {}
    cands: dict[int, list[dict]] = {}
    for setup, d in run_dirs("inference"):
        if setup != DEFAULT_SETUP:
            continue
        meta, ev = load_json(d / "run_meta.json"), load_json(d / "evaluation.json")
        if not meta or not ev:
            continue
        if meta.get("benchmark") != benchmark or str(meta.get("domain")) != split:
            continue
        full_model = str(meta.get("model") or "")
        if model.lower() not in full_model.lower():
            continue
        agg = ev.get("aggregate") or {}
        if agg.get("success_rate_mean") is None:
            continue
        k = classify(memory_block(d), retriever, pool)
        if k is None:
            continue
        m = metrics(ev.get("per_run") or [], cap)
        if m is None:
            SHORT.append((d.name, agg.get("num_runs") or 0))
            continue
        row = {
            "k": k, "name": d.name, "model": full_model,
            "n_runs": agg.get("num_runs") or 0,
            "timestamp": meta.get("timestamp", ""),
            "original_prompt": is_original_prompt(d), **m,
        }
        cands.setdefault(k, []).append(row)

    # The sweep decides which solver prompt this figure is about; the k=0 candidates are
    # then filtered to match it. Taken from the swept points rather than from a flag so the
    # figure stays correct for a sweep deliberately run on the original prompt.
    swept = [r for k, rs in cands.items() if k > 0 for r in rs]
    sweep_prompt = bool(swept and sum(r["original_prompt"] for r in swept) * 2 > len(swept))
    for k, rs in cands.items():
        found[k] = max(rs, key=lambda r: (r["original_prompt"] == sweep_prompt,
                                          r["n_runs"], r["timestamp"]))
    return found


def at_start_arm(benchmark: str, split: str, model: str, pool: str, cap: int) -> dict | None:
    """The whole-pool-at-start run on THIS pool, as a reference level for the sweep.

    Deliberately not a point on the curve (see the module docstring — it is a different
    injection mechanism, not "k = pool size"), but it is the level the sweep is climbing
    toward, so it is drawn as a horizontal reference. Matched on the memory block like
    everything else here, never on the run name.
    """
    best = None
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
        mem = memory_block(d)
        if not (mem.get("enabled") and mem.get("heuristics_at_start")):
            continue
        if not same_pool(mem.get("pool_path"), pool):
            continue
        m = metrics(ev.get("per_run") or [], cap)
        if m is None:
            continue
        row = {"name": d.name, "original_prompt": is_original_prompt(d),
               "n_runs": (ev.get("aggregate") or {}).get("num_runs") or 0,
               "timestamp": meta.get("timestamp", ""), **m}
        if best is None or (row["n_runs"], row["timestamp"]) > (best["n_runs"],
                                                               best["timestamp"]):
            best = row
    return best


def plot(rows: list[dict], n_items: int, retriever: str, metric: str, ylabel: str,
         path: Path, n_runs: int = 3, ref: tuple[float, str] | None = None,
         title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=figsize(6.4, 4.4))
    hue = TEAL
    err_key = f"{metric}_err"

    ks = [r["k"] for r in rows]
    ys = [r[metric] for r in rows]
    # Leave room for the bars, or a tall one gets clipped at the axis edge.
    lo = min(r[metric] - r[err_key] for r in rows)
    hi = max(r[metric] + r[err_key] for r in rows)

    # Symlog: linear on [0, 1] so k=0 is at its real coordinate, log above.
    ax.set_xscale("symlog", linthresh=1, linscale=0.6)
    ax.set_xlim(-0.12, max(ks) * 1.7)
    pad = max(0.18 * (hi - lo), 0.6)
    ax.set_ylim(lo - pad, hi + pad)

    # The sweep as one curve, with the baseline joined by a dashed leader: k=0 is a
    # different injection mode (none at all), so the solid line is reserved for the
    # retriever's own points.
    swept = [r for r in rows if r["k"] > 0]
    base = next((r for r in rows if r["k"] == 0), None)
    if len(swept) > 1:
        ax.plot([r["k"] for r in swept], [r[metric] for r in swept],
                color=hue, lw=1.6, zorder=2, solid_capstyle="round")
    if base and swept:
        ax.plot([base["k"], swept[0]["k"]], [base[metric], swept[0][metric]],
                color=hue, lw=1.0, ls=(0, (4, 3)), alpha=0.55, zorder=1)

    # Bars first and on their own call, so the fade applies to the whiskers only — putting
    # alpha on a combined errorbar() would wash out the mark as well.
    ax.errorbar(ks, ys, yerr=[r[err_key] for r in rows], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=2.4, capthick=0.9, alpha=ERR_ALPHA, zorder=3)
    # Points-per-y-unit, so a label can be pushed clear of its own whisker instead of
    # landing on it: axes height in inches -> points, divided by the data range.
    y0, y1 = ax.get_ylim()
    pt_per_unit = (ax.get_position().height * fig.get_figheight() * 72) / (y1 - y0)
    for r in rows:
        filled = r["k"] > 0      # hollow = no memory, filled = the retriever running
        ax.plot([r["k"]], [r[metric]], marker="o", ms=7.0, mew=1.0, mec=INK,
                mfc=hue if filled else SURFACE, linestyle="none", zorder=4)
        label = f"{r[metric]:.1f}%" + ("\nno memory" if r["k"] == 0 else "")
        drop = 9 + r[err_key] * pt_per_unit
        ax.annotate(label, (r["k"], r[metric]), xytext=(0, -drop),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6.5, color=MUTED, clip_on=False)

    ax.xaxis.set_major_locator(mtick.FixedLocator(ks))
    ax.xaxis.set_major_formatter(mtick.FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.set_xlabel(f"retrieved per turn — {retriever} top-k of {n_items}  "
                  f"(0 = no memory)")
    ax.set_ylabel(ylabel)
    # What the bars mean goes in a corner block, not the axis label: spelled out on a
    # rotated y label it overflows the figure height. Top-left is the free corner — these
    # curves rise left to right.
    note = [f"{n_runs} repeats per point",
            "bars: " + ("spread across repeats" if metric == "success"
                        else "task-sampling s.e.")]
    ax.text(0.02, 0.97, "\n".join(note), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color=MUTED, linespacing=1.5)
    # The level the sweep is climbing toward. A hairline dotted rule with its value in the
    # label, so the line never has to be read off the y axis, and so hue is not the only
    # thing telling it apart from the curve.
    if ref is not None:
        y, text = ref
        ax.axhline(y, color=SERIES[2], lw=0.9, ls=(0, (1, 2.5)), alpha=0.95, zorder=2)
        ax.text(0.995, y, f"{text} {y:.1f}%", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color=SERIES[2])
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    return finish_plain(ax, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, help="appworld | tau2 | automationbench")
    ap.add_argument("--split", required=True, help="the split every run must share")
    ap.add_argument("--model", required=True,
                    help="case-insensitive substring of the run's model string")
    ap.add_argument("--retriever", default="bm25",
                    help="the per-turn retriever slug to sweep (bm25 | dense | random | ...)")
    ap.add_argument("--pool", required=True,
                    help="the heuristic pool, as it appears in memory.pool_path")
    ap.add_argument("--runs", type=int, default=3, metavar="N",
                    help="repeats every point is cut back to, so the points are comparable "
                         "(default 3); a run with fewer is skipped and reported")
    # Each figure family owns a folder under plots/pdf/.
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/memory-use"))
    args = ap.parse_args()

    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 (pass^1 and a spread need two repeats)")

    n_items = pool_size(args.pool)
    found = collect(args.benchmark, args.split, args.model, args.retriever, args.pool,
                    args.runs)
    rows = [found[k] for k in sorted(found)]

    pool_name = Path(args.pool).parent.name or Path(args.pool).stem
    print(f"pool: {args.pool}  ({n_items} heuristics)")
    print(f"{args.benchmark} · {args.split} · {args.model} · retriever={args.retriever}")
    print(f"every point recomputed from its first {args.runs} repeat(s)\n")
    if SHORT:
        for name, n in SHORT:
            print(f"  {name}: only {n} repeat(s), needs {args.runs} — skipped")
    if not rows:
        raise SystemExit(
            f"Nothing to plot: no memory-off baseline and no {args.retriever} run on this "
            f"pool, for {args.model} on {args.benchmark}/{args.split} "
            f"with >= {args.runs} repeats.")
    k_pass = args.runs
    for r in rows:
        what = "no memory" if r["k"] == 0 else f"{args.retriever} top-{r['k']}"
        print(f"  k={r['k']:<4} success {r['success']:5.1f}% ±{r['success_err']:4.1f}   "
              f"pass^{k_pass} {r['passk']:5.1f}% ±{r['passk_err']:4.1f}   "
              f"({r['n_tasks']} tasks, {r['n_runs']} repeats on disk)  {what:22} {r['name']}")
    if 0 not in found:
        print("\n  missing: the k=0 memory-off baseline")
    if len(rows) < 2:
        raise SystemExit("\nOnly one point exists — a sweep needs at least two runs.")

    at_start = at_start_arm(args.benchmark, args.split, args.model, str(args.pool),
                            args.runs)
    if at_start is None:
        print("\n  note: no at-start run on this pool — reference line omitted")
    else:
        print(f"\n  reference (all @ start): {at_start['name']} — "
              f"success {at_start['success']:.1f}%, pass^{k_pass} {at_start['passk']:.1f}%")

    labels = {"success": "mean success rate", "passk": f"pass^{k_pass}"}
    for metric in ("success", "passk"):
        ylabel = labels[metric]
        tag = "_".join(filter(None, (args.benchmark, slug(args.split), slug(args.model),
                                     slug(pool_name), "topk", slug(args.retriever),
                                     metric if metric == "success" else f"pass{k_pass}")))
        ref = None
        if at_start is not None:
            ref = (at_start[metric], "all @ start")
        path = plot(rows, n_items, args.retriever, metric, ylabel,
                    args.out / f"{tag}.pdf", n_runs=k_pass, ref=ref,
                    title=f"{args.retriever} top-k sweep — {args.model}, "
                          f"{args.benchmark}/{args.split}")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
