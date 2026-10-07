"""Does a generated test set rank models the way the real split does?

    python plots/scripts/testset_correlation.py \
        --generation-folder appworld --split test_normal

One dot per model and metric: x = its score on the REAL split with NO memory, y = its score on
the tasks a generation run invented (`outputs/evaluation-proxy/<generation folder>/<model>/`). MSR and
pass^k share one pair of axes; hue is the metric, shape is the model.

    <benchmark>_<split>_<generation folder>_testset_combined_v2.pdf

The claim is about ORDER, not level, so the figure reports **Kendall τ-b** and the number of
model PAIRS the two axes order the same way. Every model scores higher on the generated tasks,
so a linear fit would measure the wrong thing.

The two sides repeat a different number of times (generated 3x, real split 5x), so pass^k puts
BOTH axes on the largest k every run supports (here 3) rather than each side's own maximum:
pass^k falls as k grows, and pass^5 against pass^3 would hand the y axis a structural
advantage. See `_resolve_k`.

The x side is restricted to memory-off `daedalus` runs on daedalus's own solver prompt
(`memory.enabled: false`, no `heuristics_at_start`, no `agent.original_solver_prompt`): the y
side runs every model bare on that prompt, so the x side must too. This excludes every
`references/` run, which all set `memory.enabled: false` while doing their own retrieval.
Pairing is by the full model string from each run's `run_meta.json`.
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.ticker as mtick
import yaml
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _family import finish_plain  # noqa: E402
from _runs import is_original_prompt, load_json, pass_hat_k, slug  # noqa: E402
from _style import INK, SURFACE, TEAL, figsize, plt  # noqa: E402

from daedalus.core.config import DEFAULT_SETUP, kind_root, run_dirs  # noqa: E402

DISPLAY = {"appworld": "AppWorld", "tau2": "τ²"}
OWN_HEURISTIC_SUFFIX = "_own_heuristic"  # daedalus.scripts.tasks_as_test_set
# One shape per MODEL: the legend names every model, and nine models overflow the eight
# family glyphs. Only fillable shapes, because the hue carries the metric. Shuffled once with a
# fixed seed so neighbouring models do not get visually related marks, and so a model keeps
# its shape across runs.
GLYPH_SEED = 11
MODEL_GLYPHS = tuple(random.Random(GLYPH_SEED).sample(
    ["o", "s", "D", "^", "v", "*", "<", ">", "X", "p", "P", "h"], 12))
# (metric, legend name, hue). Teal is the house single-series colour; the warm slot beside it
# is `_style.SERIES[4]`.
SERIES = (("success", "MSR", TEAL), ("passk", "pass^k", "#fdb462"))


# ── the two sides ───────────────────────────────────────────────────────────────────────
def testset_scores(generation_folder: str) -> list[dict]:
    """Every model measured on one generated test set."""
    root = kind_root("testset") / generation_folder
    if not root.is_dir():
        raise SystemExit(
            f"No test-set runs under {root}. Build them with "
            f"`python -m daedalus.scripts.tasks_as_test_set --generation "
            f"outputs/daedalus/{generation_folder} --model <model>`.")
    out = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # `tasks_as_test_set --own-heuristic` (Appendix C's counterfactual replay) measures a
        # model WITH each task's banked heuristic, not the bare model this figure ranks.
        if model_dir.name.endswith(OWN_HEURISTIC_SUFFIX):
            continue
        meta, ev = load_json(model_dir / "run_meta.json"), load_json(model_dir / "evaluation.json")
        if not meta or not ev:
            continue
        agg = ev.get("aggregate") or {}
        if agg.get("success_rate_mean") is None:
            continue
        out.append({
            "model": meta.get("model", ""),
            "tag": model_dir.name,
            "benchmark": meta.get("benchmark", ""),
            "num_tasks": (ev.get("test_set") or {}).get("num_tasks"),
            "num_runs": agg.get("num_runs"),
            "agg": agg, "per_run": ev.get("per_run") or [],
        })
    return out


def memory_off(run_dir: Path, setup: str) -> bool:
    """True when this inference run is a bare model on daedalus's solver prompt.

    `memory.enabled: false` is not sufficient: every method under `references/` turns
    daedalus's memory block off because it does its own retrieval, so a baseline has to be a
    `daedalus` run. A run on the benchmark's own solver prompt (`agent.original_solver_prompt`)
    is a prompt control, not a baseline: the generated side never uses it, and on AppWorld it
    is worth ~10 points by itself (gpt-5.4-mini: 54.8% against 44.3%).
    """
    if setup != DEFAULT_SETUP or is_original_prompt(run_dir):
        return False
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return False
    mem = cfg.get("memory") or {}
    return not mem.get("enabled") and not mem.get("heuristics_at_start")


def baseline_scores(benchmark: str, split: str) -> dict[str, list[dict]]:
    """Memory-off inference runs of `benchmark`/`split`, grouped by full model string."""
    by_model: dict[str, list[dict]] = {}
    for setup, run_dir in run_dirs("inference"):
        meta, ev = load_json(run_dir / "run_meta.json"), load_json(run_dir / "evaluation.json")
        if not meta or not ev:
            continue
        if meta.get("benchmark") != benchmark or str(meta.get("domain")) != split:
            continue
        agg = ev.get("aggregate") or {}
        if agg.get("success_rate_mean") is None or not memory_off(run_dir, setup):
            continue
        by_model.setdefault(meta.get("model", ""), []).append({
            "setup": setup, "name": run_dir.name, "agg": agg,
            "per_run": ev.get("per_run") or [],
            "num_runs": agg.get("num_runs") or 0,
            "timestamp": meta.get("timestamp", ""),
        })
    return by_model


def pick(runs: list[dict]) -> dict:
    """The most trustworthy memory-off run for one model: most repeats, then newest."""
    return sorted(runs, key=lambda r: (r["num_runs"], r["timestamp"]), reverse=True)[0]


def pair(ts_rows: list[dict], baselines: dict[str, list[dict]], metric: str) -> list[dict]:
    """Join the two sides on the model, reporting what could not be paired."""
    points, unpaired = [], []
    for row in ts_rows:
        runs = baselines.get(row["model"]) or []
        if not runs:
            # The test-set folder is named by the short tag; try that too, so a model written
            # "openrouter/qwen/x" on one side and "qwen/x" on the other still pairs.
            runs = next((v for k, v in baselines.items() if k.split("/")[-1] == row["tag"]), [])
        if not runs:
            unpaired.append(row["tag"])
            continue
        chosen = pick(runs)
        if len(runs) > 1:
            print(f"  {row['tag']}: {len(runs)} memory-off runs, using "
                  f"{chosen['setup']}:{chosen['name']} ({chosen['num_runs']} repeats); "
                  f"others: " + ", ".join(f"{r['setup']}:{r['name']}" for r in runs
                                          if r is not chosen))
        point = {"tag": row["tag"], "model": row["model"],
                 "baseline_run": f"{chosen['setup']}:{chosen['name']}",
                 "n_real": chosen["num_runs"], "n_gen": row["num_runs"]}
        if metric == "passk":
            # k is resolved below: it has to be the SAME on both axes and for every model.
            point["x_tab"] = pass_hat_k(chosen["per_run"], chosen["agg"])[0]
            point["y_tab"] = pass_hat_k(row["per_run"], row["agg"])[0]
        else:
            point["x"] = chosen["agg"].get("success_rate_mean")
            point["y"] = row["agg"].get("success_rate_mean")
        points.append(point)
    if unpaired:
        print(f"  no memory-off run on this split for: {', '.join(unpaired)} — dropped")
    if metric == "passk":
        points = _resolve_k(points)
    return [p for p in points if p["x"] is not None and p["y"] is not None]


def _resolve_k(points: list[dict]) -> list[dict]:
    """Put BOTH axes on the same k: the largest one every run on the figure supports."""
    ks = [min(max(p["x_tab"]), max(p["y_tab"])) for p in points if p["x_tab"] and p["y_tab"]]
    k = min(ks) if ks else None
    return [{**p, "x": float(p["x_tab"][k]), "y": float(p["y_tab"][k]), "k": k}
            for p in points if k is not None and k in p["x_tab"] and k in p["y_tab"]]


# ── correlation ─────────────────────────────────────────────────────────────────────────
def kendall(xs: list[float], ys: list[float]) -> dict:
    """Kendall τ-b, its p-value, and the concordant pairs counted directly.

    Counting concordant pairs rather than deriving them from τ keeps ties visible as
    "neither concordant nor discordant".
    """
    from scipy import stats

    pairs = list(combinations(range(len(xs)), 2))
    if len(xs) < 3:
        return {"kendall": None, "p_kendall": None, "concordant": None, "n_pairs": len(pairs)}
    kend = stats.kendalltau(xs, ys)
    return {"kendall": float(kend.statistic), "p_kendall": float(kend.pvalue),
            "concordant": sum(1 for i, j in pairs if (xs[i] - xs[j]) * (ys[i] - ys[j]) > 0),
            "n_pairs": len(pairs)}


def p_str(p: float | None) -> str:
    """`p<.001` rather than `p=0.000`, which reads as "exactly zero"."""
    if p is None:
        return "n/a"
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".")


# ── figure ──────────────────────────────────────────────────────────────────────────────
def _span(values: list[float]) -> tuple[float, float]:
    """Data range plus a margin, in percentage points, never clamped to [0, 100]."""
    pad = max(0.08 * (max(values) - min(values)), 1.5)
    return min(values) - pad, max(values) + pad


def plot(series: list[dict], path: Path, benchmark: str, split: str) -> Path:
    """MSR and pass^k for the same models on one pair of axes.

    Hue is the METRIC and shape is the MODEL. pass^k is drawn faded (fill 30%, outline 70%,
    outline in its own hue) so the MSR cloud reads first. Every model is named in a legend
    above the axes rather than labelled in place; the metric key sits in the lower-right
    corner, which stays empty because every model scores higher on the generated tasks.
    """
    series = [{**s, "fill_alpha": 0.3, "edge_alpha": 0.7, "edge": s["hue"]}
              if s["metric"] == "passk" else s for s in series]
    models = sorted({p["tag"] for s in series for p in s["points"]})
    if len(models) > len(MODEL_GLYPHS):
        print(f"  ! {len(models)} models past the {len(MODEL_GLYPHS)} shapes — some repeat")
    glyph = {m: MODEL_GLYPHS[i % len(MODEL_GLYPHS)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=figsize(6.4 / 1.2, 4.4 / 1.2))
    for s in series:
        for p in s["points"]:
            ax.plot([100 * p["x"]], [100 * p["y"]], glyph[p["tag"]], ms=7.6, mew=1.0,
                    mec=to_rgba(s.get("edge", INK), s.get("edge_alpha", 1.0)),
                    mfc=to_rgba(s["hue"], s.get("fill_alpha", 1.0)),
                    linestyle="none", zorder=3)
    ax.set_xlabel(f"{' / '.join(s['name'] for s in series)} on "
                  f"{DISPLAY.get(benchmark, benchmark)} {split}")
    ax.set_ylabel(f"{' / '.join(s['name'] for s in series)} on Daedalus' tasks")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
    ax.set_xlim(*_span([100 * p["x"] for s in series for p in s["points"]]))
    lo, hi = _span([100 * p["y"] for s in series for p in s["points"]])
    # Extra headroom for the two τ lines, which span most of the width.
    ax.set_ylim(lo, hi + 0.10 * (hi - lo))
    for i, s in enumerate(series):
        st_ = s["stats"]
        if st_["kendall"] is None:
            continue
        ax.text(0.02, 0.975 - 0.062 * i,
                f"{s['name']}: τ = {st_['kendall']:.2f} ({p_str(st_['p_kendall'])}), "
                f"{st_['concordant']}/{st_['n_pairs']} pairs agree",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=s["hue"],
                linespacing=1.5)

    # Two legends, because the channels are independent: the metric key takes a neutral
    # shape, the model key a neutral fill. Models are listed in order of their real-split MSR.
    x_msr = {p["tag"]: p["x"] for p in series[0]["points"]}
    ranked = sorted(models, key=lambda m: (x_msr.get(m) is None, x_msr.get(m, 0.0)))
    model_keys = [Line2D([], [], ls="none", marker=glyph[m], ms=7.0, mew=1.0, mec=INK,
                         mfc=SURFACE, label=m) for m in ranked]
    metric_keys = [Line2D([], [], ls="none", marker="o", ms=7.6, mew=1.0,
                          mec=to_rgba(s.get("edge", INK), s.get("edge_alpha", 1.0)),
                          mfc=to_rgba(s["hue"], s.get("fill_alpha", 1.0)), label=s["name"])
                   for s in series]
    ax.add_artist(ax.legend(handles=metric_keys, loc="lower right", frameon=False, fontsize=7,
                            handletextpad=0.6, borderpad=0.6))
    ax.legend(handles=model_keys, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3,
              frameon=False, fontsize=6.5, handletextpad=0.6, borderpad=0.2,
              columnspacing=1.0)
    return finish_plain(ax, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generation-folder", required=True,
                    help="the generation run's folder NAME under outputs/evaluation-proxy/")
    ap.add_argument("--split", required=True,
                    help="the real split to compare against (matched against run_meta domain)")
    ap.add_argument("--out", type=Path, default=Path("plots/pdf/evaluation-proxy"))
    args = ap.parse_args()

    folder = args.generation_folder.rstrip("/").split("/")[-1]
    ts_rows = testset_scores(folder)
    benchmarks = {r["benchmark"] for r in ts_rows}
    if len(benchmarks) != 1:
        raise SystemExit(f"{folder} mixes benchmarks {sorted(benchmarks)} — cannot pair.")
    benchmark = next(iter(benchmarks))
    print(f"{folder}: {len(ts_rows)} model(s) measured on the generated test set")
    baselines = baseline_scores(benchmark, args.split)
    print(f"memory-off {benchmark}/{args.split} runs: {len(baselines)} model(s)")

    series = []
    for metric, name, hue in SERIES:
        points = pair(ts_rows, baselines, metric)
        if len(points) < 2:
            raise SystemExit(f"{metric}: {len(points)} paired model(s) — need at least 2")
        if metric == "passk":
            name = f"pass^{points[0]['k']}"
        stats_ = kendall([p["x"] for p in points], [p["y"] for p in points])
        print(f"\n{name}: {len(points)} models, {ts_rows[0]['num_tasks']} generated tasks · "
              f"τ={stats_['kendall']:.2f} ({p_str(stats_['p_kendall'])}, "
              f"p={stats_['p_kendall']:.2e}), "
              f"{stats_['concordant']}/{stats_['n_pairs']} pairs agree")
        for p in sorted(points, key=lambda p: -p["x"]):
            print(f"  {p['tag']:24.24s} real {p['x']:6.1%} ({p['n_real']} runs)  "
                  f"generated {p['y']:6.1%} ({p['n_gen']} runs)  via {p['baseline_run']}")
        series.append({"metric": metric, "name": name, "hue": hue,
                       "points": points, "stats": stats_})

    stem = "_".join((benchmark, slug(args.split), slug(folder), "testset"))
    plot(series, args.out / f"{stem}_combined_v2.pdf", benchmark, args.split)


if __name__ == "__main__":
    main()
