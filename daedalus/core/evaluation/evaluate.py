"""Score a completed inference experiment via the benchmark's own scorer.

Runs automatically at the end of every inference run (see run_experiment). AppWorld
runs are scored with AppWorld's official evaluator; tau2 and AutomationBench aggregate
the rewards already computed at run time. Multi-run experiments (run_0/, run_1/, …) are
scored per run and reported as mean ± standard error, with pass^k and pass@k. The result
is written to the experiment's own folder as `evaluation.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig, experiment_dir, resolve_trace_dir
from daedalus.core.logging import console
from daedalus.core.logging.usage_aggregate import UsageAggregator
from daedalus.core.registry import get_benchmark

# Bookkeeping files an experiment folder carries alongside its per-task traces.
_NON_TRACE_FILES = {"evaluation.json", "run_meta.json", "run_summary.json"}


def _detect_runs(trace_dir: Path) -> list[Path]:
    """Run subdirectories (run_0/, run_1/, …), or [trace_dir] for a flat single run.

    A folder holding BOTH layouts is ambiguous and is refused: it means two experiments
    wrote to the same `experiment_name`, one multi-run and one single-run. Scoring it
    would silently read the run_*/ dirs and ignore the flat traces — and for AppWorld it
    would then look for this split's task ids inside the other split's world directory.
    """
    run_dirs = sorted(d for d in trace_dir.glob("run_*") if d.is_dir())
    if not run_dirs:
        return [trace_dir]
    flat = [p for p in trace_dir.glob("*.json") if p.name not in _NON_TRACE_FILES]
    if flat:
        raise RuntimeError(
            f"{trace_dir} holds both {len(flat)} flat trace file(s) and "
            f"{len(run_dirs)} run_*/ director(ies), so the layout is ambiguous — two runs "
            f"shared this experiment_name. Give each run its own experiment_name, then move "
            f"the flat traces into their own folder (the run_*/ dirs belong to the "
            f"multi-run experiment)."
        )
    return run_dirs


def _report_usage(usage: dict[str, Any]) -> None:
    """Print the run's spend, and say plainly when the number is known to be partial."""
    total = usage.get("estimated_total_cost_usd")
    if total is None:
        console.info(
            f"cost (estimated, INCOMPLETE): ${usage['priced_calls_estimated_cost_usd']:.4f} "
            f"over {usage['num_completed_calls']} call(s)"
        )
    else:
        console.info(
            f"cost (estimated): ${total:.4f} over {usage['num_completed_calls']} call(s)"
        )
    by_role = usage.get("cost_usd_by_role") or {}
    if by_role:
        console.info(
            "  by role: " + "  ".join(f"{k} ${v:.4f}" for k, v in by_role.items())
        )
    for reason in usage.get("cost_incompleteness_reasons") or []:
        console.info(f"  incomplete: {reason}")


def aggregate_across_runs(run_results: list[dict]) -> dict:
    """Mean/std of success rate across runs, plus unbiased pass^k and pass@k when multi-run.

    Public because the generated-test-set scorer (`daedalus.core.testset.score`) reuses the
    same combinatorics rather than re-deriving the k-curves.
    """
    if not run_results:
        return {}

    success_rates = [r.get("success_rate", 0.0) for r in run_results]
    num_tasks_list = [r.get("num_tasks", 0) for r in run_results]
    n = len(success_rates)
    mean_sr = sum(success_rates) / n
    # Sample SD across runs (Bessel, n-1) and the standard error of their mean. The SE is
    # what is reported: it answers "how precisely do these n runs pin the mean", which is the
    # question a table of arms invites. The SD is kept because it answers the other one —
    # "how far would a single rerun land" — and because the SE is derived from it.
    std_sr = (
        math.sqrt(sum((x - mean_sr) ** 2 for x in success_rates) / (n - 1))
        if n > 1
        else 0.0
    )
    se_sr = std_sr / math.sqrt(n) if n > 1 else 0.0

    # A mean over repeats that scored different numbers of tasks is not a success rate:
    # a repeat where 12 tasks crashed has a smaller denominator and silently counts as
    # much as a complete one. `pass_hat_k` below already intersects the task sets; this
    # makes the marginal mean say so too, loudly, instead of averaging them anyway.
    unequal = len(set(num_tasks_list)) > 1
    if unequal:
        console.info(
            f"  WARNING: repeats scored different task counts {num_tasks_list} — the mean "
            f"success rate below averages unequal denominators and is NOT comparable with "
            f"a complete run. pass^k is unaffected (it uses the shared tasks only). "
            f"Re-run the short repeats, or drop them."
        )

    aggregate = {
        "num_runs": n,
        "num_tasks_per_run": num_tasks_list[0] if num_tasks_list else 0,
        "num_tasks_per_run_all": num_tasks_list,
        "unequal_denominators": unequal,
        "success_rate_mean": mean_sr,
        "success_rate_std": std_sr,
        "success_rate_se": se_sr,
        "success_rate_per_run": success_rates,
        # ± is the STANDARD ERROR of the mean over runs (std_sr / sqrt(n)), not the spread.
        "success_rate_summary": f"{mean_sr:.3f} ± {se_sr:.3f}",
    }

    # Both k-curves (unbiased, tau-bench formula) over the tasks attempted in every run:
    # pass^k falls with k (all k attempts land), pass@k rises (at least one does).
    if n > 1 and all("task_success" in r for r in run_results):
        aggregate.update(pass_hat_k_with_se(run_results))
        aggregate.update(pass_at_k_with_se(run_results))

    return aggregate


def _common_task_hits(run_results: list[dict[str, Any]]) -> tuple[int, list[int]] | None:
    """(n repeats, per-task success counts) over the tasks attempted in EVERY run.

    None when there is nothing to compute a k-curve from: fewer than two repeats, a run
    without a `task_success` map, or no task shared by all of them. Shared by pass^k and
    pass@k so the two are always read off the same task set.
    """
    n = len(run_results)
    if n < 2 or not all("task_success" in r for r in run_results):
        return None
    common = sorted(set.intersection(*(set(r["task_success"]) for r in run_results)))
    if not common:
        return None
    return n, [sum(1 for r in run_results if r["task_success"][tid]) for tid in common]


def _mean_and_task_se(vals: list[float]) -> tuple[float, float]:
    """Mean of the per-task values and its standard error with respect to TASK sampling.

    `s_v / sqrt(N)` over the per-task values v_i, with `s_v` the Bessel-corrected (N-1)
    sample SD. It answers "re-draw the task set, keep these n repeats". Note what it is NOT:
    a repeat-level standard error like `success_rate_se`, which is what the mean success rate
    reports. The two columns therefore answer different questions and are not comparable to
    each other — unavoidably, for the reason in the next paragraph. Both pass^k and pass@k are
    computed from all n repeats jointly, so no single run has a value and there is nothing to
    take a spread over — and for k = n every route to a repeat-level estimate is blocked: the
    jackknife is undefined (dropping a run leaves n-1 < k), subsampling without replacement
    has exactly one subset, and the bootstrap over runs is badly biased because a size-n
    resample holds only ~0.63n distinct runs, making "all drawn runs succeeded" far easier
    than "all n runs succeeded" (measured on a 5-run AppWorld baseline: pass^5 14.88% ->
    bootstrap mean 20.94%, +6.06pp). So the task-sampling SE is what is reported.

    Bessel (N-1), not N: the framing treats v_1..v_N as a SAMPLE from a task population and
    the mean as its sample mean, so the unbiased variance estimator is the corrected one.
    Dividing by N is the finite-population variance and understates it by (N-1)/N — at N=168
    that is 0.3% on the SE, invisible in a reported figure but the wrong estimator for the
    interpretation, and inconsistent with success_rate_std, which has always used n-1.

    At k = n the pass^k values are 0/1 (`c_i == n` or not) and this reduces exactly to the
    binomial sqrt(p(1-p)/N); for k < n they are fractional and the binomial form would
    overstate it.
    """
    N = len(vals)
    mean_v = sum(vals) / N
    var = sum((v - mean_v) ** 2 for v in vals) / (N - 1) if N > 1 else 0.0
    return mean_v, math.sqrt(var / N)


def pass_hat_k_with_se(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """{"pass_hat_k": {...}, "pass_hat_k_se": {...}} over the tasks attempted in every run.

    pass^k = mean over tasks of `C(c_i, k) / C(n, k)` — the chance that k independently drawn
    attempts ALL succeed, with c_i successes out of n repeats for task i. Falls with k, and
    reads as reliability: what you get if every one of k tries has to land.

    The SE is with respect to task sampling; see `_mean_and_task_se`.
    """
    prep = _common_task_hits(run_results)
    if prep is None:
        return {}
    n, hits = prep
    out: dict[str, dict[str, float]] = {"pass_hat_k": {}, "pass_hat_k_se": {}}
    for k in range(1, n + 1):
        mean_v, se = _mean_and_task_se([math.comb(c, k) / math.comb(n, k) for c in hits])
        out["pass_hat_k"][f"pass^{k}"] = mean_v
        out["pass_hat_k_se"][f"pass^{k}"] = se
    return out


def pass_at_k_with_se(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """{"pass_at_k": {...}, "pass_at_k_se": {...}} — the mirror of pass^k.

    pass@k = mean over tasks of `1 - C(n - c_i, k) / C(n, k)` — the chance that AT LEAST ONE
    of k independently drawn attempts succeeds. Rises with k, and reads as best-of-k headroom:
    what a system that can retry, or verify and pick, has to work with. Same unbiased
    estimator family as pass^k, over the same task set and the same n repeats, so the two are
    two ends of one curve rather than two measurements: pass@1 = pass^1 = the mean success
    rate exactly, and above k = 1 they diverge.

    The SE is with respect to task sampling; see `_mean_and_task_se`.
    """
    prep = _common_task_hits(run_results)
    if prep is None:
        return {}
    n, hits = prep
    out: dict[str, dict[str, float]] = {"pass_at_k": {}, "pass_at_k_se": {}}
    for k in range(1, n + 1):
        mean_v, se = _mean_and_task_se(
            [1 - math.comb(n - c, k) / math.comb(n, k) for c in hits]
        )
        out["pass_at_k"][f"pass@{k}"] = mean_v
        out["pass_at_k_se"][f"pass@{k}"] = se
    return out


def evaluate_experiment(
    cfg: ExperimentConfig, task_ids: list[str] | None = None
) -> dict[str, Any]:
    """Score every run of an experiment, print a summary, and write evaluation.json.

    Returns the evaluation payload. Reads traces from the experiment's own folder, so
    it works for a flat single run and for multi-run (run_*/) layouts alike.
    """
    benchmark = get_benchmark(cfg)
    trace_dir = resolve_trace_dir(cfg)
    run_dirs = _detect_runs(trace_dir)
    is_multi_run = len(run_dirs) > 1 or (
        len(run_dirs) == 1 and run_dirs[0] != trace_dir
    )

    console.rule("evaluation")
    run_results: list[dict] = []
    for run_idx, run_dir in enumerate(run_dirs):
        result = benchmark.evaluate_run(
            cfg, run_dir, task_ids=task_ids, run_idx=(run_idx if is_multi_run else None)
        )
        run_results.append(result)
        ns = result.get("num_successes", 0)
        nt = result.get("num_tasks", 0)
        sr = result.get("success_rate", 0.0)
        label = run_dir.name if is_multi_run else "run"
        line = f"  {label}: {ns}/{nt} ({sr:.1%})"
        console.info(line)

    aggregate = aggregate_across_runs(run_results)
    console.info(
        f"success rate: {aggregate['success_rate_summary']} "
        f"({aggregate['num_runs']} run{'s' if aggregate['num_runs'] != 1 else ''})"
    )
    # The two k-curves side by side: pass^k (all k land) falling, pass@k (any lands) rising.
    at_k = aggregate.get("pass_at_k") or {}
    for key, value in (aggregate.get("pass_hat_k") or {}).items():
        k = key.split("^")[-1]
        line = f"  {key}: {value:.3f}"
        if f"pass@{k}" in at_k:
            line += f"   pass@{k}: {at_k[f'pass@{k}']:.3f}"
        console.info(line)

    # Whole-experiment spend from the usage ledger, NOT from the traces: the ledger also
    # holds the roles a trace never had (tau2's user simulator) and the calls of tasks that
    # crashed before writing one.
    usage = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).summary()
    _report_usage(usage)

    payload = {
        "experiment_name": cfg.name,
        "benchmark": benchmark.name,
        "multi_run": is_multi_run,
        "aggregate": aggregate,
        "per_run": run_results,
        "usage": usage,
    }
    out = experiment_dir(cfg) / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.info(f"evaluation → {out}")
    return payload
