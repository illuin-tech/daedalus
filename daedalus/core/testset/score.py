"""Score a test-set run: read the judgements, write `evaluation.json`.

The payload is deliberately the SAME shape a benchmark inference run writes (`aggregate` +
`per_run`, with `success_rate_mean/se` and both k-curves), so the run browser, the pass^k /
pass@k figures and `plots/scripts/_runs.py` read a generated-test-set run with no special
case — `aggregate_across_runs` computes the k-curves from the `task_success` maps below.
Everything specific to this kind of run is additive:

    partial_rate_*        mean fraction of success conditions met (the partial credit)
    condition_failures    how often the n-th condition of a task was the one that failed
    num_unparsed          judge replies that could not be read (counted as failures)

The judgement sidecars are the authority, not the traces: a trace exists as soon as the
rollout finishes, while a task is only *measured* once it has been judged.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

from daedalus.core.evaluation.evaluate import aggregate_across_runs

JUDGEMENTS_DIRNAME = "judgements"


def judgement_dir(base: Path, run_idx: int) -> Path:
    """`<experiment_dir>/judgements/run_<i>/` — one JSON per judged task."""
    return base / JUDGEMENTS_DIRNAME / f"run_{run_idx}"


def score_run(base: Path, run_idx: int) -> dict[str, Any]:
    """One run's results, read back from its judgement sidecars."""
    rows: list[dict[str, Any]] = []
    for path in sorted(judgement_dir(base, run_idx).glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue

    successes = sum(1 for r in rows if r.get("success"))
    partials = [float(r.get("partial") or 0.0) for r in rows]
    return {
        "run_idx": run_idx,
        "num_tasks": len(rows),
        "num_successes": successes,
        "success_rate": (successes / len(rows)) if rows else 0.0,
        "partial_rate": (sum(partials) / len(partials)) if partials else 0.0,
        # The per-task maps are what pass^k and pass@k are computed from.
        "task_success": {r["tid"]: bool(r.get("success")) for r in rows if "tid" in r},
        "task_partial": {
            r["tid"]: float(r.get("partial") or 0.0) for r in rows if "tid" in r
        },
        "num_unparsed": sum(1 for r in rows if not r.get("parsed", True)),
        # Role-separated, kept: comparing models on solver cost alone is a real question.
        "judge_cost_usd": sum(float(r.get("judge_cost_usd") or 0.0) for r in rows),
        "solver_cost_usd": sum(float(r.get("solver_cost_usd") or 0.0) for r in rows),
        # ...but the two do NOT add up to the run's bill: a tau2 test set also pays a user
        # simulator, and a memory-enabled one pays an LLM retriever. `cost_usd_by_role`
        # comes from each task's usage record and covers every role that ran.
        "cost_usd_by_role": _sum_by_role(rows),
        "cost_complete": all(
            (r.get("usage") or {}).get("cost_complete", True) for r in rows
        ),
        "avg_turns": st.mean(
            [r["num_turns"] for r in rows if r.get("num_turns") is not None]
        )
        if any(r.get("num_turns") is not None for r in rows)
        else None,
    }


def _sum_by_role(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Per-role cost across a run's task records, from each record's `usage` block."""
    by_role: dict[str, float] = {}
    for row in rows:
        for role, value in (
            (row.get("usage") or {}).get("cost_usd_by_role") or {}
        ).items():
            by_role[role] = round(by_role.get(role, 0.0) + float(value), 6)
    return dict(sorted(by_role.items()))


def condition_failures(base: Path, num_runs: int) -> dict[str, Any]:
    """Which condition position fails, and how often — a check on the tasks themselves.

    A condition that almost never holds is either genuinely hard or badly written; either
    way it is the first thing to look at when asking whether these tasks make a fair test.
    """
    met_by_index: dict[int, list[bool]] = {}
    for run_idx in range(num_runs):
        for path in sorted(judgement_dir(base, run_idx).glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for outcome in record.get("outcomes") or []:
                met_by_index.setdefault(int(outcome.get("index", 0)), []).append(
                    bool(outcome.get("met"))
                )
    return {
        f"condition_{index}": {
            "n": len(flags),
            "met_rate": (sum(flags) / len(flags)) if flags else None,
        }
        for index, flags in sorted(met_by_index.items())
    }


def build_payload(
    base: Path,
    num_runs: int,
    experiment_name: str,
    benchmark: str,
    test_set: dict[str, Any],
) -> dict[str, Any]:
    """The full `evaluation.json` payload for a test-set experiment."""
    per_run = [score_run(base, i) for i in range(num_runs)]
    per_run = [r for r in per_run if r["num_tasks"]]
    aggregate = aggregate_across_runs(per_run)

    partial = [r["partial_rate"] for r in per_run]
    if partial:
        aggregate["partial_rate_mean"] = sum(partial) / len(partial)
        aggregate["partial_rate_std"] = st.stdev(partial) if len(partial) > 1 else 0.0
        # Same convention as success_rate_se (evaluation/evaluate.py): the reported ± is the
        # standard error of the mean over runs, the SD kept beside it.
        aggregate["partial_rate_se"] = (
            st.stdev(partial) / len(partial) ** 0.5 if len(partial) > 1 else 0.0
        )
        aggregate["partial_rate_per_run"] = partial
    aggregate["num_unparsed"] = sum(r["num_unparsed"] for r in per_run)
    aggregate["judge_cost_usd"] = sum(r["judge_cost_usd"] for r in per_run)
    aggregate["solver_cost_usd"] = sum(r["solver_cost_usd"] for r in per_run)
    by_role: dict[str, float] = {}
    for r in per_run:
        for role, value in (r.get("cost_usd_by_role") or {}).items():
            by_role[role] = round(by_role.get(role, 0.0) + value, 6)
    aggregate["cost_usd_by_role"] = by_role
    aggregate["estimated_total_cost_usd"] = round(sum(by_role.values()), 6)
    aggregate["cost_complete"] = all(r.get("cost_complete", True) for r in per_run)
    turns = [r["avg_turns"] for r in per_run if r["avg_turns"] is not None]
    aggregate["avg_turns"] = (sum(turns) / len(turns)) if turns else None

    return {
        "experiment_name": experiment_name,
        "benchmark": benchmark,
        "kind": "testset",
        "multi_run": len(per_run) > 1,
        "test_set": test_set,
        "aggregate": aggregate,
        "per_run": per_run,
        "condition_failures": condition_failures(base, num_runs),
    }
