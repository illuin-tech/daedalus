"""Evaluation metrics — task success, aggregate statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> dict[str, Any]:
    """Load a single trace JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_traces(trace_dir: str | Path) -> list[dict[str, Any]]:
    """Load all traces from a directory."""
    trace_dir = Path(trace_dir)
    traces = []
    for f in sorted(trace_dir.glob("*.json")):
        traces.append(load_trace(f))
    return traces


def task_success(trace: dict[str, Any]) -> bool:
    """Whether a task was successful."""
    return trace.get("evaluation", {}).get("success", False)


def _retrieved_id_sets(trace: dict[str, Any]) -> list[set[str]]:
    """Per-turn sets of retrieved memory IDs, keeping only turns that retrieved ≥1."""
    sets = []
    for turn in trace.get("turns", []):
        ids = {m["memory_id"] for m in turn.get("retrieved_memories", []) if "memory_id" in m}
        if ids:
            sets.append(ids)
    return sets


def retrieval_diversity(trace: dict[str, Any]) -> dict[str, Any]:
    """Measure how much the injected memory *changes across turns* for one task.

    Tests the "turnover" hypothesis: conditions whose injected context renews
    every turn (random/colbert @turn) help, while those repeating the same
    nearest-neighbours every turn (bm25/qwen @turn) do not.

    Returns:
        retrieval_turns: # turns that injected ≥1 memory.
        total_injected: # memories injected across all turns (with repeats).
        unique_items: # distinct memories seen over the trajectory.
        turnover: 1 − mean Jaccard overlap of consecutive turns' ID sets
            (1.0 = fully fresh each turn, 0.0 = identical each turn).
            None when <2 retrieval turns (e.g. @start injects once).
        repeat_ratio: fraction of injections that were repeats
            (0.0 = all distinct, →1.0 = same few items reused). Defined for any
            trajectory, so it also separates @start (one set) from @turn churn.
    """
    sets = _retrieved_id_sets(trace)
    total_injected = sum(len(s) for s in sets)
    unique_items = len(set().union(*sets)) if sets else 0

    jaccards = []
    for a, b in zip(sets, sets[1:]):
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 0.0)
    turnover = (1.0 - sum(jaccards) / len(jaccards)) if jaccards else None

    return {
        "retrieval_turns": len(sets),
        "total_injected": total_injected,
        "unique_items": unique_items,
        "turnover": turnover,
        "repeat_ratio": (1.0 - unique_items / total_injected) if total_injected else 0.0,
    }


def aggregate_retrieval_diversity(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean retrieval-diversity metrics over a set of traces.

    `turnover` is averaged only over trajectories where it is defined
    (≥2 retrieval turns); `turnover_defined_tasks` reports how many that was.
    """
    per = [retrieval_diversity(t) for t in traces]
    turnovers = [p["turnover"] for p in per if p["turnover"] is not None]

    def _mean(key: str) -> float:
        return sum(p[key] for p in per) / len(per) if per else 0.0

    return {
        "num_tasks": len(per),
        "avg_retrieval_turns": _mean("retrieval_turns"),
        "avg_unique_items": _mean("unique_items"),
        "avg_repeat_ratio": _mean("repeat_ratio"),
        "avg_turnover": (sum(turnovers) / len(turnovers)) if turnovers else None,
        "turnover_defined_tasks": len(turnovers),
    }


def aggregate_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over a set of traces."""
    if not traces:
        return {"num_tasks": 0}

    successes = sum(1 for t in traces if task_success(t))
    total = len(traces)
    total_turns = sum(t.get("num_turns", 0) for t in traces)
    total_cost = sum(t.get("total_cost_usd", 0.0) for t in traces)
    total_prompt = sum(t.get("total_tokens", {}).get("prompt", 0) for t in traces)
    total_completion = sum(t.get("total_tokens", {}).get("completion", 0) for t in traces)
    # A SUBSET of total_prompt, not an addition to it: cache reads are input tokens billed
    # at a tenth the fresh rate, so the split is what separates two runs with identical
    # token counts and different bills. Every trace this repo has ever written carries it.
    total_cached = sum(t.get("total_tokens", {}).get("cached", 0) or 0 for t in traces)

    return {
        "num_tasks": total,
        "num_successes": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "avg_turns": total_turns / total if total > 0 else 0.0,
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / total if total > 0 else 0.0,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_prompt_tokens": total_cached,
    }
