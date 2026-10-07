"""The Solver loop (paper Algorithm 3): mine one heuristic from a task and validate it.

The solver attempts the task from a fresh environment. After each failure the extractor
writes (or revises) a single free-form heuristic from the failed trajectory, and the
solver retries with it in context. The loop stops after `num_success_to_continue`
consecutive successes (Ns) or `max_failures` failures (Nf). Successes never update the
heuristic.

Shared by accumulation on labeled tasks (`daedalus.scripts.accumulation`, graded by the
benchmark's verifier) and by self-play generation (graded by the LLM judge through the
`evaluator` argument), so the two differ only in where the verdict comes from.

`single_attempt_for_task` is the ablation without the loop (Table 3, row C): one
memory-free attempt, then one extraction whatever the outcome.
"""

from __future__ import annotations

import difflib
from typing import Any

from daedalus.core.config import ExperimentConfig
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging import console
from daedalus.core.memory.extraction import generate_memory_item
from daedalus.core.registry import get_benchmark


def _memory_diff(previous: str | None, current: str | None, attempt_idx: int) -> str | None:
    """Unified diff of the heuristic across attempts: None on the first one written,
    "(no change)" when unchanged."""
    if previous is None or previous == current:
        return None if previous is None else "(no change)"
    diff_lines = list(
        difflib.unified_diff(
            previous.splitlines(keepends=True),
            (current or "").splitlines(keepends=True),
            fromfile=f"attempt{attempt_idx - 1}",
            tofile=f"attempt{attempt_idx}",
            lineterm="",
        )
    )
    return "\n".join(diff_lines) if diff_lines else "(no change)"


def _usage_mark(llm: LLMClient):
    """Mark where this task's spend starts in the process's usage tally."""
    return llm.ledger.tally.snapshot() if llm.ledger is not None else None


def _usage_since(llm: LLMClient, mark) -> dict[str, Any]:
    """Canonical per-role usage for everything recorded since `mark`."""
    return llm.ledger.tally.since(mark) if llm.ledger is not None else {}


def _extraction_client(llm: LLMClient, task_id: str) -> LLMClient:
    """The extraction client, scoped to this task in the cost ledger."""
    scope = llm.scope
    if scope is None:
        return llm
    return llm.with_scope(scope.for_role("extraction").for_task(task_id=task_id))


def _new_summary(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "attempts": [],
        "final_outcome": None,
        "final_memory": None,
        "solved_without_memory": False,
        # Solver only, summed over attempts. `usage` is the canonical all-role figure.
        "cost_usd": 0.0,
        "tokens": {"prompt": 0, "completion": 0, "cached": 0},
        "usage": {},
        "last_trace_text": "",  # the final attempt's trajectory, fed to task refinement
    }


def _add_solver_cost(summary: dict[str, Any], result: dict[str, Any]) -> None:
    summary["cost_usd"] += float(result.get("total_cost_usd", 0.0) or 0.0)
    tokens = result.get("total_tokens") or {}
    for key in ("prompt", "completion", "cached"):
        summary["tokens"][key] += int(tokens.get(key, 0) or 0)


def accumulate_for_task(
    task_id: str,
    cfg: ExperimentConfig,
    extraction_llm: LLMClient,
    instruction_override: str | None = None,
    evaluator: Any | None = None,
    trace_prefix: str = "",
    task_override: Any | None = None,
) -> dict[str, Any]:
    """Run the Solver loop on one task and return its summary.

    For a generated task, `instruction_override` (AppWorld) or `task_override` (τ²,
    AutomationBench) carries the task, and `evaluator(trace) -> (success, details)` is the
    LLM judge grading it against the explorer's success conditions.

    `final_outcome` is "success" when the loop ended on the success streak. The task was
    then accepted if a heuristic was written on the way, and too easy otherwise
    (`solved_without_memory`).
    """
    benchmark = get_benchmark(cfg)
    max_failures = cfg.accumulation.max_failures
    num_success_needed = cfg.accumulation.num_success_to_continue

    summary = _new_summary(task_id)
    usage_mark = _usage_mark(extraction_llm)
    current_memory: str | None = None
    consecutive_successes = 0
    failures = 0
    attempt_idx = 0

    while True:
        attempt_idx += 1
        agent = benchmark.build_agent(cfg)
        result = agent.solve_task(
            task_id,
            heuristics=[current_memory] if current_memory else None,
            trace_suffix=f"{trace_prefix}attempt{attempt_idx}",
            instruction_override=instruction_override,
            evaluator=evaluator,
            task_override=task_override,
        )
        _add_solver_cost(summary, result)
        success = bool(result.get("success", False))
        outcome = "success" if success else "failure"
        trajectory_text = benchmark.format_trajectory(result)

        previous_memory = current_memory
        if success:
            consecutive_successes += 1
            if not current_memory and consecutive_successes >= num_success_needed:
                summary["solved_without_memory"] = True
        else:
            consecutive_successes = 0
            failures += 1
            current_memory = generate_memory_item(
                llm=_extraction_client(extraction_llm, task_id),
                instruction=result.get("task_instruction", task_id),
                trajectory_text=trajectory_text,
                outcome=outcome,
                current_memory=current_memory,
                attempt_index=attempt_idx,
                max_failures=max_failures,
                benchmark=cfg.benchmark,
            )

        summary["attempts"].append({
            "attempt": attempt_idx,
            "outcome": outcome,
            "consecutive_successes": consecutive_successes,
            "memory": current_memory,
            "memory_diff": _memory_diff(previous_memory, current_memory, attempt_idx),
            "skipped_generation": success,
        })
        console.info(
            f"    attempt {attempt_idx}: {outcome} | "
            f"consecutive_successes={consecutive_successes} | "
            f"failures={failures}/{max_failures} | "
            f"memory_len={len(current_memory) if current_memory else 0}"
        )

        if consecutive_successes >= num_success_needed:
            console.info(f"    → move on (reached {num_success_needed} consecutive success(es))")
            break
        if failures >= max_failures:
            console.info(f"    → give up (reached max_failures={max_failures})")
            break

    summary["final_outcome"] = outcome
    summary["final_memory"] = current_memory
    summary["last_trace_text"] = trajectory_text
    summary["usage"] = _usage_since(extraction_llm, usage_mark)
    return summary


def single_attempt_for_task(
    task_id: str,
    cfg: ExperimentConfig,
    extraction_llm: LLMClient,
    instruction_override: str | None = None,
    evaluator: Any | None = None,
    trace_prefix: str = "",
    task_override: Any | None = None,
) -> dict[str, Any]:
    """Table 3, row C: one memory-free attempt, one judgment, one extraction.

    No retry, and the extracted heuristic is never tested. Extraction runs on successes
    too: a success yields a procedure, a failure a lesson from the observed blocker.
    """
    benchmark = get_benchmark(cfg)
    summary = _new_summary(task_id)
    summary["mode"] = "single_attempt"
    usage_mark = _usage_mark(extraction_llm)

    result = benchmark.build_agent(cfg).solve_task(
        task_id,
        heuristics=[],
        trace_suffix=f"{trace_prefix}attempt1",
        instruction_override=instruction_override,
        evaluator=evaluator,
        task_override=task_override,
    )
    _add_solver_cost(summary, result)
    outcome = "success" if result.get("success", False) else "failure"
    trajectory_text = benchmark.format_trajectory(result)
    memory = generate_memory_item(
        llm=_extraction_client(extraction_llm, task_id),
        instruction=result.get("task_instruction", instruction_override or task_id),
        trajectory_text=trajectory_text,
        outcome=outcome,
        current_memory=None,
        attempt_index=1,
        max_failures=1,
        benchmark=cfg.benchmark,
    ).strip()

    summary["attempts"].append({
        "attempt": 1, "outcome": outcome, "memory": memory or None,
        "memory_diff": None, "skipped_generation": False,
    })
    summary.update(
        final_outcome=outcome,
        final_memory=memory or None,
        solved_without_memory=outcome == "success",
        last_trace_text=trajectory_text,
        usage=_usage_since(extraction_llm, usage_mark),
    )
    console.info(f"    single attempt: {outcome} | extracted memory_len={len(memory)}")
    return summary
