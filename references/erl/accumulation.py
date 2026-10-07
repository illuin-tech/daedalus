"""ERL experience accumulation — build the heuristic pool from labeled source tasks.

    uv run python -m references.erl.accumulation --config references/erl/configs/tau2_accumulation.yaml

One pass over the source split. Per task: ONE memory-free rollout, read the environment's
binary reward, reflect once (paper Fig. 8), bank the heuristic. Every task banks — successes
and failures alike, because the paper learns from both.

This is ERL's counterpart to `daedalus.scripts.accumulation`, and deliberately the same
shape on disk (`pool.json`, `pool.log.json`, `task_summaries/`, `traces/`), so the `serve`
viewer and any pool consumer work unchanged — one level deeper, under
`outputs/baselines/erl/memory/<name>/`, so ERL runs stay out of daedalus's tree.
The algorithm inside is what differs:

    DAEDALUS  retry loop — attempt k+1 re-runs WITH the memory written at attempt k, until a
            success streak or the failure budget; only band-appropriate tasks bank.
    ERL     one attempt, no retries, no memory in context; every task banks exactly one
            heuristic, tagged with its outcome.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from daedalus.core.config import ExperimentConfig, pool_output_path
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.logging import console
from daedalus.core.memory.pool import MemoryPool
from daedalus.core.registry import get_benchmark

from references.common import accumulation as common
from references.common.usage import write_run_summary
from references.erl.config import SETUP, ERLConfig
from references.erl.heuristics import Heuristic, reward_label, to_memory_item
from references.erl.reflection import Usage, generate_heuristic


def accumulate_task(
    task_id: str, cfg: ExperimentConfig, reflection_llm: LLMClient
) -> dict[str, Any]:
    """Run one source task once and reflect on it. Returns the task's summary record.

    The summary carries DAEDALUS's accumulation field names (`attempts`, `final_outcome`,
    `final_memory`, `solved_without_memory`) so the viewer reads ERL runs too; ERL's list
    is always exactly one attempt.
    """
    benchmark = get_benchmark(cfg)
    agent = benchmark.build_agent(cfg)

    # heuristics=[] — an EMPTY list, not None: it keeps the runner on its accumulation
    # path (re-run the task, save the attempt trace) while injecting nothing. The rollout
    # must be memory-free; that is what makes the reflection a lesson about the
    # environment rather than about earlier advice.
    result = agent.solve_task(task_id, heuristics=[], trace_suffix="attempt1")

    success = bool(result.get("success", False))
    instruction = result.get("task_instruction") or task_id
    trajectory = benchmark.format_trajectory(result)

    reflection = Usage()
    heuristic_text = generate_heuristic(
        llm=reflection_llm,
        task=instruction,
        trajectory=trajectory,
        success=success,
        usage=reflection,
    )

    outcome = "success" if success else "failure"
    tokens = result.get("total_tokens") or {}
    return {
        "task_id": task_id,
        "method": "erl",
        "task": instruction,
        "reward": reward_label(success),
        "attempts": [
            {
                "attempt": 1,
                "outcome": outcome,
                "consecutive_successes": 1 if success else 0,
                "memory": heuristic_text,
                "memory_diff": None,
                "skipped_generation": False,
            }
        ],
        "final_outcome": outcome,
        "final_memory": heuristic_text,
        # ERL never injects memory during accumulation, so this is just the outcome of
        # the single memory-free rollout — the paper's reward signal.
        "solved_without_memory": success,
        "cost_usd": float(result.get("total_cost_usd", 0.0) or 0.0),
        "tokens": {
            k: int(tokens.get(k, 0) or 0) for k in ("prompt", "completion", "cached")
        },
        "reflection": reflection.to_dict(),
    }


def _task_worker(
    task_id: str,
    config_path: str,
    task_index: int,
    n_tasks: int,
    experiment_name: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Top-level worker for ProcessPoolExecutor (must be importable and picklable)."""
    # memory.enabled is off there: the rollout is memory-free by construction.
    cfg = common.worker_config(config_path, SETUP, experiment_name)

    accounting = CostAccounting.for_worker(cfg)
    reflection_llm = accounting.client(
        cfg.accumulation.extraction_model,
        "reflection",
        temperature=0.0,
        reasoning_effort=cfg.accumulation.extraction_reasoning_effort,
    )

    console.info(f"[{task_index + 1}/{n_tasks}] Task: {task_id} (start)", flush=True)
    summary = accumulate_task(task_id, cfg, reflection_llm)
    console.info(
        f"[{task_index + 1}/{n_tasks}] Task: {task_id} → {summary['final_outcome']} "
        f"| heuristic_len={len(summary['final_memory'] or '')}",
        flush=True,
    )

    # Checkpoint immediately: the pool is only assembled once every task is done, so
    # without this a late crash would throw away every finished task's heuristic.
    common.write_json(common.summary_path(cfg, task_id), summary)
    return task_index, summary


def build_pool(summaries: list[dict[str, Any]]) -> MemoryPool:
    """Assemble the heuristic pool. Every task with a heuristic banks, either outcome."""
    now = datetime.now(timezone.utc).isoformat()
    pool = MemoryPool()
    for summary in summaries:
        if not summary.get("final_memory"):
            continue
        pool.add(
            to_memory_item(
                Heuristic(
                    scenario_id=summary["task_id"],
                    task=summary.get("task", ""),
                    reward=summary.get("reward")
                    or reward_label(summary["final_outcome"] == "success"),
                    text=summary["final_memory"],
                ),
                timestamp=now,
            )
        )
    return pool


def main() -> None:
    args, cfg = common.start(
        SETUP, "ERL accumulation: one rollout + one reflection per source task"
    )
    accounting = CostAccounting.start(
        cfg,
        extra_models=[
            ModelRequirement(
                cfg.accumulation.extraction_model,
                "reflection",
                "accumulation.extraction_model",
            )
        ],
    )
    ERLConfig.load(args.config).save(cfg)  # erl.yaml — the run folder stays self-describing

    task_ids = common.task_ids(cfg, args.task_id)
    n_workers = max(1, cfg.run.parallel or 1)
    console.header(
        "erl · accumulation",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "solver": cfg.agent.model,
            "reflection": cfg.accumulation.extraction_model,
            "tasks": len(task_ids),
            "parallel": n_workers,
            "output": pool_output_path(cfg).parent,
        },
    )

    n_tasks = len(task_ids)
    ordered, pending = common.restore(task_ids, lambda t: common.summary_path(cfg, t))
    for index, summary in common.run_tasks(
        _task_worker,
        pending,
        lambda i, tid: (tid, args.config, i, n_tasks, cfg.name),
        n_workers,
    ):
        ordered[index] = summary

    summaries = [s for s in ordered if s is not None]
    pool = build_pool(summaries)

    # overwrite=True: the pool is rebuilt from ALL checkpoints on every invocation, so a
    # resumed run's file is a superset of the previous one. Letting daedalus's default
    # suffix it (pool_1.json) would instead leave a stale, shorter pool.json behind — and
    # that is the path inference configs point at.
    pool_path = pool.save(pool_output_path(cfg), overwrite=True)
    summary_path = write_run_summary(cfg, accounting, {"num_tasks": len(summaries)})
    console.info(f"Run summary (whole-run spend, by role) → {summary_path}")
    log_path = pool_path.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rewards = Counter(s["final_outcome"] for s in summaries)
    solver_cost = sum(float(s.get("cost_usd", 0.0) or 0.0) for s in summaries)
    reflection_cost = sum(
        (s.get("reflection") or {}).get("cost_usd", 0.0) for s in summaries
    )
    console.rule("erl accumulation complete")
    console.info(
        f"Tasks: {len(summaries)} | solved: {rewards['success']} | failed: {rewards['failure']}"
    )
    console.info(f"Banked heuristics: {len(pool)} (successes and failures both bank)")
    console.info(
        f"Cost: solver ${solver_cost:.2f} + reflection ${reflection_cost:.2f} "
        f"= ${solver_cost + reflection_cost:.2f}"
    )
    console.info(f"Pool: {pool_path}")
    console.info(f"Log:  {log_path}")


if __name__ == "__main__":
    main()
