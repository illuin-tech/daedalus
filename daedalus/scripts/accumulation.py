"""DAEDALUS-curated (paper Algorithm 4): run the Solver loop on labeled training tasks.

Every task goes through `core/generation/accumulate.py::accumulate_for_task`, graded by
the benchmark's own verifier instead of the LLM judge. Heuristics of solved tasks are
banked into `outputs/daedalus-curated/<name>/pool.json`; consolidate it with
`daedalus.scripts.consolidate` before inference.

    uv run python -m daedalus.scripts.accumulation --config configs/appworld/accumulation/curated.yaml

A killed run resumes from its `task_summaries/` checkpoints when relaunched.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daedalus.core.config import (
    ExperimentConfig,
    experiment_dir,
    load_config,
    pool_output_path,
    save_run_config,
)
from daedalus.core.env import load_dotenv
from daedalus.core.generation.accumulate import accumulate_for_task
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.memory.pool import MemoryItem, MemoryPool
from daedalus.core.registry import get_benchmark


def _task_summary_path(cfg: ExperimentConfig, task_id: str) -> Path:
    return pool_output_path(cfg).parent / "task_summaries" / f"{task_id}.json"


def _accumulate_task_worker(
    task_id: str, config_path: str, task_index: int, n_tasks: int, launch_id: str = ""
) -> tuple[int, dict[str, Any]]:
    """Solve one task in a fresh process (no shared SQLite connections or HTTP sessions).

    `launch_id` is the parent's usage-ledger launch; the worker reloads the config from
    YAML, which does not carry it.
    """
    cfg = load_config(config_path)
    console.silence_third_party(cfg.logging.verbose)
    cfg.kind = "accumulation"
    cfg.memory.enabled = False  # the heuristic is injected by the loop, not retrieved
    cfg.cost_accounting.launch_id = launch_id or cfg.cost_accounting.launch_id
    get_benchmark(cfg).prepare_accumulation(cfg)

    extraction_llm = CostAccounting.for_worker(cfg).client(
        cfg.accumulation.extraction_model,
        "extraction",
        temperature=0.0,
        reasoning_effort=cfg.accumulation.extraction_reasoning_effort,
    )
    console.info(f"[{task_index + 1}/{n_tasks}] Task: {task_id} (start)", flush=True)
    summary = accumulate_for_task(task_id, cfg, extraction_llm)
    console.info(
        f"[{task_index + 1}/{n_tasks}] Task: {task_id} → {summary['final_outcome']}", flush=True
    )
    # Persist immediately: the pool is assembled at the end, so a late crash would
    # otherwise lose every finished task.
    path = _task_summary_path(cfg, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return task_index, summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Solver loop on labeled tasks")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--task-id", type=str, default=None, help="Single task ID")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to process")
    parser.add_argument("--parallel", type=int, default=None, help="Parallel workers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    console.configure(cfg.logging.verbose)
    cfg.kind = "accumulation"
    if args.max_tasks is not None:
        cfg.run.max_tasks = args.max_tasks
    if args.parallel is not None:
        cfg.run.parallel = args.parallel
    save_run_config(cfg)
    accounting = CostAccounting.start(cfg)
    launch_id = accounting.base_scope.launch_id

    task_ids = [args.task_id] if args.task_id else get_benchmark(cfg).list_task_ids(cfg)
    if cfg.run.max_tasks is not None:
        task_ids = task_ids[: cfg.run.max_tasks]
    n_tasks = len(task_ids)
    n_workers = max(cfg.run.parallel or 1, 1)
    console.header(
        "daedalus · accumulation",
        {
            "benchmark": cfg.benchmark,
            "tasks": n_tasks,
            "max_failures": cfg.accumulation.max_failures,
            "num_success": cfg.accumulation.num_success_to_continue,
            "parallel": n_workers,
        },
    )
    get_benchmark(cfg).prepare_accumulation(cfg)

    ordered: list[dict[str, Any] | None] = [None] * n_tasks
    pending: list[tuple[int, str]] = []
    for i, task_id in enumerate(task_ids):
        checkpoint = _task_summary_path(cfg, task_id)
        if checkpoint.exists():
            ordered[i] = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            pending.append((i, task_id))
    if len(pending) < n_tasks:
        console.info(f"Resuming: {n_tasks - len(pending)} task(s) restored from checkpoints")

    if n_workers == 1:
        for i, task_id in pending:
            _, ordered[i] = _accumulate_task_worker(task_id, args.config, i, n_tasks, launch_id)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_accumulate_task_worker, tid, args.config, i, n_tasks, launch_id)
                for i, tid in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                idx, summary = future.result()
                ordered[idx] = summary
    summaries = [s for s in ordered if s is not None]

    # Only heuristics of tasks the solver ended up solving are banked (too-hard tasks
    # are dropped, as in generation).
    now = datetime.now(timezone.utc).isoformat()
    pool = MemoryPool([
        MemoryItem(
            memory_id=f"m_{uuid.uuid4().hex[:8]}",
            type="reflection",
            text=s["final_memory"],
            source_task_id=s["task_id"],
            source_trajectory_success=True,
            extraction_timestamp=now,
        )
        for s in summaries
        if s["final_memory"] and s["final_outcome"] == "success"
    ])
    output_path = pool.save(pool_output_path(cfg), overwrite=True)
    output_path.with_suffix(".log.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    n_solved = sum(1 for s in summaries if s["final_outcome"] == "success")
    usage = accounting.summary()
    console.rule("accumulation complete")
    console.info(f"Tasks: {len(summaries)} | Solved: {n_solved}/{len(summaries)}")
    console.info(f"Banked heuristics: {len(pool)} → {output_path}")
    console.info(
        "cost by role: "
        + "  ".join(f"{k} ${v:.4f}" for k, v in usage["cost_usd_by_role"].items())
    )
    (experiment_dir(cfg) / "run_summary.json").write_text(
        json.dumps(
            {
                "experiment_name": cfg.name,
                "benchmark": cfg.benchmark,
                "kind": "accumulation",
                "num_tasks": len(summaries),
                "num_solved": n_solved,
                "banked_heuristics": len(pool),
                "usage": usage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
