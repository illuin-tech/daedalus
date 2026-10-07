"""Inference runner: solve the configured tasks (single- or multi-run), then auto-score.

Called by the `daedalus.scripts.inference` entry point via `run_experiment(cfg)`.
"""

from __future__ import annotations

from pathlib import Path

from daedalus.core.registry import get_benchmark
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.evaluation.evaluate import evaluate_experiment
from daedalus.core.config import (
    ExperimentConfig,
    resolve_trace_dir,
    save_run_config,
)


def _auto_evaluate(cfg: ExperimentConfig, task_ids: list[str]) -> None:
    """Score the finished run(s). A scoring failure must not discard the completed run."""
    try:
        evaluate_experiment(cfg, task_ids)
    except Exception as e:  # noqa: BLE001
        import traceback

        console.info(f"evaluation skipped: {type(e).__name__}: {e}")
        traceback.print_exc()


def _memory_label(cfg: ExperimentConfig) -> str:
    """One-word memory description for a run header."""
    if not cfg.memory.enabled:
        return "off"
    if cfg.memory.heuristics_at_start:
        return "all-at-start"
    return cfg.memory.retriever.type


def _resolve_task_ids(cfg: ExperimentConfig) -> list[str]:
    """Resolve the list of task IDs from config."""
    task_ids = get_benchmark(cfg).list_task_ids(cfg)
    if cfg.run.max_tasks is not None:
        task_ids = task_ids[: cfg.run.max_tasks]
    return task_ids


def _is_run_complete(trace_dir: Path, task_ids: list[str]) -> bool:
    """Whether every configured task has a FINISHED trace in this run directory.

    "The file exists" is not the same as "the task finished": a worker killed mid-task
    leaves a partial trace (save_every_turn writes one after every turn), and counting it
    as done meant the task was never re-run and its unfinished trajectory was scored. A
    trace is finished when it parses and carries the `success` verdict the scorer reads.
    """
    if not trace_dir.exists():
        return False
    import json

    finished: set[str] = set()
    for path in trace_dir.glob("*.json"):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # torn write from a killed worker — re-run the task
        if isinstance(trace, dict) and "success" in trace:
            finished.add(str(trace.get("task_id") or path.stem))
    return all(tid in finished for tid in task_ids)


def run_experiment(cfg: ExperimentConfig) -> None:
    """Run the agent on the configured tasks, with multi-run support."""
    console.configure(cfg.logging.verbose)
    save_run_config(
        cfg
    )  # snapshot config.yaml + run_meta.json into the experiment folder
    # Before any paid work: resolve a price for every model this path will call, and open
    # this launch's usage ledger. An unknown model stops here rather than surfacing as a
    # $0.00 line in the summary. Task workers inherit the launch id through cfg.
    CostAccounting.start(cfg)
    benchmark = get_benchmark(cfg)
    task_ids = _resolve_task_ids(cfg)
    num_runs = cfg.run.num_runs

    # One code path for every num_runs, so traces always land in run_<idx>/ — including
    # num_runs=1, which used to write a FLAT directory instead. That special case made the
    # on-disk layout depend on a value people edit between runs: raising num_runs from 1 to 3
    # after a run finished left its traces in the flat folder where the resume check (which only
    # ever looks in run_<idx>/) could not see them, so a complete 417-task run silently re-ran.
    # Reading stays backward compatible: evaluation/evaluate.py::_detect_runs still understands
    # the old flat layout, so previously-written runs still score.
    console.header(
        "daedalus · inference",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "model": cfg.agent.model,
            "memory": _memory_label(cfg),
            "tasks": len(task_ids),
            "runs": num_runs,
            "parallel": cfg.run.parallel,
            "output": resolve_trace_dir(cfg, run_idx=0).parent,
        },
    )

    completed_runs = []
    pending_runs = []

    for run_idx in range(num_runs):
        trace_dir = resolve_trace_dir(cfg, run_idx=run_idx)
        if not cfg.run.force and _is_run_complete(trace_dir, task_ids):
            completed_runs.append(run_idx)
        else:
            pending_runs.append(run_idx)

    if completed_runs:
        console.info(f"Runs already complete: {completed_runs}")
    if not pending_runs:
        console.info("All runs already complete. Use --force to re-run.")
        _auto_evaluate(cfg, task_ids)
        return

    console.info(f"Runs to launch: {pending_runs}")

    if len(pending_runs) > 1 and cfg.run.parallel > 1:
        # ONE pool over every (run, task) pair, not one pool per run. Each pool pays a
        # tail: measured on a 168-task AppWorld eval at 24 workers, the last 24 tasks
        # take 19 of 30 minutes while workers drain with nothing to pick up, so five
        # sequential runs waste 80 of 143 minutes. Batching pays that tail once.
        from daedalus.core.agents.task_pool import solve_tasks_parallel

        console.rule(f"runs {pending_runs} — one pool, {cfg.run.parallel} workers")
        agent = benchmark.build_agent(cfg, run_idx=pending_runs[0])
        pairs = [(r, t) for r in pending_runs for t in task_ids]
        solve_tasks_parallel(
            agent,
            [t for _, t in pairs],
            cfg.run.parallel,
            run_ids=[r for r, _ in pairs],
        )
        console.info(f"runs {pending_runs} done")
    else:
        for run_idx in pending_runs:
            console.rule(f"run {run_idx}/{num_runs - 1}")
            agent = benchmark.build_agent(cfg, run_idx=run_idx)
            console.info(f"output: {agent.trace_dir}")
            agent.solve_tasks(task_ids)
            console.info(f"run {run_idx} done")

    console.rule()
    console.info(
        f"All runs complete. Traces in: {resolve_trace_dir(cfg, run_idx=0).parent}"
    )
    _auto_evaluate(cfg, task_ids)
