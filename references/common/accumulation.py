"""Plumbing shared by every method's `accumulation` entry point.

What each method learns, and from what, is in its own `accumulation.py`; what is identical
across them lives here: the CLI and config loading, the task list, the per-task checkpoints
a run resumes from, the config a spawned worker rebuilds, and the process pool that runs
the tasks (one task that dies is reported and dropped, never fatal: it leaves no
checkpoint, so a re-run picks it up again).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Callable

from daedalus.core.config import (
    ExperimentConfig,
    load_config,
    pool_output_path,
    resolve_experiment_name,
    save_run_config,
)
from daedalus.core.env import load_dotenv
from daedalus.core.logging import console
from daedalus.core.registry import get_benchmark

from references.common.config import require_kind


def start(
    setup: str, description: str, parallel_help: str = "Parallel workers"
) -> tuple[argparse.Namespace, ExperimentConfig]:
    """Parse the common flags, load the config and write `config.yaml` + `run_meta.json`.

    Every method takes the same flags as `daedalus.scripts.accumulation`
    (`--config --experiment-name --task-id --max-tasks --parallel`).
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=f"Override experiment_name → outputs/baselines/{setup}/memory/<name>/. "
        "Supports {benchmark}/{domain}/{model}/{date} tokens.",
    )
    parser.add_argument("--task-id", type=str, nargs="+", default=None, help="Task ID(s)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to process")
    parser.add_argument("--parallel", type=int, default=None, help=parallel_help)
    args = parser.parse_args()

    require_kind(args.config, "accumulation", setup)
    cfg = load_config(args.config)
    console.configure(cfg.logging.verbose)
    # → outputs/baselines/<setup>/memory/<name>/
    cfg.kind, cfg.setup = "accumulation", setup
    if args.experiment_name:
        cfg.experiment_name = resolve_experiment_name(args.experiment_name, cfg)
    if args.max_tasks is not None:
        cfg.run.max_tasks = args.max_tasks
    if args.parallel is not None:
        cfg.run.parallel = args.parallel

    save_run_config(cfg)  # config.yaml + run_meta.json
    return args, cfg


def task_ids(cfg: ExperimentConfig, explicit: list[str] | None) -> list[str]:
    """`--task-id` if given, else the benchmark's source split; capped by `run.max_tasks`."""
    ids = explicit or get_benchmark(cfg).list_task_ids(cfg)
    if cfg.run.max_tasks is not None:
        ids = ids[: cfg.run.max_tasks]
    return ids


def summary_path(cfg: ExperimentConfig, task_id: str) -> Path:
    """Per-task checkpoint: `<experiment_dir>/task_summaries/<task>.json`."""
    return pool_output_path(cfg).parent / "task_summaries" / f"{task_id}.json"


def write_json(path: Path, data: Any) -> None:
    """Write a checkpoint (creating its folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def restore(
    ids: list[str], checkpoint_of: Callable[[str], Path]
) -> tuple[list[dict[str, Any] | None], list[tuple[int, str]]]:
    """Resume: reuse the per-task checkpoints an interrupted run left behind.

    Returns `ordered` (one slot per task, filled where a checkpoint exists) and `pending`
    (the `(index, task_id)` pairs still to run).
    """
    ordered: list[dict[str, Any] | None] = [None] * len(ids)
    pending: list[tuple[int, str]] = []
    for i, task_id in enumerate(ids):
        checkpoint = checkpoint_of(task_id)
        if checkpoint.exists():
            ordered[i] = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            pending.append((i, task_id))
    if len(pending) < len(ids):
        console.info(f"Resuming: {len(ids) - len(pending)} task(s) restored from checkpoints")
    return ordered, pending


def worker_config(config_path: str, setup: str, experiment_name: str | None) -> ExperimentConfig:
    """The config a spawned worker rebuilds from the config PATH.

    Each subprocess rebuilds the config, the environment and its LLM clients from scratch,
    so nothing (SQLite handles, HTTP sessions) is shared across workers. Only the path
    crosses the process boundary, so a CLI-resolved experiment name is passed along
    explicitly — it decides where traces and checkpoints land. daedalus's own retrieval is
    off: every method here injects (or withholds) its memory itself.
    """
    cfg = load_config(config_path)
    console.silence_third_party(cfg.logging.verbose)
    cfg.kind, cfg.setup = "accumulation", setup
    if experiment_name:
        cfg.experiment_name = experiment_name
    cfg.memory.enabled = False
    get_benchmark(cfg).prepare_accumulation(cfg)
    return cfg


def run_tasks(
    worker: Callable[..., tuple[int, Any]],
    jobs: list[tuple[int, str]],
    args_of: Callable[[int, str], tuple],
    max_workers: int,
) -> list[tuple[int, Any]]:
    """Run `worker(*args_of(index, task_id))` for every job; results in task order.

    `max_workers == 1` runs in this process; otherwise a ProcessPoolExecutor, so `worker`
    must be a top-level (picklable) function returning `(index, result)`. A job that raises
    is reported and dropped.
    """
    results: list[tuple[int, Any]] = []
    if max_workers == 1:
        for index, task_id in jobs:
            try:
                results.append(worker(*args_of(index, task_id)))
            except Exception as e:  # noqa: BLE001
                console.info(f"  [error] {task_id}: {type(e).__name__}: {e}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, *args_of(index, task_id)): task_id
                for index, task_id in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:  # noqa: BLE001
                    console.info(f"  [error] {futures[future]}: {type(e).__name__}: {e}")
    return sorted(results, key=lambda r: r[0])
