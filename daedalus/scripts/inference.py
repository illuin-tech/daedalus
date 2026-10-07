"""Run inference on a benchmark — the single inference entry point.

Whether memory retrieval is used is decided by the config's `memory.enabled`
(and the pool it points at). Pass --memory / --no-memory to override it for a
one-off run without editing the YAML.

    uv run python -m daedalus.scripts.inference --config configs/appworld/inference/baseline.yaml
    uv run python -m daedalus.scripts.inference --config configs/appworld/inference/daedalus.yaml
"""

from __future__ import annotations

import argparse

from daedalus.core.config import load_config, resolve_experiment_name
from daedalus.core.env import load_dotenv
from daedalus.core.evaluation.run_experiment import run_experiment


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run inference on a benchmark")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment_name → outputs/<kind>/<name>/. "
        "Supports {benchmark}/{domain}/{model}/{date} tokens.",
    )
    parser.add_argument(
        "--task-id", type=str, nargs="+", default=None, help="Task ID(s) to run"
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Max number of tasks")
    parser.add_argument("--parallel", type=int, default=None, help="Parallel workers")
    parser.add_argument("--num-runs", type=int, default=None, help="Independent runs")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")

    # Memory retrieval: default follows the config; these flags override it.
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument(
        "--memory",
        dest="memory",
        action="store_true",
        default=None,
        help="Enable memory retrieval (overrides config)",
    )
    memory.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="Disable memory retrieval (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.experiment_name:
        cfg.experiment_name = resolve_experiment_name(args.experiment_name, cfg)
    if args.memory is not None:
        cfg.memory.enabled = args.memory
    if args.task_id:
        cfg.benchmark_config.task_ids = args.task_id
    if args.max_tasks is not None:
        cfg.run.max_tasks = args.max_tasks
    if args.parallel is not None:
        cfg.run.parallel = args.parallel
    if args.num_runs is not None:
        cfg.run.num_runs = args.num_runs
    if args.force:
        cfg.run.force = True

    run_experiment(cfg)


if __name__ == "__main__":
    main()
