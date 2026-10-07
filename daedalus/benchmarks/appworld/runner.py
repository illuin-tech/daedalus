"""AppWorld implementation of the Benchmark interface.

Thin wrapper moving previously scattered AppWorld-specific logic (task listing,
agent dispatch, official evaluation) behind the benchmark seam. Only imported
when cfg.benchmark == "appworld", so tau2 runs never import appworld.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daedalus.core.benchmark import Benchmark
from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console


def split_by_world_dir(
    tasks_dir: Path, task_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Split ids into those AppWorld can grade and those whose world is missing.

    A task keeps its world under `<tasks_dir>/<id>/dbs`. When a worker is killed between
    opening the world and writing those DBs the directory never appears, and passing that id
    to `evaluate_tasks` raises "The from_db_home_path ... does not exist", which aborts the
    evaluation of EVERY run in the experiment. Filtering here turns that into one failed task.
    """
    gradable = [t for t in task_ids if (tasks_dir / t / "dbs").is_dir()]
    missing = set(task_ids) - set(gradable)
    return gradable, [t for t in task_ids if t in missing]


class AppWorldBenchmark(Benchmark):
    name = "appworld"

    def list_task_ids(self, cfg: ExperimentConfig) -> list[str]:
        from appworld.task import load_task_ids

        if cfg.appworld.task_ids:
            return cfg.appworld.task_ids
        return load_task_ids(cfg.appworld.dataset)

    def build_agent(self, cfg: ExperimentConfig, run_idx: int | None = None):
        from daedalus.benchmarks.appworld.agent import ReActAgent
        from daedalus.benchmarks.appworld.memory_agent import MemoryReActAgent

        if cfg.memory.enabled:
            return MemoryReActAgent(cfg, run_idx=run_idx)
        return ReActAgent(cfg, run_idx=run_idx)

    def evaluate_run(
        self,
        cfg: ExperimentConfig,
        trace_dir: Path,
        task_ids: list[str] | None,
        run_idx: int | None,
    ) -> dict[str, Any]:
        """Official AppWorld evaluation, reading the AppWorld on-disk outputs."""
        from appworld.evaluator import evaluate_tasks

        from daedalus.benchmarks.appworld import appworld_root
        from daedalus.benchmarks.appworld.agent import _appworld_safe_name
        from daedalus.core.evaluation.metrics import (
            aggregate_metrics,
            load_traces,
            task_success as trace_success,
        )

        # Use the same run-scoped name the agent used, and the same resolved data root
        # as the agent (appworld_root(), not a second hardcoded guess).
        safe_name = _appworld_safe_name(cfg.name)
        if run_idx is not None:
            safe_name = f"{safe_name}_run{run_idx}"
        tasks_dir = (
            Path(appworld_root() or "")
            / "experiments"
            / "outputs"
            / safe_name
            / "tasks"
        )

        if task_ids is None:
            # Prefer AppWorld's own outputs listing (trace stems can carry
            # attempt suffixes in accumulation layouts); fall back to traces.
            if tasks_dir.exists():
                task_ids = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
            else:
                task_ids = [p.stem for p in trace_dir.glob("*.json")]
        if not task_ids:
            return {"num_tasks": 0, "success_rate": 0.0}

        result: dict[str, Any] = {}
        if tasks_dir.exists():
            # A task whose world directory is missing makes AppWorld's evaluator raise
            # ("The from_db_home_path ... does not exist"), which aborts the WHOLE
            # evaluation — every other run of the experiment included. That happens
            # whenever a worker was killed between opening the world and writing its DBs.
            # Drop those ids here and let them score as failures, which is the same
            # conservative reading the `failures` default below uses for a task the
            # evaluator did not report.
            gradable, skipped = split_by_world_dir(tasks_dir, task_ids)
            if skipped:
                console.info(
                    f"  {len(skipped)} task(s) have no world directory and score as "
                    f"failures: {', '.join(skipped[:5])}"
                    + (" …" if len(skipped) > 5 else "")
                )
            evaluation = evaluate_tasks(task_ids=gradable, experiment_name=safe_name)
            individual = evaluation.get("individual", {})
            # A task is a success iff it has zero failures. The default [""] (one
            # element) makes a missing "failures" key count as NOT successful — the
            # conservative reading when the evaluator didn't report the task.
            # Keep the per-task map (not just the count): aggregate_across_runs needs
            # it to compute cross-run pass^k, same as tau2.
            per_task = {
                str(tid): isinstance(v, dict) and len(v.get("failures", [""])) == 0
                for tid, v in individual.items()
            }
            per_task.update({str(t): False for t in skipped})
            num_successes = sum(1 for ok in per_task.values() if ok)
            result["appworld_evaluation"] = evaluation
            result["task_success"] = per_task
            result["num_tasks"] = len(individual) if individual else len(task_ids)
            result["num_successes"] = num_successes
            result["success_rate"] = (
                num_successes / result["num_tasks"] if result["num_tasks"] > 0 else 0.0
            )
        else:
            # Fallback: use trace-level metrics
            traces = load_traces(trace_dir)
            metrics = aggregate_metrics(traces)
            result.update(metrics)
            result["task_success"] = {
                str(t["task_id"]): trace_success(t)
                for t in traces
                if t.get("task_id") is not None
            }

        return result

    def format_trajectory(self, trace: dict[str, Any]) -> str:
        from daedalus.core.memory.extraction import format_trajectory_text

        return format_trajectory_text(trace)

    def prepare_accumulation(self, cfg: ExperimentConfig) -> None:
        from appworld import AppWorld

        AppWorld.init_defaults.experiment_name = cfg.name
