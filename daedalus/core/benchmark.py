"""Benchmark interface: the seam a new benchmark implements to join the harness.

A Benchmark provides tasks, builds the solver agent, and scores completed runs.
The solver agent it returns is a duck-typed "TaskAgent" with the same surface
as agents.base_agent.ReActAgent:

    __init__(cfg, run_idx=None)
    solve_task(task_id, heuristics=None, trace_suffix="",
               instruction_override=None, evaluator=None,
               task_override=None) -> dict   # normalized trace dict
    solve_tasks(task_ids) -> list[dict]
    trace_dir: Path

solve_task writes/returns the normalized trace format (logging.trace_logger.Trace),
which the memory-extraction prompts, metrics, and the viewer all consume.
`instruction_override`/`task_override` + `evaluator` are the self-play generation
hooks: run a generated task and grade it (benchmark-native reward or callback).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig


class Benchmark(ABC):
    """One benchmark's integration point (tasks, solver agent, run scoring)."""

    name: str

    @abstractmethod
    def list_task_ids(self, cfg: ExperimentConfig) -> list[str]:
        """Resolve the configured task ids (dataset/split + explicit filter)."""

    @abstractmethod
    def build_agent(self, cfg: ExperimentConfig, run_idx: int | None = None) -> Any:
        """Construct the solver TaskAgent for this benchmark and config."""

    @abstractmethod
    def evaluate_run(
        self,
        cfg: ExperimentConfig,
        trace_dir: Path,
        task_ids: list[str] | None,
        run_idx: int | None,
    ) -> dict[str, Any]:
        """Score one completed run directory -> {num_tasks, num_successes, success_rate, ...}."""

    @abstractmethod
    def format_trajectory(self, trace: dict[str, Any]) -> str:
        """Render a normalized trace as text for memory-extraction prompts."""

    def prepare_accumulation(self, cfg: ExperimentConfig) -> None:
        """One-time process setup before accumulation runs (default: nothing)."""

    def prompt_dirs(self, cfg: ExperimentConfig) -> list[Path]:
        """This benchmark's prompt search path, most specific first.

        Everything that composes a prompt (core.resources) takes a search path rather
        than one directory, so a benchmark wrapping several environments can put the
        environment-bound wording in its own subdirectory and let the rest fall through
        to the shared copy. The default is the single `benchmarks/<name>/prompts`, which
        is all a benchmark with one environment ever needs.
        """
        return [Path(__file__).parent.parent / "benchmarks" / self.name / "prompts"]
