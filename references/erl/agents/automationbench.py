"""AutomationBench solver with ERL retrieval.

The AutomationBench agent builds its system prompt once per task, in
`_build_system_prompt(task, heuristics)`, which is the first point at which the task
instruction is known — the same seam the AppWorld agent uses. Overriding it is the whole
integration: the ranker picks this task's heuristics and they are rendered alongside
whatever the caller passed.

`solve_task` performs its skip check BEFORE `_solve` calls `_build_system_prompt`, so a
resumed run never pays for the ranker on a task it is about to skip. That is why there is
no `_would_skip` gate here, unlike the tau2 runner where retrieval resolves earlier.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.automationbench.agent import AutomationBenchTaskAgent
from daedalus.benchmarks.automationbench.task_loader import Task
from daedalus.core.config import ExperimentConfig

from references.erl.agents.base import ERLSolverMixin


class ERLAutomationBenchAgent(ERLSolverMixin, AutomationBenchTaskAgent):
    """AutomationBench agent whose system prompt carries the heuristics ERL selected."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_erl(cfg)

    def _build_system_prompt(self, task: Task, heuristics: list[str]) -> str:
        selected = self._erl_heuristics(task.instruction)
        return super()._build_system_prompt(task, (heuristics or []) + selected)

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
