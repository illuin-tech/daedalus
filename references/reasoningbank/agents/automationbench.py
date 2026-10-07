"""AutomationBench solver with ReasoningBank retrieval.

One embedding search per task picks the top-k most similar past experiences and injects
their memory items into the system prompt. `_build_system_prompt(task, heuristics)` is
where that happens, which `solve_task` reaches only after its skip check — so a resumed
run pays no embedding pass for a task it is about to skip.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.automationbench.agent import AutomationBenchTaskAgent
from daedalus.benchmarks.automationbench.task_loader import Task
from daedalus.core.config import ExperimentConfig

from references.reasoningbank.agents.base import ReasoningBankSolverMixin


class ReasoningBankAutomationBenchAgent(
    ReasoningBankSolverMixin, AutomationBenchTaskAgent
):
    """AutomationBench agent whose system prompt carries the retrieved memory items."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_reasoningbank(cfg)

    def _build_system_prompt(self, task: Task, heuristics: list[str]) -> str:
        retrieved = self._reasoningbank_blocks(task.instruction)
        return super()._build_system_prompt(task, (heuristics or []) + retrieved)

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
