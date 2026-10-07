"""AutomationBench solver driven by ExpeL's insights and recalled trajectories.

Same seam as the ERL and ReasoningBank agents: `_build_system_prompt(task, heuristics)`
runs once per task, after `solve_task` has made its skip decision, so recall never fires
for a task a resumed run is about to skip.

A note on this benchmark. ExpeL injects the top-`fewshot_k` most task-similar SUCCESSFUL
trajectories verbatim. On tau2 and AppWorld every task sees the same environment, so a
recalled trajectory is always executable in the world the solver is now in. AutomationBench
seeds a different subset of its 22 services per task, so a recalled trajectory can call an
API that this task's world does not expose at all. That is the method's own behaviour on a
multi-world benchmark and it is left unmodified; `references/expel/README.md` records how
to measure how often it happens.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.automationbench.agent import AutomationBenchTaskAgent
from daedalus.benchmarks.automationbench.task_loader import Task
from daedalus.core.config import ExperimentConfig

from references.expel.agents.base import ExpeLSolverMixin


class ExpeLAutomationBenchAgent(ExpeLSolverMixin, AutomationBenchTaskAgent):
    """AutomationBench agent whose system prompt carries insights + few-shot examples."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_expel(cfg)

    def _build_system_prompt(self, task: Task, heuristics: list[str]) -> str:
        blocks = self._expel_blocks(task.instruction)
        return super()._build_system_prompt(task, (heuristics or []) + blocks)

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
