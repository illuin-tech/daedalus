"""τ²-bench solver carrying ACE's trained playbook.

Same as the other two: `_render_system_prompt` reads `self.solver_playbook`, so there is
nothing to override but construction and the per-task record. Unlike ExpeL and ERL this
agent needs no `_would_skip` gate — ACE selects nothing per task, so a task the runner is
about to skip costs no embedding pass and no LLM call.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.tau2.runner import Tau2TaskRunner
from daedalus.core.config import ExperimentConfig

from references.ace.agents.base import ACESolverMixin


class ACETau2Runner(ACESolverMixin, Tau2TaskRunner):
    """τ² runner whose system prompt carries the whole playbook, verbatim, at task start."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_ace(cfg)

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._mark_playbook()
            self._save_retrieval(task_id)
        return trace
