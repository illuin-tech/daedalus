"""AppWorld solver carrying ACE's trained playbook."""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.appworld.agent import ReActAgent
from daedalus.core.config import ExperimentConfig

from references.ace.agents.base import ACESolverMixin


class ACEAppWorldAgent(ACESolverMixin, ReActAgent):
    """ReAct agent whose system prompt carries the whole playbook, verbatim, at task start.

    No prompt-building override is needed: `_build_initial_messages` renders
    `self.solver_playbook` into the shared template's `playbook` slot, and `_init_ace` is
    what fills that attribute.
    """

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
