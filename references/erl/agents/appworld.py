"""AppWorld solver with ERL retrieval.

The base ReAct agent owns the whole solve loop; the only override is where the initial
messages are built, which is the first point at which the task instruction is known.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.appworld.agent import ReActAgent
from daedalus.core.config import ExperimentConfig

from references.erl.agents.base import ERLSolverMixin


class ERLAppWorldAgent(ERLSolverMixin, ReActAgent):
    """ReAct agent whose system prompt carries the heuristics selected for this task."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_erl(cfg)

    def _build_initial_messages(
        self,
        world,
        heuristics: list[str] | None = None,
        instruction_override: str | None = None,
    ) -> list[dict[str, str]]:
        instruction = instruction_override or world.task.instruction
        selected = self._erl_heuristics(instruction)
        return super()._build_initial_messages(
            world,
            heuristics=(heuristics or []) + selected,
            instruction_override=instruction_override,
        )

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
