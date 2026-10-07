"""AppWorld solver driven by ExpeL's insights and recalled trajectories."""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.appworld.agent import ReActAgent
from daedalus.core.config import ExperimentConfig

from references.expel.agents.base import ExpeLSolverMixin


class ExpeLAppWorldAgent(ExpeLSolverMixin, ReActAgent):
    """ReAct agent whose system prompt carries ExpeL's insights + few-shot examples."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_expel(cfg)

    def _build_initial_messages(
        self,
        world,
        heuristics: list[str] | None = None,
        instruction_override: str | None = None,
    ) -> list[dict[str, str]]:
        instruction = instruction_override or world.task.instruction
        return super()._build_initial_messages(
            world,
            heuristics=(heuristics or []) + self._expel_blocks(instruction),
            instruction_override=instruction_override,
        )

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        self._retrieval = None
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
