"""τ²-bench solver with ERL retrieval.

The runner renders `(heuristics or []) + self._start_heuristics` into the system prompt at
task start, so ERL fills `_start_heuristics` per task and leaves the `heuristics` argument
alone — that argument doubles as the runner's "accumulation mode" switch, and flipping it
would disable the skip-on-existing-trace resume that inference relies on.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.tau2.runner import Tau2TaskRunner
from daedalus.core.config import ExperimentConfig

from references.erl.agents.base import ERLSolverMixin


class ERLTau2Runner(ERLSolverMixin, Tau2TaskRunner):
    """tau2 runner whose system prompt carries the heuristics selected for this task."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_erl(cfg)

    def solve_task(
        self,
        task_id: str,
        heuristics: list[str] | None = None,
        trace_suffix: str = "",
        instruction_override: str | None = None,
        evaluator: Any | None = None,
        task_override: Any | None = None,
    ) -> dict[str, Any]:
        self._retrieval = None
        task = task_override if task_override is not None else self._tasks_by_id.get(task_id)
        # The ranker sees the user scenario — the same text the trace records as the task
        # instruction. In conversational mode the SOLVER never receives it (it discovers
        # the request through the dialogue); only heuristic selection reads it, exactly as
        # daedalus's own retrievers do (`on_task_start(str(task.user_scenario))`).
        # Retrieval happens before the runner's own skip check, so gate it here too.
        retrieve = task is not None and not self._would_skip(task_id)
        self._start_heuristics = (
            self._erl_heuristics(str(task.user_scenario)) if retrieve else []
        )
        trace = super().solve_task(
            task_id,
            heuristics=heuristics,
            trace_suffix=trace_suffix,
            instruction_override=instruction_override,
            evaluator=evaluator,
            task_override=task_override,
        )
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._save_retrieval(task_id)
        return trace
