"""τ²-bench solver driven by AutoGuide's context-aware guidelines.

The runner hands its agent a `retrieve(query)` callback and injects whatever it returns
into the turn's payload, so AutoGuide only has to own that callback. Two details:

* the callback's `query` is normally the last-turn window, while context identification
  needs the trajectory so far — the retriever's `whole_trace_query` makes the runner pass
  the whole conversation instead;
* the runner only wires the callback up when `self.retriever` is not None, which is why the
  retriever is installed on the instance rather than built from `memory.enabled`.
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.tau2.runner import Tau2TaskRunner
from daedalus.core.config import ExperimentConfig
from daedalus.core.logging.trace_logger import RetrievedMemoryRecord

from references.autoguide.agents.base import AutoGuideSolverMixin


class AutoGuideTau2Runner(AutoGuideSolverMixin, Tau2TaskRunner):
    """tau2 runner whose per-turn injection is AutoGuide's context + guidelines."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_autoguide(cfg)

    def _retrieve(self, query: str) -> list[RetrievedMemoryRecord]:
        retrieved = self.retriever.retrieve(
            query=query, reasoning_trace=query, top_k=self.method_cfg.k
        )
        return [
            RetrievedMemoryRecord(memory_id=r.memory_id, text=r.text, score=r.score, source=r.source)
            for r in retrieved
        ]

    def solve_task(
        self,
        task_id: str,
        heuristics: list[str] | None = None,
        trace_suffix: str = "",
        instruction_override: str | None = None,
        evaluator: Any | None = None,
        task_override: Any | None = None,
    ) -> dict[str, Any]:
        trace = super().solve_task(
            task_id,
            heuristics=heuristics,
            trace_suffix=trace_suffix,
            instruction_override=instruction_override,
            evaluator=evaluator,
            task_override=task_override,
        )
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._finish_task(task_id)
        return trace
