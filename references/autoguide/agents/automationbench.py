"""AutomationBench solver driven by AutoGuide's context-aware guidelines.

AutoGuide retrieves EVERY turn, so unlike the other three methods it does not go through
the system prompt. AutomationBench routes its per-turn retrieval through `SolverMemory`
(daedalus/core/agents/memory.py), which is a `Retriever` seam — so the integration is to
install `ContextAwareRetriever` there and let the agent's own loop call it.

`memory.enabled: false` in the config: `SolverMemory.__init__` then builds no daedalus
retriever and no pool, since the bank AutoGuide injects is its own (`autoguide.bank_path`).
The retriever's `whole_trace_query` makes `SolverMemory.build_query` hand over the
trajectory so far rather than the last-turn window (context identification needs τ:t).
"""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.automationbench.agent import AutomationBenchTaskAgent
from daedalus.core.config import ExperimentConfig

from references.autoguide.agents.base import AutoGuideSolverMixin


class AutoGuideAutomationBenchAgent(AutoGuideSolverMixin, AutomationBenchTaskAgent):
    """AutomationBench agent whose per-turn injection is AutoGuide's context + guidelines."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_autoguide(cfg)
        # SolverMemory.retrieve passes cfg.memory.retriever.top_k, and AutoGuide's k is
        # the number of guidelines the paper injects per turn.
        self.cfg.memory.retriever.top_k = self.method_cfg.k
        # Hand the agent's own per-turn seam to AutoGuide's retriever.
        self.memory.retriever = self.retriever
        self.memory._injection = self._injection_tracker

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._finish_task(task_id)
        return trace
