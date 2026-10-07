"""AppWorld solver driven by AutoGuide's context-aware guidelines."""

from __future__ import annotations

from typing import Any

from daedalus.benchmarks.appworld.memory_agent import MemoryReActAgent
from daedalus.core.config import ExperimentConfig
from daedalus.core.logging.trace_logger import RetrievedMemoryRecord

from references.autoguide.agents.base import AutoGuideSolverMixin


def _trajectory_text(messages: list[dict[str, str]]) -> str:
    """The trajectory so far, as the context identification module should see it.

    Everything except the system prompt: the task, the agent's thoughts and code, and the
    execution output of each turn — i.e. the paper's τ:t.
    """
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


class AutoGuideAppWorldAgent(AutoGuideSolverMixin, MemoryReActAgent):
    """ReAct agent whose per-turn injection is AutoGuide's context + guidelines."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        super().__init__(cfg, run_idx=run_idx)
        self._init_autoguide(cfg)

    def _get_retrieved_memories(
        self, query: str, messages: list[dict[str, str]], turn_idx: int
    ) -> list[RetrievedMemoryRecord]:
        retrieved = self.retriever.retrieve(
            query=query, reasoning_trace=_trajectory_text(messages), top_k=self.method_cfg.k
        )
        return [
            RetrievedMemoryRecord(memory_id=r.memory_id, text=r.text, score=r.score, source=r.source)
            for r in retrieved
        ]

    def solve_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        trace = super().solve_task(task_id, **kwargs)
        if trace:  # empty dict = the task was skipped (a completed trace already exists)
            self._finish_task(task_id)
        return trace
