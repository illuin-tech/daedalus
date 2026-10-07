"""tau2-bench task-selection + simulation config (imported by daedalus.core.config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tau2Config:
    """tau2-bench task selection and simulation settings.

    Solver modes:
      conversational — official tau2 setup: LLM user simulator plays the scenario
        (used for eval-time inference, comparable to published numbers).
      ticket — deterministic scripted user delivers the full task as a ticket in
        its first message, then stops. No user-sim LLM cost/noise; used for memory
        accumulation and self-play generation ("learn before talking to users").
        Runs on the standard orchestrator path, so it works on every domain.
    """

    domain: str = "retail"  # the paper uses retail only
    task_split: str | None = "base"  # tau2 split name; None = all tasks
    task_ids: list[str] | None = None
    solver_mode: str | None = None  # conversational | ticket | None = by kind
    user_llm: str = "gpt-4.1-2025-04-14"  # litellm model string (conversational)
    user_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    max_steps: int = 100  # orchestrator hops (agent+user+env), not LLM turns
    max_errors: int = 10
    seed: int = 42
