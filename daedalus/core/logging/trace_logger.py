"""Structured trace logging for agent runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daedalus.core.logging.usage import TokenUsage


@dataclass
class RetrievedMemoryRecord:
    """A single retrieved memory item as logged in a trace."""

    memory_id: str
    text: str
    score: float
    source: str = ""


@dataclass
class Turn:
    """One turn of the agent loop."""

    turn_idx: int
    thought: str = ""
    reasoning_with_memory: str = ""
    query: str = ""
    retrieved_memories: list[RetrievedMemoryRecord] = field(default_factory=list)
    code: str = ""
    execution_output: str = ""
    execution_success: bool = True
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0}
    )


@dataclass
class EvaluationResult:
    """Task evaluation outcome."""

    success: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    details: str = ""
    # Scalar reward for benchmarks that produce one (tau2); None for pass/fail
    # benchmarks (AppWorld). success stays the boolean gate everywhere.
    reward: float | None = None


@dataclass
class Trace:
    """Full trace for a single task run."""

    task_id: str
    experiment_name: str
    config: dict[str, Any] = field(default_factory=dict)
    task_instruction: str = ""
    timestamp: str = ""
    success: bool = False
    num_turns: int = 0
    turns: list[Turn] = field(default_factory=list)
    # A PROJECTION of this task's usage events, not an independent estimate: the numbers
    # are the per-call figures the ledger already recorded, summed for the convenience of
    # per-task analysis. The ledger under <experiment_dir>/usage/ stays the source of
    # truth, and it is what survives a crash between the last call and this file.
    total_cost_usd: float = 0.0
    total_tokens: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "cached": 0}
    )
    cost_accounting_version: int = 2
    usage_event_ids: list[str] = field(default_factory=list)
    # True when every solver call in this task was priced locally; False means the
    # projection above under-reports and the run summary says why.
    cost_complete: bool = True
    # Cost of the benchmark-side user simulator LLM (tau2 conversational mode),
    # kept separate from total_cost_usd so agent costs stay comparable across
    # benchmarks. 0.0 when there is no user simulator.
    user_sim_cost_usd: float = 0.0
    # Benchmark-specific extras (e.g. tau2 termination_reason, xor fixup count).
    extra: dict[str, Any] = field(default_factory=dict)
    evaluation: EvaluationResult = field(default_factory=EvaluationResult)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class TraceLogger:
    """Accumulates turns and saves structured trace JSON for a task."""

    def __init__(
        self,
        task_id: str,
        experiment_name: str,
        config: dict[str, Any],
        trace_dir: Path,
        model: str = "",
        save_every_turn: bool = True,
        trace_suffix: str = "",
    ):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.model = model
        self.save_every_turn = save_every_turn
        self.trace_dir = trace_dir
        self.trace_suffix = trace_suffix
        self.trace = Trace(
            task_id=task_id,
            experiment_name=experiment_name,
            config=config,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._current_turn: Turn | None = None
        trace_dir.mkdir(parents=True, exist_ok=True)

    def log_task_instruction(self, instruction: str) -> None:
        self.trace.task_instruction = instruction

    def start_turn(self, turn_idx: int) -> None:
        self._current_turn = Turn(turn_idx=turn_idx)

    def log_thought(self, thought: str) -> None:
        if self._current_turn:
            self._current_turn.thought = thought

    def log_reasoning_with_memory(self, reasoning: str) -> None:
        if self._current_turn:
            self._current_turn.reasoning_with_memory = reasoning

    def log_query(self, query: str) -> None:
        if self._current_turn:
            self._current_turn.query = query

    def log_retrieved_memories(self, memories: list[RetrievedMemoryRecord]) -> None:
        if self._current_turn:
            self._current_turn.retrieved_memories = memories

    def log_code(self, code: str) -> None:
        if self._current_turn:
            self._current_turn.code = code

    def log_execution(self, output: str, success: bool) -> None:
        if self._current_turn:
            self._current_turn.execution_output = output
            self._current_turn.execution_success = success

    def log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> None:
        """Record one call's token counts on the current turn and the trace total.

        Cost is NOT computed here. It is whatever the call's own usage event says, which
        `log_llm_call` folds in — re-estimating from these three numbers is how the trace
        used to lose the service tier, the cache-write charge and the returned model.
        """
        if self._current_turn:
            # Accumulate: a turn can make several calls (reason-then-retrieve makes two),
            # and overwriting left the turn showing only the last one's tokens.
            turn_usage = self._current_turn.token_usage
            turn_usage["prompt"] = turn_usage.get("prompt", 0) + prompt_tokens
            turn_usage["completion"] = (
                turn_usage.get("completion", 0) + completion_tokens
            )
            turn_usage["cached"] = turn_usage.get("cached", 0) + cached_tokens
        self.trace.total_tokens["prompt"] += prompt_tokens
        self.trace.total_tokens["completion"] += completion_tokens
        self.trace.total_tokens["cached"] = (
            self.trace.total_tokens.get("cached", 0) + cached_tokens
        )

    def log_llm_call(self, response: Any) -> None:
        """Fold one `LLMResponse` into the turn and the trace's usage projection."""
        usage: TokenUsage = response.usage
        self.log_token_usage(
            usage.prompt_tokens, usage.completion_tokens, usage.cache_read_tokens
        )
        if response.usage_event_id:
            self.trace.usage_event_ids.append(response.usage_event_id)
        if response.estimated_cost_usd is None:
            self.trace.cost_complete = False
        else:
            self.trace.total_cost_usd += response.estimated_cost_usd

    def end_turn(self) -> None:
        if self._current_turn:
            self.trace.turns.append(self._current_turn)
            self.trace.num_turns = len(self.trace.turns)
            self._current_turn = None
            if self.save_every_turn:
                self.save()

    def log_evaluation(self, result: EvaluationResult) -> None:
        self.trace.evaluation = result
        self.trace.success = result.success

    def save(self) -> Path:
        stem = self.task_id
        if self.trace_suffix:
            stem = f"{self.task_id}_{self.trace_suffix}"
        path = self.trace_dir / f"{stem}.json"
        path.write_text(self.trace.to_json(), encoding="utf-8")
        return path

    def trace_exists(self) -> bool:
        return (self.trace_dir / f"{self.task_id}.json").exists()
