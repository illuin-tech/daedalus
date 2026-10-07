"""Applying context-aware guidelines at test time (paper §3.3, Algorithm 2).

At every timestep: identify the current CONTEXT from the trajectory so far, match it
against the bank's keys, and — when the context is known — inject that context's
guidelines (top-k selected by an LLM when there are more than k) into the prompt for that
turn. When the context matches nothing, the agent gets no guidance, exactly as in
Algorithm 2 (`GUIDELINES ← ∅`).

Implemented as a daedalus `Retriever` so each benchmark's existing per-turn retrieval and
injection plumbing carries it: `ephemeral = True` means the block goes into the current
turn's prompt only, which is what "incorporates both the context and relevant guidelines
into the agent's action generation prompt" describes — a fresh decision every turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daedalus.core.memory.retrieval import RetrievedMemory, Retriever

from references.autoguide import modules
from references.autoguide.guidelines import GuidelineBank
from references.common.usage import Usage


@dataclass
class TurnRecord:
    """What the three modules decided at one turn."""

    turn: int
    context: str
    matched: str | None
    injected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "context": self.context,
            "matched": self.matched,
            "num_injected": len(self.injected),
            "injected": self.injected,
        }


@dataclass
class Retrieval:
    """One task's per-turn record, written to the run's `retrieval/` folder."""

    task: str = ""
    k: int = 3
    model: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        matched = [t for t in self.turns if t.matched]
        return {
            "task": self.task,
            "k": self.k,
            "model": self.model,
            "num_turns": len(self.turns),
            "num_turns_with_context_match": len(matched),
            "turns": [t.to_dict() for t in self.turns],
            "usage": self.usage.to_dict(),
        }


class ContextAwareRetriever(Retriever):
    """Algorithm 2's per-timestep guideline lookup."""

    slug = "autoguide"
    # The context and its guidelines are re-decided every turn, so they belong to that
    # turn's prompt only — never accumulated in the conversation.
    ephemeral = True
    whole_trace_query = True  # context identification reads the trajectory so far

    def __init__(
        self,
        bank: GuidelineBank,
        k: int = 3,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str | None = None,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self.bank = bank
        self.k = k
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.description = description
        # Injected by the solver mixin (Retriever.attach_llm_client) so the identify /
        # match / select calls land in the run's usage ledger under the `retriever` role.
        self._llm = None
        # Context matching is an LLM call, and the same context string recurs constantly
        # within a task (and across tasks in one worker). The mapping is a property of the
        # string, so cache it: same answer, far fewer calls.
        self._match_cache: dict[str, str | None] = {}
        self.record = Retrieval(k=k, model=model)

    def index(self, texts: list[str], ids: list[str]) -> None:
        """No-op: the bank is a context dictionary, not a flat searchable corpus."""

    def on_task_start(self, task_instruction: str = "", task_id: str = "") -> None:
        # `task_id` is unused here — AutoGuide re-seeds nothing per task — but the base
        # signature carries it and every caller passes it by keyword, so omitting it
        # raised TypeError on every task and scored the whole run 0/0.
        self.record = Retrieval(task=task_instruction, k=self.k, model=self.model)

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        """Identify → match → select for the current turn (`top_k` is ignored; k is ours)."""
        trajectory = reasoning_trace or query
        if not trajectory or len(self.bank) == 0:
            return []
        llm = self._tracked_llm()

        usage = self.record.usage
        effort = self.reasoning_effort
        context = modules.identify_context(llm, trajectory, usage, effort)
        turn = TurnRecord(turn=len(self.record.turns), context=context, matched=None)
        self.record.turns.append(turn)
        if not context:
            return []

        if context in self._match_cache:
            matched = self._match_cache[context]
        else:
            matched = modules.match_context(
                llm, context, self.bank.contexts(), self.description, usage, effort
            )
            self._match_cache[context] = matched
        turn.matched = matched
        if matched is None:
            return []  # Algorithm 2: no matching context → no guidelines

        guidelines = [g.text for g in self.bank.get(matched)]
        if len(guidelines) > self.k:
            picked = modules.select_guidelines(
                llm, self.description, guidelines, trajectory, self.k, usage, effort
            )
            guidelines = [guidelines[i] for i in picked]
        turn.injected = guidelines
        if not guidelines:
            return []

        text = "Context: {}\nContext-Aware Guideline:\n{}".format(
            matched, "\n".join(f"- {g}" for g in guidelines)
        )
        return [
            RetrievedMemory(
                memory_id=f"context_{len(self.record.turns)}",
                text=text,
                score=1.0,
                source=f"autoguide:{self.model}",
            )
        ]
