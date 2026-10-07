"""Shared memory-retrieval wiring for benchmark solver agents.

Every benchmark's memory agent builds the same thing from `cfg.memory`: a retriever
indexed over the pool, plus the injection tracker that decides how what it surfaces is
placed in the conversation. This centralizes that so the connectors don't each
re-implement it (and drift).

Two layers, so a connector takes only what it needs:

  - `build_retriever(cfg, pool)` — the (retriever, tracker) pair. AppWorld and τ² use it
    and own their injection loop.
  - `SolverMemory` — the whole solver-side policy on top of that: pool loading, the
    @start / @turn choice, query building, and placing the notes in a `list[dict]`
    message history. Used by AutomationBench.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.trace_logger import RetrievedMemoryRecord
from daedalus.core.memory.pool import MemoryPool
from daedalus.core.memory.retrieval import (
    CUMULATIVE_NO_REPLACEMENT,
    TRANSIENT,
    InjectionTracker,
    Retriever,
    WithoutReplacementRetriever,
)


def load_solver_pool(cfg: ExperimentConfig, benchmark: str) -> MemoryPool:
    """The pool this solver will inject — never silently empty.

    Every connector calls this instead of `MemoryPool.load_or_empty`, which returns an
    empty pool for a path that is not there. That turned a typo, or a run launched before
    the pool it depends on had been built, into a memory-FREE run reported as a memory arm
    — and in `heuristics_at_start` mode the trace records nothing about the injection, so
    it could not be caught afterwards either.
    """
    return MemoryPool.load_for_solver(cfg.memory.pool_path, benchmark)


def injected_heuristics_record(
    pool: MemoryPool, pool_path: str | None
) -> dict[str, Any]:
    """A witness, for `trace.extra`, that memory really was injected at task start.

    `heuristics_at_start` puts the whole bank in the system prompt and builds no
    retriever, so nothing lands in `turns[*].retrieved_memories` and such a trace was
    indistinguishable from a baseline one. The digest makes two runs' injected banks
    comparable without copying the bank into every trace.
    """
    import hashlib

    texts = pool.texts()
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    return {
        "count": len(texts),
        "chars": sum(len(t) for t in texts),
        "sha256": digest,
        "pool_path": pool_path or "",
    }


def build_retriever(
    cfg: ExperimentConfig, pool: MemoryPool
) -> tuple[Retriever | None, InjectionTracker | None]:
    """Build the configured (retriever, injection_tracker) from a memory pool.

    Returns (None, None) when there is nothing to retrieve with: memory disabled, an empty
    pool, or heuristics_at_start (the whole bank goes into the system prompt instead).
    `memory.injection.policy` is applied here: `cumulative_no_replacement` wraps the
    retriever, and an inherently ephemeral retriever (all_every_turn) forces `transient`.
    """
    if not cfg.memory.enabled or len(pool) == 0 or cfg.memory.heuristics_at_start:
        return None, None
    r = cfg.memory.retriever
    policy = cfg.memory.injection.policy
    retriever = Retriever.from_slug(r.type, model_name=r.model_name, seed=r.seed,
                                    base_url=r.base_url)
    if policy == CUMULATIVE_NO_REPLACEMENT:
        if retriever.ephemeral:
            raise ValueError(f"{policy!r} needs a per-turn top-k retriever, not {r.type!r}")
        retriever = WithoutReplacementRetriever(retriever)
    retriever.index(texts=pool.texts(), ids=pool.ids())
    return retriever, InjectionTracker(policy=TRANSIENT if retriever.ephemeral else policy)


@dataclass
class TurnMemory:
    """What one turn's retrieval produced, for a message-list-owning solver.

    `query` is what the retriever was asked (logged on the turn, truncated by the
    caller); `retrieved` is everything it surfaced (logged in full); `ephemeral` holds
    the injection message(s) to append to a COPY of the history for this one
    generation call. A persistent injection is appended to the real history by
    `SolverMemory.turn` itself, so the caller has nothing to do for it.
    """

    query: str = ""
    retrieved: list[RetrievedMemoryRecord] = field(default_factory=list)
    ephemeral: list[dict[str, Any]] = field(default_factory=list)


class SolverMemory:
    """The solver-side memory plumbing for a benchmark that owns a `list[dict]` history.

    Encapsulates what a native tool-calling connector needs from `cfg.memory`: load the
    pool, build the retriever, and place the notes, either

      - `memory.heuristics_at_start: true` — the WHOLE pool into the system prompt once
        (no retriever is built), where it persists for every turn; or
      - re-retrieve every turn and inject as the trajectory evolves, as the injection
        policy says (see `InjectionTracker`).

    Usage, once per task:

        heuristic_texts = mem.start(task.instruction, heuristics)   # -> system prompt
        for i in range(max_iterations):
            tm = mem.turn(i, messages)        # appends persistent injections itself
            resp = llm.generate(messages + tm.ephemeral, tools=...)

    `heuristics` (the accumulation loop's single mined note) short-circuits retrieval:
    when it is given, those texts are the ones injected and nothing is retrieved.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        benchmark: str,
        accounting: CostAccounting | None = None,
    ):
        self.cfg = cfg
        self.benchmark = benchmark
        self.accounting = accounting
        self.pool: MemoryPool | None = None
        self.retriever: Retriever | None = None
        self._injection: InjectionTracker | None = None
        # Per-task state, set by start().
        self._at_start = False
        self._start_memories: list[RetrievedMemoryRecord] = []

        if not cfg.memory.enabled:
            return
        if cfg.memory.retrieval_mode != "pre_generation":
            raise ValueError(
                f"{benchmark} supports memory.retrieval_mode='pre_generation' only; "
                f"got {cfg.memory.retrieval_mode!r}."
            )
        self.pool = load_solver_pool(cfg, benchmark)
        self.retriever, self._injection = build_retriever(cfg, self.pool)
        # A missing or empty pool now raises in load_solver_pool. What is left to warn
        # about is a config that builds no retriever AND does not inject at start.
        if self.retriever is None and not cfg.memory.heuristics_at_start:
            console.info(
                f"[{benchmark}] memory.enabled with no retriever and "
                "heuristics_at_start=false — nothing will be injected."
            )

    # ----- per-task -----

    def start(
        self,
        instruction: str,
        heuristics: list[str] | None = None,
        task_id: str = "",
    ) -> list[str]:
        """Begin a task; return the heuristic texts to render into the system prompt.

        Resets the per-task retrieval state (random draws, no-replacement exclusions).
        """
        whole_pool_at_start = (
            heuristics is None
            and self.cfg.memory.enabled
            and self.cfg.memory.heuristics_at_start
            and self.pool is not None
            and len(self.pool) > 0
        )
        self._at_start = whole_pool_at_start
        if self.retriever is not None:
            self.retriever.on_task_start(instruction, task_id=task_id)
            self._injection.reset()  # forget which heuristics were injected last task

        if whole_pool_at_start:
            self._start_memories = [
                RetrievedMemoryRecord(
                    memory_id=mid, text=txt, score=1.0, source="heuristics_at_start"
                )
                for mid, txt in zip(self.pool.ids(), self.pool.texts())
            ]
        else:
            self._start_memories = []

        return (
            heuristics
            if heuristics is not None
            else [m.text for m in self._start_memories]
        )

    def turn(self, turn_idx: int, messages: list[dict[str, Any]]) -> TurnMemory:
        """Retrieve for this turn and place what it surfaced.

        @start modes have nothing to retrieve; the notes are already in the system
        prompt, and are reported on turn 0 only so the trace records them once.
        """
        if self._at_start:
            return TurnMemory(retrieved=self._start_memories if turn_idx == 0 else [])
        if self.retriever is None:
            return TurnMemory()
        query = self.build_query(messages)
        retrieved = self.retrieve(query)
        to_inject, persist = self._injection.select(retrieved)
        if not to_inject:
            return TurnMemory(query=query, retrieved=retrieved)
        msg = {"role": "user", "content": format_memory(to_inject)}
        if persist:
            messages.append(msg)
            return TurnMemory(query=query, retrieved=retrieved)
        return TurnMemory(query=query, retrieved=retrieved, ephemeral=[msg])

    # ----- retrieval primitives -----

    def build_query(self, messages: list[dict[str, Any]]) -> str:
        """The retrieval query for the current turn.

        Default: the last user message plus the last assistant message (the local
        window), capped so a long observation cannot dominate the query. A retriever with
        `whole_trace_query` (AutoGuide) gets the entire conversation so far instead.
        """
        if self.retriever is not None and self.retriever.whole_trace_query:
            return "\n".join(
                f"{m['role']}: {m['content']}" for m in messages if m.get("content")
            )
        parts = []
        for m in reversed(messages):
            if m["role"] == "user" and m.get("content"):
                parts.append(f"User: {m['content']}")
                break
        for m in reversed(messages):
            if m["role"] == "assistant" and m.get("content"):
                parts.append(f"Agent: {m['content']}")
                break
        return " ".join(reversed(parts))[:1000]

    def retrieve(self, query: str) -> list[RetrievedMemoryRecord]:
        """Retrieve for `query`, as trace records (empty when there is no retriever)."""
        if self.retriever is None or not query:
            return []
        hits = self.retriever.retrieve(
            query=query, reasoning_trace=query, top_k=self.cfg.memory.retriever.top_k
        )
        return [
            RetrievedMemoryRecord(
                memory_id=h.memory_id,
                text=h.text,
                score=h.score,
                source=getattr(h, "source", ""),
            )
            for h in hits
        ]


def format_memory(memories: list[RetrievedMemoryRecord]) -> str:
    """Render retrieved notes as the message the solver sees.

    Deliberately non-committal ("may or may not be relevant"): a retriever that
    cannot abstain will surface something for every turn, and an injected note that
    reads as authoritative on a task it does not fit costs accuracy.
    """
    lines = [
        (
            "The following learned notes may or may not be relevant to the current "
            "task. Use any that help; ignore the rest."
        ),
    ]
    for i, m in enumerate(memories, 1):
        lines.append(f"  {i}. {m.text}")
    return "\n".join(lines)
