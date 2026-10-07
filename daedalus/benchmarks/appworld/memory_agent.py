"""Memory-augmented ReAct agent — adds per-turn retrieval to the base agent.

The base `ReActAgent` owns the whole solve loop (execution, observation, scoring); this
subclass only overrides how a single turn is generated (`_generate_turn`) and how the
retrieval query/memories are produced. The default retrieval mode injects memory before a
single generation; `reason_then_retrieve` thinks first, retrieves on that reasoning, then
generates the action with the memory in context.
"""

from __future__ import annotations

import re

from daedalus.benchmarks.appworld.agent import ReActAgent, parse_thought_and_code
from daedalus.core.agents.memory import build_retriever
from daedalus.core.config import ExperimentConfig
from daedalus.core.logging.trace_logger import RetrievedMemoryRecord, TraceLogger
from daedalus.core.memory.pool import MemoryPool


# `reason_first.txt` ends with the label "Thought:", which the model almost always echoes;
# the generation prompt and the stored turn prepend "Thought: " themselves, so one leading
# label is stripped to avoid "Thought: Thought: ...".
_THOUGHT_LABEL = re.compile(r"^\s*thought\s*:\s*", re.IGNORECASE)


class MemoryReActAgent(ReActAgent):
    """ReAct agent with memory retrieval before each turn's code generation."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        memory_pool: MemoryPool | None = None,
        run_idx: int | None = None,
    ):
        super().__init__(cfg, run_idx=run_idx)
        from daedalus.core.agents.memory import (
            injected_heuristics_record,
            load_solver_pool,
        )

        # A configured pool that is not on disk is a hard error, not an empty pool: an
        # empty one is a memory-FREE run reported as a memory arm, and at-start injection
        # leaves nothing in the trace to catch it with.
        self.memory_pool = memory_pool or load_solver_pool(cfg, "appworld")
        # Witness, stamped onto every trace's `extra` (see ReActAgent.solve_task).
        self.memory_record = injected_heuristics_record(
            self.memory_pool, cfg.memory.pool_path
        )
        # heuristics_at_start: inject the whole bank into the system prompt at task
        # start and skip per-turn retrieval (build_retriever returns no retriever).
        self.heuristics_at_start = bool(cfg.memory.heuristics_at_start)
        self.retriever, self._injection_tracker = build_retriever(cfg, self.memory_pool)

        # Prompts for reason-then-retrieve mode (shared, benchmark-neutral).
        from daedalus.core.resources import CORE_PROMPTS

        self._reason_first_prompt = (
            CORE_PROMPTS / "agent" / "reason_first.txt"
        ).read_text(encoding="utf-8")
        self._generate_with_memory_prompt = (
            CORE_PROMPTS / "agent" / "generate_with_memory.txt"
        ).read_text(encoding="utf-8")

    def _build_initial_messages(
        self, world, heuristics=None, instruction_override=None
    ):
        """In heuristics_at_start mode, prepend the ENTIRE bank to the (optional)
        per-task heuristics so it is injected into the system prompt at task start."""
        if self.heuristics_at_start and len(self.memory_pool) > 0:
            heuristics = (heuristics or []) + list(self.memory_pool.texts())
        return super()._build_initial_messages(
            world, heuristics=heuristics, instruction_override=instruction_override
        )

    def _get_retrieved_memories(
        self,
        query: str,
        messages: list[dict[str, str]],
        turn_idx: int,
    ) -> list[RetrievedMemoryRecord]:
        """Retrieve relevant memories based on the accumulated context."""
        if not self.cfg.memory.enabled or self.retriever is None or not query:
            return []
        retrieved = self.retriever.retrieve(
            query=query,
            reasoning_trace=query,
            top_k=self.cfg.memory.retriever.top_k,
        )
        return [
            RetrievedMemoryRecord(
                memory_id=r.memory_id, text=r.text, score=r.score, source=r.source
            )
            for r in retrieved
        ]

    def _generate_turn(
        self, messages: list[dict[str, str]], logger: TraceLogger, turn_idx: int
    ) -> tuple[str, str]:
        """Reason-then-retrieve turn: think first (no memory), retrieve on that reasoning,
        then generate the action with the memory in context.

        Falls back to the base single-pass turn for heuristics_at_start (no retriever) and
        every non-reason_then_retrieve mode.
        """
        if (
            self.heuristics_at_start
            or self.cfg.memory.retrieval_mode != "reason_then_retrieve"
        ):
            return super()._generate_turn(messages, logger, turn_idx)

        # Step 1: reason first (thought only, no memory yet).
        reason_response = self.llm.generate(
            messages + [{"role": "user", "content": self._reason_first_prompt}]
        )
        reasoning = _THOUGHT_LABEL.sub("", reason_response.content.strip(), count=1)

        # Step 2: retrieve using the reasoning as the query.
        memories = self._get_retrieved_memories(reasoning, messages, turn_idx)
        logger.log_query(reasoning[:200])
        logger.log_retrieved_memories(memories)

        # Injection policy: persistent retrievers append into the real history (kept for
        # the whole task); ephemeral ones go into this turn's generation prompt only.
        to_inject, persist = (
            self._injection_tracker.select(memories)
            if self._injection_tracker
            else (memories, True)
        )
        if to_inject and persist:
            messages.append(
                {
                    "role": "user",
                    "content": self._format_memory_injection(to_inject).strip(),
                }
            )

        # Step 3: generate the action, with the reasoning and memory in context.
        gen_messages = list(messages)
        gen_messages.append({"role": "assistant", "content": f"Thought: {reasoning}"})
        gen_prompt_parts = []
        if to_inject and not persist:  # ephemeral: this turn's prompt only
            gen_prompt_parts.append(self._format_memory_injection(to_inject).strip())
        gen_prompt_parts.append(self._generate_with_memory_prompt)
        gen_messages.append({"role": "user", "content": "\n\n".join(gen_prompt_parts)})

        response = self.llm.generate(gen_messages)
        thought, code = parse_thought_and_code(response.content)

        logger.log_thought(reasoning)
        if thought:
            logger.log_reasoning_with_memory(thought)
        # Both LLM calls (reason + generate) count toward this turn's usage, cached
        # tokens included — otherwise prompt-cached runs under-report cost. Each also
        # wrote its own ledger event, so the turn links to both.
        logger.log_llm_call(reason_response)
        logger.log_llm_call(response)

        full_thought = f"{reasoning}\n\n{thought}" if thought else reasoning
        assistant_content = (
            f"Thought: {full_thought}\n\n```python\n{code}\n```"
            if code
            else response.content
        )
        return assistant_content, code
