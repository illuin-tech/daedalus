"""tau2 conversation participants backed by the harness LLMClient.

HarnessTau2Agent is a HalfDuplexAgent driven by tau2's Orchestrator; its LLM
calls go through the harness LLMClient (flex tier, retries, unified cost
tracking) instead of tau2's litellm. TicketUser is the deterministic scripted
user for ticket mode (accumulation/generation): it delivers the whole task as
its first message, then stops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import STOP

from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.trace_logger import RetrievedMemoryRecord
from daedalus.core.memory.retrieval import InjectionTracker


def tau2_to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert tau2 message objects to OpenAI chat-completions format.

    Mirrors tau2's to_litellm_messages (llm_utils.py) — litellm dicts are the
    OpenAI chat format.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
            msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif isinstance(message, ToolMessage):
            out.append(
                {"role": "tool", "content": message.content, "tool_call_id": message.id}
            )
        elif isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.content})
        else:
            raise ValueError(f"Cannot convert message type {type(message).__name__}")
    return out


@dataclass
class AgentCallRecord:
    """Per-LLM-call bookkeeping the runner turns into normalized trace turns."""

    query: str = ""
    retrieved_memories: list[RetrievedMemoryRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    dropped_text: str = ""  # text discarded by the XOR fixup, if any
    # Per-call figures taken from the usage events these calls wrote, so the trace is a
    # projection of the ledger rather than a second, independently-estimated number.
    estimated_cost_usd: float = 0.0
    cost_complete: bool = True
    usage_event_ids: list[str] = field(default_factory=list)


class HarnessTau2Agent(HalfDuplexAgent[list]):
    """tau2 agent whose completions come from the harness LLMClient.

    State is the list of conversation messages (excluding the system prompt).
    Memory injection follows the shared policy (InjectionTracker): persistent
    retrievers keep an agent-side accumulator re-included in every turn's payload
    (kept for the whole task); the ephemeral whole-pool-every-turn retriever
    (all_every_turn) injects only the current turn's set. Either way the injection
    goes into the API payload, never into tau2's conversation state (owned by the
    orchestrator).
    """

    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: LLMClient,
        system_prompt: str,
        retrieve: "Callable[[str], list[RetrievedMemoryRecord]] | None" = None,
        retriever_ephemeral: bool = False,
        whole_trace_context: bool = False,
    ):
        super().__init__(tools, domain_policy)
        self.llm = llm
        self.system_prompt = system_prompt
        self.retrieve = retrieve
        # A retriever with `whole_trace_query` (AutoGuide) gets the whole conversation as
        # the retrieval context instead of the last-turn window.
        self._whole_trace_context = whole_trace_context
        self._tool_schemas = [t.openai_schema for t in tools]
        self.call_records: list[AgentCallRecord] = []
        # Injection policy (see daedalus.core.memory.retrieval.InjectionTracker).
        # tau2's message history is owned by the orchestrator, so persistent
        # heuristics are kept in an agent-side accumulator and re-included in every
        # turn's payload (they stay present for the whole task). Ephemeral retrievers
        # (all_every_turn) inject only the current turn's set.
        self._injection = InjectionTracker(retriever_ephemeral)
        self._persistent: list[RetrievedMemoryRecord] = []

    def get_init_state(
        self, message_history: list[Message] | None = None
    ) -> list[Message]:
        return list(message_history or [])

    def set_seed(self, seed: int) -> None:
        pass  # sampling seed is provider-side; nothing to seed locally

    def _build_retrieval_query(self, state: list[Message]) -> str:
        """Retrieval context. Default: last user utterance + last assistant text (mirrors
        the AppWorld agent's last-turn window). With whole_trace_context: the full
        conversation so far."""
        if self._whole_trace_context:
            parts: list[str] = []
            for msg in state:
                content = getattr(msg, "content", None)
                if content:
                    role = type(msg).__name__.replace("Message", "")
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)
        parts = []
        for msg in reversed(state):
            if isinstance(msg, UserMessage) and msg.content:
                parts.append(f"User: {msg.content}")
                break
        for msg in reversed(state):
            if isinstance(msg, AssistantMessage) and msg.content:
                parts.append(f"Agent: {msg.content}")
                break
        return "\n".join(parts)[:1000]

    def _format_memory_injection(self, memories: list[RetrievedMemoryRecord]) -> str:
        lines = [
            "The following memories from previous tasks are available. "
            "They may or may not be relevant to your current situation. "
            "Use them if they help you avoid repeating past mistakes or "
            "inform a better approach — otherwise, ignore them.",
            "",
        ]
        for i, mem in enumerate(memories, 1):
            lines.append(f"  {i}. {mem.text}")
        return "\n".join(lines)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: list[Message]
    ) -> tuple[AssistantMessage, list[Message]]:
        if isinstance(message, MultiToolMessage):
            state.extend(message.tool_messages)
        else:
            state.append(message)

        record = AgentCallRecord()

        # Pre-generation retrieval + injection. Persistent retrievers accumulate the
        # newly-surfaced (deduped) heuristics and re-inject the whole accumulated set
        # each turn, so they stay available for the whole task; ephemeral retrievers
        # (all_every_turn) inject only this turn's set.
        gen_extra: list[dict[str, Any]] = []
        if self.retrieve is not None:
            query = self._build_retrieval_query(state)
            record.query = query[:200]
            memories = self.retrieve(query) if query else []
            record.retrieved_memories = memories
            to_inject, persist = self._injection.select(memories)
            if persist:
                self._persistent.extend(to_inject)
                block = self._persistent
            else:
                block = memories
            if block:
                gen_extra.append(
                    {"role": "user", "content": self._format_memory_injection(block)}
                )

        openai_msgs = [{"role": "system", "content": self.system_prompt}]
        openai_msgs.extend(tau2_to_openai_messages(state))
        openai_msgs.extend(gen_extra)

        # tau2 rejects messages with neither content nor tool_calls; gpt-5.4-mini
        # occasionally returns such empty completions, so retry before giving up.
        # Every attempt's tokens/cost count (they were all billed), so accumulate them.
        prompt_tokens = completion_tokens = cached_tokens = 0
        cost = 0.0
        for _ in range(3):
            resp = self.llm.generate(openai_msgs, tools=self._tool_schemas)
            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            cached_tokens += resp.cached_tokens
            if resp.estimated_cost_usd is None:
                record.cost_complete = False
            else:
                cost += resp.estimated_cost_usd
            if resp.usage_event_id:
                record.usage_event_ids.append(resp.usage_event_id)
            if resp.content or resp.tool_calls:
                break
        record.prompt_tokens = prompt_tokens
        record.completion_tokens = completion_tokens
        record.cached_tokens = cached_tokens
        record.estimated_cost_usd = cost

        content: str | None = resp.content or None
        tool_calls: list[ToolCall] | None = None
        if resp.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.name, arguments=json.loads(tc.arguments))
                for tc in resp.tool_calls
            ]
            if content is not None:
                # tau2 messages must be text XOR tool calls. Keep the tool
                # calls, stash the text for the trace.
                record.dropped_text = content
                content = None
        if content is None and tool_calls is None:
            # Still empty after 3 retries: emit a minimal placeholder so the message is
            # valid (tau2 would reject an empty one) and the simulation ends cleanly
            # instead of crashing.
            content = "(no response)"

        assistant_msg = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            raw_data={"dropped_text": record.dropped_text}
            if record.dropped_text
            else None,
        )
        self.call_records.append(record)
        state.append(assistant_msg)
        return assistant_msg, state


class TicketUser(UserSimulator):
    """Deterministic scripted user for ticket mode. No LLM. Reply sequence:
    the full task (ticket) → `confirmations` blanket-confirmation replies (so a
    policy-following agent that asks before write actions is not cut off) → the
    stop token."""

    CONFIRMATION = (
        "Yes, I confirm — please proceed with everything exactly as described in "
        "my request. Once it is all done, let me know and we are finished."
    )

    def __init__(self, ticket: str, confirmations: int = 1):
        super().__init__(llm="none", instructions=ticket)
        self.ticket = ticket
        self.confirmations = confirmations
        self._replies = 0

    def set_seed(self, seed: int) -> None:
        pass

    def generate_next_message(self, message, state):
        if self._replies == 0:
            content = self.ticket
        elif self._replies <= self.confirmations:
            content = self.CONFIRMATION
        else:
            content = STOP
        self._replies += 1
        msg = UserMessage(role="user", content=content, cost=0.0)
        state.messages.append(msg)
        return msg, state
