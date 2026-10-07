"""LLM-as-a-Judge: the proxy correctness signal ReasoningBank learns from (§3.2, Fig. 10).

"no ground truth is available during test-time, so the agent must continually evolve by
only leveraging its own past trajectories and any self-verification without relying on
external labels." So the label that decides WHICH extraction prompt a trajectory gets is
the agent's own judgement of it, produced by a binary classifier on the same backbone LLM
"with decoding temperature setting to 0.0 for determinism".

This is the sharpest difference from the other methods in `references/`: ERL, AutoGuide and
ExpeL all read the environment's reward during accumulation. ReasoningBank never does — the
benchmark's own verdict is recorded next to the judge's (accumulation reports how often
they agree) but is not used to build the bank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.common.usage import Usage
from references.reasoningbank.domains import domain

PROMPTS = Path(__file__).parent / "prompts"

_STATUS_RE = re.compile(r"status\s*[:\-]\s*[\"'*]*\s*(success|failure)", re.IGNORECASE)
_THOUGHTS_RE = re.compile(r"thoughts\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL)
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass
class Verdict:
    """The judge's answer. `outcome` is None when it could not be read."""

    outcome: str | None  # "success" | "failure" | None
    thoughts: str = ""
    raw: str = ""

    @property
    def success(self) -> bool:
        return self.outcome == "success"


def _render(name: str, **variables: object) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).render(**variables).strip()


def judge(
    llm: LLMClient,
    benchmark: str,
    query: str,
    trajectory: str,
    usage: Usage | None = None,
    effort: str | None = None,
) -> Verdict:
    """Label one trajectory success or failure, without any ground truth.

    One corrective retry when the reply does not carry the `Status:` line the prompt asks
    for; after that the verdict is `None` and the caller banks nothing for that task —
    guessing a label would put an item extracted with the wrong prompt into the bank.
    """
    clauses = domain(benchmark)
    system = _render(
        "judge_system.txt",
        agent_description=clauses["agent_description"],
        num_task_types=_NUMBER_WORDS.get(_count_task_types(clauses["task_types"]), "several"),
        task_types=clauses["task_types"],
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _render("judge_user.txt", intent=query, trajectory=trajectory)},
    ]
    for attempt in (1, 2):
        response = llm.generate(messages, reasoning_effort=effort)
        if usage is not None:
            usage.add(llm.model, response)
        verdict = _parse(response.content or "")
        if verdict.outcome or attempt == 2:
            return verdict
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": (
                    'That could not be read. Reply with exactly two lines: a "Thoughts:" '
                    'line, then a "Status:" line whose value is success or failure.'
                ),
            },
        ]
    return Verdict(outcome=None)  # unreachable; keeps the return type honest


def _parse(content: str) -> Verdict:
    status = _STATUS_RE.search(content)
    thoughts = _THOUGHTS_RE.search(content)
    return Verdict(
        outcome=status.group(1).lower() if status else None,
        thoughts=(thoughts.group(1).strip() if thoughts else "").split("Status:")[0].strip(),
        raw=content.strip(),
    )


def _count_task_types(task_types: str) -> int:
    return len(re.findall(r"^\d+\.", task_types, re.MULTILINE))
