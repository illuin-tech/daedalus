"""Heuristic generation: one reflection per completed trajectory (paper §2, Fig. 8).

The agent runs a source task ONCE, the environment returns success/failure, and this
module turns (task, trajectory, reward) into one structured heuristic. That single-attempt
assumption is ERL's differentiator: no Reflexion retries (ExpeL, AutoGuide) and no
retry-until-banked loop (DAEDALUS) — every source task yields exactly one heuristic,
whatever its outcome.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.common.usage import Usage

PROMPTS = Path(__file__).parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def generate_heuristic(
    llm: LLMClient,
    task: str,
    trajectory: str,
    success: bool,
    usage: Usage | None = None,
) -> str:
    """Reflect on one trajectory and return the heuristic text (analysis + guideline).

    `success` is the environment's own verdict — the paper's outcome signal, and the thing
    the prompt branches on (breakpoint + correction rule for a failure, winning move +
    best practice for a success).
    """
    user = Template(_prompt("heuristic_generation.txt")).render(
        task=task,
        outcome="success" if success else "failure",
        trajectory=trajectory,
    )
    response = llm.generate(
        [
            {"role": "system", "content": _prompt("heuristic_generation_system.txt").strip()},
            {"role": "user", "content": user},
        ]
    )
    if usage is not None:
        usage.add(llm.model, response)
    return response.content.strip()
