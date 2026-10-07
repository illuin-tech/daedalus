"""Memory extraction: a trajectory in, up to three memory items out (§3.2, Fig. 9).

Two prompts, both the paper's, chosen by the judge's label:

    success   "analyzing why the trajectory led to success" (Fig. 9 left)
    failure   "reflecting on the causes of failure and articulating lessons" (Fig. 9 right)

The extractor runs on the agent's own backbone at temperature 1.0 (Appendix A.2), which is
what `ReasoningBankConfig.extraction_model = None` selects.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.common.usage import Usage
from references.reasoningbank.domains import domain
from references.reasoningbank.memory import Item, parse_items

PROMPTS = Path(__file__).parent / "prompts"


def _render(name: str, **variables: object) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).render(**variables).strip()


def extract_items(
    llm: LLMClient,
    benchmark: str,
    query: str,
    trajectory: str,
    success: bool,
    max_items: int = 3,
    usage: Usage | None = None,
    effort: str | None = None,
) -> tuple[list[Item], str]:
    """Fig. 9: the success prompt or the failure prompt, per the judge's label."""
    clauses = domain(benchmark)
    system = _render(
        "extraction_success.txt" if success else "extraction_failure.txt",
        expert=clauses["expert"],
        specifics=clauses["specifics"],
        generalize_beyond=clauses["generalize_beyond"],
        max_items=max_items,
    )
    user = _render("extraction_user.txt", query=query, trajectory=trajectory)
    response = llm.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        reasoning_effort=effort,
    )
    if usage is not None:
        usage.add(llm.model, response)
    return parse_items(response.content or "", max_items), (response.content or "").strip()
