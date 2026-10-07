"""AutoGuide's four LLM modules, one per prompt in the paper's Appendix C.

    identify_context   Mcontext   (Eq. 1, Fig. 6)  — abstract a trajectory into its CONTEXT
    match_context                 (§3.2/3.3, Fig. 11) — is this CONTEXT one we have seen?
    extract_guideline  Mguideline (Eq. 2, Fig. 9)  — contrast τ+ and τ− into a guideline
    select_guidelines  Mselect    (Eq. 3, Fig. 12) — pick the top-k for the current turn

The prompt text is the paper's ALFWorld variant (the closest of its three domains to a
text-and-tool-calls environment); see README.md for the two deviations.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.common.usage import Usage

PROMPTS = Path(__file__).parent / "prompts"


def _render(name: str, **variables: object) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).render(**variables).strip()


def task_description(benchmark: str, domain: str) -> str:
    """The paper's `{Task description}` slot: one line naming the environment."""
    return _render("task_description.txt", benchmark=benchmark, domain=domain)


def _ask(llm: LLMClient, prompt: str, usage: Usage | None, effort: str | None = None) -> str:
    response = llm.generate([{"role": "user", "content": prompt}], reasoning_effort=effort)
    if usage is not None:
        usage.add(llm.model, response)
    return (response.content or "").strip()


# ── Mcontext (Eq. 1) ────────────────────────────────────────────────────────
_SUMMARIZATION_RE = re.compile(r"SUMMARIZATION:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def identify_context(
    llm: LLMClient, trajectory: str, usage: Usage | None = None, effort: str | None = None
) -> str:
    """Abstract a (partial) trajectory into a short natural-language status.

    The prompt asks for `SUMMARIZATION: <status>`; anything after the last such marker is
    the context. A reply without the marker is used as-is (first line), since the context
    is only ever a dictionary key.
    """
    answer = _ask(llm, _render("context_identification.txt", trajectory=trajectory), usage, effort)
    match = _SUMMARIZATION_RE.search(answer)
    context = match.group(1) if match else answer
    return " ".join(context.split()).strip()  # one line: it is a dictionary key


# ── context matching (§3.2 / §3.3) ─────────────────────────────────────────
_ANSWER_RE = re.compile(r"Answer:\s*(None|\d+)", re.IGNORECASE)


def match_context(
    llm: LLMClient,
    context: str,
    seen: list[str],
    description: str,
    usage: Usage | None = None,
    effort: str | None = None,
) -> str | None:
    """Return the seen context describing the same status, or None.

    "the context identification module occasionally produces contexts that describe the
    same situation but are expressed slightly differently. To minimize redundancy, we
    employ an LLM to determine if the current context corresponds to any previously
    identified context." An exact string hit skips the call.
    """
    if not seen:
        return None
    if context in seen:
        return context
    prompt = _render(
        "context_matching.txt",
        task_description=description,
        seen="\n".join(f"{i}. {c}" for i, c in enumerate(seen, 1)),
        context=context,
    )
    answer = _ask(llm, prompt, usage, effort)
    match = _ANSWER_RE.search(answer) or re.search(r"^\s*(None|\d+)\s*$", answer, re.IGNORECASE)
    if match is None:
        return None
    value = match.group(1)
    if value.lower() == "none":
        return None
    index = int(value)
    return seen[index - 1] if 1 <= index <= len(seen) else None


# ── Mguideline (Eq. 2) ─────────────────────────────────────────────────────
_GUIDELINE_RE = re.compile(r"Reasoning:\s*(?P<reasoning>.*?)\s*Guideline:\s*(?P<guideline>.+)",
                           re.IGNORECASE | re.DOTALL)


def extract_guideline(
    llm: LLMClient,
    description: str,
    context: str,
    desired: str,
    undesired: str,
    usage: Usage | None = None,
    effort: str | None = None,
) -> tuple[str, str]:
    """Contrast a desired and an undesired trajectory into one guideline.

    Returns (reasoning, guideline). The prompt asks for 'Reasoning: … Guideline: …'; if
    only prose comes back it is taken as the guideline, so a reply that skips the format
    still contributes rather than being silently dropped.
    """
    prompt = _render(
        "guideline_extraction.txt",
        task_description=description,
        context=context,
        desired=desired,
        undesired=undesired,
    )
    answer = _ask(llm, prompt, usage, effort)
    match = _GUIDELINE_RE.search(answer)
    if match is None:
        return "", " ".join(answer.split())
    return (
        " ".join(match.group("reasoning").split()),
        " ".join(match.group("guideline").split()),
    )


# ── Mselect (Eq. 3) ────────────────────────────────────────────────────────
_LIST_RE = re.compile(r"\[[^\[\]]*\]")


def select_guidelines(
    llm: LLMClient,
    description: str,
    guidelines: list[str],
    trajectory: str,
    k: int,
    usage: Usage | None = None,
    effort: str | None = None,
) -> list[int]:
    """Pick at most k of this context's guidelines for the current turn (0-based indices).

    Only called when the context holds more than k guidelines ("If there are more than k
    guidelines in G[CONTEXT], Mselect prompts an LLM to choose top-k"). An unparseable
    reply falls back to the first k, so a turn never silently loses its guidance.
    """
    prompt = _render(
        "guideline_selection.txt",
        task_description=description,
        k=k,
        guidelines="\n".join(f"{i}. {g}" for i, g in enumerate(guidelines, 1)),
        trajectory=trajectory,
    )
    answer = _ask(llm, prompt, usage, effort)
    match = _LIST_RE.search(answer)
    if match is None:
        return list(range(min(k, len(guidelines))))
    try:
        picked = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return list(range(min(k, len(guidelines))))
    out: list[int] = []
    for value in picked if isinstance(picked, (list, tuple)) else []:
        if isinstance(value, int) and 1 <= value <= len(guidelines) and value - 1 not in out:
            out.append(value - 1)
    return out[:k]
