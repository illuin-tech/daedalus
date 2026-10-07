"""ACE's Reflector and Curator: the two LLM calls that grow the playbook.

`adaptation_react.py::reflector_call` and `::curator_call` in `ace-appworld`. The
Reflector reads the finished trajectory and writes a diagnosis; the Curator turns that
diagnosis into ADD operations against the playbook. Neither sees a success signal on the
no-GT path: the reflector template's `<<<TEST_REPORT>>>` and ground-truth slots are
literal `[not applicable]` upstream (there is no `{{test_report}}` placeholder in it at
all, so the `.replace` that would fill it is a no-op), and the curator prompt takes only
the playbook, the reflection, the trajectory and the task instruction.

The trajectory renderer here is the one place this port deliberately departs from
daedalus's shared plumbing — see `full_trajectory`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.ace.domains import domain
from references.ace.playbook import (
    ALLOWED_SECTIONS,
    extract_json_from_text,
    get_playbook_stats,
    validate_operations,
)
from references.common.usage import Usage

PROMPTS = Path(__file__).parent / "prompts"

# ACE's own default (`playbook_token_budget`). Rendered into the curator prompt as stated
# context; nothing enforces it, upstream or here — the playbook is append-only.
TOKEN_BUDGET = 80_000


def _render(name: str, **variables: Any) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).render(**variables)


def full_trajectory(trace: dict[str, Any]) -> str:
    """The AppWorld trajectory, UNTRUNCATED.

    daedalus's shared renderer (`core/memory/extraction.py::format_trajectory_text`, which
    `benchmarks/appworld/runner.py` inherits unchanged) cuts every `execution_output` at
    500 characters. ACE's reflector and curator read the agent's own message list, where
    outputs were capped at 20,000 characters when the turn happened and not cut again
    afterwards — so handing them the 500-char version would starve exactly the material
    they are asked to diagnose, so ACE gets the full text.

    τ²'s and AutomationBench's own `format_trajectory` overrides already truncate nothing,
    so `trajectory_renderer` uses theirs unchanged.
    """
    lines: list[str] = []
    for turn in trace.get("turns", []) or []:
        if turn.get("thought"):
            lines.append(f"Thought: {turn['thought']}")
        if turn.get("code"):
            lines.append(f"Code: {turn['code']}")
        if turn.get("execution_output"):
            lines.append(f"Output: {turn['execution_output']}")
        lines.append("")
    return "\n".join(lines)


def trajectory_renderer(benchmark: Any):
    """The renderer ACE's reflector/curator read a finished task through.

    AppWorld's own `format_trajectory` truncates, so it is replaced; every other benchmark
    keeps its own, which is both untruncated and shaped for its interaction style.
    """
    if getattr(benchmark, "name", "") == "appworld":
        return full_trajectory
    return benchmark.format_trajectory


def reflect(
    llm: LLMClient,
    benchmark: str,
    *,
    trajectory: str,
    playbook: str,
    usage: Usage | None = None,
    effort: str | None = None,
) -> str:
    """One Reflector call: diagnose the trajectory. Returns the raw reply."""
    clauses = domain(benchmark)
    prompt = _render(
        "reflector.txt",
        expert=clauses["expert"],
        artifact=clauses["artifact"],
        schema_note=clauses["schema_note"],
        reflector_examples=clauses["reflector_examples"],
        trajectory=trajectory,
        playbook=playbook or "N/A",
    )
    response = llm.generate(
        [{"role": "user", "content": prompt}], reasoning_effort=effort
    )
    if usage is not None:
        usage.add(llm.model, response)
    return (response.content or "").strip()


def _curator_prompt(
    benchmark: str,
    *,
    playbook: str,
    reflection: str,
    trajectory: str,
    instruction: str,
    step: int,
    total: int,
) -> str:
    clauses = domain(benchmark)
    return _render(
        "curator.txt",
        schema_note=clauses["schema_note"],
        curator_examples=clauses["curator_examples"],
        attempt_label=clauses["attempt_label"],
        token_budget=TOKEN_BUDGET,
        current_step=step,
        total_samples=total,
        playbook_stats=json.dumps(get_playbook_stats(playbook), indent=2),
        question_context=instruction,
        current_playbook=playbook,
        trajectory=trajectory,
        guidebook=reflection,
        allowed_sections=", ".join(sorted(ALLOWED_SECTIONS)),
    )


def curate(
    llm: LLMClient,
    benchmark: str,
    *,
    playbook: str,
    reflection: str,
    trajectory: str,
    instruction: str,
    step: int,
    total: int,
    usage: Usage | None = None,
    effort: str | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """One Curator call. Returns `(operations, raw reply, error)`.

    An unparseable or malformed reply is NOT fatal: `error` is set, `operations` is empty,
    and the caller keeps the playbook as it stands — upstream's behaviour, and the reason
    a long adaptation run is not lost to one bad JSON reply.
    """
    prompt = _curator_prompt(
        benchmark,
        playbook=playbook,
        reflection=reflection,
        trajectory=trajectory,
        instruction=instruction,
        step=step,
        total=total,
    )
    response = llm.generate(
        [{"role": "user", "content": prompt}], reasoning_effort=effort
    )
    if usage is not None:
        usage.add(llm.model, response)
    raw = (response.content or "").strip()
    return _parse(raw)


def _parse(raw: str) -> tuple[list[dict[str, Any]], str, str | None]:
    parsed = extract_json_from_text(raw)
    if not parsed:
        return [], raw, "Failed to extract valid JSON from curator response"
    if "reasoning" not in parsed:
        return [], raw, "JSON missing required 'reasoning' field"
    if "operations" not in parsed:
        return [], raw, "JSON missing required 'operations' field"
    if not isinstance(parsed["reasoning"], str):
        return [], raw, "'reasoning' field must be a string"
    try:
        return validate_operations(parsed["operations"]), raw, None
    except ValueError as e:
        return [], raw, str(e)


def parse_operations(raw: str) -> tuple[list[dict[str, Any]], str | None]:
    """The curator reply → `(operations, error)`, with no LLM call. Testable on its own."""
    operations, _, error = _parse(raw)
    return operations, error


def reduce_operations(
    llm: LLMClient,
    *,
    playbook: str,
    proposals: list[dict[str, Any]],
    step: int,
    total: int,
    usage: Usage | None = None,
    effort: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """ACE's ComBEE reducer: merge ADD proposals made against one frozen playbook.

    Only reached when a wave holds more than one task (`run.parallel > 1`). Upstream:
    `ace/prompts/curator.py::CURATOR_OPERATIONS_AGGREGATION_PROMPT`, applied by
    `adaptation_react_parallel.py` under `curator_parallel=True`. The shipped configs run
    one task per wave, which is the paper's sequential loop and needs no reducer.
    """
    prompt = _render(
        "curator_reduce.txt",
        token_budget=TOKEN_BUDGET,
        current_step=step,
        total_samples=total,
        playbook_stats=json.dumps(get_playbook_stats(playbook), indent=2),
        current_playbook=playbook,
        proposed_operations=json.dumps(proposals, indent=2, ensure_ascii=False),
        allowed_sections=", ".join(sorted(ALLOWED_SECTIONS)),
    )
    response = llm.generate(
        [{"role": "user", "content": prompt}], reasoning_effort=effort
    )
    if usage is not None:
        usage.add(llm.model, response)
    operations, error = parse_operations((response.content or "").strip())
    return operations, error
