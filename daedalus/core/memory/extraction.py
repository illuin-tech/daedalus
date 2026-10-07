"""LLM steps of the Solver loop: writing or revising the heuristic after a failure.

`generate_memory_item` is the Extractor (paper Prompt 7). Its prompt is shared by every
benchmark, with a per-benchmark override in `benchmarks/<name>/prompts/` (see
`_benchmark_override`). The module also holds small parsing helpers shared by the
pipeline, the judges and the reference methods.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"


def _benchmark_override(benchmark: str | None, subdir: str, name: str) -> Path | None:
    """Path of a benchmark's own copy of a shared prompt, if that file exists.

    Every prompt a benchmark overrides lives with the rest of that benchmark's prompts,
    in benchmarks/<benchmark>/prompts/, named "<subdir>_<name>" — so tau2's variant of
    accumulation/memory_extraction.txt is
    benchmarks/tau2/prompts/accumulation_memory_extraction.txt. One directory per
    benchmark holds everything specific to it; core/prompts holds only what is shared.
    """
    if not benchmark:
        return None
    candidate = BENCHMARKS_DIR / benchmark / "prompts" / f"{subdir}_{name}"
    return candidate if candidate.exists() else None


def _load_prompt(subdir: str, name: str, benchmark: str | None = None) -> str:
    """Load a prompt, preferring the benchmark's own variant when it has one."""
    override = _benchmark_override(benchmark, subdir, name)
    if override is not None:
        return override.read_text(encoding="utf-8")
    return (PROMPTS_DIR / subdir / name).read_text(encoding="utf-8")


def _system_prompt(subdir: str, stem: str, benchmark: str | None, default: str) -> str:
    """The benchmark's own <stem>_system.txt, falling back to the shared inline default."""
    override = _benchmark_override(benchmark, subdir, f"{stem}_system.txt")
    if override is not None:
        return override.read_text(encoding="utf-8").strip()
    return default


# ---------------------------------------------------------------------------
# Text cleanup helpers
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (possibly with language tag)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


_HEURISTICS_BLOCK_RE = re.compile(r"<heuristics>\s*(.*?)\s*</heuristics>", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HEURISTIC_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def has_empty_heuristics_block(text: str) -> bool:
    """Whether the response holds an explicit, unfenced, empty `<heuristics>` block."""
    match = _HEURISTICS_BLOCK_RE.search(_ANY_FENCE_RE.sub("\n", text))
    return match is not None and not match.group(1).strip()


def parse_heuristics_block(text: str, min_chars: int = 25) -> list[str]:
    """The bullets of the `<heuristics>` block, in order, whitespace-folded and deduped.

    Fenced code is removed first, so a block printed from inside a python cell cannot
    splice the surrounding code into a heuristic. Continuation lines stay attached to their
    bullet, and bullets shorter than `min_chars` are fragments, not lessons.
    """
    match = _HEURISTICS_BLOCK_RE.search(_ANY_FENCE_RE.sub("\n", text))
    if not match:
        return []
    bullets: list[str] = []
    for line in match.group(1).splitlines():
        if _HEURISTIC_BULLET_RE.match(line):
            bullets.append(_HEURISTIC_BULLET_RE.sub("", line).strip())
        elif bullets and line.strip():
            bullets[-1] += " " + line.strip()
    out: list[str] = []
    for bullet in bullets:
        heuristic = " ".join(bullet.split())
        if len(heuristic) >= min_chars and heuristic not in out:
            out.append(heuristic)
    return out


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _balanced_objects(text: str):
    """Yield every top-level `{...}` span, tracking nesting and string literals.

    The previous fallback was a `{[^{}]*}` regex, which by construction cannot
    match an object containing another object. On a NESTED reply wrapped in prose — the
    test-set judge emits `{"outcomes": [{...}, ...]}` — it matched the first INNER object
    instead, so `_outcomes` found no list, fell back to the overall boolean, and every
    condition was recorded as failed with `parsed=True`. A brace scan cannot make that
    mistake, and string-awareness keeps a `{` inside a judge's reason from unbalancing it.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: return the first parseable JSON OBJECT in text (nesting included)."""
    text = _strip_markdown_fences(text.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    for span in _balanced_objects(text):
        try:
            parsed = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# LLM-driven steps (system + user prompt)
# ---------------------------------------------------------------------------

_SELF_REFLECTION_SYSTEM = (
    "You are analyzing an agent's attempt at a task. The agent interacts with "
    "APIs by writing Python code. Given the trajectory and outcome, produce a "
    "concise self-reflection identifying what went wrong (or right) and what "
    "actionable lesson can be drawn. Focus on concrete, reusable insights — "
    "not task-specific details like IDs or names."
)

def generate_self_reflection(
    llm: LLMClient,
    instruction: str,
    trajectory_text: str,
    outcome: str,
    benchmark: str | None = None,
) -> str:
    """Generate a self-reflection on the agent's attempt."""
    template = Template(_load_prompt("accumulation", "self_reflection.txt", benchmark))
    user_content = template.render(
        instruction=instruction,
        trajectory_text=trajectory_text,
        outcome=outcome,
    )
    system = _system_prompt(
        "accumulation", "self_reflection", benchmark, _SELF_REFLECTION_SYSTEM
    )
    response = llm.generate(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    )
    return response.content


_MEMORY_EXTRACTION_SYSTEM = (
    "You help an agent build reusable memory from task attempts. "
    "Given a trajectory and outcome, produce or refine a single memory item "
    "that captures actionable lessons for future attempts. "
    "Output ONLY the memory text — no headers, no JSON, no fences."
)


def _solver_env(benchmark: str | None) -> str:
    """What the solver is told before every task: its env facts AND its protocol.

    Passed to the extraction prompt as `known` so a mined lesson does not restate it. Returns
    "" when the benchmark ships no such block."""
    if not benchmark:
        return ""
    from pathlib import Path

    # BOTH the env facts and the PROTOCOL. Only solver_env.txt was passed before, so the
    # extractor never saw the rules — including AppWorld's answer convention ("if the task
    # only requires side effects, call complete_task with no arguments"). A glm-5.3 run
    # mined 6 heuristics telling the solver to spell every requirement out in the final
    # answer, which is correct under generation's LLM judge and wrong against AppWorld's
    # evaluator: median complete_task answer went 1 -> 198 chars and MSR fell 76.5% -> 34.8%.
    d = Path(__file__).parent.parent.parent / "benchmarks" / benchmark / "prompts"
    parts = []
    for name in ("solver_env.txt", "solver_protocol.txt"):
        try:
            parts.append((d / name).read_text(encoding="utf-8").strip())
        except OSError:
            continue
    return "\n\n".join(p for p in parts if p)


def generate_memory_item(
    llm: LLMClient,
    instruction: str,
    trajectory_text: str,
    outcome: str,
    current_memory: str | None = None,
    attempt_index: int = 1,
    max_failures: int = 1,
    benchmark: str | None = None,
) -> str:
    """Generate or refine a single free-form memory item from a task attempt."""
    template = Template(
        _load_prompt("accumulation", "memory_extraction.txt", benchmark)
    )
    user_content = template.render(
        instruction=instruction,
        trajectory_text=trajectory_text,
        outcome=outcome,
        current_memory=current_memory or "",
        attempt_index=attempt_index,
        max_failures=max_failures,
        # What the solver is handed on every task anyway. Without it, 10-25% of every mined
        # lesson set re-teaches authentication and bootstrap that solver_env.txt already
        # states — measured across the train and 50-session pools.
        known=_solver_env(benchmark),
    )
    response = llm.generate(
        messages=[
            {"role": "system", "content": _MEMORY_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )
    return response.content.strip()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def format_trajectory_text(trace_dict: dict) -> str:
    """Format a trace dict into readable trajectory text."""
    lines = []
    for turn in trace_dict.get("turns", []):
        if turn.get("thought"):
            lines.append(f"Thought: {turn['thought']}")
        if turn.get("code"):
            lines.append(f"Code: {turn['code']}")
        if turn.get("execution_output"):
            output = turn["execution_output"]
            if len(output) > 500:
                output = output[:500] + "..."
            lines.append(f"Output: {output}")
        lines.append("")
    return "\n".join(lines)
