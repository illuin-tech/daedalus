"""LLM judge — grades a trace against a generated task's success conditions.

Replaces AppWorld's programmatic evaluator for generated tasks (which have no
ground truth). `make_evaluator` returns the closure the accumulation retry loop
calls in place of `world.evaluate()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from jinja2 import Template

from daedalus.core.llm.client import LLMClient
from daedalus.core.memory.extraction import (
    _extract_json_object,
    format_trajectory_text,
)

# Fallback renderer, used only when a caller supplies none. It is AppWorld-shaped
# (Thought/Code/Output) and truncates each output to 500 chars, so every benchmark should
# pass its own `Benchmark.format_trajectory`.
_DEFAULT_TRAJECTORY = format_trajectory_text

_PROMPT = Path(__file__).parent.parent / "prompts" / "generation" / "judge.txt"

_JUDGE_SYSTEM = (
    "You verify whether an agent achieved a task's intended outcomes. Judge the "
    "end effect, not exhaustive proof: a successful tool call (optionally confirmed "
    "by a read) is enough evidence its effect happened. Fail only on clear evidence "
    "a required outcome was missed or wrong; give competent solutions the benefit of "
    "the doubt and don't penalize incidental details you can't see."
)


def judge_trace(
    llm: LLMClient,
    instruction: str,
    success_conditions: list[str],
    trajectory_text: str,
    reasoning_effort: str | None = None,
) -> tuple[bool, str]:
    """Return (success, reason) for one trace against the success conditions."""
    user = Template(_PROMPT.read_text(encoding="utf-8")).render(
        instruction=instruction,
        success_conditions=success_conditions,
        trajectory_text=trajectory_text,
    )
    content = llm.generate(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        reasoning_effort=reasoning_effort,
    ).content

    obj = _extract_json_object(content)
    if obj is None:
        return False, f"unparseable_judge_output: {content[:100]}"
    return bool(obj.get("success", False)), str(obj.get("reason", ""))


def make_evaluator(
    llm: LLMClient,
    instruction: str,
    success_conditions: list[str],
    reasoning_effort: str | None = None,
    format_trajectory: Callable[[dict[str, Any]], str] | None = None,
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """Build the `evaluator(trace_dict) -> (success, details)` the agent calls.

    `format_trajectory` is the benchmark's own renderer (`Benchmark.format_trajectory`).
    Pass it: the default is AppWorld's, which labels turns Thought/Code/Output and cuts
    each output at 500 characters, so grading a tau2 conversation or an AutomationBench tool
    call through it both mislabels the trace and hides the evidence the judge needs.
    """
    render = format_trajectory or _DEFAULT_TRAJECTORY

    def _evaluator(trace: dict[str, Any]) -> tuple[bool, str]:
        return judge_trace(
            llm,
            instruction=instruction,
            success_conditions=success_conditions,
            trajectory_text=render(trace),
            reasoning_effort=reasoning_effort,
        )

    return _evaluator
