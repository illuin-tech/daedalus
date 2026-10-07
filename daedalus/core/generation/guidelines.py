"""Difficulty-balance guideline updater — the explorer's distilled memory.

Benchmark-neutral: after a session that entered refinement, the pipeline calls
`update_explorer_memory` to re-derive the design guidelines shown to the explorer
on future sessions, so it learns to propose tasks that are challenging yet
solvable. Shared by every benchmark's generation backend.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

_MEMORY_PROMPT = Path(__file__).parent.parent / "prompts" / "generation" / "explorer_memory.txt"

_MEMORY_SYSTEM = (
    "You curate concise design guidelines that steer a task designer toward tasks "
    "that are challenging yet solvable — neither trivial nor impossible."
)


def update_explorer_memory(
    llm: LLMClient,
    guidelines: list[str],
    too_easy: list[str],
    too_hard: list[str],
    attempts: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    task_guidelines: str = "",
) -> list[str]:
    """Re-derive the explorer's guidelines (add/modify/delete).

    `attempts` is the whole session trajectory — the original off-target task and
    every refinement variant — each a dict {task, result, trace}. The updater sees
    the current guidelines, ALL too-easy/too-hard task descriptions, and this full
    trajectory (including how each variant turned out) so its lessons are causal.

    `task_guidelines` is the benchmark's fixed "what a good task looks like here" block —
    the SAME text the explorer is shown. Passing it in is what stops this loop drifting: with
    difficulty as its only objective, the cheapest repair for a too-hard task is "name the
    records, spell out the steps, narrow the scope", and that is what the 50/90-session runs
    actually converged on (measured: ids per task rose 2.2 -> 8.1 from the first quartile of a
    run to the last). Shown the rules, the updater has to calibrate within them.

    The prompt keeps the playbook compact by consolidation, not a numeric cap.
    """
    user = Template(_MEMORY_PROMPT.read_text(encoding="utf-8")).render(
        guidelines=guidelines,
        too_easy=too_easy,
        too_hard=too_hard,
        attempts=attempts,
        task_guidelines=task_guidelines,
    )
    content = llm.generate(
        [
            {"role": "system", "content": _MEMORY_SYSTEM},
            {"role": "user", "content": user},
        ],
        reasoning_effort=reasoning_effort,
    ).content
    # Only lines that ARRIVE as list items become guidelines. Stripping the marker off
    # every line and keeping whatever was left meant a preamble ("Here are the updated
    # guidelines:") or a markdown header became a guideline of its own, and a wrapped
    # guideline was split into two. Fall back to all non-empty lines when the model
    # emitted no markers at all, which is how it usually replies.
    marker = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
    lines = [ln for ln in content.strip().splitlines() if ln.strip()]
    marked = [marker.sub("", ln).strip() for ln in lines if marker.match(ln)]
    if marked:
        return [ln for ln in marked if ln]
    return [ln.strip() for ln in lines if ln.strip()]
