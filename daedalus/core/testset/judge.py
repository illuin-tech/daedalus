"""The test set's grader: one LLM judge verdict per success condition.

Same judge, same doctrine and same wording as generation's own
(`prompts/generation/judge.txt` → `judge_conditions.txt`); the only change is the output
contract, which now reports a verdict per intended outcome instead of one boolean. That is
what makes **partial credit** possible: a task with three conditions of which two hold
scores 2/3 rather than simply "failed", which is the difference between a benchmark that
ranks models and one that only separates the perfect from the rest.

`success` still means every condition met, so the headline success rate is comparable with
the generation pipeline's own judgements and with a real benchmark's pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

from daedalus.core.llm.client import LLMClient
from daedalus.core.memory.extraction import _extract_json_object

_PROMPT = Path(__file__).parent.parent / "prompts" / "generation" / "judge_conditions.txt"

# Verbatim from daedalus.core.generation.judge, so the two graders share a doctrine.
_JUDGE_SYSTEM = (
    "You verify whether an agent achieved a task's intended outcomes. Judge the "
    "end effect, not exhaustive proof: a successful tool call (optionally confirmed "
    "by a read) is enough evidence its effect happened. Fail only on clear evidence "
    "a required outcome was missed or wrong; give competent solutions the benefit of "
    "the doubt and don't penalize incidental details you can't see."
)


@dataclass
class Verdict:
    """What the judge said about one trajectory."""

    success: bool
    met: list[bool]  # one entry per success condition, in order
    reason: str
    reasons: list[str] = field(default_factory=list)  # per condition
    parsed: bool = True  # False when the reply could not be read
    model: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def num_met(self) -> int:
        return sum(1 for m in self.met if m)

    @property
    def partial(self) -> float:
        """Fraction of intended outcomes achieved. 1.0 exactly when `success`."""
        return (self.num_met / len(self.met)) if self.met else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "partial": self.partial,
            "num_met": self.num_met,
            "num_conditions": len(self.met),
            "outcomes": [
                {"index": i + 1, "met": m, "reason": r}
                for i, (m, r) in enumerate(zip(self.met, self.reasons or [""] * len(self.met)))
            ],
            "reason": self.reason,
            "parsed": self.parsed,
            "judge_model": self.model,
            "judge_cost_usd": self.cost_usd,
            "judge_tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
        }

    @property
    def details(self) -> str:
        """One line for the trace's evaluation block, readable in the run browser."""
        head = f"{self.num_met}/{len(self.met)} conditions met"
        return f"{head} — {self.reason}" if self.reason else head


def judge_conditions(
    llm: LLMClient,
    instruction: str,
    success_conditions: list[str],
    trajectory_text: str,
    reasoning_effort: str | None = None,
) -> Verdict:
    """Grade one trajectory, condition by condition.

    A reply that cannot be read is recorded as a FAILURE with `parsed=False` rather than
    guessed at — a benchmark that silently turns judge errors into passes is worse than one
    that reports them, and the count surfaces in `evaluation.json`.
    """
    user = Template(_PROMPT.read_text(encoding="utf-8")).render(
        instruction=instruction,
        success_conditions=success_conditions,
        trajectory_text=trajectory_text,
    )
    response = llm.generate(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        reasoning_effort=reasoning_effort,
    )
    # The judge call's own usage event already priced it (with its service tier and cache
    # breakdown); re-deriving a second number here is what let the two disagree.
    cost = response.estimated_cost_usd or 0.0
    n = len(success_conditions)
    obj = _extract_json_object(response.content)
    if not isinstance(obj, dict):
        return Verdict(
            success=False, met=[False] * n,
            reason=f"unparseable_judge_output: {(response.content or '')[:120]}",
            reasons=[""] * n, parsed=False, model=llm.model, cost_usd=cost,
            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        )

    met, reasons = _outcomes(obj, n)
    # The prompt says "SUCCEEDS only if every intended outcome is met", so the per-outcome
    # verdicts are the authority: they are what partial credit is computed from, and a
    # model-level `success: true` that contradicts them would make the two metrics
    # inconsistent. Both are kept when they agree, and the outcomes win when they don't.
    success = all(met) if met else bool(obj.get("success", False))
    return Verdict(
        success=success, met=met, reason=str(obj.get("reason", "")), reasons=reasons,
        model=llm.model, cost_usd=cost,
        prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
    )


def _outcomes(obj: dict[str, Any], n: int) -> tuple[list[bool], list[str]]:
    """Read the per-outcome verdicts, tolerating a judge that answers loosely.

    Falls back to the overall boolean when the list is missing entirely — an older judge
    reply, or a model that ignored the format — so a run degrades to pass/fail rather than
    dying.
    """
    raw = obj.get("outcomes") or obj.get("conditions") or []
    met, reasons = [False] * n, [""] * n
    if not isinstance(raw, list) or not raw:
        overall = bool(obj.get("success", False))
        return [overall] * n, [""] * n
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        i = (int(index) - 1) if isinstance(index, (int, float)) else position
        if not 0 <= i < n:
            continue
        met[i] = bool(entry.get("met", entry.get("success", False)))
        reasons[i] = str(entry.get("reason", ""))
    return met, reasons
