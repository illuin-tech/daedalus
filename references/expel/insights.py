"""ExpeL's insight list and the four operations that maintain it (paper §4.2, Alg. 2).

The insight list is not append-only: each extraction call may AGREE with, REMOVE, EDIT or
ADD rules, and every rule carries an importance count — ADD starts it at 2, AGREE and EDIT
increment, REMOVE decrements, and a rule is dropped when the count reaches 0. "This
particular design choice robustifies the process since even successful trajectories can be
suboptimal and mislead the generated insights."

`parse_operations` and `apply_operations` follow ExpeL's released implementation
(`agent/expel.py::parse_rules` / `update_rules`) rather than the paper's prose, including
its guards: an ADD whose text already exists is dropped, an EDIT that restates an existing
rule becomes an AGREE, operations naming a rule that does not exist are dropped, and the
operations are applied in the order REMOVE → AGREE → EDIT → ADD. The one number the paper
gives and the code parameterizes is the extra weight a REMOVE carries once the list is over
budget (3 instead of 1).

Storage is daedalus's `MemoryPool`, so the insight list on disk is an ordinary `pool.json`
that the `serve` viewer and every daedalus retriever accept; the count lives in the tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from daedalus.core.memory.pool import MemoryItem, MemoryPool

# ExpeL's own parser: an operation line, optionally numbered, whose text must end in a
# period (its guard against sentences the model cut off) and must not name another
# operation (its guard against nested/garbled formatting).
_OPERATION_RE = re.compile(r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)")
_BANNED_IN_TEXT = ("ADD", "AGREE", "EDIT")


@dataclass
class Insight:
    """One rule and its importance count."""

    text: str
    count: int = 2


def parse_operations(llm_text: str) -> list[tuple[str, str]]:
    """Parse an extraction reply into (operation, rule text) pairs."""
    out: list[tuple[str, str]] = []
    for operation, text in _OPERATION_RE.findall(llm_text or ""):
        text = text.strip()
        if not text or any(word in text for word in _BANNED_IN_TEXT) or not text.endswith("."):
            continue
        out.append(("ADD" if "ADD" in operation else operation.strip(), text))
    return out


def _rule_index(insights: list[Insight], text: str) -> int | None:
    """Index of the existing rule contained in `text` (ExpeL matches by substring)."""
    for i, insight in enumerate(insights):
        if insight.text in text:
            return i
    return None


def apply_operations(
    insights: list[Insight], operations: list[tuple[str, str]], list_full: bool = False
) -> list[Insight]:
    """Apply one extraction call's operations, returning the new list.

    `list_full` triples the weight of a REMOVE — ExpeL sets it once the list has grown
    past its budget, so an over-long list shrinks faster than it grows.
    """
    insights = [Insight(i.text, i.count) for i in insights]
    kept: list[tuple[str, str]] = []
    for operation, text in operations:
        kind = operation.split(" ")[0]
        number = int(operation.split(" ")[1]) if " " in operation else None
        if kind == "ADD":
            if _rule_index(insights, text) is None:  # skip an ADD of something we have
                kept.append((operation, text))
        elif kind == "EDIT":
            index = _rule_index(insights, text)
            if index is not None:  # an EDIT that restates a rule is really an AGREE
                kept.append((f"AGREE {index + 1}", insights[index].text))
            elif number is not None and number <= len(insights):
                kept.append((operation, text))
        elif kind in ("REMOVE", "AGREE"):
            if _rule_index(insights, text) is not None:
                kept.append((operation, text))

    for kind in ("REMOVE", "AGREE", "EDIT", "ADD"):  # order matters
        for operation, text in kept:
            if operation.split(" ")[0] != kind:
                continue
            if kind == "REMOVE":
                index = _rule_index(insights, text)
                if index is not None:
                    insights[index].count -= 3 if list_full else 1
            elif kind == "AGREE":
                index = _rule_index(insights, text)
                if index is not None:
                    insights[index].count += 1
            elif kind == "EDIT":
                index = int(operation.split(" ")[1]) - 1
                if 0 <= index < len(insights):
                    insights[index] = Insight(text, insights[index].count + 1)
            elif kind == "ADD":
                insights.append(Insight(text, 2))

    return [i for i in insights if i.count > 0]


def to_pool(insights: list[Insight]) -> MemoryPool:
    """The insight list as a daedalus pool, most important first (ExpeL's own ordering)."""
    pool = MemoryPool()
    ordered = sorted(insights, key=lambda i: i.count, reverse=True)
    for n, insight in enumerate(ordered):
        pool.add(
            MemoryItem(
                memory_id=f"i{n}",
                type="insight",
                text=insight.text,
                tags={"method": "expel", "count": insight.count},
            )
        )
    return pool


def load_insights(path: str | Path | None) -> list[Insight]:
    """Load an insight list written by `references.expel.accumulation`."""
    if not path:
        raise ValueError("No insight list path given (expel.insights_path is unset).")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ExpeL insight list not found: {p}. Build one with "
            f"`python -m references.expel.accumulation --config <accumulation yaml>`."
        )
    out = []
    for item in MemoryPool.load(p).items:
        count = (item.tags or {}).get("count", 2)
        out.append(Insight(text=item.text, count=int(count)))
    return out
