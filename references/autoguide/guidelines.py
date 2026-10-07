"""The context-aware guideline and the bank that holds them (paper §3.2).

A guideline is a conditional statement — "When in <status>, you should …" — paired with the
CONTEXT it applies to. The bank is a dictionary keyed by context (the paper's G), so test
time is: identify the current context, look it up, and inject only that context's
guidelines.

Storage is daedalus's `MemoryPool`, so the bank on disk is an ordinary `pool.json` that the
`serve` viewer and every daedalus retriever accept; the context lives in `MemoryItem.tags`
and the dictionary is rebuilt from it on load.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from daedalus.core.memory.pool import MemoryItem, MemoryPool


@dataclass
class Guideline:
    """One context-aware guideline."""

    context: str  # the CONTEXT this guideline is conditioned on (the bank's key)
    text: str  # the guideline itself, in the paper's "When in what status, …" form
    source_task_id: str = ""
    reasoning: str = ""  # the extractor's own "Reasoning: …" half, kept for inspection


class GuidelineBank:
    """The paper's G: context → guidelines, in insertion order."""

    def __init__(self) -> None:
        self._by_context: "OrderedDict[str, list[Guideline]]" = OrderedDict()

    def contexts(self) -> list[str]:
        return list(self._by_context)

    def add(self, guideline: Guideline) -> None:
        self._by_context.setdefault(guideline.context, []).append(guideline)

    def get(self, context: str) -> list[Guideline]:
        return list(self._by_context.get(context, []))

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_context.values())

    @property
    def num_contexts(self) -> int:
        return len(self._by_context)

    # ── on-disk form: a normal daedalus pool ─────────────────────────────────
    def to_pool(self) -> MemoryPool:
        pool = MemoryPool()
        for i, (context, guidelines) in enumerate(self._by_context.items()):
            for j, g in enumerate(guidelines):
                pool.add(
                    MemoryItem(
                        memory_id=f"g{i}_{j}",
                        type="guideline",
                        # Self-contained: the context is what makes a guideline readable
                        # on its own, and what any daedalus retriever would match against.
                        text=f"Context: {context}\nContext-Aware Guideline: {g.text}",
                        source_task_id=g.source_task_id,
                        tags={
                            "method": "autoguide",
                            "context": context,
                            "guideline": g.text,
                            "reasoning": g.reasoning,
                        },
                    )
                )
        return pool

    @classmethod
    def load(cls, path: str | Path) -> "GuidelineBank":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"AutoGuide guideline bank not found: {p}. Build one with "
                f"`python -m references.autoguide.accumulation --config <accumulation yaml>`."
            )
        bank = cls()
        for item in MemoryPool.load(p).items:
            tags = item.tags or {}
            context = tags.get("context")
            text = tags.get("guideline")
            if not context or not text:
                continue  # not an AutoGuide item; a bank is only contexts + guidelines
            bank.add(
                Guideline(
                    context=context,
                    text=text,
                    source_task_id=item.source_task_id,
                    reasoning=tags.get("reasoning", ""),
                )
            )
        return bank
