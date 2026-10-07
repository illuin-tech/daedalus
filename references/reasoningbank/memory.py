"""The ReasoningBank memory item, and the bank it lives in (paper §3.2).

A memory item is a "structured knowledge unit that abstracts away low-level execution
details while preserving transferrable reasoning patterns": a **title** (a concise
identifier), a **description** (one sentence), and the **content** (the distilled reasoning
steps or operational insights). Up to 3 come out of one trajectory.

The bank is organized by EXPERIENCE, not by item: an entry is one past task — its query,
the outcome the judge assigned it, and the items extracted from it. That grouping is what
retrieval works on ("we select memory items of the top-k most similar experiences"), so it
is what `Experience` models here.

Storage is daedalus's `MemoryPool`, so a ReasoningBank is an ordinary `pool.json` that the
`serve` viewer and every daedalus retriever accept: one pool item per memory item, with the
experience it came from recorded in `source_task_id` and the three fields duplicated into
`tags` so they can be read back without parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Template

from daedalus.core.memory.pool import MemoryItem, MemoryPool

PROMPTS = Path(__file__).parent / "prompts"


@dataclass
class Item:
    """One memory item: {title, description, content}."""

    title: str
    description: str
    content: str

    @property
    def block(self) -> str:
        """The item as the extractor wrote it — what `pool.json` stores as its text."""
        return (
            f"## Title {self.title}\n"
            f"## Description {self.description}\n"
            f"## Content {self.content}"
        )


@dataclass
class Experience:
    """One banked experience: a past task, its judged outcome, and its memory items."""

    task_id: str
    query: str  # what the embedding index is built over
    outcome: str  # "success" | "failure" — the JUDGE's label, not the environment's
    items: list[Item] = field(default_factory=list)


# The extractor's output format (Fig. 9): "# Memory Item i" then three "## " sections. The
# regexes are deliberately loose about what follows the heading — models write both
# "## Title Foo" (the paper's format) and "## Title\nFoo".
_ITEM_SPLIT_RE = re.compile(r"^#{1,3}\s*Memory\s+Item.*$", re.MULTILINE | re.IGNORECASE)
_FIELD_RE = {
    "title": re.compile(r"^#{1,3}\s*Title\b:?\s*(.*?)(?=^#{1,3}\s|\Z)", re.MULTILINE | re.DOTALL),
    "description": re.compile(
        r"^#{1,3}\s*Description\b:?\s*(.*?)(?=^#{1,3}\s|\Z)", re.MULTILINE | re.DOTALL
    ),
    "content": re.compile(
        r"^#{1,3}\s*Content\b:?\s*(.*?)(?=^#{1,3}\s|\Z)", re.MULTILINE | re.DOTALL
    ),
}
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)


def parse_items(llm_text: str, max_items: int) -> list[Item]:
    """Parse an extraction reply into memory items, capped at `max_items`.

    An item needs a title and a content to be useful; anything else is dropped rather than
    banked half-formed. The model's leading "why did this succeed/fail" thinking sits
    before the first `# Memory Item` heading and is discarded here.
    """
    text = _FENCE_RE.sub("", llm_text or "")
    items: list[Item] = []
    for chunk in _ITEM_SPLIT_RE.split(text)[1:]:  # [0] is the pre-item thinking
        fields = {}
        for name, pattern in _FIELD_RE.items():
            match = pattern.search(chunk)
            fields[name] = _clean(match.group(1)) if match else ""
        if not fields["title"] or not fields["content"]:
            continue
        items.append(Item(**fields))
        if len(items) >= max_items:
            break
    return items


def _clean(value: str) -> str:
    """Collapse a field's whitespace and drop the angle brackets of the format example."""
    return " ".join(value.split()).strip().strip("<>").strip()


def injection_block(items: list[Item]) -> str:
    """The paper's formatting template: its instruction, then title + content per item.

    "The retrieved items are concatenated into the agent's system prompt with a simple
    formatting template (each item represented by its title and content) and instruction"
    (Appendix A.2) — the description is extraction-time metadata and is not injected.
    """
    template = Template((PROMPTS / "memory_injection.txt").read_text(encoding="utf-8"))
    return template.render(items=items).strip()


def to_pool(experiences: list[Experience]) -> MemoryPool:
    """The bank as a daedalus pool: one item per memory item, in insertion order."""
    pool = MemoryPool()
    for experience in experiences:
        for index, item in enumerate(experience.items):
            pool.add(
                MemoryItem(
                    memory_id=f"{experience.task_id}#{index}",
                    type="memory_item",
                    text=item.block,
                    source_task_id=experience.task_id,
                    # The JUDGE's label, not the environment's: only an explicitly
                    # judged failure sets this flag to False.
                    source_trajectory_success=(experience.outcome != "failure"),
                    tags={
                        "method": "reasoningbank",
                        "title": item.title,
                        "description": item.description,
                        "content": item.content,
                        "query": experience.query,
                        "judged_outcome": experience.outcome,
                    },
                )
            )
    return pool


def load_bank(pool_path: str | Path) -> list[Experience]:
    """Load a bank written by `references.reasoningbank.accumulation`.

    Items are regrouped into their source experiences, in first-seen order, because that
    is the unit retrieval scores. A missing file is an empty bank, not an error: the bank
    starts empty and accumulation's first task retrieves from it before it exists.
    """
    path = Path(pool_path)
    if not path.exists():
        return []
    experiences: dict[str, Experience] = {}
    for pool_item in MemoryPool.load(path).items:
        tags = pool_item.tags or {}
        task_id = pool_item.source_task_id or pool_item.memory_id.split("#")[0]
        experience = experiences.get(task_id)
        if experience is None:
            experience = Experience(
                task_id=task_id,
                query=tags.get("query", ""),
                outcome=tags.get("judged_outcome")
                or ("success" if pool_item.source_trajectory_success else "failure"),
            )
            experiences[task_id] = experience
        experience.items.append(_item_from(pool_item))
    return list(experiences.values())


def require_bank(bank_path: str | None) -> Path:
    """Validate an inference run's bank before anything is written.

    Called by `run.py` up front, so a typo'd or missing bank fails immediately instead of
    deep inside a spawned solver worker — and without leaving an empty run folder behind.
    """
    if not bank_path:
        raise ValueError(
            "reasoningbank.bank_path is not set — an inference run needs the bank written "
            "by `references.reasoningbank.accumulation`. (An accumulation run sets it to "
            "its own pool.json automatically.)"
        )
    path = Path(bank_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ReasoningBank not found: {path}. Build one with `python -m "
            f"references.reasoningbank.accumulation --config <accumulation yaml>`."
        )
    return path


def _item_from(pool_item: MemoryItem) -> Item:
    """Recover an item from a pool item — tags first, the stored block as the fallback."""
    tags = pool_item.tags or {}
    if tags.get("title") and tags.get("content"):
        return Item(
            title=tags["title"],
            description=tags.get("description", ""),
            content=tags["content"],
        )
    parsed = parse_items(f"# Memory Item 1\n{pool_item.text}", max_items=1)
    return parsed[0] if parsed else Item(title="", description="", content=pool_item.text)
