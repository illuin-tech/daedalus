"""The ERL heuristic: one structured lesson per source experience.

A heuristic is what accumulation banks and what retrieval selects. Its rendered form is
the paper's (Fig. 6): the source scenario id, the task it came from, the reward that task
earned, and the reflection text (`1. Analysis` + `2. Learned Guideline` with `Trigger:` /
`Action:`). That whole block is both what the ranker reads and what gets injected into the
solver's system prompt, because the paper's selection criteria are stated over the task
description and reward, not over the guideline alone.

Storage is daedalus's `MemoryPool`, so an ERL pool is an ordinary `pool.json`: the `serve`
viewer, `consolidate`, and every daedalus retriever accept it unchanged. The structured
fields are duplicated into `MemoryItem.tags` so they can be read back without parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from daedalus.core.memory.pool import MemoryItem, MemoryPool

SUCCESS = "success"
FAILURE = "failure"


@dataclass
class Heuristic:
    """One banked experience: where it came from, how it ended, what it teaches."""

    scenario_id: str  # the source task's id (the ranker's handle on this heuristic)
    task: str  # the source task description
    reward: str  # "success" | "failure"
    text: str  # the reflection: analysis + learned guideline

    @property
    def block(self) -> str:
        """The paper's rendering (Fig. 6) — self-contained, id first."""
        return (
            f"Scenario ID: {self.scenario_id}\n"
            f"Task: {self.task}\n"
            f"Reward: {self.reward}\n"
            f"{self.text.strip()}"
        )


def reward_label(success: bool) -> str:
    return SUCCESS if success else FAILURE


def to_memory_item(h: Heuristic, timestamp: str) -> MemoryItem:
    """Wrap a heuristic as a pool item (text = the full block)."""
    return MemoryItem(
        memory_id=h.scenario_id,  # the ranker refers to heuristics by scenario id
        type="heuristic",
        text=h.block,
        source_task_id=h.scenario_id,
        source_trajectory_success=(h.reward == SUCCESS),
        extraction_timestamp=timestamp,
        tags={"method": "erl", "reward": h.reward, "task": h.task, "reflection": h.text},
    )


def from_memory_item(item: MemoryItem) -> Heuristic:
    """Recover a heuristic from a pool item, tags first and block header as fallback.

    The fallback matters for a pool hand-edited or produced by another tool: the block is
    the source of truth on disk, the tags are a convenience.
    """
    tags = item.tags or {}
    task = tags.get("task")
    reward = tags.get("reward")
    text = tags.get("reflection")
    if task is None or reward is None or text is None:
        parsed = _parse_block(item.text)
        task = task if task is not None else parsed["task"]
        reward = reward if reward is not None else parsed["reward"]
        text = text if text is not None else parsed["text"]
    return Heuristic(
        scenario_id=item.memory_id or item.source_task_id,
        task=task,
        reward=reward,
        text=text,
    )


_HEADER_RE = re.compile(
    r"^Scenario ID:[^\n]*\nTask:\s*(?P<task>.*?)\nReward:\s*(?P<reward>[^\n]*)\n(?P<text>.*)$",
    re.DOTALL,
)


def _parse_block(block: str) -> dict[str, str]:
    match = _HEADER_RE.match(block.strip())
    if match is None:
        # Not an ERL block (e.g. a DAEDALUS pool): treat the whole item as the lesson.
        return {"task": "", "reward": "", "text": block.strip()}
    return {k: v.strip() for k, v in match.groupdict().items()}


def load_heuristics(pool_path: str | Path) -> list[Heuristic]:
    """Load an ERL pool as heuristics, in pool order."""
    path = Path(pool_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ERL heuristic pool not found: {path}. Build one with "
            f"`python -m references.erl.accumulation --config <accumulation yaml>`."
        )
    return [from_memory_item(item) for item in MemoryPool.load(path).items]
