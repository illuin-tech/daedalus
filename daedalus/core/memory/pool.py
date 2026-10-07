"""Memory pool management — load, save, index memory items."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    """A single memory item in the pool."""

    memory_id: str
    type: str  # "reflection", "sub_trajectory", "workflow"
    text: str
    source_task_id: str = ""
    source_trajectory_success: bool = True
    extraction_timestamp: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryPool:
    """A flat collection of memory items with load/save and indexing support."""

    def __init__(self, items: list[MemoryItem] | None = None):
        self.items: list[MemoryItem] = items or []
        self._by_id: dict[str, MemoryItem] = {m.memory_id: m for m in self.items}

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self._by_id[item.memory_id] = item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._by_id.get(memory_id)

    def texts(self) -> list[str]:
        return [m.text for m in self.items]

    def ids(self) -> list[str]:
        return [m.memory_id for m in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save the pool to a JSON file.

        If *overwrite* is False (default) and *path* already exists, a numeric
        suffix is appended (e.g. pool_1.json, pool_2.json) to avoid data loss.
        Returns the actual path written.
        """
        path = Path(path)
        if not overwrite and path.exists():
            stem = path.stem
            suffix = path.suffix
            parent = path.parent
            counter = 1
            while path.exists():
                path = parent / f"{stem}_{counter}{suffix}"
                counter += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"items": [item.to_dict() for item in self.items]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPool":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [MemoryItem(**item_data) for item_data in data["items"]]
        return cls(items)

    @classmethod
    def load_or_empty(cls, path: str | Path | None) -> "MemoryPool":
        """The pool at `path`, or an EMPTY pool when there is none.

        For a pool being BUILT (generation/accumulation read their own growing bank, which
        does not exist before the first task banks something). A solver must not use this —
        an empty pool there is a memory-free run wearing a memory run's name. Use
        `load_for_solver` instead.
        """
        if path is None:
            return cls()
        p = Path(path)
        if p.exists():
            return cls.load(p)
        return cls()

    @classmethod
    def load_for_solver(cls, path: str | Path | None, benchmark: str = "") -> "MemoryPool":
        """The pool a solver will inject, refusing to substitute an empty one.

        A missing `memory.pool_path` used to yield an empty pool and a run that scored
        exactly like the baseline while being filed and reported as a memory arm — and in
        `heuristics_at_start` mode nothing in the trace records what was injected, so it
        could not be detected afterwards either. A configured path that is not on disk is
        now a hard error, and a pool that is present but empty is too: neither can be what
        the config meant.
        """
        where = f"[{benchmark}] " if benchmark else ""
        if path is None:
            raise FileNotFoundError(
                f"{where}memory.enabled is true but memory.pool_path is not set — there is "
                f"nothing to inject. Set the pool path, or set memory.enabled: false."
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{where}memory.pool_path does not exist: {p}. Build the pool first "
                f"(generation / accumulation, then consolidate), or fix the path. "
                f"Running on would silently produce a memory-free run under a memory "
                f"run's name."
            )
        pool = cls.load(p)
        if len(pool) == 0:
            raise ValueError(
                f"{where}memory.pool_path exists but holds no items: {p}. That is a "
                f"memory-free run; set memory.enabled: false if that is what you want."
            )
        return pool
