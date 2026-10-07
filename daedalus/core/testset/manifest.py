"""The frozen test set: generated tasks turned into a fixed, citable benchmark.

A generation run's `tasks.json` is a *bank* — it grows every time the run is extended, and
its order reflects when each task happened to be banked. A benchmark cannot be that: the
same task must keep the same id across models and across repeats, or pass^k and any
cross-model comparison is meaningless.

So the first thing `tasks_as_test_set` does is freeze the bank into a manifest
(`test_set.json`): each banked task gets a stable `g###` id, and the manifest records which
generation run it came from. Every later run of the same test set reuses that file rather
than re-reading the bank, so extending the generation run does not silently change what a
model was measured on.

What a test task carries is what the judge needs (the instruction and the success
conditions) plus what the runner needs to reproduce the world (`sandbox_id`, the real
benchmark task whose *starting* database the generated instruction runs against).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "test_set.json"


@dataclass
class TestTask:
    """One task of the frozen test set."""

    tid: str  # stable id, g000…  (never derived from position in the live bank)
    instruction: str
    success_conditions: list[str]
    sandbox_id: str  # the benchmark task whose starting world this runs in
    tags: list[str] = field(default_factory=list)
    answer: Any = None  # the expected answer when the generator recorded one

    @property
    def num_conditions(self) -> int:
        return len(self.success_conditions)


@dataclass
class TestSet:
    """A frozen test set and where it came from."""

    generation_run: str
    benchmark: str
    dataset: str  # the split the sandbox worlds live in (appworld: train/dev/…)
    generator_model: str  # the solver the difficulty band was measured with
    tasks: list[TestTask]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_run": self.generation_run,
            "benchmark": self.benchmark,
            "dataset": self.dataset,
            "generator_model": self.generator_model,
            "num_tasks": len(self.tasks),
            "tasks": [asdict(t) for t in self.tasks],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TestSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            generation_run=data["generation_run"],
            benchmark=data["benchmark"],
            dataset=data["dataset"],
            generator_model=data.get("generator_model", ""),
            tasks=[TestTask(**t) for t in data["tasks"]],
        )


def from_generation_run(run_dir: str | Path) -> TestSet:
    """Freeze the BANKED tasks of a generation run into a test set.

    Only `tasks.json["tasks"]` is used: those are the specs that survived the difficulty
    band, and they are the only entries that carry `success_conditions` (the `too_easy` /
    `too_hard` buckets keep the instruction text alone). Tasks are ordered by their
    sandbox id so the ids are reproducible from the same bank, not from its write order.
    """
    base = Path(run_dir)
    bank_path = base / "tasks.json"
    if not bank_path.exists():
        raise FileNotFoundError(
            f"{bank_path} not found — `--generation` wants a generation run folder "
            f"(one of outputs/daedalus/<name>/)."
        )
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    banked = bank.get("tasks") or []
    if not banked:
        raise ValueError(f"{bank_path} has no banked tasks to evaluate on.")

    meta = _read_json(base / "run_meta.json")
    cfg = _read_yaml(base / "config.yaml")
    benchmark = meta.get("benchmark") or cfg.get("benchmark") or "appworld"
    dataset = meta.get("domain") or ((cfg.get(benchmark) or {}).get("dataset")) or ""

    tasks = []
    for i, spec in enumerate(sorted(banked, key=lambda s: (str(s.get("sandbox_id")), s["task"]))):
        conditions = [c for c in (spec.get("success_conditions") or []) if str(c).strip()]
        if not conditions:
            continue  # nothing to judge against; it cannot be a test item
        tasks.append(TestTask(
            tid=f"g{i:03d}",
            instruction=spec["task"],
            success_conditions=conditions,
            sandbox_id=str(spec.get("sandbox_id") or ""),
            tags=list(spec.get("tags") or []),
            answer=spec.get("answer"),
        ))
    return TestSet(
        generation_run=base.name,
        benchmark=benchmark,
        dataset=dataset,
        generator_model=meta.get("model") or (cfg.get("agent") or {}).get("model", ""),
        tasks=tasks,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
