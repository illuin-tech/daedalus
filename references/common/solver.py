"""Shared solver-side plumbing for the methods under `references/`.

Each benchmark's solver already exposes the two hooks a memory method needs — a
`heuristics` list rendered into the system prompt at task start, and a per-turn retrieval
callback — so a method needs only a small subclass per benchmark. What every one of those
subclasses shares lives here: loading the method's config inside the worker, recording what
retrieval selected, and not paying for retrieval on a task that is about to be skipped.
"""

from __future__ import annotations

import json
from typing import Any

from daedalus.core.config import ExperimentConfig

from references.common.config import MethodConfig, retrieval_dir


class MethodSolverMixin:
    """Mixin for a benchmark TaskAgent driven by one of the reference methods."""

    method_config_cls: type[MethodConfig]

    def _init_method(self, cfg: ExperimentConfig) -> None:
        """Load this method's settings from the run folder's snapshot."""
        self.method_cfg = self.method_config_cls.load_for(cfg)
        self._retrieval: Any | None = None

    def _would_skip(self, task_id: str) -> bool:
        """True when the benchmark's `solve_task` will skip this task and return `{}`.

        Mirrors `TraceLogger.trace_exists` (same dir, same stem). Retrieval costs LLM
        calls, so a resumed run must not pay them for tasks it is about to skip. Only
        benchmarks whose retrieval resolves before their own skip check need this —
        AppWorld already skips before its first hook runs.
        """
        if self.cfg.run.force:
            return False
        return (self.trace_dir / f"{task_id}.json").exists()

    def _save_retrieval(self, task_id: str) -> None:
        """Record what retrieval selected for this task, alongside the run's traces.

        The trace keeps daedalus's normal shape, so this file is where a run's retrieval
        decisions — what was selected, why, and what the extra LLM calls cost — stay
        auditable after the fact. `self._retrieval` must expose `to_dict()`.
        """
        if self._retrieval is None:
            return
        out_dir = retrieval_dir(self.cfg, run_idx=self.run_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{task_id}.json").write_text(
            json.dumps(self._retrieval.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
