"""AutomationBench Benchmark implementation (see daedalus/core/benchmark.py).

Grades are computed at solve time by the benchmark's own assertion sweep over the final
world and baked into each trace's `evaluation` / `extra` (as for tau2),
so `evaluate_run` only aggregates. On top of the strict pass rate it reports the
benchmark's second metric — mean `partial_credit`, the fraction of assertions met — and
a per-domain breakdown, since the six domains are the axis the published table uses.

Self-play generation is available through `backend.py`, graded by the LLM judge.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from daedalus.core.benchmark import Benchmark
from daedalus.core.config import ExperimentConfig

from . import ensure_automationbench_importable, task_loader


class AutomationBenchBenchmark(Benchmark):
    name = "automationbench"

    def list_task_ids(self, cfg: ExperimentConfig) -> list[str]:
        return task_loader.list_task_ids(cfg)

    def build_agent(self, cfg: ExperimentConfig, run_idx: int | None = None) -> Any:
        from .agent import AutomationBenchTaskAgent

        return AutomationBenchTaskAgent(cfg, run_idx=run_idx)

    def evaluate_run(
        self,
        cfg: ExperimentConfig,
        trace_dir: Path,
        task_ids: list[str] | None,
        run_idx: int | None,
    ) -> dict[str, Any]:
        trace_dir = Path(trace_dir)
        task_success: dict[str, bool] = {}
        partials: list[float] = []
        step_capped = 0
        by_domain: dict[str, list[bool]] = defaultdict(list)
        for path in sorted(trace_dir.glob("*.json")):
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if "success" not in trace:
                continue
            tid = trace.get("task_id", path.stem)
            success = bool(trace.get("success"))
            task_success[tid] = success
            extra = trace.get("extra") or {}
            partials.append(float(extra.get("partial_credit") or 0.0))
            step_capped += 1 if extra.get("hit_step_cap") else 0
            by_domain[str(extra.get("domain", ""))].append(success)

        n = len(task_success)
        ns = sum(1 for s in task_success.values() if s)
        return {
            "num_tasks": n,
            "num_successes": ns,
            # task_completed_correctly: every scored assertion passed. The official
            # AutomationBench pass rate.
            "success_rate": (ns / n) if n else 0.0,
            # partial_credit: the mean fraction of scored assertions met. Denser than
            # the pass rate and the benchmark's own training signal — read it next to
            # the pass rate, never instead of it.
            "partial_rate": (sum(partials) / len(partials)) if partials else 0.0,
            # How often the solver ran out of steps rather than deciding it was done.
            "step_cap_rate": (step_capped / n) if n else 0.0,
            "success_rate_by_domain": {
                d: (sum(1 for s in v if s) / len(v)) for d, v in sorted(by_domain.items())
            },
            "num_tasks_by_domain": {d: len(v) for d, v in sorted(by_domain.items())},
            "task_success": task_success,
        }

    def format_trajectory(self, trace: dict[str, Any]) -> str:
        lines: list[str] = []
        for turn in trace.get("turns", []):
            thought = (turn.get("thought") or "").strip()
            code = (turn.get("code") or "").strip()
            if code:
                if thought:
                    lines.append(f"Agent (thinking): {thought}")
                lines.append(f"Tool call: {code}")
                status = "ok" if turn.get("execution_success", True) else "ERROR"
                out = (turn.get("execution_output") or "").strip()
                lines.append(f"Tool result ({status}): {out}")
            elif thought:
                lines.append(f"Agent: {thought}")
        return "\n".join(lines)

    def prepare_accumulation(self, cfg: ExperimentConfig) -> None:
        # Resolve the checkout once, in the parent process, so a bad repo_root fails
        # fast instead of inside every worker.
        ensure_automationbench_importable(cfg.automationbench.repo_root)
