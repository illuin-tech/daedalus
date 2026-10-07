"""τ² generation backend: ticket-style scenario specs graded by the LLM judge.

A grounded spec converts to a native τ² Task (its reference actions define the gold DB
end state), which the solver attempts against a simulated customer. Every attempt is
graded by the shared LLM judge over the spec's `success_conditions`. The judge LLM also
triages failures into capability vs spec defect, so defective specs do not pollute the
too-hard guidelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from jinja2 import Template

from daedalus.core.llm.client import LLMClient
from daedalus.core.config import ExperimentConfig
from daedalus.core.generation.backend import GenerationBackend
from daedalus.core.registry import get_benchmark
from daedalus.benchmarks.tau2.spec import ground_spec, spec_to_task

_TRIAGE_PROMPT = Path(__file__).parent / "prompts" / "triage.txt"

_TRIAGE_SYSTEM = (
    "You review failed attempts at generated customer-service tasks and attribute "
    "the failure to agent capability or to a defective task spec. Output only the "
    "requested JSON."
)


class Tau2GenerationBackend(GenerationBackend):
    name = "tau2"

    def session_contexts(self, cfg: ExperimentConfig) -> list[str]:
        return [cfg.tau2.domain]

    def make_explorer(self, cfg: ExperimentConfig, accounting: Any = None) -> Any:
        from daedalus.benchmarks.tau2.explorer import Tau2ExplorerAgent

        return Tau2ExplorerAgent(cfg, accounting)

    def spec_task_text(self, spec: dict[str, Any]) -> str:
        return spec["scenario"]["reason_for_call"]

    def spec_path(self, spec: dict[str, Any]) -> list[str]:
        return [a["name"] for a in spec["actions"]]

    def validate_spec(self, ctx: str, spec: dict[str, Any]) -> tuple[bool, str]:
        return ground_spec(spec, ctx)

    def solve(
        self,
        ctx: str,
        spec: dict[str, Any],
        cfg: ExperimentConfig,
        extraction_llm: LLMClient,
        judge_llm: LLMClient,
        trace_prefix: str,
        runner: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        from daedalus.core.generation.judge import make_evaluator

        task = spec_to_task(spec, ctx)
        evaluator = make_evaluator(
            judge_llm,
            self.spec_task_text(spec),
            spec["success_conditions"],
            reasoning_effort=cfg.generation.judge_reasoning_effort,
            format_trajectory=get_benchmark(cfg).format_trajectory,
        )
        summary = runner(
            task.id,
            cfg,
            extraction_llm,
            evaluator=evaluator,
            task_override=task,
            trace_prefix=trace_prefix,
        )
        if summary.get("final_outcome") != "success":
            defect = self._triage(judge_llm, spec, summary, cfg)
            if defect:
                summary["spec_defect"] = defect
        return summary

    def _triage(
        self,
        judge_llm: LLMClient,
        spec: dict[str, Any],
        summary: dict[str, Any],
        cfg: ExperimentConfig,
    ) -> str | None:
        """One judge call: capability failure (None) vs spec defect (reason)."""
        scenario = spec["scenario"]
        # SPEC_SCHEMA defines no `persona` (spec_to_task deliberately leaves tau2's own
        # persona field empty, matching the retail split), so it is not listed here.
        scenario_text = "\n".join(
            f"{k}: {scenario[k]}"
            for k in ("reason_for_call", "known_info", "unknown_info", "behavior")
            if scenario.get(k)
        )
        user = Template(_TRIAGE_PROMPT.read_text(encoding="utf-8")).render(
            scenario=scenario_text,
            actions=spec["actions"],
            trace=summary.get("last_trace_text", "(no trace)"),
        )
        content = judge_llm.generate(
            [
                {"role": "system", "content": _TRIAGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            reasoning_effort=cfg.generation.judge_reasoning_effort,
        ).content
        try:
            start, end = content.index("{"), content.rindex("}") + 1
            verdict = json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            return None
        if verdict.get("cause") == "spec_defect":
            return str(verdict.get("reason") or "unspecified spec defect")
        return None

    def bank_record(
        self,
        ctx: str,
        spec: dict[str, Any],
        path: list[str],
        novelty: float,
        refinements: int,
    ) -> dict[str, Any]:
        task = spec_to_task(spec, ctx)
        return {
            "task": spec["scenario"]["reason_for_call"],
            "spec": spec,
            "success_conditions": spec["success_conditions"],
            "native_task": task.model_dump(mode="json"),
            "path": path,
            "sandbox_id": ctx,
            "novelty": novelty,
            "refinements": refinements,
        }

    def persist_native(self, bank_tasks: list[dict[str, Any]], out_dir: Path) -> None:
        """Write banked tasks as a tau2-native task list (same shape as the
        domain's own tasks.json — loadable via Task.model_validate)."""
        native = [t["native_task"] for t in bank_tasks if "native_task" in t]
        (out_dir / "native_tasks.json").write_text(
            json.dumps(native, indent=2, ensure_ascii=False), encoding="utf-8"
        )
