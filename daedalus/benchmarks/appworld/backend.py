"""AppWorld generation backend: code-executing explorer with in-loop grounding, and
LLM-judge grading (no ground truth exists for a generated task)."""

from __future__ import annotations

from typing import Any, Callable

from daedalus.core.config import ExperimentConfig
from daedalus.core.generation.backend import GenerationBackend
from daedalus.core.llm.client import LLMClient


class AppWorldGenerationBackend(GenerationBackend):
    name = "appworld"

    def configure_worker(self, cfg: ExperimentConfig, worker_exp_name: str) -> None:
        from appworld import AppWorld

        cfg.appworld.experiment_name = worker_exp_name
        AppWorld.init_defaults.experiment_name = worker_exp_name

    def session_contexts(self, cfg: ExperimentConfig) -> list[str]:
        from appworld.task import load_task_ids

        return load_task_ids(cfg.generation.sandbox_dataset)

    def make_explorer(self, cfg: ExperimentConfig, accounting: Any = None) -> Any:
        from daedalus.benchmarks.appworld.explorer import ExplorerAgent

        return ExplorerAgent(cfg, accounting)

    def spec_task_text(self, spec: dict[str, Any]) -> str:
        return spec["task"]

    def spec_path(self, spec: dict[str, Any]) -> list[str]:
        from daedalus.benchmarks.appworld.paths import predicted_path

        return predicted_path(spec["expected_path"])

    def heuristic_admissible(
        self, text: str, cfg: ExperimentConfig
    ) -> tuple[bool, str]:
        """Refuse a heuristic that names a held-out app.

        The explorer never learns those apps exist (hidden_apps.py), but the SOLVER that
        attempts the generated task sees the full catalog, and the heuristic is mined from
        its trace. Measured on the 90-session run with gmail and amazon held out: 27 of
        1,225 solver traces called `apis.gmail.*`, six banked heuristics carried gmail API
        recipes, and one reached the consolidated pool. Screening here closes that path.
        """
        from daedalus.benchmarks.appworld.hidden_apps import HiddenApps

        hidden = HiddenApps(cfg.generation.excluded_apps or [])
        if hidden and hidden.mentions(text or ""):
            named = [a for a in hidden.names if HiddenApps([a]).mentions(text or "")]
            return False, f"names held-out app(s) {named}"
        return True, ""

    def excluded_used(self, spec: dict[str, Any], cfg: ExperimentConfig) -> list[str]:
        excluded = set(cfg.generation.excluded_apps or [])
        apps = {t.split(".")[0] for t in self.spec_path(spec)}
        return sorted(excluded & apps)

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
        from daedalus.core.registry import get_benchmark

        evaluator = make_evaluator(
            judge_llm,
            spec["task"],
            spec["success_conditions"],
            reasoning_effort=cfg.generation.judge_reasoning_effort,
            format_trajectory=get_benchmark(cfg).format_trajectory,
        )
        # The generated task runs inside its sandbox world `ctx`.
        return runner(
            ctx,
            cfg,
            extraction_llm,
            instruction_override=spec["task"],
            evaluator=evaluator,
            trace_prefix=trace_prefix,
        )

    def bank_record(
        self,
        ctx: str,
        spec: dict[str, Any],
        path: list[str],
        novelty: float,
        refinements: int,
    ) -> dict[str, Any]:
        return {
            "task": spec["task"],
            "expected_path": spec["expected_path"],
            "path": path,
            "success_conditions": spec.get("success_conditions"),
            "sandbox_id": ctx,
            "novelty": novelty,
            "refinements": refinements,
        }
