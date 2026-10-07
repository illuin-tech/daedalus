"""Generation backends: the benchmark-specific seam of the self-play pipeline.

The pipeline (pipeline.py) owns the benchmark-neutral machinery — worker pool,
file-locked bank, session flow, classification, refinement loop, guideline updates.
Everything a benchmark defines lives behind this interface: how a session's environment
context is chosen, how the explorer plays and what spec it emits, how a spec is validated
and attempted, and how banked tasks are persisted natively. Adding a benchmark means
implementing this class, an explorer and its prompts.

The explorer returned by make_explorer is duck-typed:
    .llm            LLMClient (for cost accounting)
    .log_prefix     str, set by workers
    .explore(ctx, prior_tasks, guidelines, coverage_goal="", coverage_tally="")
            -> (spec | None, transcript)
    .refine(ctx, prior_tasks, guidelines, exploration, current_task, difficulty,
            variant_history, coverage_goal="", coverage_tally="")
            -> (spec | None, transcript)
        `coverage_goal` / `coverage_tally` are the run's coverage plan and the live per-tag
        tally (see generation/coverage.py); both are empty when coverage tagging is off. An
        emitted spec may carry `tags` (the tags the explorer declares for its task).
    .survey_coverage(ctx) -> (CoverageGoal, transcript)
        One-time survey of a throwaway copy of the environment (which the survey is
        expected to CHANGE — calling a write operation is how it learns what it demands)
        → the run's coverage goal + closed tag vocabulary. The pipeline runs it once
        before workers fan out (generation.coverage_tags), banks it, and feeds it back to
        every explore/refine call; generation.coverage_tags_path replaces it with a
        precomputed artifact.
    .explore_direct(ctx, previous_heuristics) -> (list[heuristic], transcript)
        AppWorld only: the explorer-only ablation (generation.stage
        `direct_exploration_with_bank`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from daedalus.core.llm.client import LLMClient
from daedalus.core.config import ExperimentConfig


class GenerationBackend(ABC):
    """One benchmark's generation hooks."""

    name: str

    def configure_worker(self, cfg: ExperimentConfig, worker_exp_name: str) -> None:
        """Per-worker-process setup (default: nothing). cfg.experiment_name is
        already set to worker_exp_name by the pipeline."""

    @abstractmethod
    def session_contexts(self, cfg: ExperimentConfig) -> list[str]:
        """Environment contexts sessions rotate through (appworld: sandbox task
        ids; tau2: the domain)."""

    @abstractmethod
    def make_explorer(
        self, cfg: ExperimentConfig, accounting: Any = None
    ) -> Any:
        """Construct the explorer (see module docstring for its duck-type).

        `accounting` is the launch's `CostAccounting`; the explorer's LLM calls are
        recorded through it under the `explorer` role.
        """

    @abstractmethod
    def spec_task_text(self, spec: dict[str, Any]) -> str:
        """The task statement of an emitted spec (bank entries, prompts, logs)."""

    @abstractmethod
    def spec_path(self, spec: dict[str, Any]) -> list[str]:
        """Ordered tool tokens of the spec's intended solution (novelty)."""

    def heuristic_admissible(
        self, text: str, cfg: ExperimentConfig
    ) -> tuple[bool, str]:
        """Whether a mined heuristic may enter the pool. Default: always.

        `excluded_used` screens the emitted SPEC, which is the explorer's declared path.
        It cannot screen the LESSON, and the lesson is mined from the solver's trace — and
        the generation-time solver sees the whole environment, held-out apps included. On
        AppWorld that put six gmail-specific API recipes into a bank built with gmail held
        out, one of which survived consolidation into the pool behind the headline row.
        A backend that holds anything out should override this.
        """
        return True, ""

    def survey_contexts(self, cfg: ExperimentConfig) -> list[str]:
        """The worlds the coverage survey visits, each surveyed once. Default: one.

        One world IS the environment on τ² (a domain) and on AppWorld (every world exposes
        every app), so one survey sees everything and the default is right. Override it
        where a world shows only a slice of the app inventory: AutomationBench seeds 2-7 of
        22 services per world, and surveying one produced a manifest with no slack tag —
        slack being seeded in 16 of 30 worlds and graded in over half the test tasks. There
        is no knob for the count; the backend derives it from the environment.
        """
        return self.session_contexts(cfg)[:1]

    def tags_for_context(
        self, cfg: ExperimentConfig, ctx: str, tags: list[dict[str, Any]]
    ) -> list[str]:
        """Which coverage tags THIS context can actually serve (default: all of them).

        Right by default where every context exposes the whole environment (τ², AppWorld).
        A benchmark whose contexts each hold a slice of it should override this, or the
        steering will order sessions to build around something that is not there — on
        AutomationBench the run's most-under tag was an app seeded in 5 of 30 worlds, so
        the directive was declined session after session while the tag kept the
        escalation slot.
        """
        return [t["tag"] for t in tags]

    def excluded_used(self, spec: dict[str, Any], cfg: ExperimentConfig) -> list[str]:
        """Held-out apps this spec touches (non-empty = reject). Default: none."""
        return []

    def validate_spec(self, ctx: str, spec: dict[str, Any]) -> tuple[bool, str]:
        """Post-emission validation (default: accept — appworld grounds specs
        inside the explorer loop; tau2 replays reference actions)."""
        return True, ""

    @abstractmethod
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
        """Attempt the spec with `runner` (the Solver loop, or the single attempt of the
        row-C ablation), graded by the LLM judge over its success conditions. Returns the
        runner's summary; may set summary["spec_defect"] = <reason> when a failure is
        attributable to a defective spec rather than task difficulty."""

    @abstractmethod
    def bank_record(
        self,
        ctx: str,
        spec: dict[str, Any],
        path: list[str],
        novelty: float,
        refinements: int,
    ) -> dict[str, Any]:
        """The bank entry for a banked task. Must include "task" (text) and
        "path" (tool tokens) — the pipeline reads only those two."""

    def persist_native(self, bank_tasks: list[dict[str, Any]], out_dir: Path) -> None:
        """Write banked tasks in the benchmark's native runnable format
        (default: nothing)."""


def get_generation_backend(cfg: ExperimentConfig) -> GenerationBackend:
    """Instantiate the generation backend for cfg.benchmark (lazy imports)."""
    if cfg.benchmark == "appworld":
        from daedalus.benchmarks.appworld.backend import AppWorldGenerationBackend

        return AppWorldGenerationBackend()
    if cfg.benchmark == "tau2":
        from daedalus.benchmarks.tau2.backend import Tau2GenerationBackend

        return Tau2GenerationBackend()
    if cfg.benchmark == "automationbench":
        from daedalus.benchmarks.automationbench.backend import (
            AutomationBenchGenerationBackend,
        )

        return AutomationBenchGenerationBackend()
    raise ValueError(f"No generation backend for benchmark {cfg.benchmark!r}")
