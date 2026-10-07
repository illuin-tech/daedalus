"""AutomationBench generation backend — judge-graded self-play.

The pipeline owns the benchmark-neutral machinery (worker pool, bank, session flow,
novelty, too_easy/too_hard classification, refinement loop, guideline updates). This
backend supplies the AutomationBench seam: a context is one of the benchmark's own task
worlds, the explorer designs a task inside a throwaway copy of it and proves the task by
executing a solution, and the generated task is then attempted by the solver under the
Solver loop and graded by the LLM judge over its success conditions (AutomationBench's own
571 assertion types are not something an explorer can be asked to author).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from daedalus.core.config import ExperimentConfig
from daedalus.core.generation.backend import GenerationBackend
from daedalus.core.llm.client import LLMClient
from daedalus.core.registry import get_benchmark

from .spec import build_contexts, spec_path, spec_to_task
from .task_loader import Task

# Stop adding survey worlds once the next one would contribute less than this share of the
# train split's label weight. Not a world count: the number falls out of the split (five on
# operations, where world six adds 3.8%).
_SURVEY_MARGIN = 0.05


class AutomationBenchGenerationBackend(GenerationBackend):
    name = "automationbench"
    # No programmatic route for a task nobody wrote in advance — see the module docstring.

    def __init__(self) -> None:
        self._contexts: dict[str, Task] | None = None

    def _ctx_map(self, cfg: ExperimentConfig) -> dict[str, Task]:
        if self._contexts is None:
            self._contexts = build_contexts(cfg)
        return self._contexts

    def session_contexts(self, cfg: ExperimentConfig) -> list[str]:
        """The worlds sessions rotate through: the configured split's task worlds.

        Randomized (seeded) because the pipeline visits session i on contexts[i % N]:
        with fewer sessions than worlds, that would otherwise always pick the first few,
        and the first few tasks of a domain share a task family.
        """
        import random

        keys = list(self._ctx_map(cfg).keys())
        random.Random(cfg.automationbench.seed or 0).shuffle(keys)
        return keys

    def survey_contexts(self, cfg: ExperimentConfig) -> list[str]:
        """Worlds the survey visits, so the vocabulary can name what the domain grades.

        A world seeds 2-7 of the 22 services, so surveying one showed about an eighth of
        the domain: the first tagged run's manifest named gmail, sheets and asana and
        omitted slack — seeded in 16 of 30 train worlds and graded in over half the test
        tasks — so the steering could not ask for it once.

        Greedy cover, but weighted by how often a service is actually GRADED in the train
        split rather than by how many new services a world adds. Counting alone spends
        worlds on services that appear in one task (chatgpt, mailchimp) and skips asana and
        google_calendar, which appear in five each. Stop when the next world would add less
        than `_SURVEY_MARGIN` of the split's total label weight: on operations that is five
        worlds, covering 90% of train label mentions, and world six adds 3.8%.
        """
        from .mirrored import compute_allowed_services
        from .task_loader import load_tasks

        contexts = self._ctx_map(cfg)
        seeded = {
            k: set(compute_allowed_services(t.initial_state, [], []))
            for k, t in contexts.items()
        }
        graded: dict[str, int] = {}
        for t in load_tasks(cfg).values():  # contexts are stripped of assertions
            for s in compute_allowed_services({}, t.assertions, []):
                graded[s] = graded.get(s, 0) + 1
        total = sum(graded.values())
        if not total:  # no labels to weight by: fall back to plain coverage
            graded, total = {s: 1 for s in set().union(*seeded.values())}, len(seeded)

        chosen: list[str] = []
        covered: set[str] = set()
        while len(chosen) < len(seeded):
            k = max(
                (k for k in seeded if k not in chosen),
                key=lambda k: (
                    sum(graded.get(s, 0) for s in seeded[k] - covered),
                    len(seeded[k] - covered),
                    k,
                ),
            )
            gain = sum(graded.get(s, 0) for s in seeded[k] - covered) / total
            if chosen and gain < _SURVEY_MARGIN:
                break  # the tail adds one rarely-graded service per world
            chosen.append(k)
            covered |= seeded[k]
        return chosen

    def tags_for_context(
        self, cfg: ExperimentConfig, ctx: str, tags: list[dict[str, Any]]
    ) -> list[str]:
        """The coverage tags this world can serve. Drops only what it can prove absent.

        A world seeds 2-7 of the 22 services, so most tags of a domain-wide vocabulary
        name an app that is not here; steering on those wastes the session.

        The test is deliberately one-sided, because a tag's requirements are not declared
        anywhere — they have to be read off the label the survey chose:

          * label names one or more KNOWN services  ->  keep it only if this world seeds
            one of them. This is the case we can decide.
          * label names none                        ->  KEEP it. A work-shaped label like
            #vendor-onboarding, or τ²'s #PolicyLimited, says nothing about which app it
            needs, and dropping it would silence a tag that may well be buildable here.

        So the filter removes tags it can prove unreachable and never guesses about the
        rest. Definitions are not matched: they say things like "…and not Gmail- or
        Sheets-primary", which would keep a Gmail tag in a world with no Gmail. The exact
        alternative is for the survey to declare each tag's services in the manifest; that
        is a schema change, and this is accurate enough while vocabularies stay app-named.
        """
        from .mirrored import compute_allowed_services
        from .env import all_service_names

        task = self._ctx_map(cfg).get(ctx)
        if task is None:
            return [t["tag"] for t in tags]
        seeded = set(compute_allowed_services(task.initial_state, [], []))

        def aliases(service: str) -> set[str]:
            # "google_sheets" also answers to "google sheets" and "sheets"
            return {service, service.replace("_", " "), service.rsplit("_", 1)[-1]}

        known = all_service_names()
        kept: list[str] = []
        for tag in tags:
            label = str(tag["tag"]).lower()
            named = {s for s in known if any(a in label for a in aliases(s))}
            if not named or (named & seeded):
                kept.append(tag["tag"])
        return kept

    def make_explorer(self, cfg: ExperimentConfig, accounting: Any = None) -> Any:
        from .explorer import AutomationBenchExplorer

        return AutomationBenchExplorer(cfg, accounting)

    def spec_task_text(self, spec: dict[str, Any]) -> str:
        return spec["task"]

    def spec_path(self, spec: dict[str, Any]) -> list[str]:
        return spec_path(spec)

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

        task = spec_to_task(spec, self._ctx_map(cfg)[ctx])
        evaluator = make_evaluator(
            judge_llm,
            spec["task"],
            spec["success_conditions"],
            reasoning_effort=cfg.generation.judge_reasoning_effort,
            # This benchmark's own trace renderer, not AppWorld's default.
            format_trajectory=get_benchmark(cfg).format_trajectory,
        )
        return runner(
            task.task_id,
            cfg,
            extraction_llm,
            evaluator=evaluator,
            task_override=task,
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
            "success_conditions": spec["success_conditions"],
            "tags": spec.get("tags") or [],
            "path": path,
            "sandbox_id": ctx,
            "novelty": novelty,
            "refinements": refinements,
        }

    def persist_native(self, bank_tasks: list[dict[str, Any]], out_dir: Path) -> None:
        """Nothing to write: a generated task is its statement plus its success
        conditions, and the pipeline already saves those in tasks.json."""
