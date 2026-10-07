"""AutoGuide accumulation — extract context-aware guidelines from offline experience.

    uv run python -m references.autoguide.accumulation --config references/autoguide/configs/tau2_accumulation.yaml

This is the paper's Algorithm 1. Per training task: gather contrasting trajectories with
ReAct+Reflexion (Appendix B.1.3), find the timestep where the desired and undesired
trajectories first diverge, summarize the shared prefix into a CONTEXT (Eq. 1), fold that
context into the ones already seen (Fig. 11), and extract one guideline for it by
contrasting the pair (Eq. 2). The result is the dictionary G: context → guidelines.

AutoGuide's counterpart to `daedalus.scripts.accumulation`, writing the same artifacts
(`pool.json`, `pool.log.json`, `task_summaries/`, `traces/`) under
`outputs/baselines/autoguide/memory/<name>/`, plus `experiences/` (the raw trials).
What differs from DAEDALUS is the whole extraction: DAEDALUS rewrites one free-form memory per
task across retries; AutoGuide keys each guideline to the state in which it applies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daedalus.core.config import pool_output_path
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.logging import console

from references.autoguide.agents.base import _domain
from references.autoguide.config import SETUP, AutoGuideConfig
from references.autoguide.guidelines import Guideline, GuidelineBank
from references.autoguide.modules import (
    extract_guideline,
    identify_context,
    match_context,
    task_description,
)
from references.common import accumulation as common
from references.common.experience import (
    Experience,
    Trial,
    gather_experiences,
    task_summary,
)
from references.common.usage import Usage, write_run_summary


def deviation_index(desired: list[str], undesired: list[str]) -> int:
    """The first timestep at which the two trajectories take different actions.

    "we compare these two trajectories to find the deviation timestep t at which they begin
    to diverge due to different actions." When one is a prefix of the other they never
    disagree, so the deviation is where the shorter one stops.
    """
    for i, (a, b) in enumerate(zip(desired, undesired)):
        if a != b:
            return i
    return min(len(desired), len(undesired))


def shared_prefix(trial: Trial, index: int, task: str) -> str:
    """τ:t — the task plus every turn before the deviation."""
    turns = trial.turn_texts[:index]
    return "\n\n".join([f"Task: {task}", *turns]) if turns else f"Task: {task}"


def build_bank(
    experiences: list[Experience],
    context_llm: LLMClient,
    extraction_llm: LLMClient,
    autoguide: AutoGuideConfig,
    description: str,
) -> tuple[GuidelineBank, list[dict[str, Any]], Usage]:
    """Algorithm 1 over every contrastive experience. Returns (bank, log, usage).

    Sequential by construction: whether a new context is folded into an existing one
    depends on the contexts banked so far.
    """
    bank = GuidelineBank()
    log: list[dict[str, Any]] = []
    usage = Usage()

    contrastive = [e for e in experiences if e.contrastive]
    console.info(
        f"Contrastive tasks: {len(contrastive)}/{len(experiences)} "
        f"(a task needs both a success and a failure to yield a guideline)"
    )
    for experience in console.track(contrastive, desc="extracting"):
        desired = experience.successes[0]
        for undesired in experience.failures[: autoguide.max_pairs_per_task]:
            index = deviation_index(desired.actions, undesired.actions)
            prefix = shared_prefix(desired, index, experience.task)
            context = identify_context(
                context_llm, prefix, usage, autoguide.context_reasoning_effort
            )
            if not context:
                continue
            matched = match_context(
                context_llm,
                context,
                bank.contexts(),
                description,
                usage,
                autoguide.context_reasoning_effort,
            )
            key = matched or context
            reasoning, guideline = extract_guideline(
                extraction_llm,
                description,
                key,
                desired.trajectory,
                undesired.trajectory,
                usage,
                autoguide.extraction_reasoning_effort,
            )
            if not guideline:
                continue
            bank.add(
                Guideline(
                    context=key,
                    text=guideline,
                    source_task_id=experience.task_id,
                    reasoning=reasoning,
                )
            )
            log.append(
                {
                    "task_id": experience.task_id,
                    "deviation_timestep": index,
                    "context": context,
                    "merged_into": matched,
                    "guideline": guideline,
                    "reasoning": reasoning,
                }
            )
    return bank, log, usage


def main() -> None:
    args, cfg = common.start(SETUP, "AutoGuide accumulation: extract context-aware guidelines from offline experience")
    autoguide = AutoGuideConfig.load(args.config)
    autoguide.save(cfg)  # autoguide.yaml — the run folder stays self-describing
    CostAccounting.start(
        cfg,
        extra_models=[
            ModelRequirement(
                autoguide.context_model or cfg.agent.model,
                "extraction",
                "autoguide.context_model",
            ),
            ModelRequirement(
                autoguide.extraction_model, "extraction", "autoguide.extraction_model"
            ),
            ModelRequirement(
                autoguide.reflection_model or cfg.agent.model,
                "reflection",
                "autoguide.reflection_model",
            ),
        ],
    )

    task_ids = common.task_ids(cfg, args.task_id)

    console.header(
        "autoguide · accumulation",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "solver": cfg.agent.model,
            "reflection": autoguide.reflection_model,
            "extraction": autoguide.extraction_model,
            "retries": autoguide.max_retries,
            "tasks": len(task_ids),
            "parallel": max(1, cfg.run.parallel or 1),
            "output": pool_output_path(cfg).parent,
        },
    )

    experiences = gather_experiences(
        cfg=cfg,
        config_path=args.config,
        task_ids=task_ids,
        setup=SETUP,
        max_retries=autoguide.max_retries,
        reflection_model=autoguide.reflection_model,
        reflection_reasoning_effort=autoguide.reflection_reasoning_effort,
    )

    description = task_description(cfg.benchmark, _domain(cfg))
    accounting = CostAccounting.for_worker(cfg)
    # AutoGuide's two passes are both memory construction: the context describer and the
    # guideline extractor. Same role, distinct components.
    context_llm = accounting.client(
        autoguide.context_model or cfg.agent.model,
        "extraction",
        component="autoguide_context",
        temperature=0.0,
    )
    extraction_llm = accounting.client(
        autoguide.extraction_model,
        "extraction",
        component="autoguide_guidelines",
        temperature=0.0,
    )
    bank, log, usage = build_bank(
        experiences, context_llm, extraction_llm, autoguide, description
    )

    # overwrite=True: the bank is rebuilt from ALL experience checkpoints on every
    # invocation, so a resumed run's file supersedes the previous one — and it is the path
    # inference configs point at.
    pool_path = bank.to_pool().save(pool_output_path(cfg), overwrite=True)
    (pool_path.with_suffix(".log.json")).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    guidelines_by_task = {}
    for entry in log:
        guidelines_by_task.setdefault(entry["task_id"], []).append(entry["guideline"])
    summaries_dir = Path(pool_path.parent / "task_summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for experience in experiences:
        summary = task_summary(
            experience,
            "\n".join(f"- {g}" for g in guidelines_by_task.get(experience.task_id, [])),
        )
        (summaries_dir / f"{experience.task_id}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    solver_cost = sum(e.cost_usd for e in experiences)
    console.rule("autoguide accumulation complete")
    console.info(
        f"Tasks: {len(experiences)} | solved: {sum(1 for e in experiences if e.solved)} | "
        f"contrastive: {sum(1 for e in experiences if e.contrastive)}"
    )
    console.info(f"Guidelines: {len(bank)} across {bank.num_contexts} context(s)")
    console.info(
        f"Cost: rollouts+reflection ${solver_cost:.2f} + extraction ${usage.cost_usd:.2f} "
        f"= ${solver_cost + usage.cost_usd:.2f} ({usage.calls} extraction-side calls)"
    )
    summary_path = write_run_summary(
        cfg,
        accounting,
        {
            "num_tasks": len(experiences),
            "num_solved": sum(1 for e in experiences if e.solved),
            "num_guidelines": len(bank),
            "num_contexts": bank.num_contexts,
        },
    )
    console.info(f"Bank: {pool_path}")
    console.info(f"Run summary (whole-run spend, by role) → {summary_path}")


if __name__ == "__main__":
    main()
