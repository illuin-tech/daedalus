"""ExpeL accumulation — gather experiences, then extract insights from them.

    uv run python -m references.expel.accumulation --config references/expel/configs/tau2_accumulation.yaml

The paper's Algorithms 1 and 2 back to back. First the experience pool: run each training
task with ReAct+Reflexion, retrying after a failure until it is solved or the retry budget
runs out (§4.1). Then insight extraction (§4.2): walk every (success, failure) pair of the
same task, then every L-sized batch of successes, letting the extraction LLM AGREE with,
REMOVE, EDIT or ADD rules to a single evolving list.

ExpeL's counterpart to `daedalus.scripts.accumulation`, writing the same artifacts
(`pool.json` = the insight list, `pool.log.json`, `task_summaries/`, `traces/`) under
`outputs/baselines/expel/memory/<name>/`, plus `experiences/` (the raw trials) and
`demonstrations.json` (the successful trajectories, which inference recalls as few-shots).
What differs from DAEDALUS: the memory is one shared, *revisable* list of insights rather than
one heuristic per task, and successes are mined in batches as well as against failures.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from daedalus.core.config import pool_output_path
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.logging import console

from references.common import accumulation as common
from references.common.experience import Experience, gather_experiences, task_summary
from references.common.usage import Usage, write_run_summary
from references.expel.config import SETUP, ExpeLConfig
from references.expel.extraction import critique_all_success, critique_compare
from references.expel.insights import Insight, to_pool
from references.expel.retrieval import Demonstration, save_demonstrations


def compare_pairs(
    experiences: list[Experience], max_pairs_per_task: int
) -> list[tuple[str, str, str, str]]:
    """ExpeL's `Ccompare`: (task_id, task, success trajectory, failure trajectory)."""
    pairs = []
    for experience in experiences:
        if not experience.contrastive:
            continue
        success = experience.successes[0]
        for failure in experience.failures[:max_pairs_per_task]:
            pairs.append(
                (
                    experience.task_id,
                    experience.task,
                    success.trajectory,
                    failure.trajectory,
                )
            )
    return pairs


def success_batches(
    experiences: list[Experience], size: int, seed: int
) -> list[list[str]]:
    """ExpeL's `Csuccess`: successes split into L-sized chunks, sampled without replacement."""
    successes = [e.successes[0].trajectory for e in experiences if e.successes]
    rng = random.Random(seed)
    rng.shuffle(successes)
    return [successes[i : i + size] for i in range(0, len(successes), size)]


def extract_insights(
    experiences: list[Experience],
    llm: LLMClient,
    expel: ExpeLConfig,
    benchmark: str,
) -> tuple[list[Insight], list[dict[str, Any]], Usage]:
    """Algorithm 2: compare pairs first, then batches of successes.

    Sequential by construction — every call operates on the list the previous call left.
    """
    insights: list[Insight] = []
    log: list[dict[str, Any]] = []
    usage = Usage()

    pairs = compare_pairs(experiences, expel.max_pairs_per_task)
    batches = success_batches(experiences, expel.success_batch_size, expel.seed)
    console.info(
        f"Insight extraction: {len(pairs)} compare pair(s) + {len(batches)} success batch(es) "
        f"(L={expel.success_batch_size})"
    )

    for task_id, task, success, failure in console.track(pairs, desc="compare"):
        before = [i.text for i in insights]
        insights = critique_compare(
            llm,
            benchmark,
            insights,
            task=task,
            success_history=success,
            fail_history=failure,
            max_num_rules=expel.max_num_rules,
            usage=usage,
            effort=expel.extraction_reasoning_effort,
        )
        log.append(
            {
                "step": "compare",
                "task_id": task_id,
                "num_insights_before": len(before),
                "num_insights_after": len(insights),
                "insights": [{"text": i.text, "count": i.count} for i in insights],
            }
        )

    for batch in console.track(batches, desc="successes"):
        insights = critique_all_success(
            llm,
            benchmark,
            insights,
            success_histories=batch,
            max_num_rules=expel.max_num_rules,
            usage=usage,
            effort=expel.extraction_reasoning_effort,
        )
        log.append(
            {
                "step": "all_success",
                "batch_size": len(batch),
                "num_insights_after": len(insights),
                "insights": [{"text": i.text, "count": i.count} for i in insights],
            }
        )

    return insights, log, usage


def main() -> None:
    args, cfg = common.start(SETUP, "ExpeL accumulation: gather experiences, then extract insights")
    expel = ExpeLConfig.load(args.config)
    expel.save(cfg)  # expel.yaml — the run folder stays self-describing
    CostAccounting.start(
        cfg,
        extra_models=[
            ModelRequirement(
                expel.extraction_model, "extraction", "expel.extraction_model"
            ),
            ModelRequirement(
                expel.reflection_model or cfg.agent.model,
                "reflection",
                "expel.reflection_model",
            ),
        ],
    )

    task_ids = common.task_ids(cfg, args.task_id)

    console.header(
        "expel · accumulation",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "solver": cfg.agent.model,
            "reflection": expel.reflection_model or cfg.agent.model,
            "extraction": expel.extraction_model,
            "retries": expel.max_retries,
            "max rules": expel.max_num_rules,
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
        max_retries=expel.max_retries,
        reflection_model=expel.reflection_model or cfg.agent.model,
        reflection_reasoning_effort=expel.reflection_reasoning_effort,
    )

    accounting = CostAccounting.for_worker(cfg)
    extraction_llm = accounting.client(
        expel.extraction_model, "extraction", temperature=0.0
    )
    insights, log, usage = extract_insights(
        experiences, extraction_llm, expel, cfg.benchmark
    )

    # overwrite=True: both files are rebuilt from ALL experience checkpoints on every
    # invocation, so a resumed run's files supersede the previous ones — and they are the
    # paths inference configs point at.
    pool_path = to_pool(insights).save(pool_output_path(cfg), overwrite=True)
    (pool_path.with_suffix(".log.json")).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    demos = [
        Demonstration(
            task_id=e.task_id, task=e.task, trajectory=e.successes[0].trajectory
        )
        for e in experiences
        if e.successes
    ]
    demos_path = save_demonstrations(pool_path.with_name("demonstrations.json"), demos)

    summaries_dir = Path(pool_path.parent / "task_summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for experience in experiences:
        # The insight list is shared rather than per task, so the per-task record shows the
        # reflections that task produced — what it actually contributed on its own.
        memory = "\n".join(
            f"- {t.reflection}" for t in experience.trials if t.reflection
        )
        (summaries_dir / f"{experience.task_id}.json").write_text(
            json.dumps(task_summary(experience, memory), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    solver_cost = sum(e.cost_usd for e in experiences)
    console.rule("expel accumulation complete")
    console.info(
        f"Tasks: {len(experiences)} | solved: {sum(1 for e in experiences if e.solved)} | "
        f"contrastive: {sum(1 for e in experiences if e.contrastive)}"
    )
    console.info(f"Insights: {len(insights)} (cap {expel.max_num_rules})")
    console.info(
        f"Experience pool: {len(demos)} successful trajector(y/ies) → {demos_path}"
    )
    console.info(
        f"Cost: rollouts+reflection ${solver_cost:.2f} + extraction ${usage.cost_usd:.2f} "
        f"= ${solver_cost + usage.cost_usd:.2f} ({usage.calls} extraction call(s))"
    )
    summary_path = write_run_summary(
        cfg,
        accounting,
        {
            "num_tasks": len(experiences),
            "num_solved": sum(1 for e in experiences if e.solved),
            "num_insights": len(insights),
            "num_demonstrations": len(demos),
        },
    )
    console.info(f"Insight list: {pool_path}")
    console.info(f"Run summary (whole-run spend, by role) → {summary_path}")


if __name__ == "__main__":
    main()
