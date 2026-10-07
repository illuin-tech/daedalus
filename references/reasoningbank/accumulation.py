"""ReasoningBank accumulation — run the closed loop over a source split.

    uv run python -m references.reasoningbank.accumulation --config references/reasoningbank/configs/tau2_accumulation.yaml

The paper's three steps (§3.2), in a stream: for each task, (i) RETRIEVE the top-k most
similar past experiences from the bank built so far and inject their memory items, (ii)
EXTRACT new items from the finished trajectory — first labelling it success or failure with
an LLM-as-a-judge, then running the matching prompt (Fig. 9) — and (iii) CONSOLIDATE, which
here is what the paper says it is: "newly generated items are directly added without
additional pruning". One rollout per task (no MaTTS test-time scaling).

ReasoningBank's counterpart to `daedalus.scripts.accumulation`, writing the same artifacts
(`pool.json`, `pool.log.json`, `task_summaries/`, `traces/`) under
`outputs/baselines/reasoningbank/memory/<name>/`, plus `retrieval/` (what each task
retrieved from the bank as it stood). Two things differ from every other method here:

    * the loop is CLOSED — the rollouts are not memory-free, they use the bank being built,
      which is the paper's whole premise (and why the tasks run in sequential waves);
    * the outcome signal is the agent's OWN judgement, never the environment's reward. The
      benchmark's verdict is recorded beside it, and reported as judge accuracy at the end,
      but nothing is banked from it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig, pool_output_path
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.logging import console
from daedalus.core.registry import get_benchmark

from references.common import accumulation as common
from references.common.usage import Usage, write_run_summary
from references.reasoningbank.agents import build_reasoningbank_agent
from references.reasoningbank.config import SETUP, ReasoningBankConfig
from references.reasoningbank.extraction import extract_items
from references.reasoningbank.judge import judge
from references.reasoningbank.memory import Experience, Item, to_pool


def accumulate_task(
    task_id: str,
    cfg: ExperimentConfig,
    rb: ReasoningBankConfig,
    extraction_llm: LLMClient,
    judge_llm: LLMClient,
) -> dict[str, Any]:
    """One step of the stream: solve with the current bank, then extract from the result.

    The summary carries DAEDALUS's accumulation field names (`attempts`, `final_outcome`,
    `final_memory`, `solved_without_memory`) so the viewer reads these runs too, plus what
    is specific to ReasoningBank: the judge's label and the items it produced.
    """
    benchmark = get_benchmark(cfg)
    # The agent reads the bank as it stands (its `bank_path` is this run's own pool.json,
    # rewritten by the parent between waves) and injects what it retrieves.
    agent = build_reasoningbank_agent(cfg)

    # heuristics=[] keeps the runner on its accumulation path (re-run the task, save the
    # attempt trace); what ReasoningBank retrieves is added by the agent itself. One
    # rollout per task: the paper's k = 1, no MaTTS.
    result = agent.solve_task(task_id, heuristics=[], trace_suffix="attempt1")

    query = result.get("task_instruction") or task_id
    trajectory = benchmark.format_trajectory(result)
    env_outcome = "success" if result.get("success") else "failure"
    retrieval = agent._retrieval  # noqa: SLF001 — the record the agent just wrote
    num_retrieved = len(retrieval.items) if retrieval is not None else 0

    judge_usage, extraction_usage = Usage(), Usage()
    verdict = judge(
        judge_llm,
        cfg.benchmark,
        query=query,
        trajectory=trajectory,
        usage=judge_usage,
        effort=rb.judge_reasoning_effort,
    )
    judged = verdict.outcome or "unreadable"
    if verdict.outcome is None:
        # Without a label there is no prompt to choose; bank nothing rather than guess,
        # and say so.
        console.info(f"  [judge unreadable] {task_id}: no items extracted")
        items, raw = [], ""
    else:
        items, raw = extract_items(
            extraction_llm,
            cfg.benchmark,
            query=query,
            trajectory=trajectory,
            success=verdict.success,
            max_items=rb.max_items,
            usage=extraction_usage,
            effort=rb.extraction_reasoning_effort,
        )

    memory = "\n\n".join(item.block for item in items)
    return {
        "task_id": task_id,
        "method": "reasoningbank",
        "task": query,
        # What the bank was built from …
        "judged_outcome": judged,
        "judge_thoughts": verdict.thoughts,
        # … and what the environment actually said, for reference only.
        "env_outcomes": [env_outcome],
        "num_retrieved_items": num_retrieved,
        "attempts": [
            {
                "attempt": 1,
                "outcome": env_outcome,
                "consecutive_successes": 1 if env_outcome == "success" else 0,
                "memory": memory or None,
                "memory_diff": None,
                "skipped_generation": not items,
            }
        ],
        "final_outcome": env_outcome,
        "final_memory": memory or None,
        # ReasoningBank's rollouts see the bank, so this is only true when the bank
        # happened to be empty (or to retrieve nothing) for this task.
        "solved_without_memory": env_outcome == "success" and num_retrieved == 0,
        "items": [
            {"title": i.title, "description": i.description, "content": i.content}
            for i in items
        ],
        "extraction_raw": raw,
        "cost_usd": float(result.get("total_cost_usd", 0.0) or 0.0),
        "tokens": {
            key: int((result.get("total_tokens") or {}).get(key, 0) or 0)
            for key in ("prompt", "completion", "cached")
        },
        "judge_usage": judge_usage.to_dict(),
        "extraction_usage": extraction_usage.to_dict(),
        "retrieval_usage": (
            retrieval.usage.to_dict() if retrieval is not None else Usage().to_dict()
        ),
    }


def _task_worker(
    task_id: str,
    config_path: str,
    task_index: int,
    n_tasks: int,
    experiment_name: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Top-level worker for ProcessPoolExecutor (must be importable and picklable)."""
    cfg = common.worker_config(config_path, SETUP, experiment_name)
    # The snapshot the parent wrote (bank included).
    rb = ReasoningBankConfig.load_for(cfg)
    accounting = CostAccounting.for_worker(cfg)
    extraction_llm = accounting.client(
        rb.extraction_model or cfg.agent.model,
        "extraction",
        temperature=rb.extraction_temperature,  # 1.0 in the paper
        reasoning_effort=rb.extraction_reasoning_effort,
    )
    judge_llm = accounting.client(
        rb.judge_model or cfg.agent.model,
        "judge",
        temperature=0.0,  # "for determinism" (Appendix A.2)
        reasoning_effort=rb.judge_reasoning_effort,
    )

    console.info(f"[{task_index + 1}/{n_tasks}] Task: {task_id} (start)", flush=True)
    summary = accumulate_task(task_id, cfg, rb, extraction_llm, judge_llm)
    console.info(
        f"[{task_index + 1}/{n_tasks}] Task: {task_id} → env {summary['final_outcome']} / "
        f"judged {summary['judged_outcome']} | +{len(summary['items'])} item(s) "
        f"| retrieved {summary['num_retrieved_items']}",
        flush=True,
    )

    # Checkpoint immediately: a late crash must not throw away a finished task's items.
    common.write_json(common.summary_path(cfg, task_id), summary)
    return task_index, summary


def experience_of(summary: dict[str, Any]) -> Experience:
    """The bank entry one finished task contributes."""
    return Experience(
        task_id=summary["task_id"],
        query=summary.get("task", ""),
        outcome=summary.get("judged_outcome", ""),
        items=[Item(**item) for item in summary.get("items", [])],
    )


def _write_bank(cfg: ExperimentConfig, summaries: list[dict[str, Any]]) -> Path:
    """Consolidate every finished task into the bank on disk (plain addition, §3.2).

    overwrite=True: the bank is rebuilt from ALL checkpoints every time it is written, so
    each version is a superset of the last. Letting daedalus's default suffix it
    (pool_1.json) would leave a stale, shorter pool.json behind — and that is both the path
    the next wave's workers read and the path inference configs point at.
    """
    return to_pool([experience_of(s) for s in summaries]).save(
        pool_output_path(cfg), overwrite=True
    )


def main() -> None:
    args, cfg = common.start(
        SETUP,
        "ReasoningBank accumulation: the retrieve → extract → consolidate loop",
        parallel_help="Tasks per wave (see the README: 1 = the paper's single stream)",
    )
    rb = ReasoningBankConfig.load(args.config)
    # The bank this run retrieves from IS the bank it is building. Pinning it here (rather
    # than in the YAML) also means the snapshot the workers read can never point at some
    # other run's pool.
    rb.bank_path = str(pool_output_path(cfg))
    rb.save(cfg)  # reasoningbank.yaml — how the workers get these settings
    # This method's own models (judge, extraction, embedder) sit in reasoningbank.yaml,
    # not in the daedalus config, so the preflight is told about them explicitly.
    accounting = CostAccounting.start(
        cfg,
        extra_models=[
            ModelRequirement(
                rb.judge_model or cfg.agent.model, "judge", "reasoningbank.judge_model"
            ),
            ModelRequirement(
                rb.extraction_model or cfg.agent.model,
                "extraction",
                "reasoningbank.extraction_model",
            ),
            ModelRequirement(rb.embedder, "embedding", "reasoningbank.embedder"),
        ],
    )

    task_ids = common.task_ids(cfg, args.task_id)
    wave_size = max(1, cfg.run.parallel or 1)
    console.header(
        "reasoningbank · accumulation",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "solver": cfg.agent.model,
            "judge": rb.judge_model or cfg.agent.model,
            "extraction": rb.extraction_model or cfg.agent.model,
            "retrieval": f"{rb.embedder} (k={rb.k})",
            "tasks": len(task_ids),
            "tasks per wave": wave_size,
            "output": pool_output_path(cfg).parent,
        },
    )

    # Resume: reuse the per-task checkpoints an interrupted run left behind. Their items
    # are in the bank before the first new task runs, so the stream picks up where it
    # stopped (with the caveat in the README: the ORDER in which memory became available
    # is the resumed order, not the original one).
    n_tasks = len(task_ids)
    ordered, pending = common.restore(task_ids, lambda t: common.summary_path(cfg, t))

    # The stream, in waves: everything learned before a wave is available to every task in
    # it, and nothing learned inside it is. wave_size=1 is the paper's exact loop.
    waves = [pending[i : i + wave_size] for i in range(0, len(pending), wave_size)]
    for number, wave in enumerate(waves, 1):
        done = [s for s in ordered if s is not None]
        _write_bank(cfg, done)
        console.rule(
            f"wave {number}/{len(waves)} · {len(wave)} task(s) · bank: "
            f"{sum(len(s.get('items') or []) for s in done)} item(s) from {len(done)} experience(s)"
        )
        # A task that dies is dropped from the wave (no checkpoint, so a re-run picks it
        # up again) without losing the rest of the wave's items.
        for index, summary in common.run_tasks(
            _task_worker,
            wave,
            lambda i, tid: (tid, args.config, i, n_tasks, cfg.name),
            len(wave),
        ):
            ordered[index] = summary

    summaries = [s for s in ordered if s is not None]
    bank_path = _write_bank(cfg, summaries)
    summary_path = write_run_summary(cfg, accounting, {"num_tasks": len(summaries)})
    console.info(f"Run summary (whole-run spend, by role) → {summary_path}")
    log_path = bank_path.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    judged = Counter(s["judged_outcome"] for s in summaries)
    env = Counter(s["final_outcome"] for s in summaries)
    agreed = sum(1 for s in summaries if s["judged_outcome"] == s["final_outcome"])
    labelled = sum(
        1 for s in summaries if s["judged_outcome"] in ("success", "failure")
    )
    items = sum(len(s.get("items") or []) for s in summaries)
    solver_cost = sum(float(s.get("cost_usd", 0.0) or 0.0) for s in summaries)
    aux_keys = ("judge_usage", "extraction_usage", "retrieval_usage")
    aux_cost = sum(
        (s.get(k) or {}).get("cost_usd", 0.0) for s in summaries for k in aux_keys
    )
    aux_calls = sum(
        (s.get(k) or {}).get("calls", 0) for s in summaries for k in aux_keys
    )

    console.rule("reasoningbank accumulation complete")
    console.info(
        f"Tasks: {len(summaries)} | environment: {env['success']} solved / {env['failure']} failed"
    )
    if labelled:
        console.info(
            f"Judge: {judged['success']} success / {judged['failure']} failure"
            + (
                f" / {judged['unreadable']} unreadable"
                if judged.get("unreadable")
                else ""
            )
            + f" | agrees with the environment on {agreed}/{labelled}"
            f" ({100 * agreed / labelled:.0f}%)"
        )
    console.info(f"Bank: {items} memory item(s) from {len(summaries)} experience(s)")
    console.info(
        f"Cost: solver ${solver_cost:.2f} + judge/extraction/embedding ${aux_cost:.2f} "
        f"= ${solver_cost + aux_cost:.2f} ({aux_calls} auxiliary call(s))"
    )
    console.info(f"Bank: {bank_path}")
    console.info(f"Log:  {log_path}")


if __name__ == "__main__":
    main()
