"""ACE offline adaptation — grow a playbook over a training split.

    uv run python -m references.ace.accumulation --config references/ace/configs/appworld_accumulation.yaml

The paper's offline, no-ground-truth loop (`ace-appworld`, `ACE_offline_no_GT_adaptation`):
for each training task, (i) the GENERATOR solves it with the playbook as it stands, (ii) the
REFLECTOR reads the finished trajectory and writes a diagnosis, and (iii) the CURATOR turns
that diagnosis into ADD operations against the playbook. One rollout per task, no retries,
and the curator fires on EVERY task — there is no success signal anywhere in this loop. The
environment's verdict is recorded beside it, for reference only.

Two things follow from the playbook being a single mutating artifact:

  * the tasks run in WAVES, like ReasoningBank's closed loop next door. `run.parallel` is
    the wave size; everything learned before a wave is available to every task in it, and
    nothing learned inside it is. **`parallel: 1` is the paper's sequential loop** and is
    what the shipped configs use.
  * the workers PROPOSE operations and the parent APPLIES them, so `playbook.txt` has a
    single writer. With a wave wider than one task, the parent first merges the wave's
    proposals through ACE's own ComBEE reducer (`curator_parallel=True` upstream).

Artifacts under `outputs/baselines/ace/memory/<name>/`: `playbook.txt` (the native
artifact an inference run injects), `pool.json` (one item per bullet, so `serve`, the plots
and the pass^k tooling read an ACE run unchanged), `playbook_snapshots/`, `task_summaries/`,
`curator_operations.jsonl` and `run_summary.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig, experiment_dir, pool_output_path
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.registry import get_benchmark

from references.ace.agents import build_ace_agent
from references.ace.config import SETUP, ACEConfig
from references.ace.curation import curate, reduce_operations, reflect, trajectory_renderer
from references.ace.playbook import (
    apply_curator_operations,
    bullets,
    empty_playbook,
    get_next_global_id,
    get_playbook_stats,
    to_pool,
)
from references.common import accumulation as common
from references.common.usage import Usage, write_run_summary


def playbook_path(cfg: ExperimentConfig) -> Path:
    """The run's own playbook: `<experiment_dir>/playbook.txt`."""
    return experiment_dir(cfg) / "playbook.txt"


def accumulate_task(
    task_id: str,
    cfg: ExperimentConfig,
    ace: ACEConfig,
    reflector_llm: LLMClient,
    curator_llm: LLMClient,
    step: int,
    total: int,
) -> dict[str, Any]:
    """One adaptation step: solve with the current playbook, reflect, then propose ADDs.

    The summary carries DAEDALUS's accumulation field names (`attempts`, `final_outcome`,
    `final_memory`, `solved_without_memory`) so the run browser reads these runs too, plus
    what is specific to ACE: the reflection and the operations it proposed.
    """
    benchmark = get_benchmark(cfg)
    # The agent reads `playbook.txt` as the parent last wrote it and injects it verbatim.
    agent = build_ace_agent(cfg)
    playbook = agent.solver_playbook

    # heuristics=[] keeps the runner on its accumulation path (re-run the task, save the
    # attempt trace); the playbook rides in through the agent's own `playbook` slot.
    result = agent.solve_task(task_id, heuristics=[], trace_suffix="attempt1")
    instruction = result.get("task_instruction") or task_id
    trajectory = trajectory_renderer(benchmark)(result)
    env_outcome = "success" if result.get("success") else "failure"

    reflector_usage, curator_usage = Usage(), Usage()
    reflection = reflect(
        reflector_llm,
        cfg.benchmark,
        trajectory=trajectory,
        playbook=playbook,
        usage=reflector_usage,
        effort=ace.reflector_reasoning_effort,
    )
    operations, raw, error = curate(
        curator_llm,
        cfg.benchmark,
        playbook=playbook,
        reflection=reflection,
        trajectory=trajectory,
        instruction=instruction,
        step=step,
        total=total,
        usage=curator_usage,
        effort=ace.curator_reasoning_effort,
    )
    if error:
        console.info(f"  [curator] {task_id}: {error} — playbook left unchanged")

    memory = "\n".join(f"- {op.get('content', '')}" for op in operations)
    return {
        "task_id": task_id,
        "method": "ace",
        "task": instruction,
        # ACE's loop never reads this; it is recorded for reference only.
        "env_outcome": env_outcome,
        "playbook_bullets_before": get_playbook_stats(playbook)["total_bullets"],
        "attempts": [
            {
                "attempt": 1,
                "outcome": env_outcome,
                "consecutive_successes": 1 if env_outcome == "success" else 0,
                "memory": memory or None,
                "memory_diff": None,
                "skipped_generation": not operations,
            }
        ],
        "final_outcome": env_outcome,
        "final_memory": memory or None,
        # ACE's rollouts always see the playbook, so this is only true while it is empty.
        "solved_without_memory": env_outcome == "success" and not bullets(playbook),
        "reflection": reflection,
        "operations": operations,
        "curator_error": error,
        "curator_raw": raw,
        "cost_usd": float(result.get("total_cost_usd", 0.0) or 0.0),
        "tokens": {
            key: int((result.get("total_tokens") or {}).get(key, 0) or 0)
            for key in ("prompt", "completion", "cached")
        },
        "reflector_usage": reflector_usage.to_dict(),
        "curator_usage": curator_usage.to_dict(),
    }


def _task_worker(
    task_id: str,
    config_path: str,
    task_index: int,
    n_tasks: int,
    experiment_name: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Top-level worker for ProcessPoolExecutor (must be importable and picklable)."""
    # memory.enabled is off there: ACE injects its playbook, not daedalus's pool.
    cfg = common.worker_config(config_path, SETUP, experiment_name)

    ace = ACEConfig.load_for(cfg)  # the snapshot the parent wrote (playbook path included)
    accounting = CostAccounting.for_worker(cfg)
    reflector_llm = accounting.client(
        ace.reflector_model,
        "reflection",
        temperature=0.0,
        reasoning_effort=ace.reflector_reasoning_effort,
    )
    curator_llm = accounting.client(
        ace.curator_model,
        "extraction",
        temperature=0.0,
        reasoning_effort=ace.curator_reasoning_effort,
    )

    console.info(f"[{task_index + 1}/{n_tasks}] Task: {task_id} (start)", flush=True)
    summary = accumulate_task(
        task_id, cfg, ace, reflector_llm, curator_llm, task_index + 1, n_tasks
    )
    console.info(
        f"[{task_index + 1}/{n_tasks}] Task: {task_id} → env {summary['final_outcome']} "
        f"| {len(summary['operations'])} operation(s) proposed",
        flush=True,
    )

    # Checkpoint immediately: a late crash must not throw away a finished task's work.
    common.write_json(common.summary_path(cfg, task_id), summary)
    return task_index, summary


def _applied(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """What actually went into the playbook for this task.

    `applied_operations` is written by the parent after a wave (it differs from
    `operations` only when the reducer merged a wide wave's proposals). A checkpoint from a
    run that died between the worker's write and the parent's update falls back to the
    curator's own proposals.
    """
    return summary.get("applied_operations", summary.get("operations") or [])


def _write_playbook(
    cfg: ExperimentConfig, summaries: list[dict[str, Any]]
) -> str:
    """Rebuild the playbook from the seed by replaying every finished task, in order.

    Rebuilt from ALL checkpoints each time rather than appended to, for the same reason
    ReasoningBank rebuilds its bank: a resumed run's artifact is then a superset of the
    last, and there is never a stale shorter file at the path inference configs point at.
    Ids are assigned by replay, so the same checkpoints always produce the same playbook.
    """
    # Always from the empty 8-section skeleton (the paper's setting; ACE ships an AppWorld
    # seed of 8 bullets that restate the benchmark's own prompt instructions).
    text = empty_playbook()
    next_id = get_next_global_id(text)
    for summary in summaries:
        text, next_id = apply_curator_operations(text, _applied(summary), next_id)
    playbook_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    playbook_path(cfg).write_text(text, encoding="utf-8")
    to_pool(text, source=cfg.name).save(pool_output_path(cfg), overwrite=True)
    return text


def _apply_wave(
    cfg: ExperimentConfig,
    ace: ACEConfig,
    playbook: str,
    wave_summaries: list[dict[str, Any]],
    curator_llm: LLMClient | None,
    step: int,
    total: int,
    usage: Usage,
) -> None:
    """Decide what this wave contributes, writing `applied_operations` onto each summary.

    One task per wave (the shipped configs) is upstream's sequential loop and needs no
    reducer. A wider wave goes through ACE's ComBEE aggregation: every task proposed
    against the same frozen playbook, so the proposals overlap and are merged once.
    """
    if len(wave_summaries) == 1:
        wave_summaries[0]["applied_operations"] = wave_summaries[0]["operations"]
        return

    proposals = [op for s in wave_summaries for op in (s.get("operations") or [])]
    if not proposals or curator_llm is None:
        for summary in wave_summaries:
            summary["applied_operations"] = []
        return
    merged, error = reduce_operations(
        curator_llm,
        playbook=playbook,
        proposals=proposals,
        step=step,
        total=total,
        usage=usage,
        effort=ace.curator_reasoning_effort,
    )
    if error:
        console.info(f"  [reducer] {error} — falling back to the raw proposals")
        merged = proposals
    # The merged set belongs to the wave, not to one task; attribute it to the last one so
    # the replay order stays a pure function of the task order.
    for summary in wave_summaries[:-1]:
        summary["applied_operations"] = []
    wave_summaries[-1]["applied_operations"] = merged


def main() -> None:
    args, cfg = common.start(
        SETUP,
        "ACE offline adaptation: generator → reflector → curator over a train split",
        parallel_help="Tasks per wave (1 = the paper's sequential loop; >1 uses ACE's ComBEE reducer)",
    )
    ace = ACEConfig.load(args.config)
    # The playbook the workers inject IS the playbook this run is building. Pinning it here
    # rather than in the YAML also means the snapshot a worker reads can never point at
    # another run's artifact.
    ace.playbook_path = str(playbook_path(cfg))
    ace.save(cfg)  # ace.yaml — how the workers get these settings
    accounting = CostAccounting.start(
        cfg,
        extra_models=[
            ModelRequirement(ace.reflector_model, "reflection", "ace.reflector_model"),
            ModelRequirement(ace.curator_model, "extraction", "ace.curator_model"),
        ],
    )

    task_ids = common.task_ids(cfg, args.task_id)
    wave_size = max(1, cfg.run.parallel or 1)
    console.header(
        "ace · accumulation",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "generator": cfg.agent.model,
            "reflector": f"{ace.reflector_model} ({ace.reflector_reasoning_effort})",
            "curator": f"{ace.curator_model} ({ace.curator_reasoning_effort})",
            "seed playbook": "(empty skeleton)",
            "tasks": len(task_ids),
            "tasks per wave": wave_size,
            "output": experiment_dir(cfg),
        },
    )

    # Resume: reuse the per-task checkpoints an interrupted run left behind. Their
    # operations are replayed into the playbook before the first new task runs.
    n_tasks = len(task_ids)
    ordered, pending = common.restore(task_ids, lambda t: common.summary_path(cfg, t))

    reducer_llm = (
        accounting.client(
            ace.curator_model,
            "extraction",
            component="ace_curator_reducer",
            temperature=0.0,
            reasoning_effort=ace.curator_reasoning_effort,
        )
        if wave_size > 1
        else None
    )
    reducer_usage = Usage()

    snapshots = experiment_dir(cfg) / "playbook_snapshots"
    waves = [pending[i : i + wave_size] for i in range(0, len(pending), wave_size)]
    for number, wave in enumerate(waves, 1):
        done = [s for s in ordered if s is not None]
        playbook = _write_playbook(cfg, done)
        stats = get_playbook_stats(playbook)
        console.rule(
            f"wave {number}/{len(waves)} · {len(wave)} task(s) · playbook: "
            f"{stats['total_bullets']} bullet(s), {len(playbook)} chars"
        )
        # A task that dies is dropped from the wave (no checkpoint, so a re-run picks it
        # up again). Replay order follows task order, not completion order.
        finished = []
        for index, summary in common.run_tasks(
            _task_worker,
            wave,
            lambda i, tid: (tid, args.config, i, n_tasks, cfg.name),
            len(wave),
        ):
            ordered[index] = summary
            finished.append(summary)
        if not finished:
            continue
        _apply_wave(
            cfg, ace, playbook, finished, reducer_llm, wave[0][0] + 1, n_tasks, reducer_usage
        )
        for summary in finished:
            common.write_json(common.summary_path(cfg, summary["task_id"]), summary)
        if ace.snapshot_every and number % max(1, ace.snapshot_every // wave_size) == 0:
            snapshots.mkdir(parents=True, exist_ok=True)
            text = _write_playbook(cfg, [s for s in ordered if s is not None])
            (snapshots / f"after_{len([s for s in ordered if s])}_tasks.txt").write_text(
                text, encoding="utf-8"
            )

    summaries = [s for s in ordered if s is not None]
    playbook = _write_playbook(cfg, summaries)
    stats = get_playbook_stats(playbook)

    (experiment_dir(cfg) / "curator_operations.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"task_id": s["task_id"], "operations": _applied(s), "error": s.get("curator_error")},
                ensure_ascii=False,
            )
            for s in summaries
        ),
        encoding="utf-8",
    )
    pool_output_path(cfg).with_suffix(".log.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_path = write_run_summary(
        cfg,
        accounting,
        {
            "num_tasks": len(summaries),
            "num_bullets": stats["total_bullets"],
            "playbook_chars": len(playbook),
        },
    )

    env = Counter(s["final_outcome"] for s in summaries)
    failed_curations = sum(1 for s in summaries if s.get("curator_error"))
    console.rule("done")
    console.info(f"Tasks: {len(summaries)} — env {dict(env)} (recorded, never read by ACE)")
    console.info(
        f"Playbook: {stats['total_bullets']} bullet(s), {len(playbook)} chars → {playbook_path(cfg)}"
    )
    for section, data in stats["by_section"].items():
        console.info(f"  {data['count']:>4}  {section}")
    if failed_curations:
        console.info(f"Curator replies that failed to parse: {failed_curations}")
    console.info(f"Pool (one item per bullet) → {pool_output_path(cfg)}")
    console.info(f"Run summary (whole-run spend, by role) → {summary_path}")


if __name__ == "__main__":
    main()
