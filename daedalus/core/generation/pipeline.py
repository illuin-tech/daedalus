"""Self-play memory generation (paper Algorithms 1-2).

Once per run, before the workers fan out, the Surveyor explores a throwaway copy of the
environment and fixes the run's coverage goal: a closed set of tags with target shares
(`generation.coverage_tags`; `coverage_tags_path` reuses a saved survey). Every session
then sees the goal and a live per-tag tally of the accepted tasks. See coverage.py.

Per session, in a sandbox world:
  1. the Explorer explores and emits a grounded, feasible task with success conditions;
  2. the Solver loop (accumulate.py) attempts it, graded by the LLM judge (judge.py);
  3. a task that is too easy or too hard is refined by the Explorer, up to
     `max_refinements` times;
  4. an accepted task enters the task bank and its heuristic the heuristic bank; a
     session that needed refinement updates the explorer guidelines (guidelines.py).

`num_explorers` worker processes share the banks through a file lock. A run is resumable
(`--resume` runs `num_sessions` more sessions). `generation.stage` selects the ablations of
Table 3 (rows B and C) instead of the full method.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import json
import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daedalus.core.config import (
    ExperimentConfig,
    config_from_dict,
    experiment_dir,
    generation_paths,
    save_run_config,
)
from daedalus.core.generation import coverage
from daedalus.core.generation.accumulate import accumulate_for_task, single_attempt_for_task
from daedalus.core.generation.backend import GenerationBackend, get_generation_backend
from daedalus.core.generation.guidelines import update_explorer_memory
from daedalus.core.generation.novelty import novelty_score
from daedalus.core.llm.client import FatalLLMError, LLMClient
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.usage_aggregate import UsageAggregator
from daedalus.core.memory.pool import MemoryItem, MemoryPool

# generation.stage values. The two ablations are AppWorld-only.
DAEDALUS = "daedalus"
EXPLORER_WITH_BANK = "direct_exploration_with_bank"  # Table 3, row B
SOLVER_TRACES = "task_generation_solver"  # Table 3, row C
STAGES = (DAEDALUS, EXPLORER_WITH_BANK, SOLVER_TRACES)


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Cross-process exclusive lock guarding shared bank/pool reads and writes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── the shared bank (tasks.json) ─────────────────────────────────────────────

_BANK_KEYS = (
    "tasks", "too_easy", "too_hard", "guidelines",
    "coverage_goal", "coverage_tags", "tag_counts", "next_session",
)


def _read_bank(path: Path) -> dict[str, Any]:
    """Accepted tasks, off-target task texts, explorer guidelines, the coverage goal with
    the per-tag counts of accepted tasks, and the next session index (the claim counter)."""
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "tasks": d.get("tasks", []),
        "too_easy": d.get("too_easy", []),
        "too_hard": d.get("too_hard", []),
        "guidelines": d.get("guidelines", []),
        "coverage_goal": d.get("coverage_goal", ""),
        "coverage_tags": d.get("coverage_tags", []),
        "tag_counts": d.get("tag_counts", {}),
        "next_session": d.get("next_session", 0),
    }


def _write_bank(path: Path, bank: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: bank[k] for k in _BANK_KEYS}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_guidelines(tasks_path: Path, lock_path: Path) -> list[str]:
    """The guidelines banked right now. A refinement can run minutes after its session
    claimed a snapshot, after another worker distilled new guidelines."""
    with _file_lock(lock_path):
        return list(_read_bank(tasks_path)["guidelines"])


def _merge_guidelines(base: list[str], distilled: list[str], current: list[str]) -> list[str]:
    """The distilled list, plus any guideline another worker banked since `base` was read
    (the distillation runs outside the lock, so the write-back must be additive)."""
    return distilled + [g for g in current if g not in base and g not in distilled]


def _bank_coverage(bank: dict[str, Any]) -> coverage.CoverageGoal:
    return coverage.CoverageGoal.from_dict(
        {"report": bank["coverage_goal"], "tags": bank["coverage_tags"]}
    )


def _coverage_prompts(
    bank: dict[str, Any], gen: Any, progress: float, available: list[str] | None = None
) -> tuple[str, str]:
    """The (coverage_goal, coverage_tally) blocks of one session's explorer prompt.

    `progress` is the fraction of the run already claimed: past
    `coverage_escalate_after`, the most under-covered tag becomes a directive. Both
    strings are empty when the run has no coverage goal.
    """
    goal = _bank_coverage(bank)
    if goal.empty:
        return "", ""
    tally = coverage.render_tally(
        goal,
        bank["tag_counts"],
        num_banked=len(bank["tasks"]),
        progress=progress,
        available=available,
        escalate_after=gen.coverage_escalate_after,
        escalate_ratio=gen.coverage_escalate_ratio,
    )
    return goal.report, tally


def _save_session(sessions_dir: Path, label: str, record: dict[str, Any]) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"session_{label}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _next_session_index(sessions_dir: Path) -> int:
    """Resume checkpoint: (max saved session index) + 1, else 0."""
    idxs = [
        int(f.stem.split("_")[1])
        for f in sessions_dir.glob("session_*.json")
        if f.stem.split("_")[1].isdigit()
    ] if sessions_dir.exists() else []
    return max(idxs) + 1 if idxs else 0


def _new_pool_item(text: str, source: str, tags: dict[str, Any] | None = None) -> MemoryItem:
    return MemoryItem(
        memory_id=f"m_{uuid.uuid4().hex[:8]}",
        type="reflection",
        text=text,
        source_task_id=source,
        source_trajectory_success=True,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        tags=tags or {},
    )


def _task_guidelines(cfg: ExperimentConfig) -> str:
    """The benchmark's fixed "what a good task looks like here" block (or "").

    Read from the same prompt search path the explorer renders from, so the guideline
    updater sees exactly the block the explorer works to."""
    from daedalus.core.registry import get_benchmark
    from daedalus.core.resources import task_guidelines

    try:
        return task_guidelines(get_benchmark(cfg).prompt_dirs(cfg), "explorer")
    except OSError:
        return ""


# ── one session ──────────────────────────────────────────────────────────────


def _classify(summary: dict[str, Any]) -> tuple[str, str | None]:
    """Solver-loop summary → (outcome, heuristic).

    banked       = solved after at least one failure: the heuristic turned it around
    too_easy     = solved without ever failing (no heuristic)
    too_hard     = still failing after max_failures failures
    invalid_spec = a failure the backend attributes to a defective spec (τ² triage)
    """
    heuristic = summary.get("final_memory")
    if summary.get("final_outcome") == "success":
        return ("banked", heuristic) if heuristic else ("too_easy", None)
    if summary.get("spec_defect"):
        return "invalid_spec", None
    return "too_hard", None


def _run_one_session(
    cfg: ExperimentConfig,
    backend: GenerationBackend,
    explorer: Any,
    judge_llm: LLMClient,
    extraction_llm: LLMClient,
    ctx: str,
    i: int,
    prior_tasks: list[str],
    guidelines: list[str],
    covered_paths: list[list[str]],
    solver_trace_dir: Path,
    pfx: str,
    coverage_goal: str = "",
    coverage_tally: str = "",
    reload_guidelines=None,
) -> dict[str, Any]:
    """Explore → Solver loop → refine, on a snapshot of the shared state.

    Touches no shared state: returns a bundle the worker merges under the lock,
    {result, heuristic, current (final spec), tokens, trajectory, entered_refinement, record}.
    """
    gen = cfg.generation
    single_attempt = gen.stage == SOLVER_TRACES
    runner = single_attempt_for_task if single_attempt else accumulate_for_task
    record: dict[str, Any] = {
        "session_index": i,
        "sandbox_id": ctx,
        "kind": gen.stage,
        "prior_tasks": prior_tasks,
        "guidelines_before": list(guidelines),
    }

    def stop(result: str, current: Any = None, tokens: list[str] | None = None) -> dict:
        return {"result": result, "heuristic": None, "current": current,
                "tokens": tokens or [], "trajectory": [], "entered_refinement": False,
                "record": record}

    def admissible(spec: dict[str, Any] | None, what: str) -> str | None:
        """Why `spec` cannot be attempted, or None."""
        if spec is None:
            return f"{what} produced no spec"
        ok, reason = backend.validate_spec(ctx, spec)
        if not ok:
            record["invalid_spec_reason"] = reason
            return f"{what} failed validation ({reason})"
        bad = backend.excluded_used(spec, cfg)
        if bad:
            record["excluded_apps_used"] = bad
            return f"{what} uses excluded app(s)/tool(s) {bad}"
        return None

    artifact, transcript = explorer.explore(
        ctx, prior_tasks=prior_tasks, guidelines=guidelines,
        coverage_goal=coverage_goal, coverage_tally=coverage_tally,
    )
    record["explorer"] = {"num_turns": len(transcript), "artifact": artifact,
                          "transcript": transcript}
    why = admissible(artifact, "explorer")
    if why:
        console.info(f"{pfx} [{i}] {why}", flush=True)
        if artifact is None:
            return stop("no_task")
        return stop("excluded_app" if "excluded" in why else "invalid_spec", artifact)

    tokens = backend.spec_path(artifact)
    nov = novelty_score(tokens, covered_paths)
    record.update(predicted_path=tokens, novelty=nov)
    task_text = backend.spec_task_text(artifact)
    console.info(
        f"{pfx} [{i}] task: {task_text[:100]} | path={' → '.join(tokens)} | nov={nov:.2f}",
        flush=True,
    )
    if nov < gen.min_novelty:
        console.info(f"{pfx} [{i}] near-duplicate — skip", flush=True)
        return stop("near_duplicate", artifact, tokens)

    def solve(spec: dict, trace_prefix: str) -> tuple[dict, str, str | None]:
        summary = backend.solve(ctx, spec, cfg, extraction_llm, judge_llm, trace_prefix, runner)
        if single_attempt:
            solved = summary.get("final_outcome") == "success"
            return summary, "solved" if solved else "failed", summary.get("final_memory")
        return (summary, *_classify(summary))

    def trajectory_entry(spec: dict, result: str, summary: dict) -> dict:
        return {"task": backend.spec_task_text(spec), "result": result,
                "trace": summary.get("last_trace_text", ""), "spec": spec,
                "defect": summary.get("spec_defect")}

    summary, result, heuristic = solve(artifact, f"s{i}_" if single_attempt else "")
    console.info(f"{pfx} [{i}] solve → {result}", flush=True)
    trajectory = [trajectory_entry(artifact, result, summary)]
    refine_records: list[dict] = []
    current = artifact
    entered_refinement = result not in ("banked", "solved", "failed")

    r = 0
    while result in ("too_easy", "too_hard", "invalid_spec") and r < gen.max_refinements:
        r += 1
        direction = "harder" if result == "too_easy" else "easier"
        console.info(
            f"{pfx} [{i}] refine {r}/{gen.max_refinements} ({result} → {direction})",
            flush=True,
        )
        revised, rtranscript = explorer.refine(
            ctx,
            prior_tasks=prior_tasks,
            guidelines=reload_guidelines() if reload_guidelines else guidelines,
            exploration=transcript,
            current_task=backend.spec_task_text(current),
            difficulty=result,
            variant_history=[
                {"task": a["task"], "result": a["result"], "defect": a.get("defect")}
                for a in trajectory
            ],
            coverage_goal=coverage_goal,
            coverage_tally=coverage_tally,
        )
        refine_records.append({"refinement": r, "artifact": revised,
                               "num_turns": len(rtranscript), "transcript": rtranscript})
        why = admissible(revised, "refinement")
        if why:
            console.info(f"{pfx} [{i}] {why} — stopping", flush=True)
            break
        summary, result, heuristic = solve(revised, f"r{r}_")
        current = revised
        trajectory.append(trajectory_entry(revised, result, summary))
        console.info(f"{pfx} [{i}] refine {r} → {result}", flush=True)

    record["accumulation_summary"] = summary
    record["solver_traces"] = str(solver_trace_dir / f"{ctx}_*attempt*.json")
    record["refinements"] = refine_records
    record["trajectory"] = [{"task": a["task"], "result": a["result"]} for a in trajectory]
    return {
        "result": result,
        "heuristic": heuristic,
        "current": current,
        "tokens": backend.spec_path(current),
        "trajectory": trajectory,
        "entered_refinement": entered_refinement,
        "record": record,
    }


def _merge_bank(
    tasks_path: Path,
    pool_path: Path,
    lock_path: Path,
    cfg: ExperimentConfig,
    backend: GenerationBackend,
    ctx: str,
    b: dict[str, Any],
    extraction_llm: LLMClient,
) -> dict[str, Any]:
    """Merge one session's bundle into the shared bank and pool, under the lock.

    Returns the (possibly reclassified) result and the bank sizes. The guideline update, an
    LLM call, runs after the lock is released.
    """
    gen = cfg.generation
    single_attempt = gen.stage == SOLVER_TRACES
    with _file_lock(lock_path):
        bank = _read_bank(tasks_path)
        result = b["result"]
        novelty, heuristic_rejected, heuristic_banked = None, "", False
        # Row C keeps every task it attempted, solved or not, and every extracted heuristic.
        accepted = result in ("solved", "failed") if single_attempt else result == "banked"
        if accepted:
            novelty = novelty_score(b["tokens"], [t["path"] for t in bank["tasks"]])
            if novelty < gen.min_novelty:
                result = "near_duplicate"  # another worker banked a near-duplicate meanwhile
                accepted = False
        if accepted:
            # The heuristic is mined from the SOLVER's trace, and the solver sees the whole
            # environment, held-out apps included: screen the heuristic too, not only the spec.
            heuristic = (b.get("heuristic") or "").strip()
            ok, why = backend.heuristic_admissible(heuristic, cfg)
            if not ok:
                heuristic_rejected = why
                console.info(f"   [screen] heuristic dropped — {why}", flush=True)
            elif heuristic:
                tags = {"source": gen.stage, "solver_outcome": result} if single_attempt else {}
                pool = MemoryPool.load_or_empty(pool_path)
                pool.add(_new_pool_item(heuristic, f"gen:{ctx}", tags))
                pool.save(pool_path, overwrite=True)
                heuristic_banked = True
            rec = backend.bank_record(ctx, b["current"], b["tokens"], novelty,
                                      len(b["trajectory"]) - 1)
            if heuristic_rejected:
                rec["heuristic_rejected"] = heuristic_rejected
            # Coverage tags as the explorer declared them, filtered to the closed vocabulary.
            kept, dropped = _bank_coverage(bank).keep_known((b["current"] or {}).get("tags"))
            rec["tags"] = kept
            for t in kept:
                bank["tag_counts"][t] = bank["tag_counts"].get(t, 0) + 1
            if dropped:
                console.info(f"   [coverage] ignored out-of-vocabulary tag(s) {dropped}",
                             flush=True)
            bank["tasks"].append(rec)
            backend.persist_native(bank["tasks"], tasks_path.parent)
        elif result in ("too_easy", "too_hard"):
            bank[result].append(backend.spec_task_text(b["current"]))
        _write_bank(tasks_path, bank)
        merged = {
            "result": result,
            "novelty": novelty,
            "heuristic_banked": heuristic_banked,
            "heuristic_rejected": heuristic_rejected or None,
            "guidelines_after": list(bank["guidelines"]),
            "n_tasks": len(bank["tasks"]),
            "n_too_easy": len(bank["too_easy"]),
            "n_too_hard": len(bank["too_hard"]),
        }
        update_inputs = (
            (list(bank["guidelines"]), list(bank["too_easy"]), list(bank["too_hard"]))
            if gen.explorer_guidelines and b["entered_refinement"] else None
        )

    if update_inputs is not None:
        guidelines_in, too_easy_in, too_hard_in = update_inputs
        distilled = update_explorer_memory(
            extraction_llm, guidelines_in, too_easy_in, too_hard_in,
            attempts=b["trajectory"],
            reasoning_effort=gen.judge_reasoning_effort,
            task_guidelines=_task_guidelines(cfg),
        )
        with _file_lock(lock_path):
            bank = _read_bank(tasks_path)
            if distilled:
                bank["guidelines"] = _merge_guidelines(guidelines_in, distilled,
                                                       bank["guidelines"])
                _write_bank(tasks_path, bank)
            else:
                console.info("   [guidelines] distillation came back empty — keeping the "
                             f"{len(bank['guidelines'])} banked guideline(s)", flush=True)
            merged["guidelines_after"] = list(bank["guidelines"])
    return merged


# ── workers ──────────────────────────────────────────────────────────────────


def _claim_session(tasks_path: Path, lock_path: Path, target_total: int) -> tuple[int, dict] | None:
    """Claim the next session index (atomic under the lock) and snapshot the bank."""
    with _file_lock(lock_path):
        bank = _read_bank(tasks_path)
        i = bank["next_session"]
        if i >= target_total:
            return None
        bank["next_session"] = i + 1
        _write_bank(tasks_path, bank)
        return i, bank


def _explorer_worker(
    worker_id: int,
    cfg_dict: dict[str, Any],
    worker_exp_name: str,
    paths: dict[str, str],
    target_total: int,
    multi: bool,
) -> None:
    """Claim a session, run it, merge it into the shared banks; repeat until the session
    ceiling `target_total` is reached."""
    cfg = config_from_dict(cfg_dict)
    console.silence_third_party(cfg.logging.verbose)
    gen = cfg.generation
    cfg.kind = "generation"
    cfg.memory.enabled = False
    cfg.experiment_name = worker_exp_name
    backend = get_generation_backend(cfg)
    backend.configure_worker(cfg, worker_exp_name)
    tasks_path, pool_path = Path(paths["tasks"]), Path(paths["pool"])
    sessions_dir, lock_path = Path(paths["sessions"]), Path(paths["lock"])
    # Every worker writes its solver traces and its usage ledger into the run folder
    # (its own experiment_name is `<name>_w<k>`).
    solver_trace_dir = tasks_path.parent / "traces"
    cfg.logging.trace_dir = str(solver_trace_dir)
    cfg.cost_accounting.ledger_dir = str(tasks_path.parent)
    accounting = CostAccounting.for_worker(cfg)
    explorer = backend.make_explorer(cfg, accounting)
    pfx = f"[w{worker_id}]" if multi else "   "
    explorer.log_prefix = f"{pfx} " if multi else ""
    contexts = backend.session_contexts(cfg)
    if not contexts:
        raise RuntimeError(f"No environment contexts to generate from for {cfg.benchmark!r}")

    if gen.stage == EXPLORER_WITH_BANK:
        _explorer_with_bank_worker(cfg, backend, explorer, accounting, contexts, paths,
                                   target_total, pfx, worker_exp_name)
        return

    # The paper's DAEDALUS runs sent no reasoning effort to the generation-time extractor
    # (provider default); the row-C ablation sent `extraction_reasoning_effort`. Both are
    # kept as they ran.
    extraction_llm = accounting.client(
        cfg.accumulation.extraction_model, "extraction", temperature=0.0,
        reasoning_effort=None if gen.stage == DAEDALUS
        else cfg.accumulation.extraction_reasoning_effort,
    )
    judge_llm = accounting.client(gen.judge_model, "judge", temperature=0.0)
    # Row C runs without explorer guidelines and without coverage steering.
    guided = gen.explorer_guidelines and gen.stage == DAEDALUS

    while (claim := _claim_session(tasks_path, lock_path, target_total)) is not None:
        i, bank = claim
        ctx = contexts[i % len(contexts)]
        guidelines = list(bank["guidelines"]) if guided else []
        prior_tasks = [t["task"] for t in bank["tasks"]]
        covered = [t["path"] for t in bank["tasks"]]
        # The tally may be slightly stale by the time the session ends — by design.
        cov_goal, cov_tally = ("", "") if gen.stage != DAEDALUS else _coverage_prompts(
            bank, gen, progress=i / max(1, target_total),
            available=backend.tags_for_context(cfg, ctx, _bank_coverage(bank).tags),
        )
        # One failing session is skipped, not allowed to kill the worker.
        try:
            usage_mark = accounting.tally_snapshot()
            session_id = f"{worker_exp_name}:{i:02d}"
            for role, client in (("explorer", explorer.llm), ("judge", judge_llm),
                                 ("extraction", extraction_llm)):
                client.scope = accounting.scope(role, session_id=session_id)
            console.info(
                f"{pfx} [{i}] explore in {ctx} | T={len(prior_tasks)} | "
                + (f"{len(guidelines)} guidelines ..." if guided else "guidelines off ..."),
                flush=True,
            )
            b = _run_one_session(
                cfg, backend, explorer, judge_llm, extraction_llm, ctx, i,
                prior_tasks, guidelines, covered, solver_trace_dir, pfx,
                coverage_goal=cov_goal, coverage_tally=cov_tally,
                reload_guidelines=(lambda: _read_guidelines(tasks_path, lock_path))
                if guided else None,
            )
            m = _merge_bank(tasks_path, pool_path, lock_path, cfg, backend, ctx, b,
                            extraction_llm)
            usage = accounting.usage_since(usage_mark)
            record = b["record"]
            record.update(
                result=m["result"],
                heuristic=b["heuristic"] if m["heuristic_banked"] else None,
                guidelines_after=m["guidelines_after"],
                cost_usd=round(usage.get("priced_calls_estimated_cost_usd", 0.0), 4),
                usage=usage,
            )
            _save_session(sessions_dir, f"{i:02d}", record)
            console.info(
                f"{pfx} [{i}] final={m['result']} | T={m['n_tasks']} "
                f"pool={len(MemoryPool.load_or_empty(pool_path))} "
                f"too_easy={m['n_too_easy']} too_hard={m['n_too_hard']} | "
                f"${record['cost_usd']:.4f}",
                flush=True,
            )
        except FatalLLMError:
            raise
        except Exception as e:  # noqa: BLE001 — one session must not kill the worker
            console.info(f"{pfx} [{i}] session failed, skipping: {type(e).__name__}: {e}",
                         flush=True)
            traceback.print_exc()
            _save_session(sessions_dir, f"{i:02d}", {
                "session_index": i, "sandbox_id": ctx, "kind": "session_error",
                "stage": gen.stage, "error": f"{type(e).__name__}: {e}",
            })


def _explorer_with_bank_worker(
    cfg: ExperimentConfig,
    backend: GenerationBackend,
    explorer: Any,
    accounting: CostAccounting,
    contexts: list[str],
    paths: dict[str, str],
    target_total: int,
    pfx: str,
    worker_exp_name: str,
) -> None:
    """Table 3, row B: explorer-only sessions, each seeing the heuristics banked so far.

    No task, solver, judge or extractor. A session's bullets become ONE pool item, so the
    session is the sampling unit as in the full method.
    """
    tasks_path, pool_path = Path(paths["tasks"]), Path(paths["pool"])
    sessions_dir, lock_path = Path(paths["sessions"]), Path(paths["lock"])
    while True:
        with _file_lock(lock_path):
            bank = _read_bank(tasks_path)
            i = bank["next_session"]
            if i >= target_total:
                return
            bank["next_session"] = i + 1
            _write_bank(tasks_path, bank)
            prior = list(MemoryPool.load_or_empty(pool_path).items)
        ctx = contexts[i % len(contexts)]
        try:
            usage_mark = accounting.tally_snapshot()
            explorer.llm.scope = accounting.scope(
                "explorer", session_id=f"{worker_exp_name}:{i:02d}"
            )
            console.info(f"{pfx} [{i}] direct exploration in {ctx} | prior bank="
                         f"{len(prior)} item(s) ...", flush=True)
            heuristics, transcript = explorer.explore_direct(
                ctx, previous_heuristics=[item.text for item in prior]
            )
            text = "\n".join(f"- {h}" for h in heuristics)
            ok, why = backend.heuristic_admissible(text, cfg) if text else (True, "")
            result = "banked" if text and ok else ("heuristic_rejected" if text else "empty")
            if result == "banked":
                with _file_lock(lock_path):
                    pool = MemoryPool.load_or_empty(pool_path)
                    pool.add(_new_pool_item(
                        text, f"{EXPLORER_WITH_BANK}:{ctx}:session:{i}",
                        {"source": EXPLORER_WITH_BANK, "session_index": i,
                         "sandbox_id": ctx, "num_heuristics": len(heuristics)},
                    ))
                    pool.save(pool_path, overwrite=True)
            usage = accounting.usage_since(usage_mark)
            _save_session(sessions_dir, f"{i:02d}", {
                "session_index": i,
                "sandbox_id": ctx,
                "kind": EXPLORER_WITH_BANK,
                "explorer": {
                    "num_turns": sum(1 for e in transcript if e.get("kind") != "emission"),
                    "heuristics": heuristics,
                    "prior_memory_ids": [item.memory_id for item in prior],
                    "transcript": transcript,
                },
                "result": result,
                "heuristic": text if result == "banked" else None,
                "heuristic_rejected": why or None,
                "cost_usd": round(usage.get("priced_calls_estimated_cost_usd", 0.0), 4),
                "usage": usage,
            })
            console.info(f"{pfx} [{i}] final={result} | {len(heuristics)} heuristic(s) | "
                         f"${usage.get('priced_calls_estimated_cost_usd', 0.0):.4f}",
                         flush=True)
        except FatalLLMError:
            raise
        except Exception as e:  # noqa: BLE001 — one session must not kill the worker
            console.info(f"{pfx} [{i}] session failed, skipping: {type(e).__name__}: {e}",
                         flush=True)
            traceback.print_exc()
            _save_session(sessions_dir, f"{i:02d}", {
                "session_index": i, "sandbox_id": ctx, "kind": "session_error",
                "stage": EXPLORER_WITH_BANK, "error": f"{type(e).__name__}: {e}",
            })


# ── the run ──────────────────────────────────────────────────────────────────


def _resolve_coverage_goal(cfg: ExperimentConfig, sessions_dir: Path) -> coverage.CoverageGoal:
    """The run's coverage goal: loaded from `coverage_tags_path`, or surveyed once.

    A survey runs in a throwaway copy of the environment; its transcript is saved as the
    run's `session_coverage` record. Any failure degrades to an empty goal (no steering).
    """
    gen = cfg.generation
    if gen.coverage_tags_path:
        try:
            goal = coverage.load(gen.coverage_tags_path)
        except (OSError, ValueError) as e:
            console.info(f"Coverage goal: could not read {gen.coverage_tags_path} "
                         f"({type(e).__name__}: {e}) — continuing without tag steering.")
            return coverage.CoverageGoal()
        console.info(f"Coverage goal: {len(goal.tags)} tag(s) loaded from "
                     f"{gen.coverage_tags_path}")
        # A goal surveyed without the current hold-out would reveal the hidden apps.
        held = [a for a in gen.excluded_apps
                if re.search(rf"\b{re.escape(a)}\b", goal.to_text(), re.IGNORECASE)]
        if held:
            console.info(f"Coverage goal: WARNING — {gen.coverage_tags_path} mentions "
                         f"held-out {', '.join(held)}; re-survey with --exclude-apps.")
        return goal

    backend = get_generation_backend(cfg)
    explorer = backend.make_explorer(cfg) if backend.session_contexts(cfg) else None
    if explorer is None:
        console.info("Coverage goal: no surveyable context — continuing without steering.")
        return coverage.CoverageGoal()
    # One survey per world, merged. The backend decides which worlds: one world is the
    # whole environment on AppWorld and τ², not on AutomationBench.
    survey_ctxs = backend.survey_contexts(cfg)
    goals = []
    for n, ctx in enumerate(survey_ctxs):
        console.info(f"Coverage survey {n + 1}/{len(survey_ctxs)}: exploring {ctx} "
                     f"(≤{gen.coverage_tags_max_turns} turns) ...")
        mark = explorer.accounting.tally_snapshot()
        goal, transcript = explorer.survey_coverage(ctx)
        usage = explorer.accounting.usage_since(mark)
        goals.append(goal)
        _save_session(sessions_dir, "coverage" if len(survey_ctxs) == 1 else f"coverage_{n}", {
            "kind": "coverage_survey",
            "sandbox_id": str(ctx),
            "coverage_goal": goal.report,
            "coverage_tags": goal.tags,
            "num_turns": len(transcript),
            "transcript": transcript,
            "cost_usd": round(usage.get("priced_calls_estimated_cost_usd", 0.0), 4),
        })
    return coverage.merge(goals)


def generate(cfg: ExperimentConfig, resume: bool = False) -> dict[str, Any]:
    """Run `num_sessions` sessions with `num_explorers` concurrent worker processes.

    A fresh run refuses to touch an existing run folder. `resume=True` continues it,
    running `num_sessions` more sessions on top of its banks.
    """
    console.configure(cfg.logging.verbose)
    cfg.kind = "generation"
    gen = cfg.generation
    if gen.stage not in STAGES:
        raise ValueError(f"Unknown generation.stage {gen.stage!r}; choose one of {STAGES}")
    if gen.stage != DAEDALUS and cfg.benchmark != "appworld":
        raise NotImplementedError(f"generation.stage={gen.stage!r} is AppWorld-only")
    gpaths = generation_paths(cfg)
    tasks_path, pool_path = gpaths["tasks"], gpaths["pool"]
    sessions_dir, lock_path = gpaths["sessions"], gpaths["lock"]
    cfg.memory.enabled = False
    n_explorers = max(1, gen.num_explorers)
    # Price every model this run will call before the first call, and open this launch's
    # usage ledger (a resume gets a new launch, so spend accumulates).
    CostAccounting.start(cfg)

    if resume and tasks_path.exists():
        # From the last SAVED session, not the claim counter: a run stopped mid-flight
        # leaves claimed-but-unfinished indices, which must be re-run.
        start_next = _next_session_index(sessions_dir)
        bank = _read_bank(tasks_path)
        bank["next_session"] = start_next
        _write_bank(tasks_path, bank)
        console.info(f"Resuming after {start_next} prior session(s).")
    elif experiment_dir(cfg).exists() and not resume:
        raise FileExistsError(
            f"A generation run already exists at {experiment_dir(cfg)}. Pass --resume to "
            f"continue it, or change experiment_name to start fresh."
        )
    else:
        start_next = 0
    save_run_config(cfg)
    target_total = start_next + gen.num_sessions

    b0 = _read_bank(tasks_path)
    console.header("daedalus · generation", {
        "stage": gen.stage,
        "benchmark": cfg.benchmark,
        "sessions": f"{gen.num_sessions} (#{start_next}–{target_total - 1})",
        "explorers": n_explorers,
        "solver": "single attempt" if gen.stage == SOLVER_TRACES
        else "off" if gen.stage == EXPLORER_WITH_BANK
        else f"Ns={cfg.accumulation.num_success_to_continue} Nf={cfg.accumulation.max_failures}",
        "tasks": len(b0["tasks"]),
        "pool": len(MemoryPool.load_or_empty(pool_path)),
        "guidelines": len(b0["guidelines"]) if gen.explorer_guidelines else "off",
        "coverage": f"{len(b0['coverage_tags'])} tags" if b0["coverage_tags"]
        else ("survey" if gen.coverage_tags else "off"),
    })

    # The coverage goal is resolved once, before the workers fan out, and skipped on a
    # resume whose bank already carries one.
    if gen.stage == DAEDALUS and gen.coverage_tags and _bank_coverage(b0).empty:
        goal = _resolve_coverage_goal(cfg, sessions_dir)
        if not goal.empty:
            with _file_lock(lock_path):
                bank = _read_bank(tasks_path)
                bank["coverage_goal"], bank["coverage_tags"] = goal.report, goal.tags
                _write_bank(tasks_path, bank)
            console.info(f"Coverage goal: {len(goal.tags)} tag(s) → "
                         f"{coverage.save(gpaths['coverage'], goal)}")

    multi = n_explorers > 1
    worker_args = [
        (w, cfg.to_dict(), f"{cfg.name}_w{w}" if multi else cfg.name,
         {k: str(gpaths[k]) for k in ("tasks", "pool", "sessions", "lock")},
         target_total, multi)
        for w in range(n_explorers)
    ]
    if not multi:
        _explorer_worker(*worker_args[0])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_explorers) as ex:
            for f in concurrent.futures.as_completed(
                [ex.submit(_explorer_worker, *a) for a in worker_args]
            ):
                # One dead worker does not abort the run: the others keep drawing sessions.
                try:
                    f.result()
                except FatalLLMError:
                    raise
                except Exception as e:  # noqa: BLE001
                    console.info(f"[generate] a worker failed: {type(e).__name__}: {e}")
                    traceback.print_exc()

    final = _read_bank(tasks_path)
    pool = MemoryPool.load_or_empty(pool_path)
    usage = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).summary()
    tag_coverage = coverage.realized(_bank_coverage(final), final["tag_counts"],
                                     len(final["tasks"]))
    console.rule("generation complete")
    console.info(f"T={len(final['tasks'])} tasks → {tasks_path} | {len(pool)} heuristics → "
                 f"{pool_path}")
    console.info("cost by role (all launches): " + "  ".join(
        f"{k} ${v:.4f}" for k, v in usage["cost_usd_by_role"].items()))
    for r in tag_coverage:
        console.info(f"    {r['tag']:24s} {r['count']:3d}/{r['num_banked']}  "
                     f"{r['share'] * 100:5.1f}% vs {r['target_share'] * 100:5.1f}%  {r['flag']}")

    summary = {
        "stage": gen.stage,
        "num_tasks": len(final["tasks"]),
        "num_heuristics": len(pool),
        "tag_coverage": tag_coverage,
        "usage": usage,
        "total_cost_usd": usage["priced_calls_estimated_cost_usd"],
        "cost_usd_by_role": usage["cost_usd_by_role"],
    }
    gpaths["run_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
