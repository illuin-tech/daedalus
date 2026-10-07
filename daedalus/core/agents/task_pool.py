"""Parallel task execution shared by all benchmark solver agents.

Benchmark-agnostic: any agent exposing (cfg, run_idx, solve_task) can fan its
tasks out over a process pool. Moved out of base_agent so tau2 runs don't
import appworld transitively.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from daedalus.core.config import config_from_dict
from daedalus.core.llm.client import FatalLLMError
from daedalus.core.logging import console


def agent_class_path(agent: Any) -> str:
    """Return the fully-qualified class name for parallel spawning."""
    cls = type(agent)
    return f"{cls.__module__}.{cls.__qualname__}"


# No task legitimately takes this long: the slowest measured here is a 30-turn AppWorld task
# on a weak open model, about 12 minutes. A gap larger than this means the pool is frozen.
_STALL_TIMEOUT_S = 1500


def solve_tasks_parallel(
    agent: Any,
    task_ids: list[str],
    num_workers: int,
    run_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run tasks in parallel using ProcessPoolExecutor.

    `run_ids` batches SEVERAL runs through ONE pool: pass it parallel to `task_ids`, so
    entry i is the run index for task i. Without it every task uses `agent.run_idx` and a
    5-run experiment opens five pools in sequence.

    That matters because each pool pays its own tail. Measured on a 168-task AppWorld
    eval at 24 workers: 30 min per run, of which the last 24 tasks take 19 min (62%) as
    workers drain with nothing left to pick up. Five sequential pools spend 80 of 143
    minutes that way. One pool over all 840 (run, task) pairs pays the tail once.

    The worker rebuilds the agent from `cfg_dict` and its own `run_idx`, so a single
    parent-side agent serves every run; only its config and class path are used here.
    """
    import multiprocessing as mp

    results: list[dict[str, Any]] = []
    cfg_dict = agent.cfg.to_dict()
    cls_path = agent_class_path(agent)

    # Use 'spawn' (not Linux's default 'fork'): each worker gets a fresh
    # process/CUDA context, so GPU retrievers (ColBERT) work in parallel —
    # fork after the parent inits CUDA raises "Cannot re-initialize CUDA in
    # forked subprocess". (macOS already defaults to spawn; this is a no-op there.)
    ctx = mp.get_context("spawn")
    verbose = agent.cfg.logging.verbose
    n_ok = 0
    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
    # A SIGTERM to the parent otherwise leaves every worker running, reparented to launchd:
    # 142 of them (13.8 GB) accumulated across one day of killed launches and starved the
    # next run of memory, which is itself what triggers the freeze.
    _install_sigterm_reaper(executor)
    try:
        pairs = (
            list(zip(run_ids, task_ids, strict=True))
            if run_ids is not None
            else [(agent.run_idx, t) for t in task_ids]
        )
        futures = {
            executor.submit(_solve_single_task, cfg_dict, task_id, cls_path, run_idx): task_id
            for run_idx, task_id in pairs
        }
        bar = console.progress_bar(len(task_ids), desc="solving")
        # STALL_TIMEOUT, not a per-task budget: `as_completed` raises when NOTHING has
        # finished for this long, which is the signature of the sem_wait freeze described in
        # _shutdown_pool — every worker at 0% CPU and one future that never resolves. Left
        # alone it hangs the run forever (observed three times on 2026-09-17/18, costing a
        # night). Raising it here ends the launch with the tasks that did finish; the caller's
        # resume re-runs the rest, and the agent skips every task that already has a trace.
        try:
            completed = as_completed(futures, timeout=_STALL_TIMEOUT_S)
            for future in completed:
                task_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    success = bool(result.get("success", False))
                    n_ok += success
                    if verbose:  # per-task detail without corrupting the bar
                        bar.write(f"  [done] {task_id} — {'✓' if success else '✗'}")
                except FatalLLMError:
                    # Not this task's failure: the key or the spend limit is refusing every
                    # call, so recording an item here would build a score out of API errors.
                    bar.close()
                    raise
                except Exception as e:
                    bar.write(f"  [error] {task_id}: {e}")
                    results.append({})
                bar.update(1)
                bar.set_postfix(ok=n_ok)
        except TimeoutError:
            bar.write(
                f"  [stalled] no task finished in {_STALL_TIMEOUT_S // 60} min with "
                f"{len(futures) - len(results)} outstanding — ending this launch so the "
                f"caller can resume; finished tasks keep their traces"
            )
        bar.close()
    finally:
        _shutdown_pool(executor)

    return results


def _install_sigterm_reaper(executor: ProcessPoolExecutor) -> None:
    """Tear the pool down on SIGTERM/SIGINT, then re-raise for the default behaviour."""
    import signal

    def handler(signum, frame):  # noqa: ANN001
        _shutdown_pool(executor)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except ValueError:
            pass  # not the main thread; the finally block still runs on a clean exit


def _shutdown_pool(executor: ProcessPoolExecutor, grace: float = 15.0) -> None:
    """Tear down the pool without ever waiting on it, then reap the workers.

    `with ProcessPoolExecutor(...)` exits via shutdown(wait=True), which deadlocks: the
    parent joins the executor-manager thread while that thread sits in select.poll waiting
    for worker sentinels that never arrive, and the workers sit in sem_wait on the call
    queue waiting for work they are never sent. Observed repeatedly on macOS/spawn once the
    box starts swapping — a multi-run experiment froze at each run boundary with every
    worker at 0% CPU, and each freeze leaked a pool's worth of processes (138 orphans
    holding 3.5 GB of RSS and 30 GB of swap accumulated that way, which made the next
    freeze more likely — the failure feeds itself).

    So: never wait. Grab the worker handles before shutting down (shutdown clears them),
    ask for a no-wait shutdown, then escalate join → terminate → kill so nothing outlives
    this call. Every task has already been collected by the time we get here, so dropping
    the pool early costs no results.
    """
    procs = list(getattr(executor, "_processes", {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001 - teardown must not mask the run's own outcome
        pass
    deadline = time.monotonic() + grace
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc.join(timeout=remaining)
        except Exception:  # noqa: BLE001
            pass
    stragglers = [p for p in procs if p.is_alive()]
    for proc in stragglers:
        proc.terminate()
    if stragglers:
        time.sleep(1.0)
        for proc in stragglers:
            if proc.is_alive():
                proc.kill()
        console.info(
            f"[task_pool] force-reaped {len(stragglers)} worker(s) that ignored shutdown"
        )


def _resolve_agent_class(cls_path: str):
    """Import and return the agent class from its dotted path."""
    import importlib

    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _solve_single_task(
    cfg_dict: dict[str, Any],
    task_id: str,
    agent_cls_path: str,
    run_idx: int | None = None,
) -> dict[str, Any]:
    """Standalone function for parallel execution (must be picklable)."""
    cfg = config_from_dict(cfg_dict)
    console.silence_third_party(cfg.logging.verbose)  # quiet tau2/gym logs in the worker
    agent_cls = _resolve_agent_class(agent_cls_path)
    agent = agent_cls(cfg, run_idx=run_idx)
    return agent.solve_task(task_id)
