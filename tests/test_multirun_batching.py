"""All pending runs share ONE pool, so the drain tail is paid once, not per run.

Measured on a 168-task AppWorld eval at 24 workers: 30 min per run, of which the last 24
tasks take 19 min (62%) while workers idle with nothing left to pick up. Five sequential
pools spent 80 of 143 minutes that way.
"""

from unittest.mock import MagicMock, patch

from daedalus.core.agents.task_pool import solve_tasks_parallel


def _submitted(agent, task_ids, run_ids=None):
    """The (run_idx, task_id) pairs handed to the executor."""
    calls = []

    class _Exec:
        def __init__(self, *a, **k):
            pass

        def submit(self, _fn, _cfg, task_id, _cls, run_idx):
            calls.append((run_idx, task_id))
            f = MagicMock()
            f.result.return_value = {"success": True}
            return f

        def shutdown(self, **k):
            pass

    with patch("daedalus.core.agents.task_pool.ProcessPoolExecutor", _Exec), patch(
        "daedalus.core.agents.task_pool.as_completed", lambda d, timeout=None: list(d)
    ):
        solve_tasks_parallel(agent, task_ids, 4, run_ids=run_ids)
    return calls


def _agent(run_idx=0):
    a = MagicMock()
    a.run_idx = run_idx
    a.cfg.to_dict.return_value = {}
    a.cfg.logging.verbose = False
    return a


def test_without_run_ids_every_task_uses_the_agents_own_run() -> None:
    assert _submitted(_agent(3), ["a", "b"]) == [(3, "a"), (3, "b")]


def test_run_ids_pair_elementwise_with_tasks() -> None:
    # The whole point: three runs of two tasks enter ONE pool, so a slow task in run 0
    # overlaps with run 2's work instead of stalling a pool of its own.
    pairs = _submitted(_agent(0), ["a", "b"] * 3, run_ids=[0, 0, 1, 1, 2, 2])
    assert pairs == [(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b")]


def test_mismatched_lengths_are_refused() -> None:
    # Silent truncation here would drop tasks from a run and score the run as complete.
    import pytest

    with pytest.raises(ValueError):
        _submitted(_agent(0), ["a", "b", "c"], run_ids=[0, 1])
