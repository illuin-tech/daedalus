"""The pool must end a launch that freezes, and must not leave workers behind."""

from __future__ import annotations

import signal
from concurrent.futures import ProcessPoolExecutor

import daedalus.core.agents.task_pool as task_pool


def test_stall_timeout_is_longer_than_the_slowest_task():
    # 30 turns on a weak open model is about 12 minutes; the guard must sit above that or a
    # healthy run gets cut off mid-cell.
    assert task_pool._STALL_TIMEOUT_S >= 900


def test_sigterm_reaper_restores_default_handling(monkeypatch):
    executor = ProcessPoolExecutor(max_workers=1)
    killed: list[int] = []
    monkeypatch.setattr(task_pool, "_shutdown_pool", lambda e, grace=15.0: killed.append(1))
    monkeypatch.setattr(task_pool.os, "kill", lambda pid, sig: killed.append(sig))
    previous = signal.getsignal(signal.SIGTERM)
    try:
        task_pool._install_sigterm_reaper(executor)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        # The pool is torn down, then the signal is re-raised with the default handler.
        assert killed == [1, signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, previous)
        executor.shutdown(wait=False)
