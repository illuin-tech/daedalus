"""Append-only, per-worker ledger of LLM usage events.

Layout, one directory per process launch:

    <experiment_dir>/usage/<launch_id>/worker_<worker_id>.jsonl

Why this shape:

  * ONE WRITER PER FILE. `worker_id` defaults to the OS pid, and benchmark parallelism is
    a spawn-based process pool, so two workers never append to the same file and no
    cross-process lock is needed on the hot path of every LLM call.
  * WRITTEN BEFORE THE CALL RETURNS. A worker killed after a completed response still
    leaves that spend on disk — which is correct, because the provider billed it whether
    or not the task trace, session record or launch summary was ever written.
  * IMMUTABLE. A resume writes a NEW launch directory; earlier launches are never
    rewritten, so whole-run spend is the sum over launches and a resumed call is counted
    in addition to the abandoned one it replaces.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from daedalus.core.logging.usage import STATUS_COMPLETED, LLMUsageEvent, TokenUsage

USAGE_DIR_NAME = "usage"


def new_launch_id() -> str:
    """A launch id that sorts chronologically and is unique per launch.

    The random suffix matters: two launches of the same experiment can start inside the
    same second in the same process (a script that resumes in a loop), and sharing a
    launch directory would make one launch's ledger append to the other's file.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:6]}"


def usage_root(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / USAGE_DIR_NAME


@dataclass(frozen=True)
class TallySnapshot:
    """A point-in-time copy of a ledger's per-role running totals."""

    calls: Mapping[str, int]
    cost_usd: Mapping[str, float]
    tokens: Mapping[str, TokenUsage]
    calls_missing_estimate: int


class UsageTally:
    """Per-role running totals of what THIS process recorded, for unit-of-work summaries.

    The ledger files are the whole-run truth, but a session/task summary wants "what did
    this piece of work cost", and re-reading every launch's ledger per task would be
    quadratic. Snapshot before, `since()` after: that is the same delta trick the old
    `_client_cost(llm)` used, except it is per ROLE and per CALL, so a shared client no
    longer smears three roles into one number.
    """

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.cost_usd: dict[str, float] = defaultdict(float)
        self.tokens: dict[str, TokenUsage] = defaultdict(TokenUsage)
        self.calls_missing_estimate = 0

    def add(self, event: LLMUsageEvent) -> None:
        if event.status != STATUS_COMPLETED:
            return
        role = event.role
        self.calls[role] += 1
        if event.estimated_cost_usd is None:
            self.calls_missing_estimate += 1
        else:
            self.cost_usd[role] += event.estimated_cost_usd
        if event.tokens is not None:
            self.tokens[role] = self.tokens[role] + event.tokens

    def snapshot(self) -> TallySnapshot:
        return TallySnapshot(
            calls=dict(self.calls),
            cost_usd=dict(self.cost_usd),
            tokens=dict(self.tokens),
            calls_missing_estimate=self.calls_missing_estimate,
        )

    def since(self, snapshot: TallySnapshot | None) -> dict[str, Any]:
        """The canonical usage fields for the work done since `snapshot`."""
        before = snapshot or TallySnapshot({}, {}, {}, 0)
        roles = set(self.calls) | set(before.calls)
        calls = {
            r: self.calls.get(r, 0) - before.calls.get(r, 0)
            for r in sorted(roles)
            if self.calls.get(r, 0) - before.calls.get(r, 0) > 0
        }
        cost = {
            r: round(self.cost_usd.get(r, 0.0) - before.cost_usd.get(r, 0.0), 6)
            for r in sorted(roles)
            if r in calls
        }
        zero = TokenUsage()
        tokens_by_role: dict[str, dict[str, Any]] = {}
        total = TokenUsage()
        for r in calls:
            now = self.tokens.get(r, zero)
            was = before.tokens.get(r, zero)
            delta = TokenUsage(
                prompt_tokens=now.prompt_tokens - was.prompt_tokens,
                completion_tokens=now.completion_tokens - was.completion_tokens,
                cache_read_tokens=now.cache_read_tokens - was.cache_read_tokens,
                cache_write_tokens=now.cache_write_tokens - was.cache_write_tokens,
                reasoning_tokens=now.reasoning_tokens - was.reasoning_tokens,
                cache_details_available=now.cache_details_available,
            )
            tokens_by_role[r] = delta.to_dict()
            total = total + delta
        missing = self.calls_missing_estimate - before.calls_missing_estimate
        return {
            "estimated_total_cost_usd": (
                round(sum(cost.values()), 6) if not missing else None
            ),
            "priced_calls_estimated_cost_usd": round(sum(cost.values()), 6),
            "cost_usd_by_role": cost,
            "calls_by_role": calls,
            "tokens_total": total.to_dict(),
            "tokens_by_role": tokens_by_role,
            "cost_complete": missing == 0,
            "calls_missing_estimate": missing,
        }


class UsageLedger:
    """The writer side. One instance per (launch, worker)."""

    def __init__(
        self, experiment_dir: Path, launch_id: str, worker_id: str | None = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.launch_id = launch_id
        self.worker_id = worker_id or str(os.getpid())
        self.path = (
            usage_root(self.experiment_dir)
            / launch_id
            / f"worker_{self.worker_id}.jsonl"
        )
        self._fh = None
        # Only needed for the few thread-pool call sites (e.g. scripts/consolidate.py);
        # process workers each own their own file.
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        # In-process per-role totals, for per-task/per-session summaries (see UsageTally).
        self.tally = UsageTally()

    def record(self, event: LLMUsageEvent) -> LLMUsageEvent:
        """Append one event and flush it to the OS before returning."""
        event.launch_id = event.launch_id or self.launch_id
        event.worker_id = event.worker_id or self.worker_id
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            fh = self._open()
            fh.write(line + "\n")
            fh.flush()
            self.tally.add(event)
        return event

    def _open(self):
        # Reopen if this ledger object crossed a fork: the inherited file handle would
        # interleave two processes' writes in one file.
        if self._fh is not None and self._owner_pid == os.getpid():
            return self._fh
        if self._fh is not None:
            self._fh = None
            self._owner_pid = os.getpid()
            self.worker_id = str(os.getpid())
            self.path = (
                usage_root(self.experiment_dir)
                / self.launch_id
                / f"worker_{self.worker_id}.jsonl"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self._fh

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"UsageLedger(launch={self.launch_id!r}, worker={self.worker_id!r})"
