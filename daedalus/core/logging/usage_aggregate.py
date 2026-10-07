"""Read side of the usage ledger: whole-run, per-role and per-task cost summaries.

Every experiment kind (inference, accumulation, generation, generated test set,
reference method) builds its cost figures here, from the same events, so
`estimated_total_cost_usd` means the same thing in all of them.

Two rules that shape the code:

  * A REPEATED CALL IS REAL SPEND. Events are never de-duplicated by task, attempt,
    session or prompt: a call made before a crash and made again after a resume was billed
    twice, so it counts twice. Only a duplicate `event_id` is impossible, and that means a
    corrupt ledger, so it stops aggregation.
  * COVERAGE IS NOT A TOTAL. A provider-reported total is published only when every
    completed call carries one; partial coverage is reported as coverage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from daedalus.core.logging.usage import (
    STATUS_COMPLETED,
    STATUS_FAILED_WITHOUT_USAGE,
    USAGE_SCHEMA_VERSION,
    LLMUsageEvent,
    TokenUsage,
)
from daedalus.core.logging.usage_ledger import usage_root

COST_ACCOUNTING_VERSION = 2


class LedgerCorruptionError(RuntimeError):
    """Duplicate event ids or an unreadable schema version — aggregation must stop."""


@dataclass
class _Bucket:
    """Running totals for one grouping key."""

    calls: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    calls_missing_estimate: int = 0
    provider_cost_usd: float = 0.0
    calls_with_provider_cost: int = 0
    failed_without_usage: int = 0

    def add(self, event: LLMUsageEvent) -> None:
        if event.status == STATUS_FAILED_WITHOUT_USAGE:
            self.failed_without_usage += 1
            return
        self.calls += 1
        if event.tokens is not None:
            self.tokens = self.tokens + event.tokens
        if event.estimated_cost_usd is None:
            self.calls_missing_estimate += 1
        else:
            self.estimated_cost_usd += event.estimated_cost_usd
        if event.provider_reported_cost_usd is not None:
            self.provider_cost_usd += event.provider_reported_cost_usd
            self.calls_with_provider_cost += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "calls_missing_estimate": self.calls_missing_estimate,
            "provider_reported_cost_usd": (
                round(self.provider_cost_usd, 6)
                if self.calls_with_provider_cost == self.calls and self.calls
                else None
            ),
            "calls_with_provider_cost": self.calls_with_provider_cost,
            "failed_without_usage": self.failed_without_usage,
            "tokens": self.tokens.to_dict(),
        }


def read_events(path: Path) -> tuple[list[LLMUsageEvent], list[str]]:
    """Events in one ledger file, plus a reason per unreadable line.

    A torn final line is what a killed worker leaves behind: it is reported as an
    incompleteness reason rather than silently dropped or raised on, because everything
    before it is real spend that must still be counted.
    """
    events: list[LLMUsageEvent] = []
    problems: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            problems.append(f"unreadable (torn) ledger line {path.name}:{i}")
            continue
        version = int(data.get("schema_version") or 0)
        if version > USAGE_SCHEMA_VERSION:
            raise LedgerCorruptionError(
                f"{path}:{i} has usage schema version {version}, newer than this code "
                f"understands ({USAGE_SCHEMA_VERSION})"
            )
        events.append(LLMUsageEvent.from_dict(data))
    return events, problems


# Files that mean "this folder holds finished work", as opposed to a config-only stub.
_WORK_MARKERS = ("evaluation.json", "run_summary.json", "usage_migrated.json")


def _holds_completed_work(experiment_dir: Path) -> bool:
    """Whether this experiment folder has artifacts a run leaves behind.

    Used only to tell a pre-ledger run (has traces, cost unknown) from a folder that
    genuinely spent nothing (a fresh launch mid-flight, or a config-only stub).
    """
    if any((experiment_dir / name).exists() for name in _WORK_MARKERS):
        return True
    for sub in ("", "traces", "sessions"):
        base = experiment_dir / sub if sub else experiment_dir
        if base.is_dir() and any(base.glob("*.json")):
            return True
    return any(
        d.is_dir() and any(d.glob("*.json")) for d in experiment_dir.glob("run_*")
    )


class UsageAggregator:
    """Aggregates every ledger event of one experiment."""

    def __init__(self, events: Iterable[LLMUsageEvent], problems: Iterable[str] = ()):
        self.events: list[LLMUsageEvent] = list(events)
        self.problems: list[str] = list(problems)
        seen: dict[str, LLMUsageEvent] = {}
        for event in self.events:
            if event.event_id in seen:
                raise LedgerCorruptionError(
                    f"duplicate usage event id {event.event_id!r}: the ledger has been "
                    f"copied or rewritten. Every completed API response gets a fresh id, "
                    f"so a real repeat of a call is a DIFFERENT event, not this."
                )
            seen[event.event_id] = event

    # ---- construction ----------------------------------------------------------------

    @classmethod
    def from_experiment_dir(cls, experiment_dir: Path) -> "UsageAggregator":
        """Every launch ledger under `<experiment_dir>/usage/`.

        A folder that holds finished WORK but no ledger is a pre-ledger run, not a free
        one. It used to aggregate to `estimated_total_cost_usd: 0` with
        `cost_complete: true`, and `run_cost.resolve_run_cost` prefers that block over the
        correct `usage_migrated.json` beside it — so simply re-scoring an old run replaced
        its cost with a confident $0.00. The absence is recorded as an incompleteness
        reason instead, which makes the total `None` and lets the resolver fall through.
        """
        root = usage_root(Path(experiment_dir))
        events: list[LLMUsageEvent] = []
        problems: list[str] = []
        if root.is_dir():
            for path in sorted(root.rglob("worker_*.jsonl")):
                got, bad = read_events(path)
                events.extend(got)
                problems.extend(bad)
        if not events and _holds_completed_work(Path(experiment_dir)):
            problems.append(
                "no usage ledger for this run: it predates per-call accounting, so its "
                "cost cannot be derived here — see usage_migrated.json beside it, or "
                "rebuild it with `python -m daedalus.scripts.migrate_cost_accounting`"
            )
        return cls(events, problems)

    # ---- selection -------------------------------------------------------------------

    def filter(self, **equals: Any) -> "UsageAggregator":
        """A sub-aggregator over events matching every given field value."""
        keep = [
            e
            for e in self.events
            if all(getattr(e, key, None) == value for key, value in equals.items())
        ]
        return UsageAggregator(keep, self.problems)

    # ---- grouping --------------------------------------------------------------------

    def group_by(self, attr: str) -> dict[str, dict[str, Any]]:
        buckets: dict[str, _Bucket] = defaultdict(_Bucket)
        for event in self.events:
            buckets[str(getattr(event, attr, "") or "")].add(event)
        return {k: v.to_dict() for k, v in sorted(buckets.items())}

    def cost_by(self, attr: str) -> dict[str, float]:
        buckets: dict[str, float] = defaultdict(float)
        for event in self.events:
            if (
                event.status == STATUS_COMPLETED
                and event.estimated_cost_usd is not None
            ):
                buckets[str(getattr(event, attr, "") or "")] += event.estimated_cost_usd
        return {k: round(v, 6) for k, v in sorted(buckets.items())}

    def tokens_by(self, attr: str) -> dict[str, dict[str, Any]]:
        buckets: dict[str, TokenUsage] = defaultdict(TokenUsage)
        for event in self.events:
            if event.tokens is not None:
                key = str(getattr(event, attr, "") or "")
                buckets[key] = buckets[key] + event.tokens
        return {k: v.to_dict() for k, v in sorted(buckets.items())}

    # ---- headline numbers ------------------------------------------------------------

    @property
    def completed(self) -> list[LLMUsageEvent]:
        return [e for e in self.events if e.status == STATUS_COMPLETED]

    def total_tokens(self) -> TokenUsage:
        total = TokenUsage()
        for event in self.events:
            if event.tokens is not None:
                total = total + event.tokens
        return total

    def incompleteness_reasons(self) -> list[str]:
        reasons: list[str] = list(self.problems)
        missing = [e for e in self.completed if e.estimated_cost_usd is None]
        if missing:
            why = sorted({e.estimate_unavailable_reason or "unknown" for e in missing})
            reasons.append(
                f"{len(missing)} completed call(s) could not be priced locally ({'; '.join(why)})"
            )
        billable_failures = [
            e
            for e in self.events
            if e.status == STATUS_FAILED_WITHOUT_USAGE and e.provider_may_have_billed
        ]
        if billable_failures:
            kinds = sorted({e.error_type or "unknown" for e in billable_failures})
            reasons.append(
                f"{len(billable_failures)} failed request(s) may have been billed with no "
                f"usage payload ({', '.join(kinds)}); reconcile against provider billing"
            )
        missing_tokens = [e for e in self.completed if e.tokens is None]
        if missing_tokens:
            reasons.append(
                f"{len(missing_tokens)} completed call(s) reported cost but no token detail"
            )
        return reasons

    def provider_coverage(self) -> dict[str, Any]:
        completed = self.completed
        covered = [e for e in completed if e.provider_reported_cost_usd is not None]
        covered_est = sum(e.estimated_cost_usd or 0.0 for e in covered)
        total_est = sum(e.estimated_cost_usd or 0.0 for e in completed)
        by_source: dict[str, int] = defaultdict(int)
        for e in covered:
            by_source[e.provider_cost_source or "unknown"] += 1
        return {
            "covered_calls": len(covered),
            "total_completed_calls": len(completed),
            "covered_estimated_cost_usd": round(covered_est, 6),
            "coverage_by_calls": (len(covered) / len(completed)) if completed else 0.0,
            "coverage_by_estimated_cost": (covered_est / total_est)
            if total_est
            else 0.0,
            # WHERE each figure came from. "provider_usage_cost" is the provider's own
            # charge for that request (OpenRouter returns one); "litellm_response_cost"
            # and "tau2_completion_cost" are litellm's computation from ITS price table —
            # a second opinion on our estimate, not an invoice.
            "covered_calls_by_source": dict(sorted(by_source.items())),
        }

    def provider_native_coverage(self) -> dict[str, Any]:
        """Coverage by charges returned by the provider/router itself.

        LiteLLM's ``response_cost`` is intentionally excluded: it is another price-table
        estimate, not a charge reported by OpenAI/Anthropic.  OpenRouter's ``usage.cost``
        is currently the native source emitted by :mod:`daedalus.core.llm.client`.
        Keeping this separate prevents a fully populated LiteLLM estimate from being
        presented in a paper as provider-observed spend.
        """
        completed = self.completed
        native = [
            e for e in completed
            if e.provider_cost_source == "provider_usage_cost"
            and e.provider_reported_cost_usd is not None
        ]
        native_total = sum(e.provider_reported_cost_usd or 0.0 for e in native)
        native_estimate = sum(e.estimated_cost_usd or 0.0 for e in native)
        total_estimate = sum(e.estimated_cost_usd or 0.0 for e in completed)
        return {
            "cost_usd": round(native_total, 6) if native else None,
            "covered_calls": len(native),
            "total_completed_calls": len(completed),
            "coverage_by_calls": len(native) / len(completed) if completed else 0.0,
            "covered_estimated_cost_usd": round(native_estimate, 6),
            "coverage_by_estimated_cost": (
                native_estimate / total_estimate if total_estimate else 0.0
            ),
            "complete": bool(completed) and len(native) == len(completed),
        }

    def summary(self) -> dict[str, Any]:
        """The canonical `usage` object every experiment summary embeds."""
        completed = self.completed
        reasons = self.incompleteness_reasons()
        priced = [e for e in completed if e.estimated_cost_usd is not None]
        estimates_complete = len(priced) == len(completed)
        estimated_total = round(sum(e.estimated_cost_usd or 0.0 for e in priced), 6)
        coverage = self.provider_coverage()
        native_coverage = self.provider_native_coverage()
        provider_total = (
            round(sum(e.provider_reported_cost_usd or 0.0 for e in completed), 6)
            if completed and coverage["covered_calls"] == len(completed)
            else None
        )
        # Zero recorded calls plus a known problem means the ledger is ABSENT, not that
        # the run was free — the pre-ledger case. Publishing $0.00 there let one re-score
        # overwrite a real cost, so the total is withheld and the resolver falls through to
        # usage_migrated.json. A torn line alongside real calls is different: the priced
        # sum is still the best available number and stays published, flagged by `reasons`.
        nothing_recorded = not completed and bool(reasons)
        publish_total = estimates_complete and not nothing_recorded
        return {
            "cost_accounting_version": COST_ACCOUNTING_VERSION,
            "estimated_total_cost_usd": estimated_total if publish_total else None,
            # Always published, complete or not: the partial sum is what the priced calls
            # actually cost, and hiding it would leave no number at all to act on.
            "priced_calls_estimated_cost_usd": estimated_total,
            "provider_reported_total_cost_usd": provider_total,
            "provider_cost_coverage": coverage,
            "provider_native_cost_usd": native_coverage["cost_usd"],
            "provider_native_cost_coverage": native_coverage,
            "cost_complete": not reasons,
            "cost_incompleteness_reasons": reasons,
            "cost_usd_by_role": self.cost_by("role"),
            "tokens_total": self.total_tokens().to_dict(),
            "tokens_by_role": self.tokens_by("role"),
            "calls_by_role": {k: v["calls"] for k, v in self.group_by("role").items()},
            "by_model": self.group_by("requested_model"),
            "by_provider": self.group_by("provider"),
            "num_events": len(self.events),
            "num_completed_calls": len(completed),
            "num_failed_without_usage": len(self.events) - len(completed),
            "num_launches": len({e.launch_id for e in self.events if e.launch_id}),
            "num_workers": len({(e.launch_id, e.worker_id) for e in self.events}),
            "price_snapshot_ids": sorted(
                {e.price_snapshot_id for e in self.events if e.price_snapshot_id}
            ),
            # Deprecated: kept so consumers that have not migrated keep reading a number.
            # Equals estimated_total_cost_usd when the estimate is complete.
            "total_cost_usd": estimated_total,
        }

    def task_projection(self, task_id: str) -> dict[str, Any]:
        """The per-trace projection: a task-local view of the SAME events."""
        sub = self.filter(task_id=task_id)
        summary = sub.summary()
        return {
            "usage_event_ids": [e.event_id for e in sub.events],
            "estimated_total_cost_usd": summary["priced_calls_estimated_cost_usd"],
            "provider_reported_total_cost_usd": summary[
                "provider_reported_total_cost_usd"
            ],
            "cost_usd_by_role": summary["cost_usd_by_role"],
            "tokens_total": summary["tokens_total"],
            "cost_complete": summary["cost_complete"],
        }


def experiment_usage_summary(experiment_dir: Path) -> dict[str, Any]:
    """Convenience: the canonical usage object for an experiment folder."""
    return UsageAggregator.from_experiment_dir(experiment_dir).summary()
