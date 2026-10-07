"""Auditable run-cost resolution shared by the UI, reports, and figure scripts.

No accounting method is silently substituted for another. The result exposes the live
per-call ledger estimate, provider-native response charges, a billing reconciliation,
the historical migration, and the old artifact total side by side. ``total_usd`` is a
documented best-available choice kept for compatibility; paper tables should retain the
method columns and ``paper_cost_method``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daedalus.core.logging.usage_aggregate import LedgerCorruptionError, UsageAggregator

_USAGE_FILES = ("evaluation.json", "run_summary.json")
MIGRATED_NAME = "usage_migrated.json"
RECONCILIATION_NAME = "provider_reconciliation.json"

_MIGRATION_LABELS = {
    "recomputed": "recomputed from traces, all recoverable roles",
    "recomputed+carried": "recomputed from traces + carried auxiliary roles",
    "solver-only": "recomputed solver share only",
}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _stored_usage(run_dir: Path) -> dict[str, Any]:
    for name in _USAGE_FILES:
        usage = (_load(run_dir / name) or {}).get("usage") or {}
        if not usage or usage.get("num_completed_calls") == 0:
            continue
        events, calls = usage.get("num_events"), usage.get("num_completed_calls")
        if (events is not None or calls is not None) and not (events or calls):
            continue
        return usage
    return {}


def _native_from_summary(usage: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    native = usage.get("provider_native_cost_coverage") or {}
    if native:
        return usage.get("provider_native_cost_usd"), native
    # Compatibility with summaries written before native and LiteLLM costs were split.
    coverage = usage.get("provider_cost_coverage") or {}
    sources = coverage.get("covered_calls_by_source") or {}
    completed = int(usage.get("num_completed_calls") or 0)
    native_calls = int(sources.get("provider_usage_cost") or 0)
    value = usage.get("provider_reported_total_cost_usd")
    if native_calls != completed:
        value = None
    return value, {
        "cost_usd": value,
        "covered_calls": native_calls,
        "total_completed_calls": completed,
        "coverage_by_calls": native_calls / completed if completed else 0.0,
        "complete": bool(completed) and native_calls == completed,
    }


def _reconciliation(run_dir: Path) -> dict[str, Any]:
    report = _load(run_dir / RECONCILIATION_NAME) or {}
    if not report:
        return {}
    amount = report.get("billing_reconciled_cost_usd")
    if amount is None:
        amount = report.get("matched_export_cost_usd")
    # Version-1 reports only had export_cost_usd. It is safe as a run total solely when
    # the report proves a complete one-to-one match and is not provider-filtered.
    complete = bool(report.get("reconciliation_complete"))
    if "reconciliation_complete" not in report:
        complete = (
            not report.get("provider_filter")
            and int(report.get("num_unmatched_export_rows") or 0) == 0
            and int(report.get("num_unmatched_ledger_calls") or 0) == 0
            and int(report.get("num_matched_requests") or 0) > 0
        )
    if amount is None and complete:
        amount = report.get("export_cost_usd")
    return {
        "cost_usd": float(amount) if amount is not None else None,
        "complete": complete,
        "matched_calls": int(report.get("num_matched_requests") or 0),
        "ledger_calls": int(report.get("num_ledger_calls") or 0),
        "unmatched_ledger_calls": int(report.get("num_unmatched_ledger_calls") or 0),
        "unmatched_export_rows": int(report.get("num_unmatched_export_rows") or 0),
        "provider_filter": report.get("provider_filter"),
        "report_version": report.get("reconciliation_schema_version", 1),
    }


def resolve_run_cost(
    run_dir: Path,
    legacy_total: float | None = None,
    legacy_label: str = "legacy artifact total",
) -> dict[str, Any]:
    """Return every available cost method and a conservative best-available choice.

    Selection order is complete billing reconciliation, complete provider-native charge,
    live ledger estimate, migrated reconstruction, then legacy artifact total. A ledger
    containing only post-processing roles is added to a legacy total because the scopes
    are demonstrably non-overlapping. Partial provider coverage is never extrapolated.
    """
    run_dir = Path(run_dir)
    stored = _stored_usage(run_dir)

    if legacy_total is None:
        saved = (_load(run_dir / "run_summary.json") or {}).get("total_cost_usd")
        if saved is not None:
            legacy_total = float(saved)
            legacy_label = "pre-ledger run_summary total"

    migrated = _load(run_dir / MIGRATED_NAME) or {}
    migrated_total = migrated.get("recomputed_partial_cost_usd")
    migration_basis = str(migrated.get("migration_basis") or "")

    # The LIVE ledger, not the snapshot a run embedded when it finished: calls made after
    # that snapshot (a resume, a later `consolidate`) are real spend and are only on disk
    # here. A ledger this reader cannot trust must not take the whole browser down with
    # it, so corruption degrades to the stored snapshot and says so.
    live, ledger_error = {}, ""
    try:
        aggregator = UsageAggregator.from_experiment_dir(run_dir)
        live = aggregator.summary() if aggregator.events else {}
    except (LedgerCorruptionError, OSError) as exc:
        ledger_error = f"usage ledger unreadable ({exc})"
    usage = live or stored
    live_total = live.get("priced_calls_estimated_cost_usd") if live else None
    stored_total = stored.get("estimated_total_cost_usd")
    if stored_total is None:
        stored_total = stored.get("priced_calls_estimated_cost_usd")
    estimate_total = live_total if live_total is not None else stored_total
    estimate_complete = bool(usage.get("cost_complete")) if usage else False
    estimate_notes = list(usage.get("cost_incompleteness_reasons") or [])
    estimate_origin = "live usage ledger" if live else ("stored usage snapshot" if stored else "")

    native_total, native_coverage = _native_from_summary(usage) if usage else (None, {})
    native_complete = bool(native_coverage.get("complete")) and estimate_complete
    reconciliation = _reconciliation(run_dir)

    notes = list(estimate_notes)
    if ledger_error:
        notes.insert(0, ledger_error)
        estimate_complete = False
    source = "legacy"
    total = legacy_total
    complete = False
    label = legacy_label
    by_role: dict[str, float] = {}

    if reconciliation.get("complete") and reconciliation.get("cost_usd") is not None:
        total = reconciliation["cost_usd"]
        source, complete = "billing_reconciled", True
        label = "provider billing export, completely matched"
        by_role = usage.get("cost_usd_by_role") or {}
    elif native_complete and native_total is not None:
        total = native_total
        source, complete = "provider_native", True
        label = "provider-native response charges, all calls"
        by_role = usage.get("cost_usd_by_role") or {}
    elif estimate_total is not None:
        roles = set((usage.get("cost_usd_by_role") or {}).keys())
        post_only = bool(roles) and roles <= {"consolidation"}
        if live and post_only and legacy_total is not None:
            total = round(float(legacy_total) + float(estimate_total), 6)
            source = "hybrid"
            label = "legacy execution total + non-overlapping live consolidation ledger"
            notes.append("hybrid historical total is not billing-reconciled")
        else:
            total = float(estimate_total)
            source = "ledger_live" if live else "ledger"
            label = f"{estimate_origin}, all recorded roles"
        complete = estimate_complete and source != "hybrid"
        by_role = usage.get("cost_usd_by_role") or {}
        if not complete and "incomplete" not in label:
            label += " (incomplete)"
    elif migrated_total is not None and not (
        migration_basis == "solver-only" and legacy_total is not None
    ):
        total = float(migrated_total)
        source = "migrated"
        label = _MIGRATION_LABELS.get(migration_basis, f"migrated ({migration_basis})")
        notes = list(migrated.get("cost_incompleteness_reasons") or [])
        by_role = migrated.get("cost_usd_by_role") or {}
    elif legacy_total is None:
        notes = list(migrated.get("cost_incompleteness_reasons") or []) or [
            "no cost record: run the migration for pre-ledger artifacts or reconcile a provider export"
        ]

    if native_total is not None and not native_complete:
        notes.append(
            "provider-native charge covers "
            f"{native_coverage.get('covered_calls', 0)}/{native_coverage.get('total_completed_calls', 0)} calls"
        )

    stale = bool(
        live
        and stored_total is not None
        and abs(float(live_total or 0.0) - float(stored_total)) > 0.0000005
    )
    if stale:
        # The run finished, wrote its `usage` block, and then more calls were billed
        # against it — a resume, or `scripts/consolidate` run afterwards. The ledger is
        # the one that grew; the snapshot is a frozen view and is never preferred.
        notes.append(
            f"embedded usage snapshot is stale (${float(stored_total):.4f}); the live "
            f"ledger has ${float(live_total or 0.0):.4f}"
        )

    out = {
        "source": source,
        "total_usd": total,
        "complete": complete,
        "label": label,
        "by_role": by_role,
        "notes": list(dict.fromkeys(notes)),
        "version": usage.get("cost_accounting_version", 2 if migrated else 1),
        "paper_cost_method": source,
        "ledger_estimated_cost_usd": estimate_total,
        "ledger_estimate_complete": estimate_complete,
        "ledger_estimate_origin": estimate_origin or None,
        "ledger_num_events": usage.get("num_events") if usage else None,
        "ledger_num_completed_calls": usage.get("num_completed_calls") if usage else None,
        # Tokens from the SAME events the estimate was priced from, so a table showing
        # cost beside token counts is showing one run's worth of both. A trace-derived
        # sum covers only the solver, which on a tau2 run is a different scope from the
        # cost printed next to it.
        "ledger_tokens": (usage.get("tokens_total") or {}) if usage else {},
        "provider_native_cost_usd": native_total,
        "provider_native_complete": native_complete,
        "provider_native_coverage": native_coverage,
        "billing_reconciled_cost_usd": reconciliation.get("cost_usd"),
        "billing_reconciliation_complete": bool(reconciliation.get("complete")),
        "billing_reconciliation": reconciliation,
        "migrated_cost_usd": float(migrated_total) if migrated_total is not None else None,
        "migration_basis": migration_basis or None,
        "legacy_cost_usd": float(legacy_total) if legacy_total is not None else None,
        "legacy_cost_label": legacy_label if legacy_total is not None else None,
        "stored_usage_estimate_usd": stored_total,
        "stored_usage_is_stale": stale,
        "ledger_unreadable": bool(ledger_error),
    }
    if migration_basis == "solver-only" and migrated_total is not None:
        out["recomputed_solver_share_usd"] = float(migrated_total)
    return out
