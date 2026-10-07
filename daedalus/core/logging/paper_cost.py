"""The cost reported in the paper: every call at OpenAI standard-tier prices, with each
conversation's earlier prompt assumed to be fully cached (paper Section 4.1).

`cost_per_run(run_dir)` is the "$" column of Tables 1, 3, 6, 9 and 10: `full_cache_cost`
at the standard tier, divided by the number of repeats (`run_<i>/` folders). Calls are
priced one by one with `PriceSnapshot.price_call` (the long-context surcharge is decided
per request). Tokens come from the per-call usage ledger, else from the traces' per-turn
records and the per-call `usage` records a reference method writes under `retrieval/` or
`embedding/`. Spend recorded without tokens (τ²'s simulated user, pre-ledger auxiliary
roles) is carried at its recorded value and makes the result approximate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from daedalus.core.logging.cost import PricingError, get_price_snapshot
from daedalus.core.logging.usage import LLMUsageEvent, TokenUsage
from daedalus.core.logging.usage_aggregate import LedgerCorruptionError, UsageAggregator

# Subdirectories where a reference method writes one per-call record with its own `usage`.
_CALL_RECORD_DIRS = ("retrieval", "embedding")
# Trace/session json that is bookkeeping rather than a task trace.
_NOT_A_TRACE = {"evaluation", "run_meta", "config", "usage_migrated", "run_summary"}


def _tokens(counts: dict[str, Any], prompt: str, completion: str, cached: str) -> TokenUsage:
    """A recorded token triple as canonical usage.

    `cache_read` is clamped to the prompt total: it is a SUBSET of it, and a record whose
    cached count exceeded its prompt would otherwise price negative fresh tokens.
    `cache_details_available` preserves "this record has no cache field" as distinct from
    "it reported zero".
    """
    total = int(counts.get(prompt) or 0)
    return TokenUsage(
        prompt_tokens=total,
        completion_tokens=int(counts.get(completion) or 0),
        cache_read_tokens=min(int(counts.get(cached) or 0), total),
        cache_details_available=(cached in counts),
    )


def _trace_calls(trace: dict[str, Any], model: str) -> list[tuple[str, TokenUsage]]:
    """One entry per priceable call in a trace, at the finest granularity it recorded.

    Per TURN where the turns carry token usage. A turn that made several calls had their
    tokens summed into it, so the long-context surcharge is an upper bound, never an
    under-count.

    The exception is a turn set that records prompt and completion but NO cache field while
    the trace total has one: pricing those turns would bill every cache read at the fresh
    rate, so the (coarser but complete) trace total wins.
    """
    turns = [
        tokens
        for turn in (trace.get("turns") or [])
        if (tokens := _tokens(turn.get("token_usage") or {}, "prompt", "completion", "cached"))
        and (tokens.prompt_tokens or tokens.completion_tokens)
    ]
    totals = trace.get("total_tokens") or {}
    if turns and (
        all(t.cache_details_available for t in turns) or "cached" not in totals
    ):
        return [(model, t) for t in turns]
    whole = _tokens(totals, "prompt", "completion", "cached")
    if whole.prompt_tokens or whole.completion_tokens:
        return [(model, whole)]
    return []


def _record_calls(run_dir: Path) -> tuple[list[tuple[str, TokenUsage]], float, float]:
    """Per-call records, tokenless spend, and their original recorded spend.

    ExpeL's retrieval records carry an `embedder` and no `usage` at all — that retrieval is
    not an LLM chat call, so there is nothing to price and nothing missing. ReasoningBank
    writes an embedding record whose model string is empty; its cost cannot be re-priced
    against a snapshot that keys on the model, so it is returned as tokenless.
    """
    calls: list[tuple[str, TokenUsage]] = []
    tokenless = 0.0
    recorded_spend = 0.0
    for name in _CALL_RECORD_DIRS:
        base = run_dir / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            usage = (record or {}).get("usage") or {}
            if not usage:
                continue
            recorded_spend += float(usage.get("cost_usd") or 0.0)
            model = record.get("model") or ""
            if not model:
                tokenless += float(usage.get("cost_usd") or 0.0)
                continue
            calls.append((
                model,
                _tokens(usage, "prompt_tokens", "completion_tokens", "cached_tokens"),
            ))
    return calls, tokenless, recorded_spend


def _read_traces(run_dir: Path) -> Iterable[dict[str, Any]]:
    """Every task trace of any experiment kind, parsed once.

    Inference uses ``run_<i>/`` (or a legacy flat folder); accumulation and generation
    use ``traces/``.  Keeping that layout knowledge here is what makes the paper-cost
    policy usable by all three tabs instead of being an inference-only calculation.
    """
    runs = sorted(d for d in run_dir.glob("run_*") if d.is_dir())
    if not runs:
        traces_dir = run_dir / "traces"
        runs = [traces_dir] if traces_dir.is_dir() else [run_dir]
    for base in runs:
        for path in sorted(base.glob("*.json")):
            if path.stem in _NOT_A_TRACE:
                continue
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue


def _load_dict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _historical_total(run_dir: Path) -> float | None:
    """Largest explicit pre-ledger whole-run total available in the artifacts.

    It is used only to recover the residual roles whose tokens were never written.  The
    priceable trace share is subtracted before this value is carried, so solver spend is
    never counted both at its historical price and at the uniform snapshot price.
    """
    candidates: list[float] = []
    summary = _load_dict(run_dir / "run_summary.json")
    if summary.get("total_cost_usd") is not None:
        candidates.append(float(summary["total_cost_usd"]))

    sessions = run_dir / "sessions"
    if sessions.is_dir():
        values = [
            float(_load_dict(path).get("cost_usd") or 0.0)
            for path in sorted(sessions.glob("*.json"))
        ]
        if values:
            candidates.append(sum(values))

    task_summaries = run_dir / "task_summaries"
    if task_summaries.is_dir():
        values = [
            float(_load_dict(path).get("cost_usd") or 0.0)
            for path in sorted(task_summaries.glob("*.json"))
        ]
        if values:
            candidates.append(sum(values))
    return max(candidates) if candidates else None


def _artifact_calls(
    run_dir: Path, traces: Iterable[dict[str, Any]], model: str
) -> tuple[list[tuple[str, TokenUsage]], float, float]:
    """Priceable calls, recorded tokenless spend, and the old trace-priced share."""
    calls: list[tuple[str, TokenUsage]] = []
    tokenless = 0.0
    old_trace_spend = 0.0
    for trace in traces:
        if not isinstance(trace, dict) or "total_tokens" not in trace:
            continue
        trace_model = (((trace.get("config") or {}).get("agent") or {}).get("model")) or model
        calls.extend(_trace_calls(trace, trace_model))
        user_cost = float(trace.get("user_sim_cost_usd") or 0.0)
        tokenless += user_cost
        old_trace_spend += float(trace.get("total_cost_usd") or 0.0) + user_cost

    record_calls, record_tokenless, record_spend = _record_calls(run_dir)
    calls.extend(record_calls)
    tokenless += record_tokenless
    old_trace_spend += record_spend

    aggregate = _historical_total(run_dir)
    if aggregate is not None:
        # What remains after removing the trace-associated spend is an explicitly recorded
        # auxiliary-role total (explorer, judge, extraction, reflection, ...), but its
        # tokens no longer exist. Carry it once and mark the result reconstructed.
        tokenless += max(0.0, aggregate - old_trace_spend)
    return calls, tokenless, old_trace_spend


_CACHE_PREFIX_BLOCK = 128
_CONVERSATIONAL_ROLES = frozenset({"solver", "explorer", "user_simulator"})
_HISTORICAL_AUX_ROLES = frozenset(
    {"explorer", "judge", "extraction", "reflection", "retry_decision"}
)


def _historical_auxiliary_lower_bound(
    run_dir: Path, tier: str, snapshot_id: str | None = None
) -> tuple[float, TokenUsage, float, list[str]]:
    """Reprice recoverable pre-ledger auxiliary roles without mixing service tiers.

    Old generation/accumulation summaries retained aggregate tokens by role but not the
    individual calls.  They are still enough for a standard-price LOWER BOUND: explorer
    prompts use the user's requested maximally cached assumption; one-shot judge and
    extraction prompts retain their observed cache split.  Base (non-long-context) rates
    are deliberate because an aggregate cannot reveal whether any individual call crossed
    a per-request surcharge threshold.

    The third return value is the old dollar spend represented by these token aggregates;
    the caller removes it from the otherwise-tokenless historical residual so it is not
    counted twice.
    """
    summary = _load_dict(run_dir / "run_summary.json")
    tokens_by_role = summary.get("tokens_by_role") or {}
    cost_by_role = summary.get("cost_usd_by_role") or {}
    if not tokens_by_role or not cost_by_role:
        return 0.0, TokenUsage(), 0.0, []

    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        cfg = {}
    agent = cfg.get("agent") or {}
    generation = cfg.get("generation") or {}
    accumulation = cfg.get("accumulation") or {}
    extraction_model = accumulation.get("extraction_model") or ""
    role_models = {
        "explorer": generation.get("explorer_model") or "",
        "judge": generation.get("judge_model") or "",
        "extraction": extraction_model,
        "reflection": extraction_model,
        "retry_decision": extraction_model,
        "solver": agent.get("model") or "",
    }

    snapshot = get_price_snapshot(snapshot_id)
    total = 0.0
    aggregate = TokenUsage()
    represented_old_spend = 0.0
    notes: list[str] = []
    priced_roles: list[str] = []
    for role in sorted(_HISTORICAL_AUX_ROLES & tokens_by_role.keys() & cost_by_role.keys()):
        model = role_models.get(role) or ""
        counts = tokens_by_role.get(role) or {}
        prompt = int(counts.get("prompt_tokens", counts.get("prompt", 0)) or 0)
        completion = int(
            counts.get("completion_tokens", counts.get("completion", 0)) or 0
        )
        observed_cached = min(
            int(counts.get("cache_read_tokens", counts.get("cached", 0)) or 0),
            prompt,
        )
        cached = prompt if role == "explorer" else observed_cached
        tokens = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_read_tokens=cached,
            cache_details_available=True,
        )
        try:
            record = snapshot.resolve(model)
            multiplier = record.tier_multiplier(tier)
        except PricingError as exc:
            notes.append(f"cannot reprice historical {role} tokens: {exc}")
            continue

        # Aggregate role totals cannot safely select a per-request long-context tier.
        # Base rates are the defensible lower bound requested for this historical data.
        micro = (
            tokens.uncached_prompt_tokens * record.input_per_mtok
            + tokens.cache_read_tokens * record.cache_read_per_mtok
            + tokens.completion_tokens * record.output_per_mtok
        )
        total += micro * multiplier / 1_000_000
        aggregate = aggregate + tokens
        represented_old_spend += float(cost_by_role.get(role) or 0.0)
        priced_roles.append(role)

    if priced_roles:
        notes.append(
            "historical auxiliary role totals repriced at "
            f"{tier}: {', '.join(priced_roles)}"
        )
    if "explorer" in priced_roles:
        notes.append(
            "explorer prompt uses the maximal-cache lower bound; per-call history was not saved"
        )
    return round(total, 6), aggregate, represented_old_spend, notes


def _historical_residual_tier_ratio(
    run_dir: Path, from_tier: str, to_tier: str, snapshot_id: str | None = None
) -> float | None:
    """Uniform tier conversion for an un-tokenized historical residual, if known.

    The residual contains calls from the configured solver/extractor and, for generation,
    explorer/judge. When all have the same tier ratio (GPT flex runs are 2x from flex to
    standard), it can be normalized without conflating it with separately recorded
    tokenless trace spend such as tau2's user simulator. Mixed ratios remain unknown.
    """
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return None
    kind = cfg.get("kind")
    if kind not in {"inference", "accumulation", "generation"}:
        return None
    agent = cfg.get("agent") or {}
    generation = cfg.get("generation") or {}
    accumulation = cfg.get("accumulation") or {}
    models = {
        agent.get("model"),
    } - {None, ""}
    if kind in {"accumulation", "generation"}:
        models.add(accumulation.get("extraction_model"))
    if kind == "generation":
        models.update(
            {generation.get("explorer_model"), generation.get("judge_model")}
        )
    models -= {None, ""}
    if not models:
        return None
    snapshot = get_price_snapshot(snapshot_id)
    ratios: list[float] = []
    try:
        for model in models:
            record = snapshot.resolve(str(model))
            before = record.tier_multiplier(from_tier)
            after = record.tier_multiplier(to_tier)
            ratios.append(after / before)
    except (PricingError, ZeroDivisionError):
        return None
    return ratios[0] if ratios and all(abs(r - ratios[0]) < 1e-12 for r in ratios) else None


def _counterfactual_sequence(
    calls: list[tuple[str, TokenUsage, str | None, float | None]]
) -> list[tuple[str, TokenUsage, str | None, float | None]]:
    """Apply the paper's optimistic conversation-prefix caching policy to one dialogue.

    The first request keeps its observed cache read. Thereafter the preceding request's
    complete prompt is the reusable prefix, rounded down to a 128-token block; everything
    by which the next prompt grew remains fresh. A prompt decrease denotes a reset and is
    treated as a new first request rather than as cacheable history.
    """
    adjusted = []
    previous_prompt: int | None = None
    for model, tokens, tier, fallback in calls:
        if previous_prompt is None or tokens.prompt_tokens < previous_prompt:
            cached = tokens.cache_read_tokens
        else:
            prefix = min(previous_prompt, tokens.prompt_tokens)
            cached = (
                prefix // _CACHE_PREFIX_BLOCK * _CACHE_PREFIX_BLOCK
                if prefix >= 1024 else 0
            )
        adjusted.append((
            model,
            replace(tokens, cache_read_tokens=cached, cache_details_available=True),
            tier,
            fallback,
        ))
        previous_prompt = tokens.prompt_tokens
    return adjusted


def full_cache_cost(
    run_dir: Path,
    model: str = "",
    assumed_tier: str = "",
    snapshot_id: str | None = None,
    traces: Iterable[dict[str, Any]] | None = None,
    pricing_tier_override: str | None = None,
) -> dict[str, Any]:
    """Counterfactual cost when every conversation's prior prompt remains cached.

    This is deliberately separate from :func:`formula_cost`: it answers a what-if, not
    what the provider reported. Trace call order defines solver conversations. Remaining
    ledger events use their scope and timestamp; conversational scopes restart whenever
    prompt size decreases, while one-shot roles retain observed caching. When
    ``pricing_tier_override`` is set, every call is priced at that tier; the run browser
    uses ``standard`` for its raw, undiscounted comparison column.
    """
    run_dir = Path(run_dir)
    snapshot = get_price_snapshot(snapshot_id)
    tier = assumed_tier or "flex"
    trace_list = list(_read_traces(run_dir) if traces is None else traces)
    try:
        aggregator = UsageAggregator.from_experiment_dir(run_dir)
    except (LedgerCorruptionError, OSError) as exc:
        return {
            "cost_usd": None, "complete": False, "cache_rate": None,
            "fresh_prompt_tokens": 0, "cached_prompt_tokens": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "num_priced_calls": 0,
            "price_snapshot_id": snapshot.snapshot_id,
            "pricing_tier": pricing_tier_override or "recorded/assumed per call",
            "notes": [f"usage ledger unreadable ({exc})"],
        }

    events_by_id = {event.event_id: event for event in aggregator.completed}
    roles = {event.role for event in aggregator.completed}
    ledger_is_post_only = bool(roles) and roles <= {"consolidation"}
    full_ledger = bool(aggregator.events) and not ledger_is_post_only
    covered_event_ids: set[str] = set()
    priced_calls: list[tuple[str, TokenUsage, str | None, float | None]] = []
    notes: list[str] = []
    cache_observation_missing = False

    for trace in trace_list:
        if not isinstance(trace, dict) or "total_tokens" not in trace:
            continue
        event_ids = list(trace.get("usage_event_ids") or [])
        # With a full ledger, an unlinked historical trace cannot safely be added: the
        # same solver calls are already in the ledger. Linked traces supply exact dialogue
        # boundaries and their events are removed from the ledger remainder below.
        if full_ledger and not event_ids:
            continue
        turns = []
        for turn in trace.get("turns") or []:
            counts = turn.get("token_usage") or {}
            tokens = _tokens(counts, "prompt", "completion", "cached")
            if tokens.prompt_tokens or tokens.completion_tokens:
                turns.append(tokens)
        if not turns:
            whole = _tokens(trace.get("total_tokens") or {}, "prompt", "completion", "cached")
            if whole.prompt_tokens or whole.completion_tokens:
                turns = [whole]
        trace_model = (((trace.get("config") or {}).get("agent") or {}).get("model")) or model
        sequence = []
        for idx, tokens in enumerate(turns):
            event = events_by_id.get(event_ids[idx]) if idx < len(event_ids) else None
            if event is not None:
                covered_event_ids.add(event.event_id)
                sequence.append((
                    event.requested_model, event.tokens or tokens,
                    event.effective_service_tier, event.provider_reported_cost_usd,
                ))
            else:
                sequence.append((trace_model, tokens, tier, None))
        first_event = events_by_id.get(event_ids[0]) if event_ids else None
        if turns and not turns[0].cache_details_available and (
            first_event is None or first_event.tokens is None
        ):
            cache_observation_missing = True
            notes.append("a trace did not record observed caching on its first request")
        covered_event_ids.update(event_ids)
        priced_calls.extend(_counterfactual_sequence(sequence))

    # Per-call retrieval/embedding artifacts are independent requests rather than one
    # dialogue, so their observed cache split is retained.
    if not full_ledger:
        record_calls, _, _ = _record_calls(run_dir)
        priced_calls.extend((m, t, tier, None) for m, t in record_calls)

    remaining = [
        event for event in aggregator.completed
        if event.event_id not in covered_event_ids and event.tokens is not None
    ]
    groups: dict[tuple[Any, ...], list[LLMUsageEvent]] = defaultdict(list)
    for event in remaining:
        if event.role not in _CONVERSATIONAL_ROLES:
            priced_calls.append((
                event.requested_model, event.tokens, event.effective_service_tier,
                event.provider_reported_cost_usd,
            ))
            continue
        key = (
            event.role, event.component, event.task_id, event.session_id,
            event.attempt_id, event.run_idx, event.worker_id,
            event.requested_model, event.effective_service_tier,
        )
        groups[key].append(event)
    for events in groups.values():
        sequence: list[tuple[str, TokenUsage, str | None, float | None]] = []
        previous = None
        for event in sorted(events, key=lambda item: item.timestamp):
            assert event.tokens is not None
            if previous is not None and event.tokens.prompt_tokens < previous:
                priced_calls.extend(_counterfactual_sequence(sequence))
                sequence = []
            sequence.append((
                event.requested_model, event.tokens, event.effective_service_tier,
                event.provider_reported_cost_usd,
            ))
            previous = event.tokens.prompt_tokens
        priced_calls.extend(_counterfactual_sequence(sequence))

    auxiliary_total = 0.0
    auxiliary_tokens = TokenUsage()
    auxiliary_lower_bound = False
    if full_ledger:
        tokenless = sum(
            float(event.provider_reported_cost_usd or 0.0)
            for event in aggregator.completed if event.tokens is None
        )
    else:
        _, tokenless, old_trace_spend = _artifact_calls(run_dir, trace_list, model)
        historical_total = _historical_total(run_dir)
        historical_residual = (
            max(0.0, historical_total - old_trace_spend)
            if historical_total is not None else 0.0
        )
        # Keep genuinely tokenless trace spend (notably tau2's simulator) separate from
        # the whole-run auxiliary residual. Only the latter follows the generation
        # agents' service tier.
        independent_tokenless = max(0.0, tokenless - historical_residual)
        auxiliary_tier = pricing_tier_override or tier
        (
            auxiliary_total,
            auxiliary_tokens,
            represented_old_spend,
            auxiliary_notes,
        ) = _historical_auxiliary_lower_bound(
            run_dir, auxiliary_tier, snapshot_id=snapshot_id
        )
        remaining_residual = max(0.0, historical_residual - represented_old_spend)
        if auxiliary_total or represented_old_spend:
            # `_artifact_calls` recovered these roles only as a dollar residual. Replace
            # that old-tier scalar with the role-token lower bound instead of adding both.
            auxiliary_lower_bound = True
            notes.extend(auxiliary_notes)
        residual_ratio = _historical_residual_tier_ratio(
            run_dir, tier, auxiliary_tier, snapshot_id=snapshot_id
        )
        if remaining_residual and residual_ratio is not None:
            remaining_residual *= residual_ratio
            if residual_ratio != 1.0:
                notes.append(
                    "un-tokenized historical residual converted from "
                    f"{tier} to {auxiliary_tier} pricing ({residual_ratio:g}x)"
                )
        tokenless = independent_tokenless + remaining_residual
        # The post-only ledger is outside the historical aggregate.
        if ledger_is_post_only:
            tokenless += sum(
                float(event.provider_reported_cost_usd or 0.0)
                for event in aggregator.completed if event.tokens is None
            )

    total = auxiliary_total
    calls_priced = 0
    problems: list[str] = []
    carried_unpriceable = 0.0
    aggregate_tokens = auxiliary_tokens
    unknown_cache_writes = 0
    for call_model, tokens, call_tier, fallback in priced_calls:
        aggregate_tokens = aggregate_tokens + tokens
        try:
            if not full_ledger and snapshot.resolve(call_model).rates_for(
                tokens.prompt_tokens
            )[2] > 0:
                unknown_cache_writes += 1
            total += snapshot.price_call(
                call_model, tokens, pricing_tier_override or call_tier
            )
            calls_priced += 1
        except PricingError as exc:
            problems.append(str(exc))
            carried_unpriceable += float(fallback or 0.0)
    tokenless += carried_unpriceable
    notes.extend(sorted(set(problems))[:3])
    if tokenless:
        notes.append(f"${tokenless:.4f} carried without priceable token detail")
    if unknown_cache_writes:
        notes.append(
            f"{unknown_cache_writes} historical call(s) may have unrecorded cache writes"
        )
    ledger_reasons = [
        reason for reason in aggregator.incompleteness_reasons()
        if not reason.startswith("no usage ledger for this run:")
    ]
    notes.extend(ledger_reasons)
    prompt = aggregate_tokens.prompt_tokens
    cached = aggregate_tokens.cache_read_tokens
    return {
        # Rule 1 explicitly preserves observed caching on the first request. If an old
        # trace did not persist that observation, the requested counterfactual is not
        # identifiable; a blank cell is more honest than a value that can exceed reality.
        "cost_usd": (
            round(total + tokenless, 6)
            if (calls_priced or tokenless) and not cache_observation_missing else None
        ),
        "complete": bool(calls_priced) and not (
            problems or tokenless or ledger_reasons or cache_observation_missing
            or unknown_cache_writes or auxiliary_lower_bound
        ),
        "cache_rate": cached / prompt if prompt else None,
        "fresh_prompt_tokens": prompt - cached,
        "cached_prompt_tokens": cached,
        "prompt_tokens": prompt,
        "completion_tokens": aggregate_tokens.completion_tokens,
        "num_priced_calls": calls_priced,
        "price_snapshot_id": snapshot.snapshot_id,
        "pricing_tier": pricing_tier_override or "recorded/assumed per call",
        "notes": list(dict.fromkeys(notes))[:5],
    }


def cost_per_run(run_dir: Path) -> tuple[float | None, dict[str, Any]]:
    """The paper's $ per run (standard tier, full caching), and the full computation."""
    run_dir = Path(run_dir)
    meta = _load_dict(run_dir / "run_meta.json")
    migrated = _load_dict(run_dir / "usage_migrated.json")
    info = full_cache_cost(
        run_dir,
        model=str(meta.get("model") or ""),
        assumed_tier=str(migrated.get("assumed_service_tier") or ""),
        pricing_tier_override="standard",
    )
    total = info.get("cost_usd")
    repeats = len([d for d in run_dir.glob("run_*") if d.is_dir()]) or 1
    return (total / repeats if total is not None else None), info
