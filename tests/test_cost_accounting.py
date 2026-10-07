"""Unit tests for the cost-accounting foundations.

The regressions these lock down all produced a plausible-looking WRONG number rather than
an error, which is why each one is asserted directly: a prefix match that priced
`gpt-5.4-nano` as `gpt-5.4`, an unknown model reported as $0.00, a flex run costed at
standard rates, cache writes charged as nothing, and a killed worker's spend vanishing
with its unwritten summary.
"""

from __future__ import annotations

import json

import pytest

from daedalus.core.logging.cost import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    PriceRecord,
    PriceSnapshot,
    PricingError,
    UnknownModelPriceError,
    UnknownServiceTierError,
    UnpriceableModalityError,
    estimate_cost,
    get_price_snapshot,
)
from daedalus.core.logging.usage import (
    STATUS_COMPLETED,
    STATUS_FAILED_WITHOUT_USAGE,
    LLMUsageEvent,
    TokenUsage,
    UsageError,
    UsageScope,
)
from daedalus.core.logging.usage_aggregate import (
    LedgerCorruptionError,
    UsageAggregator,
)
from daedalus.core.logging.usage_ledger import UsageLedger, new_launch_id

MTOK = 1_000_000


def _tokens(**kwargs) -> TokenUsage:
    return TokenUsage(**kwargs)


# ---- price resolution (plan §13.1) ----------------------------------------------------


def test_exact_canonical_model_resolves():
    assert get_price_snapshot().resolve("gpt-5.4").key == "gpt-5.4"


def test_provider_qualified_alias_resolves_when_declared():
    assert get_price_snapshot().resolve("openai/gpt-5.4").key == "gpt-5.4"


def test_openrouter_bare_id_and_prefixed_id_share_one_record():
    snapshot = get_price_snapshot()
    assert (
        snapshot.resolve("qwen/qwen3.8-27b").key
        == snapshot.resolve("openrouter/qwen/qwen3.8-27b").key
    )


def test_date_pinned_alias_resolves():
    """tau2's default user simulator is a date-pinned id at identical published rates."""
    assert get_price_snapshot().resolve("gpt-4.1-2025-04-14").key == "gpt-4.1"


def test_unknown_model_raises_instead_of_returning_zero():
    with pytest.raises(UnknownModelPriceError):
        estimate_cost(
            "gpt-5.9-unreleased", _tokens(prompt_tokens=10), effective_tier="standard"
        )


def test_nano_is_priced_on_its_own_record_not_the_base_models():
    """The old bidirectional prefix match priced nano at gpt-5.4's rate, 12x too high."""
    snapshot = get_price_snapshot()
    nano = snapshot.resolve("gpt-5.4-nano")
    base = snapshot.resolve("gpt-5.4")
    assert nano.key == "gpt-5.4-nano"
    assert nano.input_per_mtok == 0.2 and base.input_per_mtok == 2.5


def test_an_unlisted_suffix_still_refuses_to_borrow_a_price():
    """The protection itself: a sibling that is NOT declared must not resolve at all."""
    with pytest.raises(UnknownModelPriceError):
        get_price_snapshot().resolve("gpt-5.4-nano-turbo")


def test_two_revisions_of_one_family_stay_separate():
    snapshot = get_price_snapshot()
    base = snapshot.resolve("deepseek/deepseek-v4-flash")
    pinned = snapshot.resolve("deepseek/deepseek-v4-flash-0731")
    assert base.key != pinned.key
    assert base.input_per_mtok != pinned.input_per_mtok


def test_free_tier_variant_is_not_priced_as_the_paid_one():
    snapshot = get_price_snapshot()
    assert snapshot.resolve("z-ai/glm-5.2:free").input_per_mtok == 0.0
    assert snapshot.resolve("z-ai/glm-5.2").input_per_mtok > 0.0


def test_empty_model_name_raises():
    with pytest.raises(UnknownModelPriceError):
        get_price_snapshot().resolve("")


def test_ambiguous_alias_is_rejected_at_snapshot_construction():
    def _rec(key: str, alias: str) -> PriceRecord:
        return PriceRecord(
            key=key,
            provider="test",
            input_per_mtok=1.0,
            cache_read_per_mtok=0.1,
            cache_write_per_mtok=0.0,
            output_per_mtok=2.0,
            aliases=(alias,),
            source_url="",
            effective_date="",
        )

    with pytest.raises(PricingError):
        PriceSnapshot("t", [_rec("a", "shared"), _rec("b", "shared")])


def test_unknown_snapshot_id_raises():
    with pytest.raises(PricingError):
        get_price_snapshot("1999-01-01")


# ---- token and cache pricing (plan §13.2) --------------------------------------------


def test_no_cache_hit_charges_full_input():
    cost = estimate_cost(
        "gpt-4.1", _tokens(prompt_tokens=MTOK), effective_tier="standard"
    )
    assert cost == pytest.approx(2.0)


def test_partial_cache_hit_splits_input_between_rates():
    cost = estimate_cost(
        "gpt-4.1",
        _tokens(prompt_tokens=MTOK, cache_read_tokens=MTOK // 2),
        effective_tier="standard",
    )
    # 0.5M at $2.00 + 0.5M at $0.50
    assert cost == pytest.approx(1.0 + 0.25)


def test_full_cache_hit_charges_only_the_cached_rate():
    cost = estimate_cost(
        "gpt-4.1",
        _tokens(prompt_tokens=MTOK, cache_read_tokens=MTOK),
        effective_tier="standard",
    )
    assert cost == pytest.approx(0.5)


def test_cache_write_is_charged_at_its_own_rate():
    """Anthropic's explicit cache write is 1.25x input, and used to be free of charge."""
    cost = estimate_cost(
        "claude-sonnet-4-20250514",
        _tokens(prompt_tokens=0, cache_write_tokens=MTOK),
        effective_tier="standard",
    )
    assert cost == pytest.approx(3.75)


def test_cache_read_and_write_in_one_request():
    cost = estimate_cost(
        "claude-sonnet-4-20250514",
        _tokens(prompt_tokens=MTOK, cache_read_tokens=MTOK, cache_write_tokens=MTOK),
        effective_tier="standard",
    )
    assert cost == pytest.approx(0.30 + 3.75)


def test_reasoning_tokens_are_not_charged_twice():
    with_reasoning = estimate_cost(
        "gpt-4.1",
        _tokens(completion_tokens=MTOK, reasoning_tokens=MTOK // 2),
        effective_tier="standard",
    )
    without = estimate_cost(
        "gpt-4.1", _tokens(completion_tokens=MTOK), effective_tier="standard"
    )
    assert with_reasoning == pytest.approx(without) == pytest.approx(8.0)


def test_cache_read_above_prompt_tokens_is_rejected():
    with pytest.raises(UsageError):
        TokenUsage(prompt_tokens=10, cache_read_tokens=11)


def test_reported_zero_cached_tokens_differs_from_no_cache_details():
    reported = TokenUsage(
        prompt_tokens=10, cache_read_tokens=0, cache_details_available=True
    )
    silent = TokenUsage(prompt_tokens=10)
    assert reported.cache_details_available and not silent.cache_details_available


def test_negative_counts_are_rejected():
    with pytest.raises(UsageError):
        TokenUsage(prompt_tokens=-1)


def test_modality_without_a_rate_stops_estimation():
    with pytest.raises(UnpriceableModalityError):
        estimate_cost(
            "gpt-4.1",
            _tokens(prompt_tokens=10, image_input_tokens=5),
            effective_tier="standard",
        )


# ---- long-context surcharge ----------------------------------------------------------


def test_a_request_below_the_threshold_uses_the_base_rates():
    cost = estimate_cost(
        "gpt-5.4", _tokens(prompt_tokens=271_999), effective_tier="standard"
    )
    assert cost == pytest.approx(271_999 * 2.5 / MTOK)


def test_a_request_at_the_threshold_switches_every_rate():
    """272k is inclusive, and input, cache-read and output all change together."""
    tokens = _tokens(
        prompt_tokens=272_000, cache_read_tokens=200_000, completion_tokens=1_000
    )
    cost = estimate_cost("gpt-5.4", tokens, effective_tier="standard")
    assert cost == pytest.approx((72_000 * 5.0 + 200_000 * 0.5 + 1_000 * 22.5) / MTOK)


def test_the_threshold_counts_cached_prompt_tokens_too():
    """It is the request's context size that is billed differently, not its cache misses."""
    snapshot = get_price_snapshot()
    record = snapshot.resolve("gpt-5.4")
    # 300k prompt of which 290k came from cache is still a 300k request.
    assert record.rates_for(300_000)[0] == 5.0


def test_a_model_without_a_tier_is_unaffected_by_prompt_size():
    small = estimate_cost(
        "gpt-5.4-mini", _tokens(prompt_tokens=1_000), effective_tier="standard"
    )
    large = estimate_cost(
        "gpt-5.4-mini", _tokens(prompt_tokens=1_000_000), effective_tier="standard"
    )
    assert large == pytest.approx(small * 1000)


def test_flex_still_halves_the_long_context_rates():
    tokens = _tokens(prompt_tokens=400_000, completion_tokens=1_000)
    assert estimate_cost("gpt-5.4", tokens, effective_tier="flex") == pytest.approx(
        estimate_cost("gpt-5.4", tokens, effective_tier="standard") / 2
    )


def test_lunas_long_context_tier_also_re_prices_cache_writes():
    tokens = _tokens(prompt_tokens=300_000, cache_write_tokens=10_000)
    cost = estimate_cost("gpt-5.6-luna", tokens, effective_tier="standard")
    assert cost == pytest.approx((300_000 * 0.4 + 10_000 * 0.5) / MTOK)


def test_only_the_two_documented_models_carry_a_tier():
    """A surcharge invented for a model that has none would over-report it."""
    snapshot = get_price_snapshot()
    with_tier = {k for k in snapshot.model_keys if snapshot.resolve(k).long_context}
    assert with_tier == {"gpt-5.4", "gpt-5.6-luna"}


# ---- service tier (plan §13.3) -------------------------------------------------------


def test_flex_is_half_of_standard_on_every_rate():
    tokens = _tokens(
        prompt_tokens=MTOK, cache_read_tokens=MTOK // 2, completion_tokens=MTOK
    )
    flex = estimate_cost("gpt-5.4", tokens, effective_tier="flex")
    standard = estimate_cost("gpt-5.4", tokens, effective_tier="auto")
    assert flex == pytest.approx(standard / 2)


def test_flex_request_that_completed_as_auto_is_billed_as_auto():
    """An explicitly auto-tier request is priced at the tier the CALL ran at.

    Deliberately a sub-272k prompt, to isolate the tier from the long-context surcharge.
    """
    tokens = _tokens(prompt_tokens=100_000)
    assert estimate_cost("gpt-5.4", tokens, effective_tier="auto") == pytest.approx(
        0.25
    )


def test_unknown_effective_tier_on_a_tier_priced_model_raises():
    with pytest.raises(UnknownServiceTierError):
        estimate_cost("gpt-5.4", _tokens(prompt_tokens=10), effective_tier=None)


def test_unknown_tier_is_harmless_when_the_model_has_no_tier_pricing():
    assert estimate_cost(
        "qwen/qwen3.8-27b", _tokens(prompt_tokens=MTOK), effective_tier=None
    ) == pytest.approx(0.42)


def test_unsupported_tier_name_raises():
    with pytest.raises(UnknownServiceTierError):
        estimate_cost("gpt-5.4", _tokens(prompt_tokens=10), effective_tier="platinum")


# ---- ledger (plan §13.5) -------------------------------------------------------------


def _event(role: str = "solver", **kwargs) -> LLMUsageEvent:
    kwargs.setdefault("requested_model", "gpt-4.1")
    kwargs.setdefault("tokens", _tokens(prompt_tokens=1000, completion_tokens=100))
    kwargs.setdefault("estimated_cost_usd", 0.01)
    return LLMUsageEvent.from_scope(UsageScope(role=role), **kwargs)


def test_one_event_round_trips_through_the_ledger(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="7")
    ledger.record(_event())
    ledger.close()
    path = tmp_path / "usage" / "launch-a" / "worker_7.jsonl"
    assert path.is_file()
    assert json.loads(path.read_text())["role"] == "solver"


def test_events_are_flushed_before_record_returns(tmp_path):
    """A worker killed after a completed call must still leave that spend on disk."""
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="7")
    ledger.record(_event())
    # No close(): read the file while the handle is still open.
    events, problems = UsageAggregator.from_experiment_dir(tmp_path), []
    assert len(events.events) == 1 and not problems


def test_several_roles_on_one_client_stay_separate(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="1")
    ledger.record(_event("solver", estimated_cost_usd=0.10))
    ledger.record(_event("verifier", estimated_cost_usd=0.01))
    ledger.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["cost_usd_by_role"] == {"solver": 0.1, "verifier": 0.01}
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.11)


def test_workers_and_launches_are_kept_apart_and_summed(tmp_path):
    for launch in ("launch-a", "launch-b"):
        for worker in ("1", "2"):
            led = UsageLedger(tmp_path, launch, worker_id=worker)
            led.record(_event(estimated_cost_usd=1.0))
            led.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["num_launches"] == 2
    assert summary["num_workers"] == 4
    assert summary["estimated_total_cost_usd"] == pytest.approx(4.0)


def test_duplicate_event_id_stops_aggregation(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="1")
    event = _event()
    ledger.record(event)
    ledger.record(event)  # same event_id: only a corrupted/copied ledger does this
    ledger.close()
    with pytest.raises(LedgerCorruptionError):
        UsageAggregator.from_experiment_dir(tmp_path)


def test_repeated_call_after_a_resume_counts_twice(tmp_path):
    """Identical task/attempt, different event: both were billed, so both count."""
    first = UsageLedger(tmp_path, "launch-a", worker_id="1")
    first.record(_event(task_id="t1", attempt_id="0", estimated_cost_usd=0.5))
    first.close()
    second = UsageLedger(tmp_path, "launch-b", worker_id="1")
    second.record(_event(task_id="t1", attempt_id="0", estimated_cost_usd=0.5))
    second.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["num_completed_calls"] == 2
    assert summary["estimated_total_cost_usd"] == pytest.approx(1.0)


def test_torn_final_line_is_reported_not_swallowed(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="1")
    ledger.record(_event(estimated_cost_usd=0.25))
    ledger.close()
    with ledger.path.open("a", encoding="utf-8") as fh:
        fh.write('{"role": "solver", "requested_mo')
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.25)
    assert summary["cost_complete"] is False
    assert any("torn" in r for r in summary["cost_incompleteness_reasons"])


def test_failed_without_usage_makes_the_run_incomplete(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="1")
    ledger.record(_event(estimated_cost_usd=0.25))
    ledger.record(
        LLMUsageEvent.from_scope(
            UsageScope(role="solver"),
            requested_model="gpt-4.1",
            status=STATUS_FAILED_WITHOUT_USAGE,
            error_type="Timeout",
            provider_may_have_billed=True,
        )
    )
    ledger.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["num_failed_without_usage"] == 1
    assert summary["cost_complete"] is False
    assert any(
        "may have been billed" in r for r in summary["cost_incompleteness_reasons"]
    )


def test_ledger_events_carry_no_prompt_or_response_payload(tmp_path):
    ledger = UsageLedger(tmp_path, "launch-a", worker_id="1")
    ledger.record(_event())
    ledger.close()
    record = json.loads(ledger.path.read_text())
    assert not {"messages", "prompt", "content", "response", "api_key"} & set(record)


def test_unknown_role_is_rejected():
    with pytest.raises(UsageError):
        UsageScope(role="mystery")


# ---- provider cost coverage (plan §13.4) ---------------------------------------------


def test_full_provider_coverage_publishes_a_provider_total(tmp_path):
    ledger = UsageLedger(tmp_path, "l", worker_id="1")
    for _ in range(2):
        ledger.record(
            _event(
                estimated_cost_usd=0.10,
                provider_reported_cost_usd=0.09,
                provider_cost_source="provider_usage_cost",
            )
        )
    ledger.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["provider_reported_total_cost_usd"] == pytest.approx(0.18)
    assert summary["provider_cost_coverage"]["coverage_by_calls"] == 1.0
    assert summary["provider_cost_coverage"]["covered_calls_by_source"] == {
        "provider_usage_cost": 2
    }


def test_partial_provider_coverage_publishes_no_provider_total(tmp_path):
    ledger = UsageLedger(tmp_path, "l", worker_id="1")
    ledger.record(_event(estimated_cost_usd=0.10, provider_reported_cost_usd=0.09))
    ledger.record(_event(estimated_cost_usd=0.10))
    ledger.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["provider_reported_total_cost_usd"] is None
    assert summary["provider_cost_coverage"]["coverage_by_calls"] == pytest.approx(0.5)
    # A missing provider cost is never back-filled from the estimate.
    assert summary["estimated_total_cost_usd"] == pytest.approx(0.20)


def test_unpriced_completed_call_withholds_the_estimated_total(tmp_path):
    ledger = UsageLedger(tmp_path, "l", worker_id="1")
    ledger.record(_event(estimated_cost_usd=0.10))
    ledger.record(
        _event(
            estimated_cost_usd=None,
            estimate_unavailable_reason="UnknownModelPriceError",
        )
    )
    ledger.close()
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["estimated_total_cost_usd"] is None
    assert summary["priced_calls_estimated_cost_usd"] == pytest.approx(0.10)
    assert summary["cost_complete"] is False


def test_task_projection_covers_only_that_task(tmp_path):
    ledger = UsageLedger(tmp_path, "l", worker_id="1")
    ledger.record(_event(task_id="t1", estimated_cost_usd=0.10))
    ledger.record(_event(task_id="t2", estimated_cost_usd=0.30))
    ledger.close()
    projection = UsageAggregator.from_experiment_dir(tmp_path).task_projection("t1")
    assert projection["estimated_total_cost_usd"] == pytest.approx(0.10)
    assert len(projection["usage_event_ids"]) == 1


def test_launch_ids_are_unique_even_within_one_second():
    """Two launches in the same second in the same process must not share a directory."""
    ids = {new_launch_id() for _ in range(50)}
    assert len(ids) == 50


def test_default_snapshot_is_the_dated_one():
    assert (
        get_price_snapshot(DEFAULT_PRICE_SNAPSHOT_ID).snapshot_id
        == DEFAULT_PRICE_SNAPSHOT_ID
    )


def test_completed_status_is_the_default():
    assert _event().status == STATUS_COMPLETED
