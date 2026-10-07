"""What `LLMClient.generate()` records for one call, using fake litellm responses.

These assert the four things the old client threw away: the cache breakdown, the tier the
call actually ran at, the provider's own cost figure, and the fact that a call happened at
all when it failed without a usage payload.
"""

from __future__ import annotations

import pytest

import daedalus.core.llm.client as client_mod
from daedalus.core.llm.client import LLMClient, parse_provider_cost, parse_token_usage
from daedalus.core.logging.usage import UsageError, UsageScope
from daedalus.core.logging.usage_aggregate import UsageAggregator
from daedalus.core.logging.usage_ledger import UsageLedger


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _response(
    prompt=1000,
    completion=100,
    cached=None,
    cache_creation=None,
    reasoning=None,
    service_tier=None,
    usage_cost=None,
    hidden_cost=None,
    model="gpt-5.4",
):
    prompt_details = None
    if cached is not None or cache_creation is not None:
        prompt_details = _Obj(
            cached_tokens=cached, cache_creation_tokens=cache_creation
        )
    completion_details = (
        _Obj(reasoning_tokens=reasoning) if reasoning is not None else None
    )
    usage = _Obj(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )
    if usage_cost is not None:
        usage.cost = usage_cost
    response = _Obj(
        id="req-123",
        model=model,
        usage=usage,
        choices=[_Obj(message=_Obj(content="hi", tool_calls=None))],
    )
    if service_tier is not None:
        response.service_tier = service_tier
    if hidden_cost is not None:
        response._hidden_params = {"response_cost": hidden_cost}
    return response


@pytest.fixture
def fake_completion(monkeypatch):
    """Replace litellm.completion with a scripted queue of responses/exceptions."""
    calls: list[dict] = []
    queue: list = []

    def _completion(**kwargs):
        calls.append(kwargs)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client_mod.litellm, "completion", _completion)
    return _Obj(calls=calls, queue=queue)


def _client(tmp_path, **kwargs):
    kwargs.setdefault("model", "gpt-5.4")
    kwargs.setdefault("ledger", UsageLedger(tmp_path, "launch-1", worker_id="1"))
    kwargs.setdefault("scope", UsageScope(role="solver", task_id="t1"))
    return LLMClient(**kwargs)


def _events(tmp_path):
    return UsageAggregator.from_experiment_dir(tmp_path).events


# ---- token parsing -------------------------------------------------------------------


def test_openai_style_cache_read_is_part_of_prompt_tokens():
    usage = parse_token_usage(_response(prompt=1000, cached=800))
    assert usage.prompt_tokens == 1000
    assert usage.cache_read_tokens == 800
    assert usage.uncached_prompt_tokens == 200


def test_anthropic_style_cache_read_is_added_to_prompt_tokens():
    """There `input_tokens` EXCLUDES cache reads, so the two must be summed."""
    usage = parse_token_usage(_response(prompt=200, cached=800))
    assert usage.prompt_tokens == 1000
    assert usage.uncached_prompt_tokens == 200


def test_cache_write_tokens_are_captured():
    usage = parse_token_usage(_response(prompt=1000, cache_creation=500))
    assert usage.cache_write_tokens == 500


def test_missing_cache_fields_are_distinguishable_from_zero():
    assert parse_token_usage(_response()).cache_details_available is False
    assert parse_token_usage(_response(cached=0)).cache_details_available is True


def test_reasoning_tokens_are_capped_at_the_completion_total():
    usage = parse_token_usage(_response(completion=100, reasoning=400))
    assert usage.reasoning_tokens == 100


def test_no_usage_payload_parses_as_none():
    assert parse_token_usage(_Obj(usage=None)) is None


# ---- provider cost -------------------------------------------------------------------


def test_provider_native_cost_wins_over_litellm():
    cost, source = parse_provider_cost(_response(usage_cost=0.04, hidden_cost=0.09))
    assert (cost, source) == (0.04, "provider_usage_cost")


def test_litellm_response_cost_is_used_when_there_is_no_native_cost():
    cost, source = parse_provider_cost(_response(hidden_cost=0.09))
    assert (cost, source) == (0.09, "litellm_response_cost")


def test_absent_provider_cost_stays_absent():
    assert parse_provider_cost(_response()) == (None, "")


# ---- one recorded call ---------------------------------------------------------------


def test_a_completed_call_writes_one_event_with_full_detail(tmp_path, fake_completion):
    fake_completion.queue.append(
        _response(prompt=1000, cached=800, completion=100, usage_cost=0.001)
    )
    llm = _client(tmp_path)
    resp = llm.generate([{"role": "user", "content": "x"}])

    (event,) = _events(tmp_path)
    assert event.role == "solver" and event.task_id == "t1"
    assert event.tokens.cache_read_tokens == 800
    assert event.request_id == "req-123"
    assert event.provider_reported_cost_usd == 0.001
    assert event.pricing_model_key == "gpt-5.4"
    # flex is requested by default for gpt-5 on OpenAI-direct, and is what it ran at.
    assert event.effective_service_tier == "flex"
    assert event.estimated_cost_usd == pytest.approx(
        (200 * 2.5 + 800 * 0.25 + 100 * 15.0) / 1e6 * 0.5
    )
    assert resp.usage_event_id == event.event_id
    assert resp.prompt_tokens == 1000  # compatibility property


def test_provider_confirmed_tier_overrides_the_requested_one(tmp_path, fake_completion):
    fake_completion.queue.append(_response(service_tier="default"))
    _client(tmp_path).generate([{"role": "user", "content": "x"}])
    (event,) = _events(tmp_path)
    assert event.effective_service_tier == "default"
    assert event.service_tier_confirmed is True


def test_a_call_that_never_asked_for_flex_is_priced_at_standard(
    tmp_path, fake_completion
):
    fake_completion.queue.append(_response(model="gpt-4.1"))
    llm = _client(tmp_path, model="gpt-4.1")
    llm.generate([{"role": "user", "content": "x"}])
    (event,) = _events(tmp_path)
    assert event.requested_service_tier == "flex"  # the client's default...
    assert event.effective_service_tier == "standard"  # ...but gpt-4.1 never sends it


def test_two_roles_on_one_client_are_billed_separately(tmp_path, fake_completion):
    fake_completion.queue.extend([_response(), _response()])
    llm = _client(tmp_path)
    llm.generate([{"role": "user", "content": "x"}])
    llm.generate(
        [{"role": "user", "content": "x"}],
        scope=UsageScope(role="verifier", task_id="t1"),
    )
    roles = sorted(e.role for e in _events(tmp_path))
    assert roles == ["solver", "verifier"]


def test_with_scope_clones_the_client_without_sharing_the_role(
    tmp_path, fake_completion
):
    fake_completion.queue.extend([_response(), _response()])
    llm = _client(tmp_path)
    verifier = llm.with_scope(UsageScope(role="verifier"))
    llm.generate([{"role": "user", "content": "x"}])
    verifier.generate([{"role": "user", "content": "x"}])
    assert sorted(e.role for e in _events(tmp_path)) == ["solver", "verifier"]
    assert llm.scope.role == "solver"  # the original is untouched


def test_a_ledger_without_a_scope_is_refused(tmp_path, fake_completion):
    fake_completion.queue.append(_response())
    llm = LLMClient("gpt-5.4", ledger=UsageLedger(tmp_path, "l", worker_id="1"))
    with pytest.raises(UsageError):
        llm.generate([{"role": "user", "content": "x"}])


def test_an_unpriceable_model_still_records_the_call(tmp_path, fake_completion):
    """The spend happened; only the price is unknown, and the event says which."""
    fake_completion.queue.append(_response(model="mystery-9"))
    llm = _client(tmp_path, model="mystery-9")
    llm.generate([{"role": "user", "content": "x"}])
    (event,) = _events(tmp_path)
    assert event.estimated_cost_usd is None
    assert "UnknownModelPriceError" in event.estimate_unavailable_reason
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["estimated_total_cost_usd"] is None and not summary["cost_complete"]


class _ServerError(Exception):
    status_code = 500


class _BadRequest(Exception):
    status_code = 400


def test_a_timeout_that_may_have_been_billed_is_recorded(
    tmp_path, fake_completion, monkeypatch
):
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    fake_completion.queue.extend([_ServerError(), _response()])
    _client(tmp_path).generate([{"role": "user", "content": "x"}])
    events = _events(tmp_path)
    assert [e.status for e in events] == ["failed_without_usage", "completed"]
    assert events[0].provider_may_have_billed is True
    assert events[0].tokens is None and events[0].estimated_cost_usd is None
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["cost_complete"] is False


def test_a_retried_client_error_is_not_recorded_as_billable(
    tmp_path, fake_completion, monkeypatch
):
    """A 400 produced no tokens, so it is not a billed attempt."""
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    fake_completion.queue.extend([_BadRequest(), _response()])
    _client(tmp_path).generate([{"role": "user", "content": "x"}])
    assert [e.status for e in _events(tmp_path)] == ["completed"]


def test_diagnostic_counters_still_track_tokens(tmp_path, fake_completion):
    fake_completion.queue.append(_response(prompt=1000, cached=800, completion=100))
    llm = _client(tmp_path)
    llm.generate([{"role": "user", "content": "x"}])
    assert (
        llm.total_prompt_tokens,
        llm.total_completion_tokens,
        llm.total_cached_tokens,
    ) == (
        1000,
        100,
        800,
    )
