"""Unified LLM client on top of litellm, with token-usage tracking.

litellm gives one call surface for every provider (OpenAI, Anthropic, vLLM,
Together, Azure, ...). Pass a litellm model string:

    "gpt-5.4-mini"                      # OpenAI (default provider)
    "anthropic/claude-sonnet-4-5"       # Anthropic
    "openai/my-model" + base_url=...    # vLLM / any OpenAI-compatible server

Reasoning-model quirks (drop temperature, use max_completion_tokens, accept
reasoning_effort) are handled by litellm's `drop_params`, so callers pass the
same kwargs for every model. OpenAI-only extras (e.g. service_tier="flex") are
forwarded as-is and dropped on a 400 that names them.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any

import logging as _logging

import litellm

from daedalus.core.logging.cost import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    PricingError,
    get_price_snapshot,
)
from daedalus.core.logging.usage import (
    STATUS_FAILED_WITHOUT_USAGE,
    LLMUsageEvent,
    TokenUsage,
    UsageError,
    UsageScope,
)
from daedalus.core.logging.usage_ledger import UsageLedger

# Let litellm silently drop parameters a given model/provider doesn't support
# (e.g. temperature for gpt-5/o-series), instead of erroring.
litellm.drop_params = True

# Quiet litellm's own console noise. It is loud in two ways that say nothing about our run:
#   - on import it tries to fetch a pricing table from GitHub and warns when that times out,
#     once per worker process. Irrelevant here: costs come from our own table in logging/cost.py.
#   - on every failed request it prints a "Provider List: ... / Give Feedback / Get Help: ..."
#     banner, before `generate` below catches the exception and retries. Transient blips
#     therefore looked like fatal errors in the log while the run carried on fine.
# The exception object is untouched, so a genuine failure still surfaces: `generate` re-raises
# it after exhausting its retries.
litellm.suppress_debug_info = True
litellm.set_verbose = False
_logging.getLogger("LiteLLM").setLevel(_logging.ERROR)


@dataclass
class ToolCallRequest:
    """A tool call requested by the model (function-calling API)."""

    id: str
    name: str
    arguments: str  # raw JSON string; the caller parses it


@dataclass
class LLMResponse:
    """Structured response from an LLM call, carrying its canonical usage record.

    `usage` is the full token breakdown (cache reads AND writes, reasoning, modalities);
    `usage_event_id` links back to the ledger line this call wrote, so a trace can point
    at its spend instead of re-accumulating it.

    `prompt_tokens` / `completion_tokens` / `cached_tokens` remain as read-only
    compatibility properties for call sites not yet migrated to `usage`.
    """

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    tool_calls: list[ToolCallRequest] | None = None
    reasoning: str = ""  # provider reasoning trace, when exposed; empty otherwise
    raw: dict[str, Any] = field(default_factory=dict)
    usage_event_id: str = ""
    provider_reported_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    effective_service_tier: str | None = None
    request_id: str = ""
    requested_model: str = ""
    returned_model: str = ""

    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

    @property
    def cached_tokens(self) -> int:
        return self.usage.cache_read_tokens


class FatalLLMError(RuntimeError):
    """A provider error that every later call will hit too: bad key, or spend cut off.

    Retrying it wastes the backoff, and swallowing it is worse — the callers that fan
    tasks out over a pool deliberately record a failed item and carry on, which for this
    class of error yields a full run of "failures" that are really API errors, and a
    confident-looking score built from none of the work. So it is raised on the first
    occurrence and re-raised by those pools instead of being counted.
    """


# Wording providers use when spend, not the request, is what was refused. Matched against
# the message because litellm does not expose a status code on every wrapped provider
# error (OpenRouter's budget refusal arrives as an APIError whose text carries the 403).
_SPEND_REFUSED = (
    "budget",
    "insufficient credit",
    "insufficient_quota",
    "exceeded your current quota",
    "quota exceeded",
    "billing",
)


# Refusals that NAME spend but clear on their own. OpenRouter's in-flight budget is the
# real case: 12 parallel workers overran it, the body says "exceed your available credits
# given your current in-flight requests" and carries Retry-After, and treating that as
# fatal killed a generation run that would have continued after a 120s wait. A transient
# marker always wins over the spend words below.
_SPEND_TRANSIENT = (
    "in_flight",
    "in-flight",
    "retry-after",
    "retry after",
    "settle",
    # An UPSTREAM rate limit relayed by OpenRouter. Its body carries
    # provider_error_code "insufficient_quota", which matches _SPEND_REFUSED, and that
    # killed a generation run on its final session with 7 of 8 tasks already banked.
    # Matched on wording, not on the 429 status: OpenAI returns 429 for a genuinely
    # exhausted quota, which IS permanent, so the status alone cannot separate the two.
    "rate-limited",
    "rate limited",
    "retry shortly",
    "shared_pool",
)


def _is_fatal_provider_error(exc: Exception) -> bool:
    """True for an authentication or spend refusal — permanent for the whole run.

    A 401 is always fatal. A 402/403 is fatal only when the text names spend or billing:
    the same codes also carry per-request refusals (content policy, a model the key may
    not use), and those must stay a single failed item.
    """
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status == 401 or "authenticationerror" in type(exc).__name__.lower():
        return True
    if any(w in msg for w in _SPEND_TRANSIENT):
        return False
    if any(w in msg for w in _SPEND_REFUSED):
        return True
    return status in (402, 403) and "403" in msg and any(
        w in msg for w in _SPEND_REFUSED
    )


def _looks_openai_reasoning(model: str) -> bool:
    """Heuristic: an OpenAI reasoning model (gpt-5/o-series) that supports flex."""
    m = model.split("/")[-1]
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _int(value: Any) -> int:
    """A provider count as a non-negative int; missing/None becomes 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def parse_token_usage(response: Any) -> TokenUsage | None:
    """The canonical token breakdown for one litellm response, or None if it reported none.

    Two provider-semantics normalizations, both explicit rather than clamped away:

      * Anthropic-style `input_tokens` EXCLUDES cache reads, while OpenAI's
        `prompt_tokens` INCLUDES them. When the reported cache reads exceed the reported
        prompt tokens we are in the first case, so prompt tokens are the sum of the two —
        which is what the pricing formula (uncached input + cache reads) then charges.
      * `reasoning_tokens` is a detail of the completion total, so it is capped at it. It
        is never priced separately, so the cap cannot change a cost.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    def _detail(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    cache_read_raw = _detail(prompt_details, "cached_tokens")
    if cache_read_raw is None:
        cache_read_raw = getattr(usage, "cache_read_input_tokens", None)
    cache_write_raw = _detail(prompt_details, "cache_creation_tokens")
    if cache_write_raw is None:
        cache_write_raw = getattr(usage, "cache_creation_input_tokens", None)
    # Distinguish "the provider said nothing about caching" from "it said zero": only the
    # second supports a cache-savings claim.
    cache_details_available = cache_read_raw is not None or cache_write_raw is not None

    prompt_tokens = _int(getattr(usage, "prompt_tokens", 0))
    cache_read = _int(cache_read_raw)
    if cache_read > prompt_tokens:
        prompt_tokens = prompt_tokens + cache_read

    completion_tokens = _int(getattr(usage, "completion_tokens", 0))
    reasoning = min(
        _int(_detail(completion_details, "reasoning_tokens")), completion_tokens
    )

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=_int(cache_write_raw),
        reasoning_tokens=reasoning,
        audio_input_tokens=_int(_detail(prompt_details, "audio_tokens")),
        audio_output_tokens=_int(_detail(completion_details, "audio_tokens")),
        image_input_tokens=_int(_detail(prompt_details, "image_tokens")),
        image_output_tokens=_int(_detail(completion_details, "image_tokens")),
        cache_details_available=cache_details_available,
    )


def parse_provider_cost(response: Any) -> tuple[float | None, str]:
    """(cost, source) as reported BY THE PROVIDER for this request, else (None, "").

    Checked in a fixed order of explicitly supported fields:
      1. `usage.cost` — OpenRouter's own charge for the request. Authoritative there,
         because OpenRouter routes across upstream providers whose rates differ, so a
         static list price is only a ceiling.
      2. litellm's computed `_hidden_params["response_cost"]`.
    Never falls back to our local estimate: a missing provider cost stays missing.
    """
    usage = getattr(response, "usage", None)
    native = getattr(usage, "cost", None) if usage is not None else None
    if isinstance(native, (int, float)):
        return float(native), "provider_usage_cost"
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        litellm_cost = hidden.get("response_cost")
        if isinstance(litellm_cost, (int, float)):
            return float(litellm_cost), "litellm_response_cost"
    return None, ""


class LLMClient:
    """Unified LLM interface via litellm, writing one usage event per billed call.

    An experiment-associated client is built with a `UsageLedger` and a base `UsageScope`.
    The scope says WHO the spend belongs to (experiment, task, role); `generate(scope=...)`
    narrows it per call, which is how one client shared by two roles still bills each role separately. A client
    built without a ledger performs UNTRACKED work and is only for ad-hoc scripts.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        ledger: UsageLedger | None = None,
        scope: UsageScope | None = None,
        price_snapshot_id: str | None = None,
        provider_routing: dict[str, Any] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        # Default reasoning effort for every call (None = provider default); a per-call
        # reasoning_effort kwarg to generate() overrides it.
        self.reasoning_effort = reasoning_effort
        self.ledger = ledger
        self.scope = scope
        self.price_snapshot_id = price_snapshot_id
        # OpenRouter's `provider` routing object (see config.ProviderRoutingConfig).
        # Only OpenRouter understands it, and it rejects unknown top-level fields, so
        # it is sent for an "openrouter/" model and dropped for every other provider.
        self.provider_routing = provider_routing or None
        # Diagnostics only. These are NOT the source of truth for run cost: they cannot
        # separate roles on a shared client, do not survive a crash, and cannot carry a
        # per-call service tier or provider-reported cost. The ledger does all four.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0

        # A model string may name its own provider ("openrouter/qwen/...",
        # "anthropic/claude-..."). Only fall back to the OPENAI_* env vars when it does
        # NOT, or when it says "openai/" (the vLLM / OpenAI-compatible-server case, which
        # needs OPENAI_BASE_URL). Passing OPENAI_API_KEY to another provider is not a
        # harmless no-op: litellm sends the explicit key instead of resolving that
        # provider's own env var, and OpenRouter answered 401 "Missing Authentication
        # header" for every call. Explicit api_key / base_url arguments always win.
        self._provider = model.split("/")[0].lower() if "/" in model else "openai"
        _openai_env = self._provider == "openai"
        self.base_url = base_url or (
            os.environ.get("OPENAI_BASE_URL") if _openai_env else None
        )
        self.api_key = api_key or (
            os.environ.get("OPENAI_API_KEY") if _openai_env else None
        )
        # Flex processing (OpenAI): ~50% cheaper, slower best-effort latency.
        # Enabled by default for OpenAI-direct reasoning models; opt out only by explicitly
        # setting OPENAI_SERVICE_TIER=auto or service_tier="auto".
        self.service_tier = service_tier or os.environ.get(
            "OPENAI_SERVICE_TIER", "flex"
        )
        # service_tier is OpenAI-only: never send it behind a custom base_url or to
        # another provider (OpenRouter rejects unknown top-level fields).
        self._openai_direct = _openai_env and not self.base_url
        # Flex requests queue, so allow a generous default timeout.
        #
        # Non-OpenAI providers used to get 120s, which was too short and failed SILENTLY:
        # one measured glm-5.3 call returning 7,939 completion tokens took 104.7s, so an
        # explorer call carrying a large history crosses 120s routinely. Each crossing
        # timed out, the retry loop re-sent the SAME expensive call up to 6 times with
        # backoff (~13 min per call), and a provider may still bill a request that timed
        # out after generating. A 90-session AppWorld generation run spent 70 minutes in
        # that loop with no log line, because the log only writes when a turn completes.
        # 600s is above the slowest legitimate call observed by a wide margin, and a real
        # hang still ends — 6 attempts cap the total.
        self.timeout = (
            timeout
            if timeout is not None
            else (
                900.0
                if (self.service_tier == "flex" and self._openai_direct)
                else 600.0
            )
        )

    def with_scope(self, scope: UsageScope) -> "LLMClient":
        """This client's ledger and transport under a different base scope."""
        clone = copy.copy(self)
        clone.scope = scope
        return clone

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        scope: UsageScope | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion, recording its usage event before returning.

        Args:
            messages: chat messages (may include assistant `tool_calls` and
                role-"tool" result messages, OpenAI format).
            tools: OpenAI function-calling tool schemas; enables tool calls.
            tool_choice: tool-choice mode when tools are given (default "auto").
            scope: whose spend this call is; overrides the client's base scope.
            kwargs: max_tokens, temperature, reasoning_effort, service_tier.
        """
        kwargs.pop(
            "reasoning_summary", None
        )  # legacy no-op (Responses-API path removed)
        call_scope = scope or self.scope
        if self.ledger is not None and call_scope is None:
            raise UsageError(
                f"LLMClient({self.model!r}) has a usage ledger but no UsageScope: pass "
                f"scope= to generate() or build the client with scope=. Cost roles are "
                f"never inferred from which client made the call."
            )

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": self.timeout,
        }
        if self.provider_routing and self._provider == "openrouter":
            # extra_body reaches OpenRouter's request body through litellm. A bad provider
            # tag comes back as a 404 naming its routing_funnel, and a filter no upstream
            # satisfies fails the call outright — both are louder than silently running on
            # weights the config did not ask for, so neither is retried away below.
            api_kwargs["extra_body"] = {"provider": dict(self.provider_routing)}
        if self.base_url:
            api_kwargs["api_base"] = self.base_url
        if self.api_key:
            api_kwargs["api_key"] = self.api_key
        if kwargs.get("max_tokens") is not None:
            api_kwargs["max_tokens"] = kwargs["max_tokens"]
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort:
            api_kwargs["reasoning_effort"] = effort
        if tools is not None:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = tool_choice or "auto"

        # Flex only for OpenAI-direct reasoning models.
        tier = kwargs.get("service_tier", self.service_tier)
        requested_tier = tier
        if not (self._openai_direct and _looks_openai_reasoning(self.model)):
            tier = None
        elif tier == "flex" and not _looks_openai_reasoning(self.model):
            tier = None

        max_attempts = 6
        last_exc: Exception | None = None
        response = None
        attempt = -1
        while True:
            attempt += 1
            if attempt >= max_attempts:
                break
            call_kwargs = dict(api_kwargs)
            if tier is not None:
                call_kwargs["service_tier"] = tier
            try:
                response = litellm.completion(**call_kwargs)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001
                last_exc = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                fatal = _is_fatal_provider_error(e)
                # A request that failed WITHOUT a usage payload may still have been billed
                # (a timeout or 5xx can arrive after the model generated). Record it as an
                # unknown-cost attempt so the run reports itself incomplete instead of
                # quietly under-counting; a 4xx never produced tokens, so it is not billed.
                self._record_failure(
                    call_scope,
                    error=e,
                    requested_tier=requested_tier,
                    sent_tier=tier,
                    final=(fatal or attempt >= max_attempts - 1),
                )
                if fatal:
                    raise FatalLLMError(
                        f"{self.model}: the provider refused the call and will refuse "
                        f"every later one — fix the key or the spend limit and re-run. "
                        f"{type(e).__name__}: {e}"
                    ) from e
                # Endpoint/model rejected service_tier or reasoning_effort (400) → drop the
                # offending parameter and retry. These grant one EXTRA attempt each, rather
                # than consuming one: dropping a parameter is not a failed try, and on the
                # last attempt the old code `continue`d past the raise below, fell out of
                # the loop, and hit an unbound `response` — turning a clear API error into
                # an UnboundLocalError.
                if tier is not None and status == 400 and "service_tier" in msg:
                    tier = None
                    max_attempts += 1
                    continue
                if (
                    "reasoning_effort" in api_kwargs
                    and status == 400
                    and "reasoning_effort" in msg
                ):
                    api_kwargs.pop("reasoning_effort")
                    max_attempts += 1
                    continue
                # Same shape for temperature: the gpt-5.6 family accepts only its default
                # (1), so a role that pins 0.0 for determinism — the generation judge and
                # the extraction writer both do — is refused outright. Drop it and let the
                # provider default apply, rather than failing the session.
                if (
                    "temperature" in api_kwargs
                    and status == 400
                    and "temperature" in msg
                ):
                    api_kwargs.pop("temperature")
                    max_attempts += 1
                    continue
                if attempt >= max_attempts - 1:
                    raise last_exc from None
                time.sleep(min(2**attempt, 30))

        if response is None:
            # Every attempt was consumed by parameter-dropping retries and none succeeded.
            raise last_exc if last_exc else RuntimeError(
                f"LLMClient({self.model!r}): no response and no exception — unreachable"
            )
        return self._parse(response, call_scope, requested_tier, tier)

    # ---- usage recording -------------------------------------------------------------

    def _effective_tier(
        self, response: Any, sent_tier: str | None
    ) -> tuple[str | None, bool]:
        """The tier the call ACTUALLY ran at, and whether the provider confirmed it.

        Order matters. The provider's echoed `service_tier` wins; otherwise the tier we
        successfully sent is what it ran at; and if we sent none at all, the call ran at
        the provider's standard tier — we know that, because asking for the flex discount
        is something only this client does.
        """
        reported = getattr(response, "service_tier", None)
        if isinstance(reported, str) and reported:
            return reported, True
        if sent_tier:
            return sent_tier, False
        return "standard", False

    def _estimate(
        self, model: str, tokens: TokenUsage | None, effective_tier: str | None
    ) -> tuple[float | None, str, str]:
        """(estimated_cost, pricing_model_key, unavailable_reason)."""
        if tokens is None:
            return None, "", "provider returned no usage payload"
        try:
            snapshot = get_price_snapshot(self.price_snapshot_id)
            record = snapshot.resolve(model)
            return snapshot.price_call(model, tokens, effective_tier), record.key, ""
        except PricingError as e:
            return None, "", f"{type(e).__name__}: {e}"

    def _record_failure(
        self,
        scope: UsageScope | None,
        error: Exception,
        requested_tier: str | None,
        sent_tier: str | None,
        final: bool,
    ) -> None:
        status = getattr(error, "status_code", None)
        client_error = isinstance(status, int) and 400 <= status < 500
        may_have_billed = not client_error
        if self.ledger is None or scope is None:
            return
        if not may_have_billed and not final:
            return  # a retried 400 produced no tokens and no bill: nothing to record
        self.ledger.record(
            LLMUsageEvent.from_scope(
                scope,
                requested_model=self.model,
                status=STATUS_FAILED_WITHOUT_USAGE,
                provider=self._provider,
                requested_service_tier=requested_tier,
                effective_service_tier=sent_tier,
                error_type=type(error).__name__,
                provider_may_have_billed=may_have_billed,
                price_snapshot_id=self.price_snapshot_id or DEFAULT_PRICE_SNAPSHOT_ID,
            )
        )

    def _parse(
        self,
        response: Any,
        scope: UsageScope | None = None,
        requested_tier: str | None = None,
        sent_tier: str | None = None,
    ) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning_content", None) or ""

        tool_calls: list[ToolCallRequest] | None = None
        raw_tcs = getattr(message, "tool_calls", None)
        if raw_tcs:
            tool_calls = [
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
                for tc in raw_tcs
            ]

        tokens = parse_token_usage(response)
        returned_model = getattr(response, "model", "") or self.model
        effective_tier, tier_confirmed = self._effective_tier(response, sent_tier)
        provider_cost, provider_cost_source = parse_provider_cost(response)
        # The REQUESTED model is what the price snapshot is keyed on: it is the string the
        # config chose and the one the preflight validated. The returned model is kept
        # beside it so a provider silently serving something else stays visible.
        estimated, price_key, estimate_reason = self._estimate(
            self.model, tokens, effective_tier
        )

        if tokens is not None:
            self.total_prompt_tokens += tokens.prompt_tokens
            self.total_completion_tokens += tokens.completion_tokens
            self.total_cached_tokens += tokens.cache_read_tokens

        event_id = ""
        if self.ledger is not None and scope is not None:
            event = self.ledger.record(
                LLMUsageEvent.from_scope(
                    scope,
                    requested_model=self.model,
                    returned_model=returned_model,
                    provider=self._provider,
                    request_id=str(getattr(response, "id", "") or ""),
                    requested_service_tier=requested_tier,
                    effective_service_tier=effective_tier,
                    service_tier_confirmed=tier_confirmed,
                    tokens=tokens,
                    provider_reported_cost_usd=provider_cost,
                    provider_cost_source=provider_cost_source,
                    estimated_cost_usd=estimated,
                    price_snapshot_id=self.price_snapshot_id
                    or DEFAULT_PRICE_SNAPSHOT_ID,
                    pricing_model_key=price_key,
                    estimate_unavailable_reason=estimate_reason,
                )
            )
            event_id = event.event_id

        return LLMResponse(
            content=content,
            usage=tokens or TokenUsage(),
            model=returned_model,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage_event_id=event_id,
            provider_reported_cost_usd=provider_cost,
            estimated_cost_usd=estimated,
            effective_service_tier=effective_tier,
            request_id=str(getattr(response, "id", "") or ""),
            requested_model=self.model,
            returned_model=returned_model,
        )

    def reset_usage(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
