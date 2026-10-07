"""Versioned LLM price snapshots and local cost estimation.

Three properties this module is built around, each a real defect in the previous
hand-maintained tuple table:

  1. EXACT resolution. A model resolves through its canonical key or an explicitly
     declared alias, and nothing else. The old bidirectional prefix match priced
     `gpt-5.4-nano` at the `gpt-5.4` rate (6x too high) purely because one string was a
     prefix of the other.
  2. NO SILENT ZERO. An unknown model raises `UnknownModelPriceError`. The preflight
     (`daedalus.core.logging.preflight`) resolves every configured model before any paid
     work starts, so the failure lands before the spend, not in the summary.
  3. EXPLICIT TIER AND CACHE RATES. Flex is 50% of standard, so a call's effective tier
     changes its price; the tier is passed in per call (LLMClient knows when it fell back
     from flex to auto) instead of being re-read from the environment at aggregation time.
     Cache READS and cache WRITES are separate rates.

Prices are per 1M tokens, in USD, and live in a dated snapshot. Recomputing a historical
run means selecting its snapshot, not editing today's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from daedalus.core.logging.usage import TokenUsage

# Tier names that mean "the provider's normal, undiscounted service".
STANDARD_TIERS: frozenset[str] = frozenset(
    {"standard", "auto", "default", "priority_off"}
)


class PricingError(ValueError):
    """Base class for anything that makes a call unpriceable locally."""


class UnknownModelPriceError(PricingError):
    """No exact price record (canonical key or declared alias) for this model."""


class UnknownServiceTierError(PricingError):
    """The effective service tier is unknown, or unsupported by the price record."""


class UnpriceableModalityError(PricingError):
    """The call used a modality the price record has no rate for."""


@dataclass(frozen=True)
class LongContextTier:
    """Rates that replace a record's base rates for a LARGE SINGLE REQUEST.

    Some models re-price a request whose prompt crosses a threshold: gpt-5.4 above 272k
    prompt tokens costs $5.00/Mtok in instead of $2.50, and $22.50 out instead of $15.00.
    The threshold is per REQUEST, so it is decided by one call's own prompt size — never by
    a task's or a run's summed tokens, which cross it constantly without any call doing so.
    """

    min_prompt_tokens: int
    input_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float
    output_per_mtok: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_prompt_tokens": self.min_prompt_tokens,
            "input_per_mtok": self.input_per_mtok,
            "cache_read_per_mtok": self.cache_read_per_mtok,
            "cache_write_per_mtok": self.cache_write_per_mtok,
            "output_per_mtok": self.output_per_mtok,
        }


@dataclass(frozen=True)
class PriceRecord:
    """Rates for one model, per 1M tokens, at the STANDARD service tier.

    `cache_read_per_mtok` is the discounted rate for prompt tokens served from the
    provider's prompt cache; `cache_write_per_mtok` is what the provider charges to PUT
    tokens into that cache (0.0 where caching is implicit and free, as on OpenAI and on
    every OpenRouter model listed here; 1.25x input on Anthropic's explicit cache).

    `tier_multipliers` scales every rate for a non-standard tier — flex processing bills
    at exactly 50% of standard, input, cache and output alike. A record with a non-empty
    `tier_multipliers` REFUSES to price a call whose effective tier is unknown, rather
    than guessing the cheaper or dearer one.
    """

    key: str
    provider: str
    input_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float
    output_per_mtok: float
    source_url: str
    effective_date: str
    aliases: tuple[str, ...] = ()
    supported_tiers: frozenset[str] = STANDARD_TIERS
    tier_multipliers: Mapping[str, float] = field(default_factory=dict)
    currency: str = "USD"
    # Modality rates. None means "this record cannot price that modality" — a call that
    # reports such tokens is rejected instead of being under-reported.
    image_input_per_mtok: float | None = None
    image_output_per_mtok: float | None = None
    audio_input_per_mtok: float | None = None
    audio_output_per_mtok: float | None = None
    # Set only for the models that publish a large-request surcharge (see LongContextTier).
    long_context: LongContextTier | None = None

    def rates_for(self, prompt_tokens: int) -> tuple[float, float, float, float]:
        """(input, cache_read, cache_write, output) for a request of this prompt size."""
        tier = self.long_context
        if tier is not None and prompt_tokens >= tier.min_prompt_tokens:
            return (
                tier.input_per_mtok,
                tier.cache_read_per_mtok,
                tier.cache_write_per_mtok,
                tier.output_per_mtok,
            )
        return (
            self.input_per_mtok,
            self.cache_read_per_mtok,
            self.cache_write_per_mtok,
            self.output_per_mtok,
        )

    def tier_multiplier(self, effective_tier: str | None) -> float:
        if not self.tier_multipliers:
            # No tier-dependent pricing: an unknown tier cannot change the number.
            return 1.0
        if effective_tier is None:
            raise UnknownServiceTierError(
                f"{self.key} is tier-priced ({sorted(self.tier_multipliers)}) but the "
                f"effective service tier of the call is unknown"
            )
        if effective_tier in self.tier_multipliers:
            return self.tier_multipliers[effective_tier]
        if effective_tier in self.supported_tiers:
            return 1.0
        raise UnknownServiceTierError(
            f"{self.key} has no rate for service tier {effective_tier!r}; "
            f"known: {sorted(set(self.supported_tiers) | set(self.tier_multipliers))}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "input_per_mtok": self.input_per_mtok,
            "cache_read_per_mtok": self.cache_read_per_mtok,
            "cache_write_per_mtok": self.cache_write_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "supported_tiers": sorted(self.supported_tiers),
            "tier_multipliers": dict(self.tier_multipliers),
            "currency": self.currency,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
            "long_context": self.long_context.to_dict() if self.long_context else None,
        }


class PriceSnapshot:
    """An immutable, dated set of price records with exact-only model resolution."""

    def __init__(self, snapshot_id: str, records: list[PriceRecord]):
        self.snapshot_id = snapshot_id
        self._records: dict[str, PriceRecord] = {}
        self._aliases: dict[str, str] = {}
        for record in records:
            if record.key in self._records:
                raise PricingError(
                    f"duplicate price key {record.key!r} in {snapshot_id}"
                )
            self._records[record.key] = record
            for alias in record.aliases:
                owner = self._aliases.get(alias)
                if owner is not None and owner != record.key:
                    raise PricingError(
                        f"alias {alias!r} in {snapshot_id} is claimed by both "
                        f"{owner!r} and {record.key!r}"
                    )
                if alias in self._records and alias != record.key:
                    raise PricingError(
                        f"alias {alias!r} in {snapshot_id} collides with canonical key "
                        f"{alias!r}"
                    )
                self._aliases[alias] = record.key

    def __contains__(self, model: str) -> bool:
        return model in self._records or model in self._aliases

    @property
    def model_keys(self) -> list[str]:
        return sorted(self._records)

    def resolve(self, model: str) -> PriceRecord:
        """The price record for `model`: exact key, then exact declared alias, else raise."""
        if not model:
            raise UnknownModelPriceError("empty model name has no price record")
        record = self._records.get(model)
        if record is not None:
            return record
        canonical = self._aliases.get(model)
        if canonical is not None:
            return self._records[canonical]
        raise UnknownModelPriceError(
            f"no price record for model {model!r} in price snapshot {self.snapshot_id!r}. "
            f"Add it to daedalus/core/logging/cost.py (or declare it as an alias of an "
            f"existing record) — prices are never inferred from a similar name."
        )

    def price_call(
        self, model: str, tokens: TokenUsage, effective_tier: str | None
    ) -> float:
        """USD for one call: uncached input + cache reads + cache writes + output.

        Reasoning tokens are already inside `completion_tokens` and are NOT charged again.
        """
        record = self.resolve(model)
        mult = record.tier_multiplier(effective_tier)

        def _modality(count: int, rate: float | None, name: str) -> float:
            if not count:
                return 0.0
            if rate is None:
                raise UnpriceableModalityError(
                    f"{record.key} reported {count} {name} tokens but the price snapshot "
                    f"{self.snapshot_id!r} has no {name} rate for it"
                )
            return count * rate

        # The long-context surcharge is decided by THIS call's prompt size.
        input_rate, cache_read_rate, cache_write_rate, output_rate = record.rates_for(
            tokens.prompt_tokens
        )
        micro = (
            tokens.uncached_prompt_tokens * input_rate
            + tokens.cache_read_tokens * cache_read_rate
            + tokens.cache_write_tokens * cache_write_rate
            + tokens.completion_tokens * output_rate
            + _modality(
                tokens.image_input_tokens, record.image_input_per_mtok, "image input"
            )
            + _modality(
                tokens.image_output_tokens, record.image_output_per_mtok, "image output"
            )
            + _modality(
                tokens.audio_input_tokens, record.audio_input_per_mtok, "audio input"
            )
            + _modality(
                tokens.audio_output_tokens, record.audio_output_per_mtok, "audio output"
            )
        )
        return micro * mult / 1_000_000


# --------------------------------------------------------------------------------------
# Snapshot 2026-08-31
# --------------------------------------------------------------------------------------
# Flex processing (https://developers.openai.com/api/docs/guides/flex-processing) bills at
# exactly 50% of standard — input, cached input and output alike — for OpenAI's gpt-5/o-series
# reasoning models, which is why those records carry {"flex": 0.5} instead of a duplicated
# rate table. LLMClient requests flex by default for them and preserves that tier across
# retries; it is the EFFECTIVE tier that reaches price_call().
_FLEX = {"flex": 0.5}
_FLEX_TIERS = STANDARD_TIERS | {"flex"}

_OPENAI = "https://platform.openai.com/docs/pricing"
_ANTHROPIC = "https://www.anthropic.com/pricing"
_OPENROUTER = "https://openrouter.ai/api/v1/models"


def _openai(
    key: str,
    inp: float,
    cache_read: float,
    out: float,
    flex: bool = False,
    pinned: tuple[str, ...] = (),
    cache_write: float = 0.0,
    long_context: LongContextTier | None = None,
) -> PriceRecord:
    """An OpenAI-direct record. Prompt caching there is usually implicit and free to write,
    which is why `cache_write` defaults to 0.0 — pass it only where the model publishes a
    cache-write rate of its own (gpt-5.6-luna does).

    `pinned` names date-pinned ids of the SAME model at the SAME rates (e.g.
    "gpt-4.1-2025-04-14"), which is the only case where a differently-spelled model may
    share a record: the price is known to be identical, not assumed to be.
    """
    return PriceRecord(
        key=key,
        provider="openai",
        input_per_mtok=inp,
        cache_read_per_mtok=cache_read,
        cache_write_per_mtok=cache_write,
        output_per_mtok=out,
        aliases=(f"openai/{key}", *pinned, *(f"openai/{a}" for a in pinned)),
        supported_tiers=_FLEX_TIERS if flex else STANDARD_TIERS,
        tier_multipliers=dict(_FLEX) if flex else {},
        long_context=long_context,
        effective_date="2026-08-31",
        source_url=_OPENAI,
    )


def _openrouter(key: str, inp: float, cache_read: float, out: float) -> PriceRecord:
    """An OpenRouter record, keyed by the litellm string; the bare id is an alias.

    The config carries "openrouter/vendor/model" while the API response's `model` field
    carries "vendor/model", and both must resolve to the same rates. These are OpenRouter
    LIST prices: routing across upstream providers means the real charge can come in
    under them, so `provider_reported_cost_usd` (which OpenRouter does return) is the
    authoritative figure and this is the ceiling.
    """
    bare = key.removeprefix("openrouter/")
    return PriceRecord(
        key=key,
        provider="openrouter",
        input_per_mtok=inp,
        cache_read_per_mtok=cache_read,
        cache_write_per_mtok=0.0,
        output_per_mtok=out,
        aliases=(bare,),
        effective_date="2026-08-31",
        source_url=_OPENROUTER,
    )


_SNAPSHOT_2026_08_31 = PriceSnapshot(
    "2026-08-31",
    [
        # ---- Anthropic: cache writes are billed at 1.25x input (5-minute TTL). ----
        PriceRecord(
            key="claude-sonnet-4-20250514",
            provider="anthropic",
            input_per_mtok=3.0,
            cache_read_per_mtok=0.30,
            cache_write_per_mtok=3.75,
            output_per_mtok=15.0,
            aliases=("anthropic/claude-sonnet-4-20250514",),
            effective_date="2026-08-31",
            source_url=_ANTHROPIC,
        ),
        PriceRecord(
            key="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_per_mtok=3.0,
            cache_read_per_mtok=0.30,
            cache_write_per_mtok=3.75,
            output_per_mtok=15.0,
            aliases=("anthropic/claude-3-5-sonnet-20241022",),
            effective_date="2026-08-31",
            source_url=_ANTHROPIC,
        ),
        # ---- OpenAI ----
        _openai("gpt-4o", 2.5, 1.25, 10.0),
        _openai("gpt-4o-mini", 0.15, 0.075, 0.60),
        # gpt-4.1-2025-04-14 is the date-pinned snapshot tau2's user simulator defaults
        # to (Tau2Config.user_llm); same model, same published rates.
        _openai("gpt-4.1", 2.0, 0.50, 8.0, pinned=("gpt-4.1-2025-04-14",)),
        _openai("gpt-4.1-mini", 0.4, 0.10, 1.6),
        _openai("gpt-5-mini", 0.25, 0.025, 2.0, flex=True),
        _openai("gpt-5.4-mini", 0.75, 0.075, 4.5, flex=True),
        _openai(
            "gpt-5.4",
            2.5,
            0.25,
            15.0,
            flex=True,
            # >272k prompt tokens in one request doubles input and output. Its override
            # publishes no cache-WRITE rate, so that stays 0.0 as at the base tier.
            long_context=LongContextTier(272_000, 5.0, 0.5, 0.0, 22.5),
        ),
        # Rates read off OpenRouter's openai/* routes, which republish OpenAI's list prices:
        # every model above that appears there matches this snapshot exactly.
        #
        _openai("gpt-5.4-nano", 0.2, 0.02, 1.25, flex=True),
        # luna is the one OpenAI model here that charges to WRITE the prompt cache.
        #
        # NOTE luna and gpt-5.4 (and NO other model in this snapshot) publish a
        # LONG-CONTEXT OVERRIDE that this record cannot express: for a request whose PROMPT
        # exceeds 272k tokens, every rate changes —
        #     gpt-5.6-luna  prompt $0.40  cache-read $0.04  output $1.80
        #     gpt-5.4       prompt $5.00  cache-read $0.50  output $22.50
        # so one request that large is UNDER-estimated by ~2x. The threshold is PER
        # REQUEST, which is why it cannot be applied to a task's summed prompt tokens: on
        # the traces in outputs/, 592 tasks exceed 272k in total while only 3 individual
        # turns do. The ledger records prompt_tokens per call, so implementing this would
        # be exact going forward.
        _openai(
            "gpt-5.6-luna",
            0.2,
            0.02,
            1.2,
            flex=True,
            cache_write=0.25,
            long_context=LongContextTier(272_000, 0.4, 0.04, 0.5, 1.8),
        ),
        # terra is the strong 5.6 model, and the one the explorer / judge / extraction
        # roles use. Like luna it charges to WRITE the prompt cache. Rates read off
        # OpenRouter's openai/gpt-5.6-terra route on 2026-09-09, which reproduces the luna
        # and gpt-5.4 rows above exactly. No long-context override was published for it
        # there, so none is modelled here — unlike luna and gpt-5.4, which do publish one.
        _openai("gpt-5.6-terra", 2.0, 0.2, 12.0, flex=True, cache_write=2.5),
        # ---- OpenRouter ----
        _openrouter("openrouter/qwen/qwen3.8-27b", 0.42, 0.085, 3.00),
        _openrouter("openrouter/qwen/qwen3.6-35b-a3b", 0.10, 0.05, 0.90),
        _openrouter("openrouter/nvidia/nemotron-3.5-lightning", 0.08, 0.04, 0.2),
        _openrouter("openrouter/stepfun/step-3.7-flash", 0.2, 0.04, 1.15),
        _openrouter("openrouter/deepseek/deepseek-v4-flash", 0.06706, 0.01341, 0.13412),
        _openrouter("openrouter/google/gemini-3.6-flash", 0.75, 0.075, 3.75),
        _openrouter("openrouter/z-ai/glm-4.6", 0.5, 0.1, 2),
        _openrouter("openrouter/mistralai/mistral-medium-3-5", 1.5, 1.5, 7.5),
        _openrouter("openrouter/openai/gpt-oss-120b", 0.03, 0.03, 0.17),
        # The two DeepSeek Flash revisions are DIFFERENT models at different prices, which is
        # exactly what exact-only resolution protects: "-0731" no longer collapses onto the base.
        _openrouter("openrouter/deepseek/deepseek-v4-flash-0731", 0.065, 0.016, 0.18),
        _openrouter("openrouter/nvidia/nemotron-3-super-120b-a12b", 0.085, 0.085, 0.40),
        _openrouter("openrouter/xiaomi/mimo-v2.5", 0.14, 0.0028, 0.28),
        _openrouter("openrouter/tencent/hy3", 0.132, 0.033, 0.528),
        # Gemma 4 31B's cache-read rate EQUALS its prompt rate, i.e. no cache discount at
        # all — expensive here, where ~85% of prompt tokens are cache hits.
        _openrouter("openrouter/google/gemma-4-31b-it", 0.10, 0.10, 0.34),
        _openrouter("openrouter/minimax/minimax-m2.7", 0.30, 0.06, 1.20),
        # The :free tier needs its own exact key so a stray free-tier run is not billed at
        # the paid rate below it. (It is unusable for agentic rollouts anyway: its shared
        # upstream pool 429s a turn or two in, even at --parallel 1.)
        _openrouter("openrouter/z-ai/glm-5.2:free", 0.0, 0.0, 0.0),
        # Paid glm-5.2 bills CACHED prompt tokens at the FULL input rate.
        _openrouter("openrouter/z-ai/glm-5.2", 0.966, 0.966, 3.036),
        # Flagship/small open-weight pairs, added 2026-09-07 for the AppWorld
        # open-model baselines. Rates read from the OpenRouter models API on that date.
        # glm-5.3-flash does give a cache discount (0.015 vs 0.075), unlike 5.2 above.
        _openrouter("openrouter/z-ai/glm-5.3-flash", 0.075, 0.015, 0.25),
        _openrouter("openrouter/qwen/qwen3.8-flash", 0.15, 0.016, 0.47),
        _openrouter("openrouter/thinkingmachines/inkling-small", 0.45, 0.10, 1.20),
        # The two DAEDALUS-on-open-models families, added 2026-09-08. DeepSeek publishes
        # date-pinned ids, so those are what the configs name; z-ai does not — "glm-5.3"
        # is the only routable id, and its canonical slug (glm-5.3-20260816) moves when
        # z-ai reposts the model, so a GLM run records the date it ran, not a pin.
        _openrouter("openrouter/deepseek/deepseek-v4-pro-0813", 1.0494, 0.03498, 3.1482),
        _openrouter("openrouter/z-ai/glm-5.3", 1.40, 0.26, 4.40),
        # ---- Embeddings: input-only, no caching, no completion tokens. ----
        PriceRecord(
            key="text-embedding-3-large",
            provider="openai",
            input_per_mtok=0.13,
            cache_read_per_mtok=0.13,
            cache_write_per_mtok=0.0,
            output_per_mtok=0.0,
            aliases=("openai/text-embedding-3-large",),
            effective_date="2026-08-31",
            source_url=_OPENAI,
        ),
        PriceRecord(
            key="text-embedding-3-small",
            provider="openai",
            input_per_mtok=0.02,
            cache_read_per_mtok=0.02,
            cache_write_per_mtok=0.0,
            output_per_mtok=0.0,
            aliases=("openai/text-embedding-3-small",),
            effective_date="2026-08-31",
            source_url=_OPENAI,
        ),
        PriceRecord(
            key="gemini/gemini-embedding-001",
            provider="gemini",
            input_per_mtok=0.15,
            cache_read_per_mtok=0.15,
            cache_write_per_mtok=0.0,
            output_per_mtok=0.0,
            aliases=("vertex_ai/gemini-embedding-001",),
            effective_date="2026-08-31",
            source_url="https://ai.google.dev/gemini-api/docs/pricing",
        ),
    ],
)

PRICE_SNAPSHOTS: dict[str, PriceSnapshot] = {
    _SNAPSHOT_2026_08_31.snapshot_id: _SNAPSHOT_2026_08_31,
}

DEFAULT_PRICE_SNAPSHOT_ID = _SNAPSHOT_2026_08_31.snapshot_id


def get_price_snapshot(snapshot_id: str | None = None) -> PriceSnapshot:
    """The named price snapshot, or the current default."""
    sid = snapshot_id or DEFAULT_PRICE_SNAPSHOT_ID
    snapshot = PRICE_SNAPSHOTS.get(sid)
    if snapshot is None:
        raise PricingError(
            f"unknown price snapshot {sid!r}; known: {sorted(PRICE_SNAPSHOTS)}"
        )
    return snapshot


def estimate_cost(
    model: str,
    tokens: TokenUsage,
    *,
    effective_tier: str | None,
    snapshot_id: str | None = None,
) -> float:
    """USD for one call under `snapshot_id`. Raises rather than returning a silent 0.0.

    `effective_tier` is the tier the request ACTUALLY ran at (LLMClient tracks its own
    flex-to-auto fallback), never the environment default read after the fact.
    """
    return get_price_snapshot(snapshot_id).price_call(model, tokens, effective_tier)
