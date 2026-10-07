"""Canonical usage data model: one record per billed LLM call.

Three concepts are kept strictly apart (see LLM_COST_ACCOUNTING_IMPLEMENTATION_PLAN.md §3.1):

  usage                  — provider-reported tokens/metadata for one completed call.
  estimated_cost_usd     — computed locally from a versioned price snapshot.
  provider_reported_cost — cost the provider (or router) returned for that request.

None of the three is ever substituted for another: a missing provider cost stays None
rather than being back-filled with the estimate, and an unpriceable call stays None
rather than being reported as $0.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

# Closed role set. A role names WHAT the call was for, never which LLMClient made it:
# one client may be shared by several roles, so role comes from the call's UsageScope.
ROLES: tuple[str, ...] = (
    "solver",
    "user_simulator",
    "retriever",
    "explorer",
    "judge",
    "extraction",
    "reflection",
    "retry_decision",
    "verifier",
    "embedding",
    "consolidation",
)

STATUS_COMPLETED = "completed"
STATUS_FAILED_WITHOUT_USAGE = "failed_without_usage"

# Bumped when the on-disk event schema changes in a way readers must notice.
USAGE_SCHEMA_VERSION = 2


class UsageError(ValueError):
    """Invalid usage record — a bug in the harness, not a provider hiccup."""


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts for one call.

    `cache_read_tokens` is the part of `prompt_tokens` served from the prompt cache;
    `uncached_prompt_tokens` is the remainder, so the two never double-count.
    `reasoning_tokens` is a DETAIL of `completion_tokens` (already included in it) and
    must not be added to the completion total a second time.

    `cache_details_available` distinguishes "the provider reported no cache fields" from
    "the provider reported zero cached tokens" — the first cannot support a cache-savings
    claim, the second can.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    image_input_tokens: int = 0
    image_output_tokens: int = 0
    cache_details_available: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "audio_input_tokens",
            "audio_output_tokens",
            "image_input_tokens",
            "image_output_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise UsageError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise UsageError(f"{name} must be non-negative, got {value}")
        if self.cache_read_tokens > self.prompt_tokens:
            raise UsageError(
                f"cache_read_tokens ({self.cache_read_tokens}) exceeds prompt_tokens "
                f"({self.prompt_tokens})"
            )
        if self.reasoning_tokens > self.completion_tokens:
            raise UsageError(
                f"reasoning_tokens ({self.reasoning_tokens}) exceeds completion_tokens "
                f"({self.completion_tokens})"
            )

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uncached_prompt_tokens"] = self.uncached_prompt_tokens
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TokenUsage | None":
        if data is None:
            return None
        fields = {
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "audio_input_tokens",
            "audio_output_tokens",
            "image_input_tokens",
            "image_output_tokens",
            "cache_details_available",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            audio_input_tokens=self.audio_input_tokens + other.audio_input_tokens,
            audio_output_tokens=self.audio_output_tokens + other.audio_output_tokens,
            image_input_tokens=self.image_input_tokens + other.image_input_tokens,
            image_output_tokens=self.image_output_tokens + other.image_output_tokens,
            cache_details_available=self.cache_details_available
            or other.cache_details_available,
        )


@dataclass(frozen=True)
class UsageScope:
    """WHO the call belongs to. Passed explicitly at every `LLMClient.generate()`.

    A scope is cheap to derive: clients carry a base scope and callers narrow it with
    `for_role()` / `for_task()`, so a client shared by two roles cannot silently bill
    both to the same one.
    """

    experiment_name: str = ""
    experiment_kind: str = ""
    benchmark: str = ""
    role: str = "solver"
    component: str = ""
    task_id: str = ""
    session_id: str = ""
    attempt_id: str = ""
    launch_id: str = ""
    worker_id: str = ""
    # Which repeat of a multi-run inference / test-set experiment this call belongs to,
    # so per-run cost lines up with the per-run traces in run_<idx>/.
    run_idx: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise UsageError(f"Unknown cost role {self.role!r}; known: {list(ROLES)}")

    def for_role(self, role: str, component: str = "") -> "UsageScope":
        return replace(self, role=role, component=component or self.component)

    def for_task(
        self, task_id: str = "", session_id: str = "", attempt_id: str = ""
    ) -> "UsageScope":
        return replace(
            self,
            task_id=task_id or self.task_id,
            session_id=session_id or self.session_id,
            attempt_id=attempt_id or self.attempt_id,
        )


@dataclass
class LLMUsageEvent:
    """One completed (or billably-failed) LLM call.

    Deliberately carries NO prompt, response, tool-argument or credential payload: the
    ledger is a financial record and is copied around more freely than a trace.
    """

    role: str
    requested_model: str
    schema_version: int = USAGE_SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    launch_id: str = ""
    worker_id: str = ""
    experiment_name: str = ""
    experiment_kind: str = ""
    benchmark: str = ""
    task_id: str = ""
    session_id: str = ""
    attempt_id: str = ""
    run_idx: int | None = None
    component: str = ""
    status: str = STATUS_COMPLETED
    returned_model: str = ""
    provider: str = ""
    request_id: str = ""
    requested_service_tier: str | None = None
    effective_service_tier: str | None = None
    # True when the provider's response echoed the tier it actually served, rather than
    # the tier merely being the one we asked for.
    service_tier_confirmed: bool = False
    tokens: TokenUsage | None = None
    provider_reported_cost_usd: float | None = None
    provider_cost_source: str = ""
    estimated_cost_usd: float | None = None
    price_snapshot_id: str = ""
    pricing_model_key: str = ""
    estimate_unavailable_reason: str = ""
    error_type: str = ""
    provider_may_have_billed: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise UsageError(f"Unknown cost role {self.role!r}; known: {list(ROLES)}")
        if self.status not in (STATUS_COMPLETED, STATUS_FAILED_WITHOUT_USAGE):
            raise UsageError(f"Unknown usage status {self.status!r}")

    @classmethod
    def from_scope(cls, scope: UsageScope, **kwargs: Any) -> "LLMUsageEvent":
        """An event carrying the scope's identity; `kwargs` may override any of it."""
        fields: dict[str, Any] = {
            "role": scope.role,
            "component": scope.component,
            "experiment_name": scope.experiment_name,
            "experiment_kind": scope.experiment_kind,
            "benchmark": scope.benchmark,
            "task_id": scope.task_id,
            "session_id": scope.session_id,
            "attempt_id": scope.attempt_id,
            "run_idx": scope.run_idx,
            "launch_id": scope.launch_id,
            "worker_id": scope.worker_id,
        }
        fields.update(kwargs)
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tokens"] = self.tokens.to_dict() if self.tokens is not None else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMUsageEvent":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["tokens"] = TokenUsage.from_dict(data.get("tokens"))
        return cls(**kwargs)
