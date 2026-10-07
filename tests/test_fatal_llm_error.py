"""A provider refusing on the key or the spend limit must stop the run, not score it.

This is the regression that cost two full AppWorld runs. OpenRouter answered every call
with `Workspace monthly budget of $1.00 exceeded` (403); `LLMClient` retried it six times
per call, and `solve_tasks_parallel` recorded each task as a failed item and carried on.
The run then reported a mean success rate computed entirely from API errors — 12.3% for a
model that had solved nothing — and only fell over three layers later, in the AppWorld
evaluator, on a world directory that no solve had written.

So both halves are pinned here: the error is classified as fatal and raised on the first
occurrence, and it is NOT confused with the per-request refusals that share its status
codes.
"""

from __future__ import annotations

import pytest

from daedalus.core.llm.client import FatalLLMError, _is_fatal_provider_error


class _ProviderError(Exception):
    """A litellm-shaped error: a message, and a status code that may be absent."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# The exact body OpenRouter returns, which arrives with no `status_code` attribute at all —
# the 403 is only in the text, which is why the detector reads the message.
_OPENROUTER_BUDGET = (
    '{"error":{"message":"Workspace monthly budget of $1.00 exceeded. '
    'Contact your org admin.","code":403}}'
)


@pytest.mark.parametrize(
    "message,status",
    [
        (_OPENROUTER_BUDGET, None),
        ("AuthenticationError: invalid api key", 401),
        ("insufficient credit remaining on your account", 402),
        ("You exceeded your current quota, please check your billing details", 429),
    ],
)
def test_spend_and_auth_refusals_are_fatal(message: str, status: int | None) -> None:
    assert _is_fatal_provider_error(_ProviderError(message, status)) is True


# OpenRouter's in-flight budget refusal. It names "credits", so it matches the spend
# words, but it clears on its own — the body carries Retry-After. Classifying it fatal
# killed a live generation run once 12 parallel workers overran the budget.
_OPENROUTER_IN_FLIGHT = (
    'OpenrouterException - {"error":{"message":"This request would exceed your '
    'available credits given your current in-flight requests. Retry after in-flight '
    'requests settle, or add credits.","code":402,'
    '"metadata":{"reason":"in_flight_budget_exhausted"}}}'
)


@pytest.mark.parametrize(
    "message,status",
    [
        (_OPENROUTER_IN_FLIGHT, None),
        # Shares the 403 with the budget refusal, but refuses THIS request only.
        ("Forbidden: content policy violation for this request", 403),
        ("RateLimitError: too many requests", 429),
        ("BadRequestError: unsupported parameter reasoning_effort", 400),
        ("APITimeoutError: request timed out", None),
        ("InternalServerError: upstream provider unavailable", 500),
    ],
)
def test_per_request_failures_stay_retryable(message: str, status: int | None) -> None:
    assert _is_fatal_provider_error(_ProviderError(message, status)) is False


def test_fatal_error_is_picklable_across_the_process_pool() -> None:
    """Workers raise it in another process, so it must survive pickling intact."""
    import pickle

    revived = pickle.loads(pickle.dumps(FatalLLMError("budget exceeded")))
    assert isinstance(revived, FatalLLMError)
    assert "budget exceeded" in str(revived)


def test_task_pool_reraises_instead_of_recording_a_failed_task() -> None:
    """`solve_tasks_parallel` counts ordinary failures; this one it must let through."""
    import inspect

    from daedalus.core.agents import task_pool

    source = inspect.getsource(task_pool.solve_tasks_parallel)
    fatal_at = source.index("except FatalLLMError")
    generic_at = source.index("except Exception")
    # Order matters: the generic handler appends {} and swallows, so the fatal clause is
    # only reachable while it comes first.
    assert fatal_at < generic_at


class TestRateLimitIsNeverFatal:
    """A 429 must not abort the run, however its body is worded.

    OpenRouter wraps an upstream 429 as a RateLimitError carrying
    provider_error_code "insufficient_quota". That phrase is in _SPEND_REFUSED, so the
    classifier called it permanent and a generation run died on its final session with
    seven of eight tasks already banked.
    """

    _BODY = (
        'RateLimitError: OpenrouterException - {"error":{"message":"Provider returned '
        'error","code":429,"metadata":{"raw":"qwen/qwen3.8-flash is temporarily '
        'rate-limited upstream. Please retry shortly","provider_name":"Alibaba",'
        '"provider_error_code":"insufficient_quota",'
        '"limit_source":"upstream_provider_shared_pool"}}}'
    )

    def test_upstream_rate_limit_naming_insufficient_quota_is_transient(self):
        exc = Exception(self._BODY)
        exc.status_code = 429
        assert _is_fatal_provider_error(exc) is False

    def test_an_exhausted_openai_quota_on_429_stays_fatal(self):
        # Same status, opposite meaning: OpenAI uses 429 for a spent balance, which no
        # amount of waiting fixes. The wording separates them, not the code.
        exc = Exception("You exceeded your current quota, please check your billing details")
        exc.status_code = 429
        assert _is_fatal_provider_error(exc) is True

    def test_a_real_budget_403_is_still_fatal(self):
        exc = Exception(
            'APIError: OpenrouterException - {"error":{"message":"Budget limit exceeded '
            '(monthly limit). Contact your org admin.","code":403}}'
        )
        exc.status_code = 403
        assert _is_fatal_provider_error(exc) is True
