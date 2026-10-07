"""The explorer's per-turn budget steering, and its default-off contract.

`deepseek-v4-pro-0813` spent all 40 turns of 4 of 4 AppWorld generation sessions executing
code, opened a NEW app on turn 40, and emitted no spec, so every session banked nothing.
gpt-5.4 converged by turn 26 unaided in all 90 sessions of the reference run, which is why
the steering is opt-in: turning it on for everyone would change the bank that the paper
already reports.

Pinned here: the default stays off, the wording escalates toward the ceiling, and nothing
is appended once the budget is spent.

`_steer` reads only the config, but its module opens a live world, so these are skipped
when AppWorld is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "appworld", reason="AppWorld not installed (uv sync --extra appworld)"
)

from daedalus.benchmarks.appworld.explorer import ExplorerAgent  # noqa: E402
from daedalus.core.config import GenerationConfig  # noqa: E402


class _Explorer:
    """Just enough of the explorer to exercise `_steer`, which reads only `self.gen`."""

    def __init__(self, steering: bool):
        self.gen = GenerationConfig(explorer_turn_steering=steering)

    _steer = ExplorerAgent._steer


def test_steering_is_off_by_default() -> None:
    assert GenerationConfig().explorer_turn_steering is False


@pytest.mark.parametrize("turn,budget,expect", [
    (0, 40, "39 turns left."),
    (18, 40, "21 turns left."),
])
def test_early_turns_get_a_bare_count(turn: int, budget: int, expect: str) -> None:
    assert _Explorer(True)._steer(turn, budget) == expect


def test_past_halfway_it_names_one_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # 40-turn budget, turn 20 -> 19 left, which is <= budget // 2.
    msg = _Explorer(True)._steer(20, 40)
    assert "19 turns left" in msg
    assert "Pick ONE task" in msg
    assert "Do not start on a new app" in msg


@pytest.mark.parametrize("turn", [36, 37, 38])
def test_last_turns_demand_the_spec(turn: int) -> None:
    msg = _Explorer(True)._steer(turn, 40)
    assert "Stop exploring" in msg
    assert "json" in msg


def test_nothing_is_appended_once_the_budget_is_spent() -> None:
    # turn == budget - 1 leaves 0 turns; a reminder then has nothing to ask for, and the
    # ceiling request that follows the loop is what asks for the spec.
    assert _Explorer(True)._steer(40, 40) == ""
    assert _Explorer(True)._steer(41, 40) == ""


def test_a_short_budget_still_escalates() -> None:
    """A 6-turn budget must not skip straight from a bare count to the ceiling."""
    seen = [_Explorer(True)._steer(t, 6) for t in range(6)]
    assert "Pick ONE task" in seen[2] or "Stop exploring" in seen[2]
    assert seen[-1] == ""
