"""The Solver loop (paper Algorithm 3) and the session outcome it implies, offline.

A scripted fake benchmark stands in for the environment: each attempt's success is read
from a list, so the loop's stopping rule and heuristic acceptance can be checked exactly.
"""

from __future__ import annotations

import pytest

from daedalus.core.config import ExperimentConfig
from daedalus.core.generation import accumulate
from daedalus.core.generation.pipeline import _classify


class _Agent:
    def __init__(self, outcomes, seen):
        self.outcomes, self.seen = outcomes, seen

    def solve_task(self, task_id, heuristics=None, **_):
        self.seen.append(heuristics)
        return {"success": self.outcomes.pop(0), "task_instruction": "do it"}


class _Benchmark:
    def __init__(self, outcomes):
        self.outcomes, self.seen = list(outcomes), []

    def build_agent(self, cfg):
        return _Agent(self.outcomes, self.seen)

    def format_trajectory(self, result):
        return "trace"


def _run(monkeypatch, outcomes, ns=3, nf=8):
    bench = _Benchmark(outcomes)
    monkeypatch.setattr(accumulate, "get_benchmark", lambda cfg: bench)
    writes = iter(f"h{i}" for i in range(1, 99))
    monkeypatch.setattr(accumulate, "generate_memory_item", lambda **kw: next(writes))
    cfg = ExperimentConfig()
    cfg.accumulation.num_success_to_continue, cfg.accumulation.max_failures = ns, nf
    return accumulate.accumulate_for_task("t", cfg, extraction_llm=_NoLedger()), bench


class _NoLedger:
    ledger = None
    scope = None


def test_a_heuristic_is_accepted_after_ns_consecutive_successes(monkeypatch):
    summary, bench = _run(monkeypatch, [False, False, True, True, True])
    assert _classify(summary) == ("banked", "h2")
    # The solver retries with the latest heuristic after each failure.
    assert bench.seen == [None, ["h1"], ["h2"], ["h2"], ["h2"]]


def test_no_failure_means_too_easy(monkeypatch):
    summary, _ = _run(monkeypatch, [True, True, True])
    assert _classify(summary) == ("too_easy", None) and summary["solved_without_memory"]


def test_nf_failures_means_too_hard(monkeypatch):
    summary, _ = _run(monkeypatch, [False, True, False], ns=3, nf=2)
    assert _classify(summary) == ("too_hard", None)


def test_a_failure_resets_the_streak(monkeypatch):
    summary, _ = _run(monkeypatch, [False, True, True, False, True, True, True])
    assert len(summary["attempts"]) == 7 and _classify(summary)[0] == "banked"


def test_single_attempt_extracts_once_whatever_the_outcome(monkeypatch):
    bench = _Benchmark([True])
    monkeypatch.setattr(accumulate, "get_benchmark", lambda cfg: bench)
    monkeypatch.setattr(accumulate, "generate_memory_item", lambda **kw: " lesson ")
    summary = accumulate.single_attempt_for_task("t", ExperimentConfig(), _NoLedger())
    assert summary["final_memory"] == "lesson" and summary["final_outcome"] == "success"
    assert bench.seen == [[]]


@pytest.mark.parametrize("defect, expected", [(None, "too_hard"), ("bad spec", "invalid_spec")])
def test_a_spec_defect_is_not_counted_as_too_hard(defect, expected):
    summary = {"final_outcome": "failure", "final_memory": "x", "spec_defect": defect}
    assert _classify(summary)[0] == expected
