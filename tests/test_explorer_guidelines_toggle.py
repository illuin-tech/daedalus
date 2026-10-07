"""`generation.explorer_guidelines: false` — the evolved-guideline ablation.

The explorer carries TWO kinds of guidance and only one of them is meant to be switchable:

- the FIXED per-benchmark `explorer_task_guidelines.txt` ("what a good task looks like
  here"), rendered unconditionally for the explorer, the refiner and the guideline updater
  alike — making that optional is what once let the three disagree;
- the EVOLVED difficulty-balance playbook an LLM re-derives after every refined session
  (`core/generation/guidelines.py`) and injects into each later session's prompt.

`explorer_guidelines` gates the second only, and gates BOTH of its halves: no distillation
call, and nothing already banked is injected either — so a run resumed with the flag off
does not quietly keep steering on the list its earlier sessions wrote.

Pinned here: the default stays on, the flag is fingerprinted, and with it off the rendered
explorer prompt carries no evolved-guideline block while the fixed block survives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daedalus.core.config import GenerationConfig
from daedalus.core.resources import render_prompt

_HEADER = "## Guidelines from past generations"
# One directory per benchmark, with the extra vars its env/protocol templates need.
_BENCHES = {
    "appworld": (
        Path("daedalus/benchmarks/appworld/prompts"),
        {"app_descriptions": "  - spotify: music", "main_user": "Alice"},
    ),
    "tau2": (
        Path("daedalus/benchmarks/tau2/prompts"),
        {"domain_policy": "P", "tool_list": "T"},
    ),
}


def _explorer_prompt(bench: str, guidelines: list[str]) -> str:
    prompts_dir, extra = _BENCHES[bench]
    return render_prompt(
        "explorer",
        prompts_dir,
        add_env_knowledge=True,
        prior_tasks=["an earlier task"],
        guidelines=guidelines,
        coverage_goal="",
        coverage_tally="",
        **extra,
    )


def test_evolved_guidelines_are_on_by_default() -> None:
    assert GenerationConfig().explorer_guidelines is True


def test_the_flag_is_fingerprinted() -> None:
    """Changing it changes execution semantics, so it must invalidate a run folder."""
    from daedalus.core.config import ExperimentConfig, config_fingerprint

    cfg = ExperimentConfig(benchmark="tau2", experiment_name="x")
    before = config_fingerprint(cfg)
    cfg.generation.explorer_guidelines = False
    assert config_fingerprint(cfg) != before


@pytest.mark.parametrize("bench", sorted(_BENCHES))
def test_an_empty_list_renders_no_evolved_block(bench: str) -> None:
    """What the pipeline passes when the flag is off is `[]`, for every benchmark."""
    off = _explorer_prompt(bench, [])
    assert _HEADER not in off
    assert "SENTINEL-GUIDELINE" not in off


@pytest.mark.parametrize("bench", sorted(_BENCHES))
def test_a_non_empty_list_still_renders(bench: str) -> None:
    """The off case must come from the empty list, not from a template that dropped it."""
    on = _explorer_prompt(bench, ["SENTINEL-GUIDELINE"])
    assert _HEADER in on
    assert "SENTINEL-GUIDELINE" in on


@pytest.mark.parametrize("bench", sorted(_BENCHES))
def test_the_fixed_task_guidelines_survive_with_the_flag_off(bench: str) -> None:
    """The knob must not take the benchmark's own task-style block down with it."""
    from daedalus.core.resources import task_guidelines

    prompts_dir, _ = _BENCHES[bench]
    fixed = task_guidelines(prompts_dir, "explorer")
    if not fixed.strip():
        pytest.skip(f"{bench} ships no explorer_task_guidelines.txt")
    off = _explorer_prompt(bench, [])
    # Compare on a distinctive line rather than the whole block: the fixed text is itself
    # a Jinja template and is rendered with the same vars inside the composed prompt.
    probe = next(
        line.strip()
        for line in fixed.splitlines()
        if len(line.strip()) > 40 and "{{" not in line and "{%" not in line
    )
    assert probe in off


class _Backend:
    """Only the two hooks `_merge_bank` reaches on the too_hard path."""

    def spec_task_text(self, spec):
        return spec["task"]


def _merge_too_hard(tmp_path: Path, update_guidelines: bool) -> dict:
    """Drive the real `_merge_bank` down its too_hard path (no pool, no extraction)."""
    from daedalus.core.config import ExperimentConfig
    from daedalus.core.generation import pipeline as P

    cfg = ExperimentConfig(benchmark="tau2", experiment_name="x",
                           generation=GenerationConfig(explorer_guidelines=update_guidelines))

    tasks_path = tmp_path / "tasks.json"
    b = {
        "result": "too_hard",
        "current": {"task": "an impossible errand"},
        "tokens": ["a", "b"],
        "trajectory": [{"task": "t", "result": "too_hard", "trace": ""}],
        "entered_refinement": True,
        "record": {},
    }
    P._merge_bank(
        tasks_path,
        tmp_path / "pool.json",
        tmp_path / "bank.lock",
        cfg,
        _Backend(),
        "ctx",
        b,
        None,  # extraction_llm — only reached if the distillation fires
    )
    return P._read_bank(tasks_path)


def test_merge_bank_distils_when_the_flag_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daedalus.core.generation import pipeline as P

    calls: list = []
    monkeypatch.setattr(
        P, "update_explorer_memory", lambda *a, **k: calls.append(k) or ["learned"]
    )
    bank = _merge_too_hard(tmp_path, True)
    assert len(calls) == 1
    assert bank["guidelines"] == ["learned"]


def test_merge_bank_skips_distillation_when_the_flag_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write half: no LLM call, and nothing written to the banked list."""
    from daedalus.core.generation import pipeline as P

    calls: list = []
    monkeypatch.setattr(
        P, "update_explorer_memory", lambda *a, **k: calls.append(k) or ["learned"]
    )
    bank = _merge_too_hard(tmp_path, False)
    assert calls == []
    assert bank["guidelines"] == []


@pytest.mark.parametrize("update_guidelines", [True, False])
def test_too_hard_is_banked_either_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, update_guidelines: bool
) -> None:
    """The off-target lists feed only the updater, but they are kept regardless, so an
    off-run's bank stays diffable against an on-run's."""
    from daedalus.core.generation import pipeline as P

    monkeypatch.setattr(P, "update_explorer_memory", lambda *a, **k: [])
    bank = _merge_too_hard(tmp_path, update_guidelines)
    assert bank["too_hard"] == ["an impossible errand"]
