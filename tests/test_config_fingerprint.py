"""The changed-config guard: what counts as a contradiction vs. an extension.

`_guard_changed_config` refuses to reuse a run folder whose recorded config differs,
because a resume REUSES completed work rather than re-running it. That is right for
anything deciding how a task is executed, and wrong for a per-launch budget: raising
`generation.num_sessions` on a `--resume` says how much MORE to do, exactly as
`run.num_runs` does for inference, and it blocked a legitimate top-up of a generation run
that had lost sessions to provider errors.
"""

from __future__ import annotations

import json

import pytest

from daedalus.core.config import (
    _FINGERPRINT_VERSION,
    _guard_changed_config,
    config_fingerprint,
    config_from_dict,
)


def _cfg():
    return config_from_dict({"benchmark": "tau2", "experiment_name": "x"})


# ── what must NOT change the fingerprint ─────────────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [("num_sessions", 14), ("num_sessions", 999)],
)
def test_per_launch_budgets_do_not_change_the_fingerprint(field, value):
    """These say how much more to do on a resume; they cannot contradict prior work."""
    c = _cfg()
    before = config_fingerprint(c)
    setattr(c.generation, field, value)
    assert config_fingerprint(c) == before


def test_run_block_does_not_change_the_fingerprint():
    c = _cfg()
    before = config_fingerprint(c)
    c.run.num_runs += 5
    c.run.parallel += 1
    c.run.max_tasks = 3
    assert config_fingerprint(c) == before


# ── what MUST change it ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "block,field,value",
    [
        ("agent", "model", "gpt-4o"),
        ("tau2", "max_steps", 30),
        ("tau2", "domain", "airline"),
        ("memory", "pool_path", "somewhere/else.json"),
        ("generation", "explorer_model", "gpt-4o"),
        ("generation", "coverage_tags", False),
        ("generation", "explorer_guidelines", False),
        ("generation", "withhold_discoverable_ids", True),
        ("accumulation", "max_failures", 2),
    ],
)
def test_execution_semantics_change_the_fingerprint(block, field, value):
    c = _cfg()
    before = config_fingerprint(c)
    setattr(getattr(c, block), field, value)
    assert config_fingerprint(c) != before, f"{block}.{field} must be fingerprinted"


# ── the version pass-through ─────────────────────────────────────────────────


def test_folder_fingerprinted_by_an_older_definition_is_passed_through(tmp_path):
    """An older hash cannot be compared against one computed here — only the hash moved."""
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"config_fingerprint": "deadbeefdeadbeef", "config_fingerprint_version": 1}),
        encoding="utf-8",
    )
    _guard_changed_config(_cfg(), tmp_path)  # must not raise


def test_current_version_mismatch_still_raises(tmp_path):
    """A genuinely different config at the CURRENT version is still refused."""
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "config_fingerprint": "deadbeefdeadbeef",
                "config_fingerprint_version": _FINGERPRINT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="DIFFERENT config"):
        _guard_changed_config(_cfg(), tmp_path)


def test_matching_fingerprint_passes(tmp_path):
    c = _cfg()
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "config_fingerprint": config_fingerprint(c),
                "config_fingerprint_version": _FINGERPRINT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    _guard_changed_config(c, tmp_path)


def test_a_resume_with_more_sessions_is_allowed(tmp_path):
    """The case this fix exists for: top up a generation run that lost sessions."""
    c = _cfg()
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "config_fingerprint": config_fingerprint(c),
                "config_fingerprint_version": _FINGERPRINT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    c.generation.num_sessions = 14  # --num-sessions 14 on a --resume
    _guard_changed_config(c, tmp_path)

def test_every_fingerprinted_block_is_covered_by_the_current_version() -> None:
    """Changing _FINGERPRINT_BLOCKS must come with a _FINGERPRINT_VERSION bump, or every
    existing run fails the changed-config guard on a hash that was merely redefined."""
    from daedalus.core.config import _FINGERPRINT_BLOCKS

    assert _FINGERPRINT_BLOCKS == (
        "agent", "memory", "accumulation", "generation",
        "appworld", "tau2", "automationbench", "provider_routing",
    ), "blocks changed: bump _FINGERPRINT_VERSION, then update this list"
    assert _FINGERPRINT_VERSION == 5
