"""AppWorld task-selection config (imported by daedalus.core.config)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppWorldConfig:
    """Which AppWorld tasks to run."""

    dataset: str = "dev"  # AppWorld split: train | dev | test_normal | test_challenge
    task_ids: list[str] | None = None
    experiment_name: str = "baseline"  # legacy; prefer top-level experiment_name
