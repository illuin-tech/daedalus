"""AutomationBench task-selection + environment config (imported by daedalus.core.config)."""

from __future__ import annotations

from dataclasses import dataclass, field

# The six scored domains. `simple` (200 single/two-step tasks) exists too, but the
# benchmark excludes it from the official pass rate, so it is not a default.
PUBLIC_DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")


@dataclass
class AutomationBenchConfig:
    """AutomationBench task selection and environment settings.

    The benchmark runs fully in-process: one pydantic `WorldState` per task, built from
    the task's `initial_state`, and grading is a deterministic assertion sweep over the
    final world. No server, no container, no account.

    Note `OPENAI_API_KEY` is read by the ENVIRONMENT as well as by the solver: the
    simulated ChatGPT app makes a real OpenAI call and falls back to a stub response
    without a key. That affects 15 of the 600 scored tasks, so keep the key set whatever
    provider the solver runs on.
    """

    # Which domains to load; [] = the six scored ones. The paper uses Operations.
    domains: list[str] = field(default_factory=lambda: ["operations"])
    task_ids: list[str] | None = None  # subset filter; None = all
    # Train/test split: split_file is a JSON {"train": [ids], "test": [ids]} and `split`
    # picks which list to keep (None = all); composes with task_ids. The paper's split is
    # splits/operations_30_70.json (daedalus.scripts.make_automationbench_split).
    split_file: str | None = None
    split: str | None = None
    # Path to the cloned AutomationBench repo; None => auto-detect (see __init__.py).
    repo_root: str | None = None
    max_steps: int = 50  # model-step cap (this, not agent.max_turns) — their --max-steps
    seed: int = 42
