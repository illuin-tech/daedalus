"""Per-role reporting aggregate for a reference method's own LLM calls.

These are the extra calls a memory method makes around the solver — reflection,
extraction, ranking, per-turn retrieval — which every paper reports separately (e.g.
ERL's Table 4).

The canonical record of all of them (the solver's calls included) is the run's usage
ledger under `<experiment_dir>/usage/`; read it with
`daedalus.core.logging.usage_aggregate.UsageAggregator`. This class only keeps the
per-method split those tables want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


@dataclass
class Usage:
    """Cumulative usage over one or more LLM calls.

    A REPORTING aggregate, not an accounting authority: every call it sees has already
    written its own usage event to the run's ledger, and `add()` takes that event's price
    rather than re-deriving one. `cost_complete` goes False if any call could not be
    priced, so a partial number is never presented as a whole one.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    cost_complete: bool = True

    def add(self, model: str, response) -> None:
        self.calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.cached_tokens += response.cached_tokens
        if response.estimated_cost_usd is None:
            self.cost_complete = False
        else:
            self.cost_usd += response.estimated_cost_usd

    def merge(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.cost_usd += other.cost_usd
        self.cost_complete = self.cost_complete and other.cost_complete

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
            "cost_complete": self.cost_complete,
        }


def write_run_summary(
    cfg, accounting, extra: dict | None = None, filename: str = "run_summary.json"
) -> "Path":
    """Write a reference method's run summary with the canonical `usage` block.

    Every experiment kind reports spend the same way (see
    `daedalus.core.logging.usage_aggregate`), so a reference method's total is directly
    comparable with daedalus's own instead of being a differently-scoped number in a
    differently-shaped file.
    """
    import json
    from pathlib import Path

    from daedalus.core.config import experiment_dir

    usage = accounting.summary()
    base = experiment_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)
    path = base / filename
    path.write_text(
        json.dumps(
            {
                "experiment_name": cfg.name,
                "benchmark": cfg.benchmark,
                "kind": cfg.kind,
                "setup": cfg.setup,
                **(extra or {}),
                "usage": usage,
                # Deprecated alias, equal to the estimated total when it is complete.
                "total_cost_usd": usage["priced_calls_estimated_cost_usd"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Path(path)
