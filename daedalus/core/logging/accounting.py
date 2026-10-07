"""One handle that ties an experiment to its usage ledger.

Every entry point (inference, accumulation, generation, generated test set, a reference
method, a standalone script) starts the same way:

    acc = CostAccounting.start(cfg)          # preflight prices, open a launch ledger
    llm = acc.client(cfg.agent.model, "solver")

and every spawned worker continues it with

    acc = CostAccounting.for_worker(cfg)     # same launch, this process's own file

`start()` mints a new launch id and stores it on the config, so the id travels into
spawn-based workers inside the serialized config and every worker of one launch writes
into the same launch directory (its own file). A resume calls `start()` again and gets a
new launch id, which is what keeps earlier launches' spend intact and additive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daedalus.core.logging.cost import DEFAULT_PRICE_SNAPSHOT_ID
from daedalus.core.logging.preflight import ModelRequirement, preflight_prices
from daedalus.core.logging.usage import UsageScope
from daedalus.core.logging.usage_aggregate import UsageAggregator
from daedalus.core.logging.usage_ledger import UsageLedger, new_launch_id

if TYPE_CHECKING:  # pragma: no cover
    from daedalus.core.config import ExperimentConfig
    from daedalus.core.llm.client import LLMClient


# One ledger per (experiment folder, launch) per PROCESS. The accumulation loop builds a
# fresh agent — and so a fresh CostAccounting — for every attempt, and each would otherwise
# open its own handle on the same file.
_LEDGERS: dict[tuple[str, str, int], UsageLedger] = {}


def _shared_ledger(experiment_dir: Path, launch_id: str) -> UsageLedger:
    key = (str(experiment_dir), launch_id, os.getpid())
    ledger = _LEDGERS.get(key)
    if ledger is None:
        ledger = _LEDGERS[key] = UsageLedger(experiment_dir, launch_id)
    return ledger


@dataclass
class CostAccounting:
    """The ledger, base scope and price snapshot for one process of one experiment."""

    experiment_dir: Path
    ledger: UsageLedger | None
    base_scope: UsageScope
    price_snapshot_id: str
    # OpenRouter `provider` routing for every client this launch builds, or None.
    provider_routing: dict[str, Any] | None = None

    # ---- construction ----------------------------------------------------------------

    @classmethod
    def start(
        cls,
        cfg: "ExperimentConfig",
        extra_models: list[ModelRequirement] | None = None,
        run_idx: int | None = None,
        announce: bool = True,
    ) -> "CostAccounting":
        """Begin a launch: validate prices, mint a launch id, open this process's ledger."""
        from daedalus.core.config import experiment_dir

        if not cfg.cost_accounting.enabled:
            raise RuntimeError(
                "cost_accounting.enabled is false: the run would perform LLM work with no "
                "usage ledger and could not report what it spent. Enable it, or use a "
                "standalone script that opens its own ledger."
            )
        preflight_prices(cfg, extra_models, announce=announce)
        cfg.cost_accounting.launch_id = cfg.cost_accounting.launch_id or new_launch_id()
        return cls._build(cfg, experiment_dir(cfg), run_idx)

    @classmethod
    def for_worker(
        cls, cfg: "ExperimentConfig", run_idx: int | None = None
    ) -> "CostAccounting":
        """Continue the launch named in the config, from a spawned worker process.

        The ledger goes to `cost_accounting.ledger_dir` when set, else to
        `experiment_dir(cfg)`. Self-play generation sets it: each explorer worker runs under
        its own `experiment_name` (`<name>_w0`, …) so their environments cannot collide, and
        EVERY accounting object built in that process would otherwise write to
        `outputs/daedalus/<name>_w0/usage/` — outside the run folder, where the run's own
        `UsageAggregator` looks. One field on the config covers all of them, whereas an
        argument only covers the call site that passes it. The worker id survives on each
        event's `experiment_name`, so provenance is not lost.
        """
        from daedalus.core.config import experiment_dir

        exp_dir = Path(cfg.cost_accounting.ledger_dir or experiment_dir(cfg))
        if not cfg.cost_accounting.enabled:
            return cls(exp_dir, None, UsageScope(), DEFAULT_PRICE_SNAPSHOT_ID)
        if not cfg.cost_accounting.launch_id:
            # No launch id means the entry point did not call start() — an unwired script,
            # or a resume path that rebuilt the config from disk. Minting one here keeps
            # the calls RECORDED (a launch of its own, summed with the rest) instead of
            # trading a missing group label for missing spend. The only thing skipped is
            # the price preflight, so an unpriceable model shows up as an incomplete
            # summary rather than a hard stop.
            cfg.cost_accounting.launch_id = new_launch_id()
        return cls._build(cfg, exp_dir, run_idx)

    @classmethod
    def standalone(
        cls,
        output_dir: Path,
        experiment_name: str = "",
        experiment_kind: str = "script",
        benchmark: str = "",
        price_snapshot: str | None = None,
    ) -> "CostAccounting":
        """A ledger for a script that has no `ExperimentConfig`.

        Standalone utilities (consolidation, the naive explorer) do real, billed LLM work;
        without this they would be the one place spend went unrecorded. The ledger lives
        under the folder the script writes to, in the same `usage/<launch_id>/` layout, so
        the same aggregator reads it.
        """
        launch_id = new_launch_id()
        return cls(
            experiment_dir=Path(output_dir),
            ledger=_shared_ledger(Path(output_dir), launch_id),
            base_scope=UsageScope(
                experiment_name=experiment_name or str(Path(output_dir).name),
                experiment_kind=experiment_kind,
                benchmark=benchmark,
                launch_id=launch_id,
            ),
            price_snapshot_id=price_snapshot or DEFAULT_PRICE_SNAPSHOT_ID,
        )

    @classmethod
    def disabled(cls, experiment_dir: Path | None = None) -> "CostAccounting":
        """A no-ledger handle, for unit tests and offline analysis."""
        return cls(
            Path(experiment_dir or "."), None, UsageScope(), DEFAULT_PRICE_SNAPSHOT_ID
        )

    @classmethod
    def _build(
        cls, cfg: "ExperimentConfig", exp_dir: Path, run_idx: int | None
    ) -> "CostAccounting":
        snapshot_id = cfg.cost_accounting.price_snapshot or DEFAULT_PRICE_SNAPSHOT_ID
        launch_id = cfg.cost_accounting.launch_id
        return cls(
            experiment_dir=exp_dir,
            ledger=_shared_ledger(exp_dir, launch_id),
            base_scope=UsageScope(
                experiment_name=cfg.name,
                experiment_kind=cfg.kind,
                benchmark=cfg.benchmark,
                launch_id=launch_id,
                run_idx=run_idx,
            ),
            price_snapshot_id=snapshot_id,
            provider_routing=cfg.provider_routing.to_body(),
        )

    # ---- use -------------------------------------------------------------------------

    def scope(self, role: str, component: str = "", **identity: Any) -> UsageScope:
        """The base scope narrowed to one role (and optionally task/session/attempt)."""
        scope = self.base_scope.for_role(role, component)
        if identity:
            scope = scope.for_task(**identity)
        return scope

    def client(
        self, model: str, role: str, component: str = "", **kwargs: Any
    ) -> "LLMClient":
        """An LLMClient that writes every call to this launch's ledger under `role`."""
        from daedalus.core.llm.client import LLMClient

        # Provider routing belongs to the run, not to a role: every role calling the
        # same OpenRouter model must reach the same upstream, or the run mixes weights.
        # A caller may still override it explicitly.
        kwargs.setdefault("provider_routing", self.provider_routing)
        return LLMClient(
            model=model,
            ledger=self.ledger,
            scope=self.scope(role, component),
            price_snapshot_id=self.price_snapshot_id,
            **kwargs,
        )

    def tally_snapshot(self):
        """Mark a point in this process's spend, to diff a unit of work against."""
        return self.ledger.tally.snapshot() if self.ledger is not None else None

    def usage_since(self, snapshot) -> dict[str, Any]:
        """Canonical usage fields for the work done since `snapshot`, by role."""
        if self.ledger is None:
            return {}
        return self.ledger.tally.since(snapshot)

    def aggregator(self) -> UsageAggregator:
        """Read back every launch's events for this experiment."""
        return UsageAggregator.from_experiment_dir(self.experiment_dir)

    def summary(self) -> dict[str, Any]:
        """The canonical `usage` object for the whole experiment, resumes included."""
        return self.aggregator().summary()

    def record_external(
        self,
        role: str,
        model: str,
        tokens: Any = None,
        third_party_cost_usd: float | None = None,
        third_party_cost_source: str = "",
        component: str = "",
        provider: str = "",
        effective_tier: str | None = "standard",
        **identity: Any,
    ) -> str:
        """Record a call made by code we do not own (e.g. tau2's user simulator).

        Two separate inputs, never merged:

          * `tokens` — what that code reported the call used. Priced HERE, with this run's
            price snapshot, so a user-simulator call is costed on the same basis as a
            solver call. When it is None the event carries no local estimate and the run
            reports itself incomplete rather than inventing token counts.
          * `third_party_cost_usd` — a cost figure that library computed itself (tau2 uses
            litellm's `completion_cost`, i.e. litellm's own price table). It is stored with
            its source named, and never becomes the local estimate.
        """
        from daedalus.core.logging.usage import LLMUsageEvent

        if self.ledger is None:
            return ""
        estimated, reason = None, ""
        if tokens is None:
            reason = f"{third_party_cost_source or 'external library'} reported no token detail"
        else:
            estimated = self._estimate(model, tokens, effective_tier)
            if estimated is None:
                reason = f"no exact price for {model!r} in snapshot {self.price_snapshot_id!r}"
        event = LLMUsageEvent.from_scope(
            self.scope(role, component, **identity),
            requested_model=model,
            returned_model=model,
            provider=provider,
            effective_service_tier=effective_tier,
            tokens=tokens,
            provider_reported_cost_usd=third_party_cost_usd,
            provider_cost_source=third_party_cost_source,
            estimated_cost_usd=estimated,
            price_snapshot_id=self.price_snapshot_id,
            estimate_unavailable_reason=reason,
        )
        self.ledger.record(event)
        return event.event_id

    def _estimate(
        self, model: str, tokens: Any, effective_tier: str | None
    ) -> float | None:
        from daedalus.core.logging.cost import PricingError, get_price_snapshot

        try:
            return get_price_snapshot(self.price_snapshot_id).price_call(
                model, tokens, effective_tier
            )
        except PricingError:
            return None
