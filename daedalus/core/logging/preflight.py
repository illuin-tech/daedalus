"""Price preflight: resolve every model an experiment will call, before it spends.

The point is WHERE the failure lands. An unknown model used to be priced at $0 and the
run finished with a plausible-looking, wrong total; now the resolution happens before the
first paid call, names the model and the config field that chose it, and stops.

The models a run bills are not all in one config field — a conversational tau2 generation
run also pays a user simulator, an explorer, a judge, an extraction model and possibly an
LLM retriever — so `required_models()` walks the config the way the run will.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from daedalus.core.logging import console
from daedalus.core.logging.cost import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    PricingError,
    get_price_snapshot,
)

if TYPE_CHECKING:  # pragma: no cover
    from daedalus.core.config import ExperimentConfig


class PricePreflightError(RuntimeError):
    """One or more configured models cannot be priced exactly."""


@dataclass(frozen=True)
class ModelRequirement:
    """A model this run will bill, and the config field that selected it."""

    model: str
    role: str
    source: str


def _tau2_conversational(cfg: "ExperimentConfig") -> bool:
    """Whether the tau2 LLM user simulator will run (and so be billed)."""
    if cfg.benchmark != "tau2":
        return False
    mode = cfg.tau2.solver_mode or (
        "conversational" if cfg.kind == "inference" else "ticket"
    )
    return mode == "conversational"


def required_models(
    cfg: "ExperimentConfig", extra: list[ModelRequirement] | None = None
) -> list[ModelRequirement]:
    """Every billed model this experiment path will call, with its role."""
    reqs: list[ModelRequirement] = [
        ModelRequirement(cfg.agent.model, "solver", "agent.model")
    ]

    if _tau2_conversational(cfg):
        reqs.append(
            ModelRequirement(cfg.tau2.user_llm, "user_simulator", "tau2.user_llm")
        )

    if cfg.kind in ("accumulation", "generation"):
        # The Extractor (and, for the reference methods, their reflection calls).
        reqs.append(
            ModelRequirement(
                cfg.accumulation.extraction_model,
                "extraction",
                "accumulation.extraction_model",
            )
        )

    if cfg.kind == "generation":
        reqs.append(
            ModelRequirement(
                cfg.generation.explorer_model, "explorer", "generation.explorer_model"
            )
        )
        reqs.append(
            ModelRequirement(
                cfg.generation.judge_model, "judge", "generation.judge_model"
            )
        )

    reqs.extend(extra or [])

    # De-duplicate on (model, role, source) while keeping declaration order.
    seen: set[tuple[str, str, str]] = set()
    out: list[ModelRequirement] = []
    for r in reqs:
        key = (r.model, r.role, r.source)
        if r.model and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def preflight_prices(
    cfg: "ExperimentConfig",
    extra: list[ModelRequirement] | None = None,
    announce: bool = True,
) -> dict[str, Any]:
    """Resolve and print the pricing snapshot for this run; raise if anything is unknown.

    Honors `cost_accounting.require_complete_estimates`: with it off, an unpriceable model
    is reported as a warning and the run proceeds with an incomplete estimate.
    """
    snapshot_id = cfg.cost_accounting.price_snapshot or DEFAULT_PRICE_SNAPSHOT_ID
    snapshot = get_price_snapshot(snapshot_id)
    resolved: list[dict[str, Any]] = []
    problems: list[str] = []

    for req in required_models(cfg, extra):
        try:
            record = snapshot.resolve(req.model)
        except PricingError as e:
            problems.append(f"{req.role} ({req.source}): {e}")
            continue
        resolved.append(
            {
                "role": req.role,
                "source": req.source,
                "requested_model": req.model,
                "pricing_model_key": record.key,
                "provider": record.provider,
                "input_per_mtok": record.input_per_mtok,
                "cache_read_per_mtok": record.cache_read_per_mtok,
                "cache_write_per_mtok": record.cache_write_per_mtok,
                "output_per_mtok": record.output_per_mtok,
                "tier_multipliers": dict(record.tier_multipliers),
                "long_context": (
                    record.long_context.to_dict() if record.long_context else None
                ),
            }
        )

    if announce:
        console.info(f"cost accounting: price snapshot {snapshot_id}")
        for r in resolved:
            tier = f"  tiers={r['tier_multipliers']}" if r["tier_multipliers"] else ""
            console.info(
                f"  {r['role']:<14} {r['requested_model']} → {r['pricing_model_key']} "
                f"(in ${r['input_per_mtok']}/Mtok, cache-read ${r['cache_read_per_mtok']}, "
                f"cache-write ${r['cache_write_per_mtok']}, out ${r['output_per_mtok']}){tier}"
            )
            # A model that re-prices large requests is worth seeing up front: a prompt over
            # the threshold roughly doubles, and nothing else in the run says so.
            lc = r["long_context"]
            if lc:
                console.info(
                    f"  {'':<14}   above {lc['min_prompt_tokens']:,} prompt tokens per "
                    f"request: in ${lc['input_per_mtok']}, cache-read "
                    f"${lc['cache_read_per_mtok']}, out ${lc['output_per_mtok']}"
                )

    if problems:
        message = (
            "cost accounting preflight failed — these models have no exact price record "
            f"in snapshot {snapshot_id!r}:\n  - " + "\n  - ".join(problems)
        )
        if cfg.cost_accounting.require_complete_estimates:
            raise PricePreflightError(
                message + "\n\nAdd a PriceRecord (or an explicit alias) in "
                "daedalus/core/logging/cost.py, or set "
                "cost_accounting.require_complete_estimates: false to run with an "
                "incomplete cost estimate."
            )
        console.info(message)

    return {
        "price_snapshot": snapshot_id,
        "models": resolved,
        "unpriceable": problems,
    }
