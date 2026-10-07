"""AutoGuide inference — guidelines chosen for the agent's state, every turn.

    uv run python -m references.autoguide.run --config references/autoguide/configs/tau2_inference.yaml

Per turn (paper Algorithm 2): identify the CONTEXT of the trajectory so far, match it
against the bank's contexts, and inject that context's top-k guidelines into this turn's
prompt. Nothing else about the solver changes. The driver is `references.common.run`;
artifacts land in `outputs/baselines/autoguide/inference/<name>/`.

Note the cost shape: AutoGuide adds one context-identification call per turn, plus a
matching call for each context string not yet seen in this worker, plus a selection call on
turns whose context holds more than k guidelines. That per-turn overhead is the method's
known trade-off, not an implementation artifact.
"""

from __future__ import annotations

import json

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console
from daedalus.core.logging.preflight import ModelRequirement

from references.autoguide.agents import build_autoguide_agent
from references.autoguide.config import SETUP, AutoGuideConfig, retrieval_dir
from references.common.run import InferenceMethod, Override, main


def _summarize(cfg: ExperimentConfig) -> None:
    """Sum what retrieval cost across every task-run on disk (the paper's cost line).

    Reads the artifacts rather than this invocation's in-memory state, so the totals cover
    resumed runs too.
    """
    totals = {"tasks": 0.0, "turns": 0.0, "matched": 0.0, "calls": 0.0, "cost_usd": 0.0}
    for path in sorted(retrieval_dir(cfg).glob("run_*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        totals["tasks"] += 1
        totals["turns"] += record.get("num_turns", 0)
        totals["matched"] += record.get("num_turns_with_context_match", 0)
        totals["calls"] += usage.get("calls", 0)
        totals["cost_usd"] += usage.get("cost_usd", 0.0)
    if totals["tasks"]:
        console.info(
            f"retrieval (all task-runs on disk): {int(totals['calls'])} module call(s) "
            f"over {int(totals['turns'])} turn(s) in {int(totals['tasks'])} task-run(s); "
            f"context matched on {totals['matched'] / max(1.0, totals['turns']):.0%} of turns; "
            f"${totals['cost_usd']:.2f}"
        )


METHOD = InferenceMethod(
    setup=SETUP,
    name="AutoGuide",
    config_cls=AutoGuideConfig,
    build_agent=build_autoguide_agent,
    memory_note="AutoGuide does its own retrieval",
    header=lambda cfg, ag: {
        "context/select": f"{ag.context_model or cfg.agent.model} (k={ag.k})",
        "bank": ag.bank_path,
    },
    overrides=(
        Override("--bank", "bank_path", str, "Override autoguide.bank_path"),
        Override("-k", "k", int, "Override autoguide.k (guidelines per turn)"),
    ),
    # The retriever runs identify/match/select on the context model at inference time, so
    # that model needs a price before the run starts, not after it.
    extra_models=lambda cfg, ag: [
        ModelRequirement(
            ag.context_model or cfg.agent.model, "retriever", "autoguide.context_model"
        )
    ],
    summarize=_summarize,
)


if __name__ == "__main__":
    main(METHOD)
