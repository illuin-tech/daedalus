"""ERL inference — solve the benchmark's tasks with retrieved heuristics in context.

    uv run python -m references.erl.run --config references/erl/configs/tau2_inference.yaml

Per task: one ranker call selects the top-k heuristics from the pool on the task
description (paper Fig. 9), and they are injected into the solver's system prompt for the
whole task. Nothing else about the solver changes, so an ERL run is directly comparable with
a DAEDALUS or baseline run. The driver is `references.common.run`; artifacts land in
`outputs/baselines/erl/inference/<name>/`.

The differences from DAEDALUS's inference are the retrieval granularity (once per task, on
the task description, whole pool ranked by an LLM, k = 20) and the injection point (system
prompt at task start, kept for the task).
"""

from __future__ import annotations

import json

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console
from daedalus.core.logging.preflight import ModelRequirement

from references.common.run import InferenceMethod, Override, main
from references.erl.agents import build_erl_agent
from references.erl.config import SETUP, ERLConfig, retrieval_dir


def _summarize(cfg: ExperimentConfig) -> None:
    """Sum what retrieval cost across every task-run on disk (the paper's cost line).

    Reads the artifacts rather than this invocation's in-memory state, so the totals cover
    resumed runs too — and `calls` can be lower than `tasks` when the pool was small
    enough for the ranker to be skipped.
    """
    totals = {"tasks": 0.0, "selected": 0.0, "calls": 0.0, "cost_usd": 0.0}
    for path in sorted(retrieval_dir(cfg).glob("run_*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        totals["tasks"] += 1
        totals["selected"] += record.get("num_selected", 0)
        totals["calls"] += usage.get("calls", 0)
        totals["cost_usd"] += usage.get("cost_usd", 0.0)
    if totals["tasks"]:
        console.info(
            f"retrieval (all task-runs on disk): {int(totals['calls'])} ranker call(s) "
            f"over {int(totals['tasks'])} task-run(s), "
            f"{totals['selected'] / totals['tasks']:.1f} heuristics/task, "
            f"${totals['cost_usd']:.2f}"
        )


METHOD = InferenceMethod(
    setup=SETUP,
    name="ERL",
    config_cls=ERLConfig,
    build_agent=build_erl_agent,
    memory_note="ERL does its own retrieval",
    header=lambda cfg, erl: {
        "ranker": f"{erl.ranker_model} (k={erl.k})",
        "pool": erl.pool_path,
    },
    overrides=(
        Override("--pool", "pool_path", str, "Override erl.pool_path"),
        Override("-k", "k", int, "Override erl.k (heuristics per task)"),
    ),
    # ERL ranks the heuristic pool with an LLM once per task; price it up front.
    extra_models=lambda cfg, erl: [
        ModelRequirement(erl.ranker_model or cfg.agent.model, "retriever", "erl.ranker_model")
    ],
    summarize=_summarize,
)


if __name__ == "__main__":
    main(METHOD)
