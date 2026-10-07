"""ExpeL inference — insights and recalled successful trajectories in context.

    uv run python -m references.expel.run --config references/expel/configs/tau2_inference.yaml

Per task (paper §4.3, Algorithm 3): inject the whole insight list, plus the top-k
successful trajectories whose source task is most similar to this one, into the solver's
system prompt — then solve the task in a single attempt, with no retries and no per-turn
retrieval. The driver is `references.common.run`; artifacts land in
`outputs/baselines/expel/inference/<name>/`.

Retrieval costs an embedding pass per task, not an LLM call; the extra input tokens are the
insight list and the few-shot trajectories, which show up in the traces' token counts.
"""

from __future__ import annotations

import json

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.common.run import InferenceMethod, Override, main
from references.expel.agents import build_expel_agent
from references.expel.config import SETUP, ExpeLConfig, retrieval_dir


def _summarize(cfg: ExperimentConfig) -> None:
    """What was injected across every task-run on disk.

    Reads the artifacts rather than this invocation's in-memory state, so the totals cover
    resumed runs too. ExpeL's recall makes no LLM calls, so there is no cost line here —
    its overhead is the extra input tokens, already counted in each trace.
    """
    totals = {"tasks": 0.0, "fewshots": 0.0, "insights": 0.0}
    for path in sorted(retrieval_dir(cfg).glob("run_*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        totals["tasks"] += 1
        totals["fewshots"] += record.get("num_selected", 0)
        totals["insights"] = max(totals["insights"], record.get("num_insights", 0))
    if totals["tasks"]:
        console.info(
            f"recall (all task-runs on disk): {int(totals['insights'])} insight(s) injected, "
            f"{totals['fewshots'] / totals['tasks']:.1f} few-shot trajector(y/ies) per task "
            f"over {int(totals['tasks'])} task-run(s)"
        )


METHOD = InferenceMethod(
    setup=SETUP,
    name="ExpeL",
    config_cls=ExpeLConfig,
    build_agent=build_expel_agent,
    memory_note="ExpeL does its own recall",
    header=lambda cfg, expel: {
        "fewshots": f"{expel.fewshot_k} via {expel.embedder.split('/')[-1]}",
        "insights": expel.insights_path,
    },
    overrides=(
        Override("--insights", "insights_path", str, "Override expel.insights_path"),
        Override("-k", "fewshot_k", int, "Override expel.fewshot_k"),
    ),
    # Recall is by similarity over stored demonstrations — no LLM of its own at inference
    # time, so the solver is the only model to price.
    summarize=_summarize,
)


if __name__ == "__main__":
    main(METHOD)
