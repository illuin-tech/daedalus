"""ReasoningBank inference — solve the benchmark's tasks with retrieved memory items.

    uv run python -m references.reasoningbank.run --config references/reasoningbank/configs/tau2_inference.yaml

Per task: one embedding search selects the top-k most similar past experiences from the bank
(k = 1, the paper's best setting), and their memory items are injected into the solver's
system prompt for the whole task. The driver is `references.common.run`; artifacts land in
`outputs/baselines/reasoningbank/inference/<name>/`.

The bank is FROZEN here — the closed loop that grows it runs in `accumulation`, over the
source split (see the README).
"""

from __future__ import annotations

import json

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console
from daedalus.core.logging.preflight import ModelRequirement

from references.common.run import InferenceMethod, Override, main
from references.reasoningbank.agents import build_reasoningbank_agent
from references.reasoningbank.config import SETUP, ReasoningBankConfig, retrieval_dir
from references.reasoningbank.memory import require_bank


def _summarize(cfg: ExperimentConfig) -> None:
    """Sum what retrieval selected and cost across every task-run on disk.

    Reads the artifacts rather than this invocation's in-memory state, so the totals cover
    resumed runs too — and `calls` is lower than `tasks` when the bank was small enough for
    the search to be skipped (a bank of ≤ k experiences is returned whole).
    """
    totals = {"tasks": 0.0, "experiences": 0.0, "items": 0.0, "calls": 0.0, "cost_usd": 0.0}
    for path in sorted(retrieval_dir(cfg).glob("run_*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        totals["tasks"] += 1
        totals["experiences"] += record.get("num_experiences", 0)
        totals["items"] += record.get("num_items", 0)
        totals["calls"] += usage.get("calls", 0)
        totals["cost_usd"] += usage.get("cost_usd", 0.0)
    if totals["tasks"]:
        console.info(
            f"retrieval (all task-runs on disk): {int(totals['calls'])} embedding call(s) "
            f"over {int(totals['tasks'])} task-run(s), "
            f"{totals['experiences'] / totals['tasks']:.1f} experience(s) and "
            f"{totals['items'] / totals['tasks']:.1f} item(s)/task, "
            f"${totals['cost_usd']:.4f}"
        )


METHOD = InferenceMethod(
    setup=SETUP,
    name="ReasoningBank",
    config_cls=ReasoningBankConfig,
    build_agent=build_reasoningbank_agent,
    memory_note="ReasoningBank retrieves itself",
    header=lambda cfg, rb: {
        "retrieval": f"{rb.embedder} (k={rb.k} experience(s))",
        "bank": rb.bank_path,
    },
    overrides=(
        Override("--bank", "bank_path", str, "Override reasoningbank.bank_path"),
        Override("-k", "k", int, "Override reasoningbank.k (experiences)"),
    ),
    extra_models=lambda cfg, rb: [
        ModelRequirement(rb.embedder, "embedding", "reasoningbank.embedder")
    ],
    validate=lambda rb: require_bank(rb.bank_path),
    summarize=_summarize,
)


if __name__ == "__main__":
    main(METHOD)
