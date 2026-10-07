"""ACE inference — the trained playbook in context, injected once at task start.

    uv run python -m references.ace.run --config references/ace/configs/appworld_inference.yaml

`ace-appworld/experiments/code/ace/evaluation_react.py`: render the WHOLE trained playbook
into the solver's system prompt before the first turn, then solve the task in a single
attempt. No retrieval, no per-turn injection, no retries, and nothing about the solver
changes. The driver is `references.common.run`; artifacts land in
`outputs/baselines/ace/inference/<name>/`.

ACE costs no extra LLM calls at inference time: its overhead is the playbook's input
tokens, on every turn of every task, and those show up in the traces' token counts.
"""

from __future__ import annotations

import json

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.ace.agents import build_ace_agent
from references.ace.config import SETUP, ACEConfig, retrieval_dir
from references.common.run import InferenceMethod, Override, main


def _summarize(cfg: ExperimentConfig) -> None:
    """What was injected across every task-run on disk.

    ACE selects nothing per task, so this is a size witness rather than a retrieval audit —
    in at-start mode the trace itself does not record the system prompt.
    """
    totals = {"tasks": 0.0, "bullets": 0.0, "chars": 0.0}
    for path in sorted(retrieval_dir(cfg).glob("run_*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        totals["tasks"] += 1
        totals["bullets"] = max(totals["bullets"], record.get("num_bullets", 0))
        totals["chars"] = max(totals["chars"], record.get("num_chars", 0))
    if totals["tasks"]:
        console.info(
            f"injected (all task-runs on disk): {int(totals['bullets'])} bullet(s), "
            f"{int(totals['chars'])} chars over {int(totals['tasks'])} task-run(s)"
        )


METHOD = InferenceMethod(
    setup=SETUP,
    name="ACE",
    config_cls=ACEConfig,
    build_agent=build_ace_agent,
    memory_note="ACE injects its own playbook",
    header=lambda cfg, ace: {"playbook": ace.playbook_path},
    overrides=(Override("--playbook", "playbook_path", str, "Override ace.playbook_path"),),
    # No LLM call of its own at inference time: the solver is the only model to price.
    summarize=_summarize,
)


if __name__ == "__main__":
    main(METHOD)
