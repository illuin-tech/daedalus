"""ReasoningBank solver agents, one per benchmark. Lazily imported (each pulls in its benchmark)."""

from __future__ import annotations

import importlib
from typing import Any

from daedalus.core.config import ExperimentConfig

# Same shape as daedalus's benchmark registry: a dotted path, imported on demand, so a
# tau2 run never imports appworld. The path is also what daedalus's parallel runner passes to
# its workers (`task_pool.agent_class_path`), so these classes must stay module-level.
_AGENTS = {
    "appworld": "references.reasoningbank.agents.appworld.ReasoningBankAppWorldAgent",
    "tau2": "references.reasoningbank.agents.tau2.ReasoningBankTau2Runner",
    "automationbench": "references.reasoningbank.agents.automationbench.ReasoningBankAutomationBenchAgent",
}


def build_reasoningbank_agent(cfg: ExperimentConfig, run_idx: int | None = None) -> Any:
    """Construct the ReasoningBank solver for `cfg.benchmark`."""
    try:
        path = _AGENTS[cfg.benchmark]
    except KeyError:
        raise ValueError(
            f"ReasoningBank has no solver for benchmark {cfg.benchmark!r}; wired: "
            f"{sorted(_AGENTS)}. Adding one is a ~20-line subclass — see "
            f"references/reasoningbank/README.md."
        ) from None
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)(cfg, run_idx=run_idx)
