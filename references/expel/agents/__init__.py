"""ExpeL solver agents, one per benchmark. Lazily imported (each pulls in its benchmark)."""

from __future__ import annotations

import importlib
from typing import Any

from daedalus.core.config import ExperimentConfig

# Dotted paths, imported on demand, so an ExpeL tau2 run never imports appworld. The path
# is also what daedalus's parallel runner passes to its workers
# (`task_pool.agent_class_path`), so these classes must stay module-level.
_AGENTS = {
    "appworld": "references.expel.agents.appworld.ExpeLAppWorldAgent",
    "tau2": "references.expel.agents.tau2.ExpeLTau2Runner",
    "automationbench": "references.expel.agents.automationbench.ExpeLAutomationBenchAgent",
}


def build_expel_agent(cfg: ExperimentConfig, run_idx: int | None = None) -> Any:
    """Construct the ExpeL solver for `cfg.benchmark`."""
    try:
        path = _AGENTS[cfg.benchmark]
    except KeyError:
        raise ValueError(
            f"ExpeL has no solver for benchmark {cfg.benchmark!r}; wired: {sorted(_AGENTS)}. "
            f"Adding one is a ~20-line subclass — see references/expel/README.md."
        ) from None
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)(cfg, run_idx=run_idx)
