"""Per-benchmark ACE solvers, resolved by name.

Values are dotted strings imported on demand, so a τ² run never imports AppWorld. The same
dotted path is what daedalus's parallel runner hands its workers
(`task_pool.agent_class_path`), so these classes must stay importable at module level.
"""

from __future__ import annotations

import importlib
from typing import Any

from daedalus.core.config import ExperimentConfig

_AGENTS = {
    "appworld": "references.ace.agents.appworld.ACEAppWorldAgent",
    "tau2": "references.ace.agents.tau2.ACETau2Runner",
    "automationbench": "references.ace.agents.automationbench.ACEAutomationBenchAgent",
}


def build_ace_agent(cfg: ExperimentConfig, run_idx: int | None = None) -> Any:
    """The ACE solver for `cfg.benchmark`."""
    try:
        path = _AGENTS[cfg.benchmark]
    except KeyError:
        raise ValueError(
            f"ACE has no solver for benchmark {cfg.benchmark!r}; wired: {sorted(_AGENTS)}. "
            "Adding one is a ~20-line subclass — see references/ace/README.md."
        ) from None
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)(cfg, run_idx=run_idx)
