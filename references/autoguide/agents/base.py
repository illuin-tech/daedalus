"""AutoGuide's solver hook: identify the context and inject its guidelines, every turn.

Both benchmarks already retrieve once per turn and inject what comes back, so AutoGuide
installs its own retriever into that seam (`ContextAwareRetriever`) instead of adding a new
code path. The per-benchmark subclasses differ only in how they hand the trajectory so far
to the context identification module.
"""

from __future__ import annotations

from daedalus.core.config import ExperimentConfig
from daedalus.core.memory.retrieval import InjectionTracker

from references.autoguide.config import AutoGuideConfig
from references.autoguide.guidelines import GuidelineBank
from references.autoguide.modules import task_description
from references.autoguide.retrieval import ContextAwareRetriever
from references.common.solver import MethodSolverMixin


def _domain(cfg: ExperimentConfig) -> str:
    if cfg.benchmark == "automationbench":
        # AutomationBench loads a list of domains; a run under this comparison uses one.
        return ", ".join(cfg.automationbench.domains) or "operations"
    return cfg.tau2.domain if cfg.benchmark == "tau2" else cfg.appworld.dataset


class AutoGuideSolverMixin(MethodSolverMixin):
    """Installs the context-aware retriever into a benchmark's per-turn retrieval seam."""

    method_config_cls = AutoGuideConfig

    def _init_autoguide(self, cfg: ExperimentConfig) -> None:
        self._init_method(cfg)
        if not self.method_cfg.bank_path:
            raise ValueError(
                "autoguide.bank_path is not set — an AutoGuide inference run needs the "
                "guideline bank written by `references.autoguide.accumulation`."
            )
        self.retriever = ContextAwareRetriever(
            bank=GuidelineBank.load(self.method_cfg.bank_path),
            k=self.method_cfg.k,
            # The paper runs context identification and guideline selection on the agent's
            # own model; only guideline EXTRACTION uses the strong one.
            model=self.method_cfg.context_model or cfg.agent.model,
            reasoning_effort=self.method_cfg.context_reasoning_effort,
            description=task_description(cfg.benchmark, _domain(cfg)),
        )
        # The identify/match/select calls are billed to the `retriever` role of this run's
        # ledger, not to the solver that happens to share the model.
        self.retriever.attach_llm_client(
            self.accounting.client(
                self.method_cfg.context_model or cfg.agent.model,
                "retriever",
                component="autoguide_context_aware",
                temperature=0.0,
            )
        )
        self._injection_tracker = InjectionTracker(self.retriever.ephemeral)

    def _finish_task(self, task_id: str) -> None:
        """Persist this task's per-turn record."""
        self._retrieval = self.retriever.record
        self._save_retrieval(task_id)
