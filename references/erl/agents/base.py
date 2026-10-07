"""ERL's solver hook: select this task's heuristics, once, before the first turn.

Each benchmark's solver already renders a `heuristics` list into its system prompt at task
start (the slot DAEDALUS's `heuristics_at_start` uses). ERL therefore needs exactly one hook
per benchmark: at the moment the task description becomes known, rank the pool against it
and hand the top-k blocks to that slot. The rest — config loading inside the worker, the
retrieval artifact, the skip check — is shared (`references.common.solver`).

Retrieval runs inside the solver, not before it, for two reasons: the task description is
only available once the benchmark has opened the task, and keeping the standard
`solve_task(task_id)` signature means daedalus's parallel runner, its resume-on-existing-trace
behaviour and its trace format all apply to ERL runs unchanged.
"""

from __future__ import annotations

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.common.solver import MethodSolverMixin
from references.erl.config import ERLConfig
from references.erl.retrieval import HeuristicSelector


class ERLSolverMixin(MethodSolverMixin):
    """Adds per-task heuristic selection to a benchmark's TaskAgent."""

    method_config_cls = ERLConfig

    def _init_erl(self, cfg: ExperimentConfig) -> None:
        self._init_method(cfg)
        self.selector = HeuristicSelector.from_config(self.method_cfg)
        # The per-task ranking call is `retriever` spend, recorded in this run's ledger.
        self.selector.attach_llm_client(
            self.accounting.client(
                self.selector.model,
                "retriever",
                component="erl_heuristic_ranker",
                temperature=0.0,
            )
        )

    def _erl_heuristics(self, task_description: str) -> list[str]:
        """Select this task's heuristics and return the blocks to inject."""
        self._retrieval = self.selector.select(task_description)
        console.detail(
            f"[erl] {len(self._retrieval.selections)} heuristics selected "
            f"({self._retrieval.mode})",
            verbose=self.cfg.logging.verbose,
        )
        return self._retrieval.texts
