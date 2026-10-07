"""ReasoningBank's solver hook: retrieve for this task, once, before the first turn.

"During memory retrieval, the agent queries ReasoningBank with the current query context to
identify the top-k relevant experiences and their corresponding memory items using
embedding-based similarity search. Retrieved items are injected into the agent's system
instruction" — so one retrieval per task, on the task query, into the slot every solver in
this harness already has for injected memory. Nothing is retrieved per turn.

The same hook serves both phases, which is the point of ReasoningBank's closed loop: an
accumulation run retrieves from the bank it is *building* (its `bank_path` is its own
`pool.json`), so task i is solved with what tasks 1..i-1 taught. Retrieval runs inside the
solver rather than before it because the task query is only available once the benchmark has
opened the task, and because keeping the standard `solve_task(task_id)` signature means
daedalus's parallel runner, its resume-on-existing-trace behaviour and its trace format all
apply unchanged.
"""

from __future__ import annotations

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.common.solver import MethodSolverMixin
from references.reasoningbank.config import ReasoningBankConfig
from references.reasoningbank.memory import load_bank, require_bank
from references.reasoningbank.retrieval import ExperienceRetriever


class ReasoningBankSolverMixin(MethodSolverMixin):
    """Adds per-task memory retrieval to a benchmark's TaskAgent."""

    method_config_cls = ReasoningBankConfig

    def _init_reasoningbank(self, cfg: ExperimentConfig) -> None:
        self._init_method(cfg)
        rb = self.method_cfg
        if cfg.kind == "inference":
            require_bank(rb.bank_path)  # `run.py` already checked; workers check again
        elif not rb.bank_path:
            raise ValueError(
                "reasoningbank.bank_path is not set. An accumulation run sets it to its "
                "own pool.json automatically — build the agent through "
                "`references.reasoningbank.accumulation`, not by hand."
            )
        # A missing file is an EMPTY bank during accumulation: the stream's first task
        # retrieves before anything has been written.
        # Named `experiences`, not `retriever`: that attribute is daedalus's own per-turn
        # retriever on these agents, and the runner calls it when it is not None.
        self.experiences = ExperienceRetriever(
            experiences=load_bank(rb.bank_path),
            k=rb.k,
            embedder=rb.embedder,
            # Every embedding request is a billed call; the handle puts it in this run's
            # usage ledger under the `embedding` role.
            accounting=self.accounting,
        )
        self.experiences.warm()

    def _reasoningbank_blocks(self, query: str) -> list[str]:
        """Retrieve for this task and return the block to inject (or nothing)."""
        self._retrieval = self.experiences.select(query)
        console.detail(
            f"[reasoningbank] {len(self._retrieval.items)} memory item(s) from "
            f"{len(self._retrieval.selected)} experience(s) ({self._retrieval.mode})",
            verbose=self.cfg.logging.verbose,
        )
        return self._retrieval.texts
