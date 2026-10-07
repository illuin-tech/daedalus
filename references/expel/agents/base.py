"""ExpeL's solver hook: insights and recalled trajectories, injected at task start.

"the task specifications will be augmented with the concatenation of the full list of
extracted insights, and the top-k trajectories with the highest task similarity will be
retrieved and used as fewshot in-context examples" (§4.3). Both go into the slot each
benchmark's solver prompt already has for injected memory, once, before the first turn —
ExpeL does no per-turn retrieval and no retries at test time.
"""

from __future__ import annotations

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.common.solver import MethodSolverMixin
from references.expel.config import ExpeLConfig
from references.expel.extraction import fewshots_block, rules_block
from references.expel.insights import load_insights
from references.expel.retrieval import (
    DemonstrationRetriever,
    load_demonstrations,
)


def demonstrations_path(expel: ExpeLConfig) -> str:
    """Where the experience pool lives: `demonstrations.json` next to the insight list."""
    from pathlib import Path

    return str(Path(expel.insights_path).with_name("demonstrations.json"))


class ExpeLSolverMixin(MethodSolverMixin):
    """Adds ExpeL's task-start injection to a benchmark's TaskAgent."""

    method_config_cls = ExpeLConfig

    def _init_expel(self, cfg: ExperimentConfig) -> None:
        self._init_method(cfg)
        expel = self.method_cfg
        if not expel.insights_path:
            raise ValueError(
                "expel.insights_path is not set — an ExpeL inference run needs the insight "
                "list written by `references.expel.accumulation`."
            )
        # "the concatenation of the full list of extracted insights" (§4.3), always.
        self.insights = load_insights(expel.insights_path)
        self.demonstrations = DemonstrationRetriever(
            demonstrations=load_demonstrations(demonstrations_path(expel)) if expel.fewshot_k else [],
            k=expel.fewshot_k,
            embedder=expel.embedder,
        )
        # Embed the pool now, while the process is still quiet (see `warm`).
        self.demonstrations.warm()

    def _expel_blocks(self, task_description: str) -> list[str]:
        """The blocks to inject for this task: the insight list, then the few-shots."""
        blocks: list[str] = []
        if self.insights:
            blocks.append(rules_block(self.cfg.benchmark, self.insights))
        trajectories, record = self.demonstrations.select(task_description)
        record.num_insights = len(self.insights)
        self._retrieval = record
        if trajectories:
            blocks.append(fewshots_block(trajectories))
        console.detail(
            f"[expel] {len(self.insights)} insight(s) + {len(trajectories)} recalled "
            f"trajector{'y' if len(trajectories) == 1 else 'ies'}",
            verbose=self.cfg.logging.verbose,
        )
        return blocks
