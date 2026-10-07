"""AutoGuide's own config block (paper §3 + Appendix B.1.3).

Everything shared with DAEDALUS stays in daedalus's blocks (`agent`, the benchmark block,
`run`); only what the paper's method adds lives here. See `references/common/config.py`
for how the block is loaded and snapshotted into the run folder.
"""

from __future__ import annotations

from dataclasses import dataclass

from references.common.config import MethodConfig, retrieval_dir  # noqa: F401 (re-export)

# This method's setup name: it namespaces every artifact under
# outputs/baselines/autoguide/{memory,inference}/<name>/ and labels these runs in the run browser and
# the figure scripts. Both entry points set it on the config, so a YAML cannot lose it.
SETUP = "autoguide"


@dataclass
class AutoGuideConfig(MethodConfig):
    """AutoGuide's knobs: the guideline bank, and the four LLM roles around it."""

    SETUP = SETUP

    # ── inference ──
    # The context-aware guideline bank an inference run reads: the `pool.json` written by
    # `references.autoguide.accumulation`. Unused by accumulation itself.
    bank_path: str | None = None
    # top-k guideline selection (Eq. 3). 2 in the paper's ALFWorld/WebShop tables, 3 in
    # the ERL paper's AutoGuide re-implementation, which is what this defaults to.
    k: int = 3
    # Context identification + context matching + guideline selection run at EVERY turn.
    # The paper runs these on the agent's own model (the guideline EXTRACTION model is the
    # strong one), so this defaults to null = reuse `agent.model`.
    context_model: str | None = None
    context_reasoning_effort: str | None = None

    # ── accumulation ──
    # The guideline extraction LLM (Eq. 2) — the paper's strong model.
    extraction_model: str = "gpt-5.4"
    extraction_reasoning_effort: str | None = "high"
    # Reflexion retries used to collect a contrastive (τ+, τ−) pair per training task
    # (Appendix B.1.3: "collect (τ+, τ−) pairs with ReAct+Reflexion"). 3 retries = up to 4
    # attempts, stopping at the first success.
    max_retries: int = 3
    # The reflection LLM for those retries (AutoGuide's Table 7 uses a strong model).
    reflection_model: str = "gpt-5.4"
    reflection_reasoning_effort: str | None = None
    # Cap on how many (success, failure) pairs one task contributes. The paper iterates
    # over "available pairs"; a task solved on its 4th attempt offers 3.
    max_pairs_per_task: int = 3
