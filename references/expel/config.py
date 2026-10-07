"""ExpeL's own config block (paper §4 + its repo's `configs/agent/expel.yaml`).

Everything shared with DAEDALUS stays in daedalus's blocks (`agent`, the benchmark block,
`run`); only what ExpeL adds lives here. See `references/common/config.py` for how the
block is loaded and snapshotted into the run folder.
"""

from __future__ import annotations

from dataclasses import dataclass

from references.common.config import MethodConfig, retrieval_dir  # noqa: F401 (re-export)

# This method's setup name: it namespaces every artifact under
# outputs/baselines/expel/{memory,inference}/<name>/ and labels these runs in the run browser and the
# figure scripts. Both entry points set it on the config, so a YAML cannot lose it.
SETUP = "expel"


@dataclass
class ExpeLConfig(MethodConfig):
    """ExpeL's knobs. Defaults follow its repo's `agent/expel.yaml` unless noted."""

    SETUP = SETUP

    # ── inference ──
    # The insight list an inference run injects: the `pool.json` written by
    # `references.expel.accumulation`. The experience pool of successful trajectories is
    # the `demonstrations.json` written next to it.
    insights_path: str | None = None
    # Number of retrieved successful trajectories used as few-shot examples. ExpeL's
    # `Max Number of Fewshot Examples k` is 2 for ALFWorld/WebShop, 6 for HotpotQA; the ERL
    # paper's ExpeL re-implementation used 3.
    fewshot_k: int = 2
    # ExpeL's embedder for task-similarity recall (`all-mpnet-base-v2`, kNN over FAISS).
    embedder: str = "sentence-transformers/all-mpnet-base-v2"

    # ── accumulation ──
    # The insight extraction LLM — ExpeL's strong model (`gpt-4-0613` in the paper).
    extraction_model: str = "gpt-5.4"
    extraction_reasoning_effort: str | None = "high"
    # Cap on the insight list; drives ExpeL's "focus on REMOVE" suffix and the extra weight
    # a REMOVE carries once the list is over budget (`max_num_rules: 20`).
    max_num_rules: int = 20
    # L: successful trials per all-success critique call. 8 for ALFWorld/HotpotQA and 4 for
    # WebShop in the paper; the ERL paper's re-implementation used 3, which is the default
    # here because a τ² conversation is far longer than an ALFWorld episode.
    success_batch_size: int = 3
    # Reflexion retries while gathering experience (`max_reflection_depth: 3`).
    max_retries: int = 3
    # ExpeL reflects with the policy model (`Reflection LLM = gpt-3.5-turbo`, same as its
    # policy); null = reuse `agent.model`.
    reflection_model: str | None = None
    reflection_reasoning_effort: str | None = None
    # Cap on (success, failure) compare pairs one task contributes.
    max_pairs_per_task: int = 3
    # Seed for the without-replacement batching of successes.
    seed: int = 42
