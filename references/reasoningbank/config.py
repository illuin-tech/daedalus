"""ReasoningBank's own config block (paper §3.2 + Appendix A.2).

Everything shared with DAEDALUS stays in daedalus's blocks (`agent`, the benchmark block,
`run`); only what the paper's method adds lives here. See `references/common/config.py`
for how the block is loaded and snapshotted into the run folder.
"""

from __future__ import annotations

from dataclasses import dataclass

from references.common.config import MethodConfig, retrieval_dir  # noqa: F401 (re-export)

# This method's setup name: it namespaces every artifact under
# outputs/baselines/reasoningbank/{memory,inference}/<name>/ and labels these runs in the run browser
# and the figure scripts. Both entry points set it on the config, so a YAML cannot lose it.
SETUP = "reasoningbank"


@dataclass
class ReasoningBankConfig(MethodConfig):
    """ReasoningBank's knobs: the bank, its embedding retrieval, and the two LLM roles."""

    SETUP = SETUP

    # ── retrieval (both phases: the closed loop retrieves during accumulation too) ──
    # The bank to retrieve from — the `pool.json` written by
    # `references.reasoningbank.accumulation`. Accumulation overrides this with its OWN
    # pool.json, because the bank it is building is also the bank it retrieves from.
    bank_path: str | None = None
    # k = how many past EXPERIENCES are retrieved; each contributes its (≤3) memory items.
    # 1 is the paper's default and its best setting (Appendix C.1: 49.7 at k=1 vs 46.0 /
    # 45.5 / 44.4 at k=2/3/4 — "excessive experiences may introduce conflicts or noise").
    k: int = 1
    # Embedding model for the similarity search over past task queries. The paper uses
    # gemini-embedding-001 via Vertex AI; any litellm embedding model works here.
    embedder: str = "text-embedding-3-large"

    # ── memory extraction (§3.2, Fig. 9) ──
    # "The backbone LLM of the extractor is set to the same as the agent system with
    # temperature 1.0" — null = reuse `agent.model`, which is the paper's setting.
    extraction_model: str | None = None
    extraction_reasoning_effort: str | None = None
    extraction_temperature: float = 1.0
    # "at most 3 memory items could be extracted" per trajectory.
    max_items: int = 3

    # ── LLM-as-a-Judge (§3.2, Fig. 10) ──
    # The proxy correctness signal: no ground truth is used to decide what is banked.
    # Same backbone as the agent, "with decoding temperature setting to 0.0 for
    # determinism" — null = reuse `agent.model`.
    judge_model: str | None = None
    judge_reasoning_effort: str | None = None
