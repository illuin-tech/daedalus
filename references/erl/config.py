"""ERL's own config block (paper §2 + Appendix C.2).

Everything shared with DAEDALUS stays in daedalus's blocks (`agent`, the benchmark block,
`run`, and `accumulation.extraction_model` / `extraction_reasoning_effort` for the
reflection LLM); only what the paper's method adds lives here. See
`references/common/config.py` for how the block is loaded and snapshotted.
"""

from __future__ import annotations

from dataclasses import dataclass

from references.common.config import MethodConfig, retrieval_dir  # noqa: F401 (re-export)

# This method's setup name: it namespaces every artifact under
# outputs/baselines/erl/{memory,inference}/<name>/, keeps ERL runs out of daedalus's own tree, and is
# what the run browser and the figure scripts label these runs with. Both entry points set
# it on the config, so it cannot be lost by editing a YAML.
SETUP = "erl"


@dataclass
class ERLConfig(MethodConfig):
    """ERL's knobs: the pool to retrieve from, and how the ranker retrieves."""

    SETUP = SETUP

    # Heuristic pool an inference run retrieves from: the `pool.json` written by
    # `references.erl.accumulation`. Unused by accumulation itself.
    pool_path: str | None = None
    # k: how many heuristics the ranker selects per task. 20 in the paper (its best
    # configuration, and the practical ceiling it reports for LLM-based ranking).
    k: int = 20
    # The ranker LLM (paper Fig. 9). One call per task over the WHOLE pool.
    ranker_model: str = "gpt-5.4"
    ranker_reasoning_effort: str | None = "high"
