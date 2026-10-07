"""Memory retrieval: the top-k most similar past experiences (§3.2, Appendix A.2).

"For retrieval, we embed each task query using gemini-embedding-001, accessed via Vertex
AI. Similarity search is conducted over the memory pool using cosine distance. We select
memory items of the top-k most similar experiences (default k = 1)."

Two things to keep in mind about that sentence. The query — not the trajectory, not the
item text — is what is embedded on both sides: a new task is matched against the *tasks* the
bank came from. And the unit of retrieval is the EXPERIENCE: k = 1 injects the (up to three)
items of one past task, not one item.

The embedder is any litellm embedding model, so it is an API call rather than a local torch
model: no weights are loaded inside a running benchmark simulation, and
`gemini-embedding-001` itself can be selected when a Gemini key is configured. Vectors are
normalized, so the dot product IS the cosine similarity, and a bank of a few hundred
experiences is ranked in numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from daedalus.core.llm.client import LLMResponse
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.usage import TokenUsage
from daedalus.core.logging import console

from references.common.usage import Usage
from references.reasoningbank.memory import Experience, Item, injection_block

_BATCH = 64  # inputs per embedding request
_ATTEMPTS = 3

# One encoded bank PER PROCESS, keyed by (embedder, the bank's queries). daedalus's parallel
# runner builds a fresh agent for EVERY task inside a worker process
# (`task_pool._solve_single_task`), so without this the whole bank is re-embedded once per
# task: an extra API round-trip on the critical path of every task, and the same tokens paid
# again each time. The cache lives for the life of the process, which is the right scope —
# a worker handles many tasks, and the bank is frozen for the run. (An accumulation run's
# bank does grow, but there each task runs in its own fresh worker process.)
_MATRICES: dict[tuple[str, int], Any] = {}


@dataclass
class Retrieval:
    """What retrieval selected for one task (written to the run's `retrieval/`)."""

    query: str = ""
    k: int = 1
    embedder: str = ""
    # "embedding" — the similarity search ran (the paper's method)
    # "empty"     — the bank holds no memory items yet (every task of a fresh stream)
    # "all"       — the bank holds at most k experiences, so ranking is a no-op
    mode: str = "embedding"
    selected: list[dict[str, Any]] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def texts(self) -> list[str]:
        """What gets injected into the solver's system prompt (one block, or nothing)."""
        return [injection_block(self.items)] if self.items else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "embedder": self.embedder,
            "mode": self.mode,
            "num_experiences": len(self.selected),
            "num_items": len(self.items),
            "selected": self.selected,
            "items": [{"title": i.title, "content": i.content} for i in self.items],
            "usage": self.usage.to_dict(),
        }


def embed(
    texts: list[str],
    model: str,
    usage: Usage | None = None,
    accounting: "CostAccounting | None" = None,
    task_id: str = "",
) -> list[list[float]]:
    """Embed `texts` with a litellm embedding model, batched, with a couple of retries.

    An embedding request is billed like any other call, so each one writes a usage event
    under the `embedding` role. It is prompt tokens and nothing else — no completion, no
    cache — which is exactly how the price snapshot's embedding records are shaped.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        response = _embed_batch(texts[start : start + _BATCH], model)
        vectors += [row["embedding"] for row in response.data]
        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        token_usage = TokenUsage(prompt_tokens=int(tokens))
        event_id = ""
        if accounting is not None:
            event_id = accounting.record_external(
                "embedding",
                model,
                tokens=token_usage,
                component="reasoningbank_embedder",
                task_id=task_id,
            )
        if usage is not None:
            usage.add(
                model,
                LLMResponse(
                    content="",
                    usage=token_usage,
                    usage_event_id=event_id,
                    estimated_cost_usd=_embedding_cost(model, token_usage),
                ),
            )
    return vectors


def _embedding_cost(model: str, tokens: TokenUsage) -> float | None:
    """Local price for one embedding call, or None when the model has no price record."""
    from daedalus.core.logging.cost import PricingError, estimate_cost

    try:
        return estimate_cost(model, tokens, effective_tier="standard")
    except PricingError:
        return None


def _embed_batch(batch: list[str], model: str):
    """One embedding request, retried — a long accumulation must not die on a blip."""
    import litellm

    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return litellm.embedding(model=model, input=batch)
        except Exception as e:  # noqa: BLE001 — transient API errors are the norm here
            if attempt == _ATTEMPTS:
                raise
            console.info(
                f"[reasoningbank] embedding failed ({type(e).__name__}: {e}), "
                f"retry {attempt}/{_ATTEMPTS - 1}"
            )
            time.sleep(2 * attempt)


class ExperienceRetriever:
    """Cosine top-k over the task queries of the banked experiences."""

    def __init__(
        self,
        experiences: list[Experience],
        k: int = 1,
        embedder: str = "text-embedding-3-large",
        accounting: "CostAccounting | None" = None,
    ) -> None:
        # An experience with no items cannot contribute anything, so it must not occupy one
        # of the k slots. Same for one whose query was lost.
        self.experiences = [e for e in experiences if e.items and e.query]
        self.k = k
        self.embedder = embedder
        self._matrix = None  # (n, dim), L2-normalized
        self._memo: tuple[str, Retrieval] | None = None
        # What indexing the bank cost. Charged to the first retrieval record that is
        # written, so a run's `retrieval/` files still add up to what was actually spent.
        self._index_usage = Usage()
        # Set by the solver mixin so each embedding request is recorded in the run's ledger.
        self.accounting = accounting

    def warm(self) -> None:
        """Embed the bank's queries once per process (one API call per 64 experiences)."""
        if self._matrix is not None or self.k <= 0 or len(self.experiences) <= self.k:
            # A bank no bigger than k is returned whole (see `_select`), so it never
            # needs an index.
            return
        import numpy as np

        queries = [e.query for e in self.experiences]
        key = (self.embedder, hash(tuple(queries)))
        matrix = _MATRICES.get(key)
        if matrix is None:
            vectors = np.asarray(
                embed(queries, self.embedder, self._index_usage, self.accounting),
                dtype=float,
            )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            matrix = _MATRICES[key] = vectors / np.clip(norms, 1e-12, None)
        self._matrix = matrix

    def select(self, query: str) -> Retrieval:
        """Retrieve for one task: the items of the k most similar past experiences."""
        if self._memo is not None and self._memo[0] == query:
            # The same query against the same (frozen-per-process) bank retrieves the same
            # experiences, so a repeat reuses the first retrieval instead of re-embedding.
            return self._memo[1]
        record = self._select(query)
        record.usage.merge(self._index_usage)
        self._index_usage = Usage()
        self._memo = (query, record)
        return record

    def _select(self, query: str) -> Retrieval:
        record = Retrieval(query=query, k=self.k, embedder=self.embedder)
        if not self.experiences or self.k <= 0 or not query:
            record.mode = "empty"
            return record

        if len(self.experiences) <= self.k:
            # Ranking cannot change the SET, only its order — skip the call (a fresh
            # stream's first tasks, or a smoke run).
            record.mode = "all"
            chosen = list(enumerate(self.experiences))
            scores = [1.0] * len(chosen)
        else:
            self.warm()
            import numpy as np

            vector = np.asarray(
                embed([query], self.embedder, record.usage, self.accounting)[0],
                dtype=float,
            )
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            similarities = np.asarray(self._matrix) @ vector
            order = np.argsort(similarities)[::-1][: self.k]
            chosen = [(int(i), self.experiences[int(i)]) for i in order]
            scores = [float(similarities[int(i)]) for i in order]

        for (_, experience), score in zip(chosen, scores):
            record.items += experience.items
            record.selected.append(
                {
                    "task_id": experience.task_id,
                    "score": score,
                    "judged_outcome": experience.outcome,
                    "num_items": len(experience.items),
                    "titles": [item.title for item in experience.items],
                }
            )
        return record
