"""Per-turn retrieval over the heuristic bank (paper Section 5.2 and Appendix D).

The method itself injects the whole bank at task start (`memory.heuristics_at_start`) and
builds no retriever. The retrievers below are the alternatives it is compared against:

    random           uniform top-k (placebo control)
    bm25             sparse lexical top-k
    dense            Qwen3-Embedding top-k, model loaded in-process
    dense_remote     the same scoring against a served embeddings endpoint
    all_every_turn   the whole bank re-injected every turn (ALL@TURN)

`InjectionTracker` decides what happens to a retrieved note after its turn (Table 10), and
`WithoutReplacementRetriever` implements the `cumulative_no_replacement` policy.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np

from daedalus.core.logging import console


@dataclass
class RetrievedMemory:
    """A retrieved memory item with score."""

    memory_id: str
    text: str
    score: float
    source: str = ""


class Retriever(ABC):
    """Abstract retriever interface with slug-based registry."""

    slug: str = ""
    # True only for retrievers that re-send the whole bank every turn: persisting those
    # would duplicate the bank into the history each turn. Forces the `transient` policy.
    ephemeral: bool = False
    # Whether the retrieval query is the whole trajectory so far rather than the local
    # last-turn window (AutoGuide's context identification needs the trajectory).
    whole_trace_query: bool = False
    _registry: ClassVar[dict[str, type["Retriever"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "slug", ""):
            Retriever._registry[cls.slug] = cls

    @classmethod
    def from_slug(cls, slug: str, **kwargs: Any) -> "Retriever":
        if slug not in cls._registry:
            raise KeyError(f"Unknown retriever slug: {slug!r}. Known: {sorted(cls._registry)}")
        return cls._registry[slug](**kwargs)

    # LLM-backed retrievers (the AutoGuide reference method) receive the experiment's
    # ledger-writing client, so their spend is recorded under the `retriever` role.
    def attach_llm_client(self, client: Any) -> None:
        self._llm_client = client

    def _tracked_llm(self) -> Any:
        client = getattr(self, "_llm_client", None)
        if client is None:
            raise RuntimeError(f"retriever {self.slug!r} calls an LLM but has no LLMClient")
        return client

    def _llm_scope(self) -> Any:
        """This call's usage scope: the client's, narrowed to role=retriever + this task."""
        scope = getattr(self._tracked_llm(), "scope", None)
        if scope is None:
            return None
        return scope.for_role("retriever", self.slug).for_task(
            task_id=getattr(self, "_usage_task_id", "")
        )

    @abstractmethod
    def index(self, texts: list[str], ids: list[str]) -> None:
        """Build the index from a corpus of texts with corresponding IDs."""

    @abstractmethod
    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        """Retrieve top-k items given a query and optional reasoning trace."""

    def on_task_start(self, task_instruction: str = "", task_id: str = "") -> None:
        """Reset per-task state before each task's solve loop."""
        self._task_instruction = task_instruction or ""
        self._usage_task_id = task_id or ""


# ---------------------------------------------------------------------------
# Injection policies (Table 10): what happens to a note after its turn
# ---------------------------------------------------------------------------
TRANSIENT = "transient"  # shown for its turn only, never kept
CUMULATIVE_DEDUP = "cumulative_dedup"  # kept in the history, deduplicated by text
CUMULATIVE_NO_REPLACEMENT = "cumulative_no_replacement"  # kept; used notes leave the pool
CUMULATIVE_VERBATIM = "cumulative_verbatim"  # kept, repeated whenever retrieved again
INJECTION_POLICIES = (TRANSIENT, CUMULATIVE_DEDUP, CUMULATIVE_NO_REPLACEMENT, CUMULATIVE_VERBATIM)


class InjectionTracker:
    """Decides how a turn's retrieved notes enter a benchmark's conversation.

    One per solver agent; `reset()` at the start of each task. `select()` returns
    `(memories_to_inject, persist)`: with `persist=False` the agent injects into a COPY of
    the messages for that one generation call, otherwise it appends to the real history.
    `cumulative_no_replacement` behaves like `cumulative_dedup` here; its exclusion happens
    in `WithoutReplacementRetriever`, because it changes what gets ranked.
    """

    def __init__(self, ephemeral: bool = False, policy: str | None = None):
        if policy is None:
            policy = TRANSIENT if ephemeral else CUMULATIVE_DEDUP
        elif policy not in INJECTION_POLICIES:
            raise ValueError(f"unknown injection policy {policy!r}; valid: {INJECTION_POLICIES}")
        self.policy = policy
        self.ephemeral = policy == TRANSIENT
        self._seen: set[str] = set()

    def reset(self) -> None:
        self._seen.clear()

    def select(self, memories: list[RetrievedMemory]) -> tuple[list[RetrievedMemory], bool]:
        if self.policy == TRANSIENT:
            return memories, False
        if self.policy == CUMULATIVE_VERBATIM:
            return memories, True
        new = [m for m in memories if m.text not in self._seen]
        self._seen.update(m.text for m in new)
        return new, True


class WithoutReplacementRetriever(Retriever):
    """Wrap a top-k retriever so each turn draws only from what this task has not surfaced.

    Every turn then contributes up to k unseen notes. It over-fetches by the number of used
    ids (capped at the pool size) and filters them out; exclusion is keyed on memory_id.
    """

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self._used: set[str] = set()
        self._pool_size = 0
        # Instance attributes, so the class itself is not registered as a slug.
        self.slug = inner.slug
        self.ephemeral = inner.ephemeral

    def index(self, texts: list[str], ids: list[str]) -> None:
        self._pool_size = len(ids)
        self._inner.index(texts, ids)

    def on_task_start(self, task_instruction: str = "", task_id: str = "") -> None:
        self._inner.on_task_start(task_instruction, task_id=task_id)
        self._used = set()

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        over_fetch = min(top_k + len(self._used), self._pool_size or top_k + len(self._used))
        hits = self._inner.retrieve(query=query, reasoning_trace=reasoning_trace,
                                    top_k=over_fetch)
        fresh = [h for h in hits if h.memory_id not in self._used][:top_k]
        self._used.update(h.memory_id for h in fresh)
        return fresh


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class RandomRetriever(Retriever):
    """Placebo control: top_k items drawn uniformly without replacement.

    Reproducible: each task's draw is seeded from (seed, task_id), so it does not depend on
    interpreter state or on how tasks are spread over workers.
    """

    slug = "random"

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._seed = int(seed or 0)
        self._rng = random.Random(self._seed)

    def on_task_start(self, task_instruction: str = "", task_id: str = "") -> None:
        super().on_task_start(task_instruction, task_id=task_id)
        self._rng = random.Random(f"{self._seed}:{task_id}")

    def index(self, texts: list[str], ids: list[str]) -> None:
        self._texts, self._ids = texts, ids

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        if not self._ids:
            return []
        indices = self._rng.sample(range(len(self._ids)), min(top_k, len(self._ids)))
        return [RetrievedMemory(self._ids[i], self._texts[i], 0.0) for i in indices]


# Word characters only, so punctuation separates tokens and `apis.venmo.show_transactions`
# is reachable by "venmo"; `_` stays a word character (`access_token` is one token).
# Queries go through the same function as documents.
_BM25_WORD = re.compile(r"[a-z0-9_]+")


def _bm25_tokenize(text: str) -> list[str]:
    return _BM25_WORD.findall(text.lower())


class BM25Retriever(Retriever):
    """Sparse retrieval using BM25 (only positive scores are returned)."""

    slug = "bm25"

    def __init__(self, **kwargs: Any) -> None:
        self._index = None
        self._ids: list[str] = []
        self._texts: list[str] = []

    def index(self, texts: list[str], ids: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._texts, self._ids = texts, ids
        self._index = BM25Okapi([_bm25_tokenize(doc) for doc in texts])

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        if self._index is None:
            return []
        scores = self._index.get_scores(_bm25_tokenize(f"{reasoning_trace} {query}".strip()))
        return [
            RetrievedMemory(self._ids[i], self._texts[i], float(scores[i]))
            for i in np.argsort(scores)[::-1][:top_k]
            if scores[i] > 0
        ]


# One embedding model per worker process: agents (and so retrievers) are built per task,
# and reloading an 8 GB embedder for every task dominated the run. Same weights, same
# vectors: this changes cost, never results.
_MODEL_CACHE: dict[str, Any] = {}


def _cached_model(model_name: str, build: Callable[[], Any]) -> Any:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = build()
    return _MODEL_CACHE[model_name]


class DenseRetriever(Retriever):
    """Dense retrieval: sentence-transformers embeddings, FAISS inner product (cosine)."""

    slug = "dense"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        self._model_name = model_name or "Qwen/Qwen3-Embedding-4B"
        self._faiss_index = None
        self._ids: list[str] = []
        self._texts: list[str] = []

    def _load_model(self) -> Any:
        def build() -> Any:
            from sentence_transformers import SentenceTransformer

            # bf16, the dtype the weights ship in, so a 4B embedder fits on a 24 GB GPU.
            return SentenceTransformer(self._model_name, model_kwargs={"dtype": "bfloat16"})

        return _cached_model(self._model_name, build)

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        # Documents take the model's empty `document` prompt: no prompt_name here.
        return self._load_model().encode(
            texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        # Qwen3-Embedding is asymmetric: its `query` prompt is applied only when named.
        model = self._load_model()
        prompt = {"prompt_name": "query"} if "query" in (getattr(model, "prompts", None) or {}) else {}
        return model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, **prompt
        ).astype(np.float32)

    def index(self, texts: list[str], ids: list[str]) -> None:
        import faiss

        self._texts, self._ids = texts, ids
        embeddings = self._embed_documents(list(texts))
        self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self._faiss_index.add(embeddings)

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        combined = f"{reasoning_trace} {query}".strip()
        if self._faiss_index is None or not combined:
            return []
        scores, indices = self._faiss_index.search(self._embed_query(combined), top_k)
        return [
            RetrievedMemory(self._ids[i], self._texts[i], float(s), source=self.slug)
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]


class RemoteDenseRetriever(DenseRetriever):
    """`dense` against a served OpenAI-compatible `/v1/embeddings` endpoint (e.g. vLLM).

    One served model is shared by every worker instead of one resident copy per worker. A
    raw endpoint applies no prompt, so the query instruction is prepended here (documents
    are sent bare), mirroring what sentence-transformers does in-process.
    """

    slug = "dense_remote"

    # Verbatim from each model's config_sentence_transformers.json ("prompts" -> "query").
    QUERY_PROMPTS: ClassVar[dict[str, str]] = {
        f"Qwen/Qwen3-Embedding-{size}": (
            "Instruct: Given a web search query, retrieve relevant passages that answer the "
            "query\nQuery:"
        )
        for size in ("0.6B", "4B", "8B")
    }

    def __init__(self, model_name: str | None = None, base_url: str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(model_name=model_name)
        self._base_url = (base_url or "http://localhost:8000/v1").rstrip("/")
        if self._model_name not in self.QUERY_PROMPTS:
            console.info(f"[dense_remote] no query instruction known for "
                         f"{self._model_name!r}; queries are embedded plain.")

    def _post(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps({"model": self._model_name, "input": texts}).encode()
        req = urllib.request.Request(f"{self._base_url}/embeddings", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"[dense_remote] cannot reach {self._base_url}/embeddings ({e})"
            ) from e
        items = sorted(body["data"], key=lambda d: d.get("index", 0))
        vecs = np.asarray([d["embedding"] for d in items], dtype=np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._post(texts)

    def _embed_query(self, text: str) -> np.ndarray:
        return self._post([f"{self.QUERY_PROMPTS.get(self._model_name, '')}{text}"])


class AllEveryTurnRetriever(Retriever):
    """ALL@TURN: the whole bank at every turn, into that turn's prompt only (`top_k`
    ignored). Ephemeral, so it is never persisted into the history."""

    slug = "all_every_turn"
    ephemeral = True

    def __init__(self, **kwargs: Any) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []

    def index(self, texts: list[str], ids: list[str]) -> None:
        self._texts, self._ids = texts, ids

    def retrieve(
        self, query: str, reasoning_trace: str = "", top_k: int = 5
    ) -> list[RetrievedMemory]:
        return [RetrievedMemory(i, t, 1.0) for i, t in zip(self._ids, self._texts)]
