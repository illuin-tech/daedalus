"""Retrievers and injection policies (paper Section 5.2 and Appendix D), offline."""

from __future__ import annotations

import numpy as np
import pytest

from daedalus.core.memory import retrieval as R

POOL = ([f"heuristic number {i}" for i in range(20)], [f"m{i}" for i in range(20)])


class _StubModel:
    """Stand-in for a SentenceTransformer that records encode() kwargs."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return np.ones((len(texts), 4), dtype="float32")


def test_dense_names_the_query_prompt_only_on_the_query_side(monkeypatch):
    pytest.importorskip("faiss")
    model = _StubModel(prompts={"query": "Instruct: ...", "document": ""})
    monkeypatch.setitem(R._MODEL_CACHE, "stub", model)
    r = R.DenseRetriever(model_name="stub")
    r.index(["a heuristic", "another"], ["m1", "m2"])
    assert "prompt_name" not in model.calls[0]
    r.retrieve(query="how do I refund", top_k=1)
    assert model.calls[-1].get("prompt_name") == "query"


def test_dense_remote_prepends_the_query_instruction(monkeypatch):
    pytest.importorskip("faiss")
    sent: list[list[str]] = []

    def fake_post(self, texts):
        sent.append(texts)
        return np.ones((len(texts), 4), dtype="float32") / 2

    monkeypatch.setattr(R.RemoteDenseRetriever, "_post", fake_post)
    r = R.RemoteDenseRetriever(model_name="Qwen/Qwen3-Embedding-4B")
    r.index(["doc"], ["m1"])
    r.retrieve(query="q", top_k=1)
    assert sent[0] == ["doc"]
    assert sent[1][0].startswith("Instruct: ") and sent[1][0].endswith("Query:q")


def _draw(seed, task_id):
    r = R.RandomRetriever(seed=seed)
    r.index(*POOL)
    r.on_task_start("an instruction", task_id=task_id)
    return [m.memory_id for m in r.retrieve(query="q", top_k=5)]


def test_random_draws_are_reproducible_per_seed_and_task():
    assert _draw(0, "t1") == _draw(0, "t1")
    assert _draw(0, "t1") != _draw(99, "t1")
    assert len({tuple(_draw(0, f"t{i}")) for i in range(6)}) > 1


def test_bm25_tokenizes_punctuation_and_dotted_calls():
    assert R._bm25_tokenize("`apis.venmo.show_transactions(access_token=...)`") == [
        "apis", "venmo", "show_transactions", "access_token",
    ]


def _mem(i):
    return R.RetrievedMemory(f"m{i}", f"text {i}", 1.0)


@pytest.mark.parametrize(
    "policy, second_turn, persist",
    [
        (R.TRANSIENT, ["m1", "m2"], False),
        (R.CUMULATIVE_DEDUP, ["m2"], True),
        (R.CUMULATIVE_VERBATIM, ["m1", "m2"], True),
    ],
)
def test_injection_policies(policy, second_turn, persist):
    tracker = R.InjectionTracker(policy=policy)
    tracker.select([_mem(1)])
    selected, keep = tracker.select([_mem(1), _mem(2)])
    assert [m.memory_id for m in selected] == second_turn and keep is persist


def test_without_replacement_surfaces_unseen_notes_until_the_pool_is_empty():
    r = R.WithoutReplacementRetriever(R.BM25Retriever())
    r.index(*POOL)
    r.on_task_start("", task_id="t")
    seen = [m.memory_id for _ in range(5) for m in r.retrieve(query="heuristic number", top_k=5)]
    assert len(seen) == len(set(seen)) == 20


def test_whole_bank_every_turn_forces_transient():
    from daedalus.core.agents.memory import build_retriever
    from daedalus.core.config import config_from_dict
    from daedalus.core.memory.pool import MemoryItem, MemoryPool

    pool = MemoryPool([
        MemoryItem(memory_id=i, type="reflection", text=t, source_task_id="x",
                   source_trajectory_success=True, extraction_timestamp="")
        for t, i in zip(*POOL)
    ])
    cfg = config_from_dict({"memory": {"enabled": True, "retriever": {"type": "all_every_turn"}}})
    retriever, tracker = build_retriever(cfg, pool)
    assert retriever.ephemeral and tracker.policy == R.TRANSIENT
