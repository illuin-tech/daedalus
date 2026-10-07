"""Tool-path novelty scoring — IDF-weighted bigram cosine.

A *path* is the ordered sequence of `app.tool` calls a task performs (e.g.
``["spotify.login", "spotify.search_songs", "spotify.add_to_playlist"]``).
We score how novel a candidate path is against the corpus of already-banked
paths. The score is logged and stored with every banked task; with
`generation.min_novelty` > -1 a task scoring below the threshold is dropped as a
near-duplicate (an early filter, not part of the method — see the paper, Appendix E).

Similarity is the cosine between IDF-weighted **bigram** vectors:
    - bigrams (consecutive tool pairs, with <s>/</s> boundary markers) capture
      *order* and *multiplicity*;
    - IDF over the corpus down-weights boilerplate tools (login, show_api_doc)
      that appear in nearly every path, so novelty reflects the discriminative
      operations.

    sim(p, q) = Σ_g w_g(p)·w_g(q) / (‖w(p)‖·‖w(q)‖),   w_g(p) = tf(g,p)·idf(g)
    idf(g)    = ln((1+N) / (1+df(g))) + 1
    Score(p)  = 1 − max_i sim(p, corpus_i)

No model calls; trivially unit-testable.
"""

from __future__ import annotations

import math
from collections import Counter

Bigram = tuple[str, str]
Path = list[str]

START, END = "<s>", "</s>"


def bigrams(path: Path) -> list[Bigram]:
    """Consecutive tool pairs with boundary markers (so length-1 paths still
    produce features)."""
    if not path:
        return []
    seq = [START, *path, END]
    return list(zip(seq, seq[1:]))


def idf_map(corpus: list[Path]) -> dict[Bigram, float]:
    """Smoothed IDF for every bigram seen in the corpus.

    Unseen bigrams (present in a candidate but not the corpus) are handled by
    `_idf` below, which assigns them the maximal weight ln(1+N)+1.
    """
    n = len(corpus)
    df: Counter[Bigram] = Counter()
    for path in corpus:
        for g in set(bigrams(path)):
            df[g] += 1
    return {g: math.log((1 + n) / (1 + d)) + 1.0 for g, d in df.items()}


def _idf(g: Bigram, idf: dict[Bigram, float], n: int) -> float:
    """IDF of a bigram, falling back to the max (df=0) weight when unseen."""
    return idf.get(g, math.log(1 + n) + 1.0)


def _weighted_vector(
    path: Path, idf: dict[Bigram, float], n: int
) -> dict[Bigram, float]:
    """tf·idf vector over the path's bigrams."""
    tf = Counter(bigrams(path))
    return {g: count * _idf(g, idf, n) for g, count in tf.items()}


def _cosine(a: dict[Bigram, float], b: dict[Bigram, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[g] * b.get(g, 0.0) for g in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity(p: Path, q: Path, idf: dict[Bigram, float], n: int) -> float:
    """IDF-weighted bigram cosine between two paths, in [0, 1]."""
    return _cosine(_weighted_vector(p, idf, n), _weighted_vector(q, idf, n))


def novelty_score(
    path: Path, corpus: list[Path], idf: dict[Bigram, float] | None = None
) -> float:
    """1 − max similarity to any corpus path. Empty corpus ⇒ 1.0 (fully novel)."""
    if not corpus:
        return 1.0
    n = len(corpus)
    idf = idf_map(corpus) if idf is None else idf
    cand = _weighted_vector(path, idf, n)
    corpus_vecs = (_weighted_vector(c, idf, n) for c in corpus)
    return 1.0 - max(_cosine(cand, cv) for cv in corpus_vecs)
