"""ExpeL's experience recall: the k most similar successful trajectories (paper §4.2, §4.3).

"we used the Faiss vectorstore as the experience pool, kNN retriever and all-mpnet-base-v2
embedder to obtain top-k successful trajectories that have the maximum inner-product task
similarity with the evaluation task." Similarity is over the TASK text, not the trajectory:
"if the agent repeats a task or does a task similar to an existing successful trajectory
from the experience pool, the agent only needs to closely imitate the successful
trajectory".

The kNN itself is a dot product over a few hundred normalized vectors, so it is done here
in numpy rather than through daedalus's FAISS-backed dense retriever: importing FAISS late,
inside a process that is already running a benchmark simulation, segfaults on macOS (two
OpenMP runtimes), and a per-worker FAISS index would buy nothing at this size. The
embedder, the similarity and the top-k are ExpeL's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Demonstration:
    """One successful trial, recalled as a few-shot example."""

    task_id: str
    task: str  # what the kNN index is built over
    trajectory: str  # what gets injected


@dataclass
class Retrieval:
    """What experience recall selected for one task (written to `retrieval/`)."""

    task: str = ""
    k: int = 2
    embedder: str = ""
    selected: list[dict[str, Any]] = field(default_factory=list)
    num_insights: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "k": self.k,
            "embedder": self.embedder,
            "num_selected": len(self.selected),
            "selected": self.selected,
            "num_insights": self.num_insights,
        }


def save_demonstrations(path: str | Path, demos: list[Demonstration]) -> Path:
    """Write the experience pool of successful trajectories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"demonstrations": [asdict(d) for d in demos]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def load_demonstrations(path: str | Path) -> list[Demonstration]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ExpeL experience pool not found: {p}. It is written next to the insight list "
            f"by `python -m references.expel.accumulation --config <accumulation yaml>`."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    return [Demonstration(**d) for d in data.get("demonstrations", [])]


# One embedder and one encoded pool PER PROCESS. daedalus's parallel runner builds a fresh
# agent for EVERY task inside a worker process (`task_pool._solve_single_task`), so without
# these caches the 420 MB model is re-loaded and the pool re-encoded once per task — ~3 s of
# CPU and a full weight read each time, in every worker at once. The keys keep two runs with
# different embedders or different pools in one process apart; the dicts live for the life of
# the process, which is exactly the scope we want (a worker handles many tasks, and the
# parent builds one agent per run).
_MODELS: dict[str, Any] = {}
_MATRICES: dict[tuple[str, int], Any] = {}


def _embedder(name: str):
    """The process's SentenceTransformer for `name`, loading it the first time only."""
    model = _MODELS.get(name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = _MODELS[name] = SentenceTransformer(name)
    return model


class DemonstrationRetriever:
    """kNN over the source tasks of the successful trajectories."""

    def __init__(
        self,
        demonstrations: list[Demonstration],
        k: int = 2,
        embedder: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self.demonstrations = demonstrations
        self.k = k
        self.embedder = embedder
        self._model = None
        self._matrix = None  # (n, dim), L2-normalized

    def warm(self) -> None:
        """Load the embedder and embed the pool's task texts, once per process.

        Called when the agent is built, deliberately not on the first task: loading torch
        weights in the middle of a live benchmark simulation is what crashes. Both the model
        and the encoded pool come from the process-level caches above, so the second and
        later tasks a worker handles pay nothing here.
        """
        if self._matrix is not None or not self.demonstrations or self.k <= 0:
            return
        texts = [d.task for d in self.demonstrations]
        self._model = _embedder(self.embedder)
        key = (self.embedder, hash(tuple(texts)))
        matrix = _MATRICES.get(key)
        if matrix is None:
            matrix = _MATRICES[key] = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        self._matrix = matrix

    def select(self, task: str) -> tuple[list[str], Retrieval]:
        """Return (trajectories to inject, the record of what was selected)."""
        record = Retrieval(task=task, k=self.k, embedder=self.embedder)
        if not self.demonstrations or self.k <= 0 or not task:
            return [], record
        self.warm()
        import numpy as np

        query = self._model.encode([task], normalize_embeddings=True, show_progress_bar=False)
        # Normalized vectors, so the inner product IS the cosine similarity ExpeL ranks by.
        scores = np.asarray(self._matrix) @ np.asarray(query)[0]
        order = np.argsort(scores)[::-1][: min(self.k, len(self.demonstrations))]
        trajectories = []
        for index in order:
            demo = self.demonstrations[int(index)]
            trajectories.append(demo.trajectory)
            record.selected.append(
                {"task_id": demo.task_id, "score": float(scores[index]), "task": demo.task[:300]}
            )
        return trajectories, record
