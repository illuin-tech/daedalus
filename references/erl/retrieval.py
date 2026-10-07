"""Retrieval-augmented execution: pick the top-k heuristics for a new task (paper §2, Fig. 9).

ERL retrieves ONCE per task, from the task description alone, with an LLM ranking the
WHOLE pool — the configuration the paper reports as its best (k = 20, LLM ranker; §C.2
shows it beating embedding and random selection). The ranker returns a JSON object mapping
scenario id to `[rationale, score]`; the top-k heuristics by score are injected into the
solver's system prompt and stay there for the whole task.

Contrast with DAEDALUS's inference, which retrieves per turn against the evolving trajectory
with a lexical/dense retriever and a much smaller k.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

from daedalus.core.llm.client import LLMClient
from daedalus.core.logging import console

from references.erl.heuristics import Heuristic, load_heuristics
from references.erl.reflection import Usage

PROMPTS = Path(__file__).parent / "prompts"

_HEURISTIC_SEPARATOR = "\n\n" + "-" * 60 + "\n\n"


@dataclass
class Selection:
    """One selected heuristic, with the ranker's own justification."""

    heuristic: Heuristic
    score: float
    rationale: str = ""


@dataclass
class Retrieval:
    """Everything one retrieval call produced, for the run's `retrieval/` artifact."""

    task: str
    k: int
    model: str
    # "llm"      — the ranker chose (the paper's method)
    # "all"      — the pool holds at most k heuristics, so ranking is a no-op
    # "fallback" — the ranker's output could not be parsed; first k of the pool
    mode: str
    selections: list[Selection] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def texts(self) -> list[str]:
        """What gets injected into the solver's system prompt."""
        return [s.heuristic.block for s in self.selections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "k": self.k,
            "model": self.model,
            "mode": self.mode,
            "num_selected": len(self.selections),
            "selected": [
                {
                    "scenario_id": s.heuristic.scenario_id,
                    "reward": s.heuristic.reward,
                    "score": s.score,
                    "rationale": s.rationale,
                }
                for s in self.selections
            ],
            "usage": self.usage.to_dict(),
        }


class HeuristicSelector:
    """Ranks a fixed heuristic pool against a task description, once per task."""

    def __init__(
        self,
        heuristics: list[Heuristic],
        k: int = 20,
        model: str = "gpt-5.4",
        reasoning_effort: str | None = "high",
    ):
        self.heuristics = heuristics
        self.k = k
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._by_id = {h.scenario_id: h for h in heuristics}
        # Injected by the solver mixin (attach_llm_client) so the ranker's per-task call
        # lands in the run's usage ledger under the `retriever` role.
        self._llm: LLMClient | None = None
        self._template = Template(
            (PROMPTS / "heuristic_retrieval.txt").read_text(encoding="utf-8")
        )

    def attach_llm_client(self, client: LLMClient | None) -> None:
        self._llm = client

    def _tracked_llm(self) -> LLMClient:
        if self._llm is None:
            raise RuntimeError(
                "ERL's HeuristicSelector ranks with an LLM but no client was attached; "
                "call attach_llm_client() with the run's ledger-writing client."
            )
        return self._llm

    @classmethod
    def from_config(cls, erl) -> "HeuristicSelector":
        """Build from an `ERLConfig` (loads its pool)."""
        if not erl.pool_path:
            raise ValueError(
                "erl.pool_path is not set — an ERL inference run needs the heuristic pool "
                "written by `references.erl.accumulation`."
            )
        return cls(
            heuristics=load_heuristics(erl.pool_path),
            k=erl.k,
            model=erl.ranker_model,
            reasoning_effort=erl.ranker_reasoning_effort,
        )

    def select(self, task: str) -> Retrieval:
        """Return the top-k heuristics for `task` (one LLM call)."""
        if len(self.heuristics) <= self.k:
            # Ranking cannot change the SET, only its order — so skip the call. Keeps a
            # small pool (a short split, or a smoke run) from paying for a no-op.
            return Retrieval(
                task=task,
                k=self.k,
                model=self.model,
                mode="all",
                selections=[Selection(h, score=100.0) for h in self.heuristics],
            )

        usage = Usage()
        listing = _HEURISTIC_SEPARATOR.join(h.block for h in self.heuristics)
        prompt = self._template.render(k=self.k, heuristics=listing, task=task)

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        scored: dict[str, tuple[float, str]] | None = None
        for attempt in (1, 2):
            response = self._tracked_llm().generate(
                messages, reasoning_effort=self.reasoning_effort
            )
            usage.add(self.model, response)
            scored = self._parse(response.content)
            if scored:
                break
            console.info(
                f"[erl] ranker returned unparseable output (attempt {attempt}/2)"
            )
            # Show the model its own bad answer before asking again — a bare repeat of the
            # prompt tends to reproduce the same malformed output.
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": (
                        "That could not be parsed. Reply with ONLY the JSON object described "
                        "above: scenario ids as keys, [rationale, score] as values. No prose, "
                        "no code fences."
                    ),
                },
            ]

        if not scored:
            console.info(
                f"[erl] ranker failed; falling back to the first {self.k} heuristics"
            )
            return Retrieval(
                task=task,
                k=self.k,
                model=self.model,
                mode="fallback",
                selections=[Selection(h, score=0.0) for h in self.heuristics[: self.k]],
                usage=usage,
            )

        ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
        selections = [
            Selection(self._by_id[sid], score=score, rationale=rationale)
            for sid, (score, rationale) in ranked[: self.k]
        ]
        return Retrieval(
            task=task,
            k=self.k,
            model=self.model,
            mode="llm",
            selections=selections,
            usage=usage,
        )

    # ── parsing ────────────────────────────────────────────────────────────

    def _parse(self, content: str) -> dict[str, tuple[float, str]]:
        """Map the ranker's JSON to {scenario_id: (score, rationale)}, dropping junk.

        Ids the pool doesn't contain are dropped rather than trusted: a hallucinated id
        would otherwise silently shrink the injection or raise mid-task.
        """
        obj = _extract_json_object(content)
        if not isinstance(obj, dict):
            return {}
        out: dict[str, tuple[float, str]] = {}
        for raw_id, value in obj.items():
            sid = self._resolve_id(str(raw_id))
            if sid is None:
                continue
            score, rationale = _parse_entry(value)
            if score is None:
                continue
            out[sid] = (score, rationale)
        return out

    def _resolve_id(self, raw: str) -> str | None:
        """Match a returned id to a pool id, tolerating quoting/labelling noise."""
        candidate = raw.strip().strip("\"'")
        if candidate in self._by_id:
            return candidate
        stripped = candidate.removeprefix("Scenario ID:").strip()
        if stripped in self._by_id:
            return stripped
        return None


def _parse_entry(value: Any) -> tuple[float | None, str]:
    """A value is `["rationale", score]`; accept a bare score or dict form too."""
    if isinstance(value, (list, tuple)):
        rationale = next((v for v in value if isinstance(v, str)), "")
        score = next((v for v in value if isinstance(v, (int, float))), None)
        return (float(score) if score is not None else None), rationale
    if isinstance(value, (int, float)):
        return float(value), ""
    if isinstance(value, dict):
        score = value.get("score")
        rationale = value.get("rationale") or value.get("justification") or ""
        return (float(score) if isinstance(score, (int, float)) else None), str(
            rationale
        )
    return None, ""


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")


def _extract_json_object(text: str) -> Any:
    """First parseable JSON object in `text` (handles fences and surrounding prose)."""
    cleaned = _FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
