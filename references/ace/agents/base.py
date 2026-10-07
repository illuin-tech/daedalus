"""ACE's solver hook: the trained playbook, injected verbatim at task start.

`ace-appworld/experiments/code/ace/evaluation_react.py` renders the WHOLE playbook into
the agent's system prompt once, before the first turn, and never touches it again — no
retrieval, no per-turn injection, no use of the bullet ids at evaluation time. That is
daedalus's `heuristics_at_start` policy with a different pool, so the only thing these
subclasses do is put the playbook text where each benchmark's prompt builder reads it.

The playbook goes in verbatim, section headers and bullet ids intact, through the
`playbook` slot in `core/prompts/agent/solver_system.txt` — NOT through the `heuristics`
list, which renumbers its entries and would flatten ACE's structure away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console

from references.ace.config import ACEConfig
from references.ace.playbook import bullets, empty_playbook
from references.common.solver import MethodSolverMixin


@dataclass
class PlaybookRecord:
    """What was injected for one task-run. ACE selects nothing, so this is a size record.

    It is the only per-task evidence of what the solver was shown — in at-start mode the
    trace does not carry the system prompt — so it is worth writing even though there is
    no selection to audit.
    """

    num_bullets: int = 0
    num_chars: int = 0
    sections: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_bullets": self.num_bullets,
            "num_chars": self.num_chars,
            "sections": self.sections,
        }


def load_playbook(path: str | None, *, required: bool) -> str:
    """The playbook text at `path`.

    `required` is on for inference, where BOTH a missing file and a bullet-less one are
    refused. An empty playbook would score exactly like the baseline while being filed and
    reported as an ACE run, and in at-start mode nothing in the trace records what was
    injected — the failure `MemoryPool.load_for_solver` exists to prevent. The bullet-less
    case is the likely one in practice: accumulation writes the bare section skeleton
    before its first wave, so the path exists and looks plausible from the moment an
    adaptation run starts, and stays that way until the first wave finishes.

    During accumulation the playbook is BEING built, so both cases simply mean "nothing
    banked yet".
    """
    text = Path(path).read_text(encoding="utf-8") if path and Path(path).exists() else None
    if not required:
        return text if text is not None else empty_playbook()
    if text is None:
        raise FileNotFoundError(
            f"ace.playbook_path does not point at a playbook ({path!r}). An ACE inference "
            "run needs the playbook.txt written by `python -m references.ace.accumulation`."
        )
    if not bullets(text):
        raise ValueError(
            f"The playbook at {path!r} has no bullets — only the empty section skeleton. "
            "An adaptation run writes that before its first wave, so this usually means it "
            "is still running or died before banking anything. Injecting it would produce a "
            "baseline-equivalent run filed as ACE."
        )
    return text


class ACESolverMixin(MethodSolverMixin):
    """Adds ACE's at-start playbook injection to a benchmark's TaskAgent."""

    method_config_cls = ACEConfig

    def _init_ace(self, cfg: ExperimentConfig) -> None:
        self._init_method(cfg)
        text = load_playbook(
            self.method_cfg.playbook_path, required=cfg.kind == "inference"
        )
        # The attribute each benchmark's prompt builder renders into the `playbook` slot.
        self.solver_playbook = text
        entries = bullets(text)
        sections: dict[str, int] = {}
        for section, _, _ in entries:
            sections[section] = sections.get(section, 0) + 1
        self._playbook_record = PlaybookRecord(
            num_bullets=len(entries), num_chars=len(text), sections=sections
        )
        console.detail(
            f"[ace] playbook: {len(entries)} bullet(s), {len(text)} chars",
            verbose=cfg.logging.verbose,
        )

    def _mark_playbook(self) -> None:
        """Record what this task-run was shown (see `MethodSolverMixin._save_retrieval`)."""
        self._retrieval = self._playbook_record
