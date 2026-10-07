"""Config plumbing shared by the competing methods under `references/`.

Every method reuses daedalus's `ExperimentConfig` for what it shares with DAEDALUS (solver
model, benchmark + split, parallelism, the extraction LLM) and adds only its own knobs, in
a block of the SAME YAML named after the method (`erl:`, `autoguide:`, `expel:`,
`reasoningbank:`). daedalus's loader ignores unknown top-level keys, so one file configures
both halves.

The block is written to `<experiment_dir>/<method>.yaml` when a run starts, for two
reasons: the run folder stays self-describing (like `config.yaml` / `run_meta.json`), and
the spawned solver workers — which receive only a serialized `ExperimentConfig` — read
their method settings back from it (`MethodConfig.load_for`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import yaml

from daedalus.core.config import ExperimentConfig, experiment_dir


@dataclass
class MethodConfig:
    """Base for a method's own config block. Subclasses set `SETUP` and add fields.

    `SETUP` is both the YAML block name and `ExperimentConfig.setup`, which namespaces
    every artifact under `outputs/baselines/<setup>/{memory,inference}/<name>/`.
    """

    SETUP: ClassVar[str] = ""

    @classmethod
    def snapshot_name(cls) -> str:
        return f"{cls.SETUP}.yaml"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        """Build from a raw block, ignoring unknown keys (as daedalus's loader does)."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, config_path: str | Path):
        """Read this method's block out of an experiment YAML."""
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data.get(cls.SETUP))

    @classmethod
    def load_for(cls, cfg: ExperimentConfig):
        """Read the snapshot written into `cfg`'s experiment folder (worker path)."""
        path = experiment_dir(cfg) / cls.snapshot_name()
        if not path.exists():
            raise FileNotFoundError(
                f"No {cls.SETUP} settings found at {path}. A run writes them there at "
                f"startup; launch it with `python -m references.{cls.SETUP}.run "
                f"--config <yaml>` rather than building an agent by hand."
            )
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, cfg: ExperimentConfig) -> Path:
        """Snapshot this block into the run folder, next to `config.yaml`."""
        base = experiment_dir(cfg)
        base.mkdir(parents=True, exist_ok=True)
        path = base / self.snapshot_name()
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path


def require_kind(config_path: str | Path, expected_kind: str, setup: str) -> None:
    """Fail early when a config written for the other entry point is passed to this one.

    Each method ships two configs per benchmark — one for `accumulation`, one for `run` —
    and their names differ by a word. Passing the wrong one used to fail deep inside the
    agent (a missing memory path), which said nothing about the actual mistake.
    """
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    declared = data.get("kind") or expected_kind
    if declared == expected_kind:
        return
    entry = {"accumulation": "accumulation", "inference": "run"}
    raise ValueError(
        f"{config_path} declares `kind: {declared}`, but this is the {expected_kind} entry "
        f"point. Run it with `python -m references.{setup}.{entry.get(declared, declared)} "
        f"--config {config_path}`, or point this command at the "
        f"{entry.get(expected_kind, expected_kind)} config next to it."
    )


def retrieval_dir(cfg: ExperimentConfig, run_idx: int | None = None) -> Path:
    """Where a run records what its retrieval step selected, per task.

    Sibling of the trace dirs: `<experiment_dir>/retrieval/run_<i>/<task>.json`. Kept out
    of the traces so those stay in daedalus's normal shape (the viewer, metrics and pass^k
    all read them unchanged).
    """
    base = experiment_dir(cfg) / "retrieval"
    return base / f"run_{run_idx}" if run_idx is not None else base
