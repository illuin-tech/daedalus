"""Shared helpers for the per-benchmark smoke tests.

A smoke test runs ONE tiny inference plus ONE tiny run of the other task that
benchmark supports — self-play generation where there is a `GenerationBackend`,
accumulation otherwise — to verify the whole harness is wired correctly end-to-end.
If the benchmark's dependencies/environment are not available, it raises a clear,
actionable error (rather than skipping) — per-benchmark, so you can run the
benchmark you have set up without needing the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import daedalus` and `import tests` work whether run as `python tests/smoke_x.py`
# or `python -m tests.smoke_x` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CONFIGS = _REPO_ROOT / "configs"


def require(module: str, extra: str, benchmark: str, note: str = "") -> None:
    """Raise a clear, actionable error if a benchmark's library isn't importable."""
    try:
        __import__(module)
    except Exception as e:  # noqa: BLE001
        msg = (
            f"[{benchmark}] not available: `import {module}` failed ({e.__class__.__name__}: {e}).\n"
            f"  Install it with:  uv sync --extra {extra}"
        )
        if note:
            msg += f"\n  {note}"
        raise RuntimeError(msg) from e


def _load(config_rel: str):
    """Load a smoke config, with the same env bootstrap the CLI entry points do."""
    from daedalus.core.config import load_config
    from daedalus.core.env import load_dotenv

    load_dotenv()  # API keys / OPENAI_SERVICE_TIER / … from a repo-root .env
    return load_config(CONFIGS / config_rel)


def run_inference(config_rel: str) -> None:
    from daedalus.core.evaluation.run_experiment import run_experiment

    cfg = _load(config_rel)
    print(f"[smoke] inference: {config_rel}")
    run_experiment(cfg)


def run_generation(config_rel: str) -> None:
    from daedalus.core.generation.pipeline import generate

    cfg = _load(config_rel)
    print(f"[smoke] generation: {config_rel}")
    generate(cfg, resume=False)


def run_accumulation(config_rel: str) -> None:
    """One tiny `accumulation` run — the second half of the smoke for a benchmark
    that has no self-play generation backend.

    Goes through the CLI entry point (rather than the loop function) so the smoke
    exercises the same config → kind → output-path wiring a real run does.
    """
    from daedalus.scripts.accumulation import main

    print(f"[smoke] accumulation: {config_rel}")
    argv = sys.argv
    sys.argv = ["accumulation", "--config", str(CONFIGS / config_rel)]
    try:
        main()
    finally:
        sys.argv = argv
