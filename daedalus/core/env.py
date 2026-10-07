"""Load a `.env` into os.environ at entry-point startup.

daedalus reads all provider config from the environment (API keys, OPENAI_BASE_URL,
OPENAI_SERVICE_TIER, …). This makes a repo-root `.env` "just work" for
`uv run python -m daedalus.scripts.*` without a manual `set -a; . ./.env` first.

Existing environment variables are NOT overridden (a real shell export wins over the
file), matching python-dotenv's default. Spawned worker processes inherit os.environ, so
loading once in the parent entry point is enough.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from a `.env` into os.environ (without overriding existing
    vars). With no path, uses the first `.env` found in the cwd or an ancestor. Returns
    the file that was loaded, or None if none was found.

    Supports `export KEY=VALUE`, `#` comments, blank lines, and single/double-quoted
    values. Not a full dotenv parser (no interpolation) — enough for provider config.
    """
    if path is None:
        for base in [Path.cwd(), *Path.cwd().parents]:
            candidate = base / ".env"
            if candidate.is_file():
                path = candidate
                break
        else:
            return None
    path = Path(path)
    if not path.is_file():
        return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:  # never override an explicit shell export
            os.environ[key] = value
    return path
