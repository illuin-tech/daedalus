"""AppWorld connector.

Importing this package selects a safe APPWORLD_ROOT before any `import appworld`
runs. AppWorld's apply_db_changes() rejects any DB path containing the substring
"memory" (our repo folder trips that), so we redirect the data root to a sibling
`appworld/` checkout. This runs only when the AppWorld benchmark is actually used
(the registry imports this package lazily), so tau2/automationbench runs never
require appworld to be installed.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path


def _find_sibling_appworld() -> str | None:
    """A sibling `appworld/` checkout with a data/ dir and no "memory" in its path
    (works from a normal checkout and from git worktrees), searching upward. None if
    not found."""
    here = _Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "appworld"
        if (
            candidate.is_dir()
            and (candidate / "data").is_dir()
            and "memory" not in str(candidate)
        ):
            return str(candidate)
    return None


def appworld_root() -> str | None:
    """The resolved AppWorld data root: the current APPWORLD_ROOT if it is set and safe
    (no "memory" in the path, has a data/ dir), otherwise a sibling checkout. None if
    neither exists. One source of truth for both the import-time env fix below and the
    evaluator's output lookup in runner.py."""
    current = _os.environ.get("APPWORLD_ROOT")
    if current and "memory" not in current and (_Path(current) / "data").is_dir():
        return current
    return _find_sibling_appworld()


# On import, point APPWORLD_ROOT at a safe root when the current one is unusable.
_root = appworld_root()
if _root and _os.environ.get("APPWORLD_ROOT") != _root:
    _os.environ["APPWORLD_ROOT"] = _root
