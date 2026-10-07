"""AppWorld tool-call paths (`app.tool` sequences) from code and generated plans.

Used to ground an explorer's spec (its expected path must have been executed) and as
the path the novelty score (`core/generation/novelty.py`) compares.
"""

from __future__ import annotations

import re
from typing import Any

# Matches `apis.<app>.<tool>(` in agent code, e.g. apis.spotify.search_songs(...)
_API_CALL_RE = re.compile(r"\bapis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(")


def path_from_code(code: str) -> list[str]:
    """Ordered `app.tool` tokens from a single code block (call order preserved)."""
    return [f"{app}.{tool}" for app, tool in _API_CALL_RE.findall(code)]


def predicted_path(expected_path: dict[str, Any] | list[Any]) -> list[str]:
    """The `app.tool` sequence a generated task's expected plan intends to make.

    Accepts the generator's expected-path shape — a list of step dicts, or a
    dict of step/tool_call pairs — and pulls the `tool_call` tokens in order.
    Each tool_call is normalized to its `app.tool` head, dropping arguments.
    """
    steps = expected_path.values() if isinstance(expected_path, dict) else expected_path
    path: list[str] = []
    for step in steps:
        call = step.get("tool_call") if isinstance(step, dict) else step
        if isinstance(call, str):
            path.extend(path_from_code(call) or [_normalize_call(call)])
    return [tok for tok in path if tok]


def _normalize_call(call: str) -> str:
    """Reduce a free-form tool_call string to its `app.tool` head.

    Handles `apis.app.tool(...)`, `app.tool(...)`, and bare `app.tool`.
    """
    call = call.strip()
    m = _API_CALL_RE.search(call)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    head = call.split("(", 1)[0].strip()
    head = head[len("apis."):] if head.startswith("apis.") else head
    parts = head.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else ""
