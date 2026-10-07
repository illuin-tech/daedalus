"""Two functions mirrored from AutomationBench's own `automationbench/runner.py`.

We cannot import that module: it imports `verifiers` at module level, and the whole
point of this connector is to run the benchmark without that dependency chain (see
`__init__.py`). Both functions below are byte-for-byte behaviour copies of theirs, and
`tests/test_automationbench_adapter.py` pins them by extracting the originals from the
checkout's source and comparing outputs on every public task.

They are not incidental detail:

* `compute_allowed_services` decides `world.meta.allowed_services`, and `api_fetch`
  answers a credentials error for any service outside that set. Get it wrong and the
  task the solver faces is not the task the benchmark defines.
* `strip_none_values` removes the `None`s HuggingFace's schema normalization adds, so
  pydantic's `default_factory` applies instead of receiving `None`.
"""

from __future__ import annotations

from typing import Any


def strip_none_values(obj: Any) -> Any:
    """Recursively strip None values from nested dicts and lists."""
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    return obj


def _service_fields() -> list[str]:
    """WorldState's service field names, longest first (so "google_sheets" wins over
    a hypothetical "google")."""
    from automationbench.schema.world import WorldState

    return sorted(
        (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
    )


def _service_for_name(name: str, fields: list[str]) -> str | None:
    """Map an assertion type or tool name to its WorldState service field."""
    for field in fields:
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict, assertions: list[dict], zapier_tools: list[str]
) -> list[str]:
    """The set of services a task's world is subscribed to.

    A service is in scope when the task seeds it (key present in initial_state, even if
    empty — presence signals intent), asserts on it, or grants one of its Zapier tools.
    """
    from automationbench.schema.world import WorldState

    fields = _service_fields()
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for a in assertions or []:
        service = _service_for_name(str(a.get("type", "")), fields)
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name, fields)
        if service:
            allowed.add(service)
    return sorted(allowed)
