"""AutomationBench session: the world, the tool surface, execution, and grading.

Everything is in-process. A session builds one `WorldState` from the task's initial
state, exposes the `api` toolset as native tool schemas, executes calls straight against that
world, and at the end scores the *live* world with the benchmark's own rubric — no
replay, no server, no LLM judge.

The tool schemas are derived here rather than taken from the benchmark: theirs are built
by `verifiers`' `convert_func_to_oai_tool`, which this connector does not depend on (see
`__init__.py`). The toolset is three functions whose parameters are
plain `str`/`int`/`Optional[str]`, so the conversion below is a signature read plus a
docstring, and `world` is stripped exactly as their `tool_wrapper` strips it.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from . import ensure_automationbench_importable
from .mirrored import compute_allowed_services
from .task_loader import Task

# Injected by us from the session's world, never exposed to the model.
_SKIPPED_ARGS = ("world",)

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class ToolCallResult:
    success: bool
    result: Any
    error: str = ""


@dataclass
class Grade:
    """The benchmark's two per-task metrics, plus the per-assertion detail."""

    # task_completed_correctly: every scored assertion passed. The official pass rate.
    success: bool
    # partial_credit: the fraction of scored assertions that passed.
    partial_credit: float
    assertions: list[dict[str, Any]] = field(default_factory=list)


def _json_type(annotation: Any) -> str:
    """The JSON-schema type for a parameter annotation (Optional[X] -> X's type)."""
    if annotation in _JSON_TYPES:
        return _JSON_TYPES[annotation]
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if len(args) == 1 and args[0] in _JSON_TYPES:
        return _JSON_TYPES[args[0]]
    raise TypeError(f"unsupported tool parameter annotation: {annotation!r}")


def openai_schema(func: Callable) -> dict[str, Any]:
    """One tool function as an OpenAI function-calling schema."""
    hints = typing.get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(func).parameters.items():
        if name in _SKIPPED_ARGS:
            continue
        properties[name] = {"type": _json_type(hints[name])}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class AutomationBenchSession:
    """One task's world: the tool surface, the calls, and the final grade."""

    def __init__(self, task: Task, cfg_ab: Any):
        ensure_automationbench_importable(cfg_ab.repo_root)
        from automationbench.schema.world import WorldState

        self.task = task
        self.world = WorldState(**task.initial_state)
        # Service gating: api_fetch answers a credentials error for a service the task
        # is not subscribed to, which is what stops a write into untracked state.
        self.world.meta.allowed_services = compute_allowed_services(
            task.initial_state, task.assertions, task.zapier_tools
        )

        from automationbench.tools.api import API_TOOLS

        self._tools = {t.__name__: t for t in API_TOOLS}
        self.openai_tools = [openai_schema(t) for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> ToolCallResult:
        """Execute one tool call against this session's world."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolCallResult(
                success=False,
                result=None,
                error=f"Tool {name!r} not found. Available: {', '.join(sorted(self._tools))}",
            )
        # An empty object is the "no value" sentinel some models emit for an optional
        # argument; dropping the key lets the parameter's own default apply. Verbatim
        # from their runner's update_tool_args, and collision-free because no parameter
        # takes a meaningful empty dict (bodies are passed as JSON strings).
        call_args = {
            k: v for k, v in args.items() if not (isinstance(v, dict) and len(v) == 0)
        }
        if "world" in inspect.signature(tool).parameters:
            call_args["world"] = self.world
        try:
            return ToolCallResult(success=True, result=tool(**call_args))
        except Exception as e:  # a tool error is an observation, not a crash
            return ToolCallResult(
                success=False, result=None, error=f"{type(e).__name__}: {e}"
            )

    def grade(self) -> Grade:
        """Score the live world with the benchmark's own rubric.

        `partial_credit` does the whole assertion sweep and caches its result in the
        state dict; `task_completed_correctly` reads that cache, so the two calls cost
        one sweep. Assertions that already held in the initial state are excluded (no
        credit for doing nothing) unless the agent broke them.

        Assertion errors raise here by default (`AUTOMATIONBENCH_STRICT_ASSERTIONS`),
        which the agent records as a failed trace with `extra.error` rather than a
        silent zero.
        """
        from automationbench.rubric import partial_credit, task_completed_correctly

        state: dict[str, Any] = {
            "info": {"assertions": self.task.assertions},
            "world": self.world,
            "initial_state": self.task.initial_state,
        }
        fraction = partial_credit(state)
        success = bool(task_completed_correctly(state))
        return Grade(
            success=success,
            partial_credit=fraction,
            # Their `_end_state` (the whole final world) is deliberately dropped: it is
            # megabytes per task and nothing downstream reads it.
            assertions=[
                {
                    "type": a["type"],
                    "passed": a["passed"],
                    "excluded": a.get("excluded", False),
                }
                for a in state.get("_assertion_results", [])
            ],
        )


def all_service_names() -> set[str]:
    """Every service `WorldState` models — the vocabulary a tag label can name."""
    from automationbench.schema.world import WorldState

    return {str(f) for f in WorldState.model_fields if f != "meta"}


def render_observation(res: ToolCallResult) -> str:
    """One tool result as the string handed back to the model."""
    if not res.success:
        return res.error
    if isinstance(res.result, str):
        return res.result
    return json.dumps(res.result, ensure_ascii=False, default=str)
