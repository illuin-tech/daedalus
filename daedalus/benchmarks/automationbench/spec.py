"""The AutomationBench task spec: what an explorer emits, and how it becomes a task.

A spec is the AppWorld shape — `{task, expected_path, success_conditions}`
— because AutomationBench cannot grade a generated task programmatically. Its own tasks
are checked by 571 registered assertion types over the final world; asking an explorer to
author assertions in that vocabulary is not realistic, so generated tasks are judged
against natural-language success conditions by the LLM judge instead.

A generation *context* is one of the benchmark's own task worlds: its `initial_state`,
with the assertions and tool grants dropped so the world is subscribed to exactly the
services it seeds. The explorer plays there, and the solver later attempts the generated
task in a fresh copy of the same world.
"""

from __future__ import annotations

import hashlib
from typing import Any

from daedalus.core.config import ExperimentConfig

from .task_loader import Task, load_tasks

SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "The task statement, exactly as a colleague would write it: what to do, "
                "against which real records, under which rules. Self-contained — the "
                "solver sees only this text and the environment."
            ),
        },
        "expected_path": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The ordered calls your reference solution actually made, one per entry, "
                'as "METHOD url" (e.g. "PATCH .../sobjects/Opportunity/006xx000004MER1").'
            ),
        },
        "success_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Outcome conditions a judge can check against the end state, one per "
                "entry. State the observable result, never the route taken to it."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The coverage tags this task belongs to, taken from the run's tag "
                "vocabulary. Without them the coverage tally cannot count the task and "
                "every tag stays UNDER for the whole run."
            ),
        },
    },
    "required": ["task", "expected_path", "success_conditions"],
}


def parse_spec(obj: Any) -> tuple[dict[str, Any] | None, str]:
    """Validate a submitted spec. Returns (spec, "") or (None, rejection reason)."""
    if not isinstance(obj, dict):
        return None, "the submission must be an object with task, expected_path and success_conditions"
    task = str(obj.get("task") or "").strip()
    if len(task) < 40:
        return None, "task is missing or too short to be a real request; write the full statement"
    path = [str(p).strip() for p in (obj.get("expected_path") or []) if str(p).strip()]
    if not path:
        return None, "expected_path is empty; list the calls your reference solution actually made"
    conditions = [str(c).strip() for c in (obj.get("success_conditions") or []) if str(c).strip()]
    if not conditions:
        return None, "success_conditions is empty; state what must hold in the end state"
    spec = {"task": task, "expected_path": path, "success_conditions": conditions}
    tags = [str(x).strip() for x in (obj.get("tags") or []) if str(x).strip()]
    if tags:
        spec["tags"] = tags
    return spec, ""


_HOST_TO_SERVICE: dict[str, str] | None = None


def _host_to_service() -> dict[str, str]:
    """host -> service name, read from the benchmark's own endpoint schemas.

    Splitting the host on dots does NOT work: gmail, sheets, drive and calendar all live
    under `*.googleapis.com`, so a naive `host.split(".")[-2]` collapses them to
    "googleapis" and the novelty measure goes blind — on the first 30-session run it saw
    only 4 distinct tokens across 23 banked tasks.
    """
    global _HOST_TO_SERVICE
    if _HOST_TO_SERVICE is None:
        import json as _json

        from . import ensure_automationbench_importable

        root = ensure_automationbench_importable(None)
        mapping: dict[str, str] = {}
        for f in (root / "automationbench" / "tools" / "api" / "schemas").glob("*.jsonc"):
            text = "\n".join(
                line for line in f.read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("//")
            )
            d = _json.loads(text)
            host = str(d.get("baseUrl", "")).split("//")[-1].split("/")[0]
            if host:
                mapping.setdefault(host, d["api"])
        _HOST_TO_SERVICE = mapping
    return _HOST_TO_SERVICE


def spec_path(spec: dict[str, Any]) -> list[str]:
    """Ordered tool tokens of the intended solution, for the novelty measure.

    A path entry is "METHOD url", too specific to compare across tasks (record ids differ
    every time). The token is the service plus the HTTP verb — "salesforce PATCH" — which
    is what distinguishes one solution shape from another. The service comes from the
    schema's own baseUrl table (see `_host_to_service`), never from the host string.
    """
    hosts = _host_to_service()
    tokens: list[str] = []
    for entry in spec.get("expected_path") or []:
        parts = str(entry).split()
        method = parts[0].upper() if parts else ""
        url = parts[1] if len(parts) > 1 else ""
        host = url.split("//")[-1].split("/")[0]
        service = hosts.get(host) or host or "?"
        tokens.append(f"{service} {method}".strip())
    return tokens


def build_contexts(cfg: ExperimentConfig) -> dict[str, Task]:
    """The worlds sessions rotate through: the configured split's tasks, stripped.

    Assertions and tool grants are dropped, so `compute_allowed_services` subscribes the
    world to exactly the services its own `initial_state` seeds — the explorer may use
    every app the company actually has, and no others.
    """
    contexts: dict[str, Task] = {}
    for task_id, task in load_tasks(cfg).items():
        contexts[task_id] = Task(
            task_id=task_id,
            domain=task.domain,
            example_id=task.example_id,
            instruction="",
            system_prompt=task.system_prompt,
            initial_state=task.initial_state,
            assertions=[],
            zapier_tools=[],
            contract_sha256=task.contract_sha256,
        )
    return contexts


SYNTHETIC_CONTEXT_ID = "synthetic_all_services"


def build_synthetic_context(cfg: ExperimentConfig) -> tuple[Task, dict[str, str]]:
    """One world seeded with every service the split's worlds seed between them.

    A real context seeds 2-7 of the 22 services, so an agent given ONE of them sees about
    an eighth of the domain. That is fine for generation, which rotates sessions across
    every world, but it silently handicaps any single-session baseline — the naive
    explorer — for a reason that has nothing to do with the loop being ablated. Merging
    the split's `initial_state` blocks, first-wins per service in sorted task order, gives
    a single world subscribed to all 22.

    Returns the world and a `service -> source task_id` provenance map, because this world
    is an ARTEFACT: no real task world looks like it, and anything measured in it should be
    traceable back to the worlds it was assembled from.

    Two properties make the merge sound rather than merely convenient, both measured on the
    operations train split:

    - Union of the 30 train worlds is exactly the 22 services, so nothing is missing and
      nothing had to be invented.
    - Identities are no less coherent than in a real world. Mean cross-service email overlap
      is 0.01 in real train worlds and 0.01 here — AutomationBench worlds already share
      almost no identities between apps, so stitching two companies' data together breaks
      nothing that was intact to begin with.

    `meta` is taken whole from the first task rather than merged: it carries `current_time`,
    and interleaving two worlds' clocks would be worse than picking one. Only 1 of 30 train
    worlds sets it at all.
    """
    tasks = load_tasks(cfg)
    if not tasks:
        raise SystemExit(
            "no tasks to build a synthetic world from — check automationbench.domains, "
            "split_file and split"
        )
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for task_id, task in sorted(tasks.items()):
        for service, state in task.initial_state.items():
            if service == "meta" or service in merged:
                continue
            merged[service] = state
            provenance[service] = task_id
    first = tasks[sorted(tasks)[0]]
    if "meta" in first.initial_state:
        merged["meta"] = first.initial_state["meta"]
    return (
        Task(
            task_id=SYNTHETIC_CONTEXT_ID,
            domain=first.domain,
            example_id=first.example_id,
            instruction="",
            system_prompt=first.system_prompt,
            initial_state=merged,
            assertions=[],
            zapier_tools=[],
            contract_sha256="",
        ),
        provenance,
    )


def spec_to_task(spec: dict[str, Any], ctx: Task) -> Task:
    """A generated spec as a runnable Task in the context's world.

    No assertions: a generated task is graded by the LLM judge over its
    `success_conditions`, never by the benchmark's own rubric.
    """
    digest = hashlib.sha256(spec["task"].encode("utf-8")).hexdigest()[:8]
    return Task(
        task_id=f"gen.{ctx.task_id}.{digest}",
        domain=ctx.domain,
        example_id=ctx.example_id,
        instruction=spec["task"],
        system_prompt=ctx.system_prompt,
        initial_state=ctx.initial_state,
        assertions=[],
        zapier_tools=[],
        contract_sha256="",
    )
