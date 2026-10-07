"""Adapter invariants for the AutomationBench connector, checked without an LLM.

Two things in this connector can be silently wrong, and both are covered here.

The first is the pair of functions mirrored from the benchmark's own runner (see
`benchmarks/automationbench/mirrored.py`). We cannot import that module — it imports
`verifiers` — so the test extracts the originals from the checkout's source with `ast`,
executes just those definitions, and compares their output to ours on every public task.
That keeps the mirror honest across upgrades without importing the dependency chain.

The second is the live path: tool exposure, argument handling, world mutation and the
assertion sweep. The test drives a real task with real `api_fetch` calls and asserts the
score moves the way the rubric says it should.

Skipped (not failed) when the checkout or `datasets` is absent, so the suite still runs
for someone who only set up the other benchmarks.
"""

from __future__ import annotations

import ast
import json

import pytest

from daedalus.core.config import config_from_dict

pytest.importorskip("datasets", reason="AutomationBench's task loaders need datasets")

try:
    from daedalus.benchmarks.automationbench import ensure_automationbench_importable

    ROOT = ensure_automationbench_importable(None)
except (FileNotFoundError, ImportError) as e:  # no checkout on this machine
    pytest.skip(f"AutomationBench checkout not available: {e}", allow_module_level=True)

from daedalus.benchmarks.automationbench.env import AutomationBenchSession
from daedalus.benchmarks.automationbench.explorer import (
    _COVERAGE_TOOL,
    AutomationBenchExplorer,
)
from daedalus.benchmarks.automationbench.mirrored import (
    compute_allowed_services,
    strip_none_values,
)
from daedalus.benchmarks.automationbench.task_loader import load_tasks

# One task whose write path is known: mark a Salesforce opportunity Closed Won.
_SALES_TASK = "sales.multi_hop_lookup"
_OPPORTUNITY_URL = (
    "https://yourinstance.salesforce.com/services/data/v61.0/sobjects/Opportunity/"
    "006xx000004MER1"
)


def _cfg(**automationbench):
    return config_from_dict(
        {"benchmark": "automationbench", "automationbench": automationbench}
    )


@pytest.fixture(scope="module")
def all_tasks():
    return load_tasks(
        _cfg(
            domains=[
                "sales",
                "marketing",
                "operations",
                "support",
                "finance",
                "hr",
                "simple",
            ]
        )
    )


@pytest.fixture(scope="module")
def upstream():
    """The originals of the two mirrored functions, executed straight from source.

    `automationbench/runner.py` imports `verifiers` at module level, so it cannot be
    imported. Only the definitions we mirror are extracted and compiled; they depend on
    nothing but `WorldState`.
    """
    from automationbench.schema.world import WorldState

    source = (ROOT / "automationbench" / "runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_funcs = {
        "strip_none_values",
        "_service_for_name",
        "compute_allowed_services",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted_funcs)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_SERVICE_FIELDS"
                for t in node.targets
            )
        )
    ]
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    assert found == wanted_funcs, f"runner.py no longer defines {wanted_funcs - found}"

    namespace: dict = {"WorldState": WorldState}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<runner>", "exec"), namespace
    )
    return namespace


def test_tasks_load(all_tasks):
    """800 public tasks, uniquely identified, each with assertions and a contract hash."""
    assert len(all_tasks) == 800
    by_domain: dict[str, int] = {}
    for task in all_tasks.values():
        by_domain[task.domain] = by_domain.get(task.domain, 0) + 1
        assert task.task_id.startswith(task.domain + ".")
        assert task.instruction and task.system_prompt
        assert task.assertions
        assert len(task.contract_sha256) == 64
    assert by_domain == {
        "sales": 100,
        "marketing": 100,
        "operations": 100,
        "support": 100,
        "finance": 100,
        "hr": 100,
        "simple": 200,
    }


def test_allowed_services_match_upstream(all_tasks, upstream):
    """Our mirror agrees with the benchmark's own function on every task.

    This is the one that matters most: `api_fetch` rejects any service outside this set,
    so a divergence changes what the solver is allowed to do.
    """
    for task in all_tasks.values():
        theirs = upstream["compute_allowed_services"](
            task.initial_state, task.assertions, task.zapier_tools
        )
        ours = compute_allowed_services(
            task.initial_state, task.assertions, task.zapier_tools
        )
        assert ours == theirs, task.task_id
        assert ours, task.task_id  # every task subscribes to something


def test_strip_none_values_matches_upstream(upstream):
    payload = {
        "a": None,
        "b": [1, None, {"c": None, "d": 2}],
        "e": {"f": {"g": None}},
        "h": 0,
        "i": False,
        "j": "",
    }
    assert strip_none_values(payload) == upstream["strip_none_values"](payload)


def test_tool_schemas(all_tasks):
    """The model sees the `api` toolset's functions, and never the injected `world`."""
    task = all_tasks[_SALES_TASK]

    api = AutomationBenchSession(task, _cfg().automationbench)
    names = [t["function"]["name"] for t in api.openai_tools]
    assert names == ["api_search", "api_fetch", "base64_encode"]
    fetch = next(t for t in api.openai_tools if t["function"]["name"] == "api_fetch")
    params = fetch["function"]["parameters"]
    assert "world" not in params["properties"]
    assert params["required"] == ["method", "url"]
    assert params["properties"]["params"]["type"] == "string"
    assert "api_search" in fetch["function"]["description"]


def test_search_read_write_and_grade(all_tasks):
    """Drive one task's real tool path and watch the rubric follow the world."""
    session = AutomationBenchSession(all_tasks[_SALES_TASK], _cfg().automationbench)

    assert session.world.meta.allowed_services == [
        "gmail",
        "google_drive",
        "google_sheets",
        "salesforce",
    ]

    # Untouched world: nothing is earned, and the negative assertions are excluded
    # rather than counted as free credit.
    before = session.grade()
    assert before.partial_credit == 0.0
    assert before.success is False
    assert any(a["excluded"] for a in before.assertions)

    found = session.call("api_search", {"query": "salesforce update opportunity stage"})
    assert found.success
    assert "sobjects/Opportunity" in found.result

    # An empty object is the "no value" sentinel, dropped so the parameter default
    # applies — the same normalization the benchmark's runner does.
    written = session.call(
        "api_fetch",
        {
            "method": "PATCH",
            "url": _OPPORTUNITY_URL,
            "body": json.dumps({"StageName": "Closed Won"}),
            "params": {},
        },
    )
    assert written.success
    assert "error" not in written.result

    after = session.grade()
    assert after.partial_credit > before.partial_credit
    stage = next(a for a in after.assertions if a["type"] == "salesforce_field_equals")
    assert stage["passed"] is True


def test_out_of_scope_service_is_refused(all_tasks):
    """A write to a service the task does not subscribe to fails like a real workspace."""
    session = AutomationBenchSession(all_tasks[_SALES_TASK], _cfg().automationbench)
    assert "slack" not in session.world.meta.allowed_services
    res = session.call(
        "api_fetch",
        {
            "method": "POST",
            "url": "https://slack.com/api/chat.postMessage",
            "body": json.dumps({"channel": "#sales", "text": "hi"}),
        },
    )
    assert res.success  # the call runs; the API answers with an error payload
    assert "401" in res.result or "credential" in res.result.lower()


def test_unknown_tool_is_an_observation(all_tasks):
    session = AutomationBenchSession(all_tasks[_SALES_TASK], _cfg().automationbench)
    res = session.call("send_email", {})
    assert res.success is False
    assert "not found" in res.error


def test_coverage_tool_uses_parser_tag_key():
    """Coverage submissions must use the key consumed by coverage.parse_tags."""
    item_schema = _COVERAGE_TOOL["function"]["parameters"]["properties"]["tags"][
        "items"
    ]
    assert item_schema["required"] == ["tag", "definition", "target_share"]
    assert "tag" in item_schema["properties"]
    assert "name" not in item_schema["properties"]


def test_explorer_tracks_api_error_payloads_as_failed_writes():
    """A tool call can succeed while the simulated vendor API rejects the write."""
    explorer = object.__new__(AutomationBenchExplorer)
    explorer._write_log = {"ok": set(), "failed": {}}
    args = {"method": "POST", "url": "https://example.test/permissions"}

    explorer._record_call("api_fetch", args, '{"error": {"code": 403}}')
    assert explorer._unresolved_writes() == [
        'POST https://example.test/permissions — {"error": {"code": 403}}'
    ]

    explorer._record_call("api_fetch", args, '{"id": "permission-1"}')
    assert explorer._unresolved_writes() == []
