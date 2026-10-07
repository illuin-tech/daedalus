"""Pin which identifier classes the tau2 leak gate treats as discoverable, per domain.

`generation.withhold_discoverable_ids` rejects a generated spec whose customer volunteers
an id the solver could have looked up. What counts as look-up-able is DECLARED per domain
in `spec.py::_ID_CLASSES`, read off the domain's tool signatures and its written policy.

Declared, not inferred — so the failure mode is an entity type or a read tool that nobody
classified, which would silently stop being gated. Two tests close that: every identifier
class in the database, and every class a read tool consumes, must be either gated or a
declared root. Retail's gated set and count are pinned exactly, because that is what the
published column was scored with.

These build the real environments (no LLM, no tool calls, no network) and are skipped when
tau2-bench is not installed.
"""

from __future__ import annotations

import pytest

tau2_runner = pytest.importorskip(
    "tau2.runner", reason="tau2-bench not installed (uv sync --extra tau2)"
)

from daedalus.benchmarks.tau2.spec import (  # noqa: E402
    _READ_PREFIXES,
    _discoverable_ids,
    _harvest,
    id_classes,
    leaked_identifier_reason,
)


def _classes(domain: str) -> set[str]:
    """The identifier CLASSES the gate marks discoverable in this domain."""
    classes, _ = _harvest(
        tau2_runner.build_environment(domain).tools.db.model_dump(mode="json")
    )
    ids = _discoverable_ids(domain)
    return {cls for cls, values in classes.items() if values & ids}


# What each domain's tools and policy support. `roots` are the classes the customer is the
# only source of, so gating one would make the task impossible rather than harder.
EXPECTED = {
    # find_user_id_by_name_zip turns a name and a zip into a user_id and everything else
    # follows, so retail has no roots. This is the domain the published results were
    # produced on: the set must not move.
    "retail": {
        "gated": {"id", "item_id", "order_id", "payment_method_id", "product_id", "user_id"},
        "roots": set(),
        "num_ids": 2836,
    },
}


@pytest.mark.parametrize("domain", sorted(EXPECTED))
def test_declaration_matches_the_code(domain):
    gated, roots = id_classes(domain)
    assert set(gated) == EXPECTED[domain]["gated"]
    assert set(roots) == EXPECTED[domain]["roots"]


@pytest.mark.parametrize("domain", sorted(EXPECTED))
def test_discoverable_classes_are_gated_minus_roots(domain):
    expected = EXPECTED[domain]["gated"] - EXPECTED[domain]["roots"]
    assert _classes(domain) == expected


@pytest.mark.parametrize("domain", sorted(EXPECTED))
def test_gated_id_count_is_pinned(domain):
    """Retail's count is the one that matters: it is what the published column was scored
    with, so a change here means the paper's retail numbers no longer describe the code."""
    assert len(_discoverable_ids(domain)) == EXPECTED[domain]["num_ids"]


@pytest.mark.parametrize("domain", sorted(EXPECTED))
def test_every_identifier_class_in_the_database_is_classified(domain):
    """The invariant that replaces the old reachability closure.

    `_ID_CLASSES` is declared rather than inferred, so the risk is a tau2-bench upgrade
    adding an entity type that nobody classifies — it would silently stop being gated.
    Every class `_harvest` finds must be either gated or a root.
    """
    classes, _ = _harvest(
        tau2_runner.build_environment(domain).tools.db.model_dump(mode="json")
    )
    gated, roots = id_classes(domain)
    unclassified = set(classes) - set(gated) - set(roots)
    assert not unclassified, (
        f"{domain}: identifier class(es) {sorted(unclassified)} exist in the database but "
        f"are neither gated nor declared roots in spec.py::_ID_CLASSES"
    )


@pytest.mark.parametrize("domain", sorted(EXPECTED))
def test_every_class_a_read_tool_consumes_is_classified(domain):
    """The other half: a new READ tool that takes an id we do not classify must fail here.

    A class a read tool consumes is either look-up-able or a root; there is no third
    option, and leaving it out is how the gate would quietly stop firing.
    """
    env = tau2_runner.build_environment(domain)
    classes, _ = _harvest(env.tools.db.model_dump(mode="json"))
    consumed: set[str] = set()
    for tool in env.get_tools():
        if not tool.name.startswith(_READ_PREFIXES):
            continue
        schema = tool.openai_schema.get("function") or tool.openai_schema
        params = (schema.get("parameters") or {}).get("properties") or {}
        consumed |= {p for p in params if p in classes}
    gated, roots = id_classes(domain)
    assert not (consumed - set(gated) - set(roots))


def test_an_unclassified_domain_fails_open():
    """A domain nobody has classified gates nothing, rather than everything.

    Over-gating is the expensive failure: demanding an id the customer is the only source
    of makes a task impossible.
    """
    assert id_classes("banking_knowledge") == (frozenset(), frozenset())


def test_retail_gates_a_real_order_id():
    """The end-to-end behaviour: a volunteered order id is refused with a reason."""
    ids = _discoverable_ids("retail")
    order_id = next(i for i in ids if i.startswith("#W"))
    spec = {
        "scenario": {
            "reason_for_call": f"Please cancel order {order_id}.",
            "known_info": "You are Sara Doe, zip 12345.",
            "behavior": "You answer questions when asked.",
        }
    }
    reason = leaked_identifier_reason(spec, "retail")
    assert order_id in reason and "looked up" in reason


def test_a_described_target_is_not_gated():
    """Describing the target instead of naming it is exactly what the gate wants."""
    spec = {
        "scenario": {
            "reason_for_call": "Please cancel the boots I ordered last month.",
            "known_info": "You are Sara Doe, zip 12345.",
            "behavior": "You do not remember your order number.",
        }
    }
    assert leaked_identifier_reason(spec, "retail") == ""


