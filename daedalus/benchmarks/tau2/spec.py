"""tau2 generated-task specs: schema, grounding replay, native Task conversion.

A tau2 explorer emits a generic conversational-task spec:

    {"purpose": str,           # what the task tests (diagnostic)
     "scenario": {             # a generic customer-conversation description
        "reason_for_call": str,   # the request itself (2nd person)
        "known_info": str,        # facts the customer can state from memory — facts only
        "behavior": str,          # the customer's script: demeanour, disclosure rule, reactions
        "unknown_info": str | None},  # optional: what the customer cannot provide
     "actions": [{"name": str, "arguments": dict}, ...],  # grounded reference solution
     "success_conditions": [str, ...],  # outcome-based checks, graded by the LLM judge
     "communicate_info": [str, ...],    # optional: substrings the agent must say to the customer
     "nl_assertions": [str, ...]}       # optional: NL claims about what the agent told them

The scenario is a benchmark-agnostic representation of a conversational task;
`spec_to_task` is the only tau2-specific adapter, mapping it onto tau2's
`StructuredUserInstructions` (`behavior` → `task_instructions`).

Grounding (`ground_spec`) replays the actions on a fresh environment: every call
must succeed, argument literals not produced by earlier tool outputs must appear in
the user-revealable scenario text (what the simulated customer can convey), and every
`communicate_info` value must appear in the replayed output (otherwise the agent is
asked to report something it cannot learn).

The DB end-state hash must change UNLESS the spec carries `nl_assertions`. tau2
grades both, and the retail train split relies on it: 112 of its 114 tasks carry
reward_basis [DB, NL_ASSERTION], 36 carry `communicate_info`, 40 carry `nl_assertions`,
and 2 have a correct end state in which nothing changed at all. So a reported-answer
task is first-class here; only a task that neither changes state nor reports anything
checkable is ungradable (a do-nothing agent would score 1.0).

A grounded spec converts to a first-class tau2 Task whose reward_basis is [DB] plus
NL_ASSERTION when the spec supplies `nl_assertions` — the shape 112 of retail's 114
train tasks use. `communicate_info` is carried but never added to the basis, matching
train (36 of its tasks have it, none grade on it): the check is a bare substring test,
so it informs rather than decides. DB stays in every basis: for an answer-only task it
is what catches an agent that mutated state it should have left alone.

Generated tasks are graded by the LLM judge over `success_conditions`. `actions` are always
required: they ground the spec (replayed on a fresh environment) and define the native task.
"""

from __future__ import annotations

import json
import re as _re
import uuid
from typing import Any

SPEC_REQUIRED_KEYS = ("purpose", "scenario", "actions", "success_conditions")
SCENARIO_REQUIRED_KEYS = ("reason_for_call", "known_info", "behavior")

# Coverage tags (see core/generation/coverage.py). Optional: empty when the run has no
# coverage manifest. Taken exactly as declared; tags outside the manifest are dropped
# when the task is banked, so no validation happens here.
TAGS_PROP: dict[str, Any] = {
    "type": "array",
    "description": (
        "The coverage tags that define this task, copied verbatim from the coverage list "
        "in your instructions — every one that genuinely applies, and no others. Omit if you "
        "were not given a coverage list."
    ),
    "items": {"type": "string"},
}

# The JSON schema handed to the explorer's submit_task_spec tool.
SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "purpose": {
            "type": "string",
            "description": "One or two sentences: what this task tests, including which policy rules it exercises.",
        },
        "scenario": {
            "type": "object",
            "description": (
                "The customer's situation, written so a role-playing user can act it out in "
                "conversation. Write every field in the second person ('You are ...', 'You "
                "want ...')."
            ),
            "properties": {
                "reason_for_call": {
                    "type": "string",
                    "description": (
                        "Why the customer is contacting support — the request itself, in the "
                        "second person. State exactly what they want done."
                    ),
                },
                "known_info": {
                    "type": "string",
                    "description": (
                        "FACTS ONLY, and only ones the customer could state from memory: who "
                        "they are, their postal code or email, a value they are introducing "
                        "from outside (a new address, a budget). One short sentence is normal. "
                        "Do NOT put behaviour, preferences, conditions or what they will do "
                        "later in here — that belongs in `behavior`. Any literal the reference "
                        "solution uses that no earlier tool call produces must appear here or "
                        "in reason_for_call."
                    ),
                },
                "behavior": {
                    "type": "string",
                    "description": (
                        "The customer's SCRIPT for the conversation — not the task steps, and "
                        "not just a mood. Cover: how much they volunteer versus only answer "
                        "when asked; and at least one REACTION keyed on what the agent says or "
                        "finds ('if the agent tells you X is not possible, you want Y instead'; "
                        "'if the agent asks you to confirm, you change your mind and keep it'; "
                        "'you are only satisfied once Z'). Write reactions so they fire only if "
                        "the agent raises the point — a customer does not pre-announce their "
                        "fallback."
                    ),
                },
                "unknown_info": {
                    "type": "string",
                    "description": (
                        "Optional: information the customer does NOT know and cannot provide "
                        "(the agent must look it up via tools)."
                    ),
                },
            },
            "required": list(SCENARIO_REQUIRED_KEYS),
        },
        "actions": {
            "type": "array",
            "description": (
                "The minimal reference solution: the exact tool calls (in order) that solve "
                "the task, as you executed them after reset_environment()."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
            "minItems": 1,
        },
        "success_conditions": {
            "type": "array",
            "description": (
                "Outcome-based, path-agnostic checks on the END STATE, each a change from the "
                "starting state (e.g. 'the order's status is cancelled and a refund to the "
                "original card exists'). Never reference which tools were used. Only require "
                "what the task determines; allow any reasonable phrasing for free text. Used to "
                "grade the attempt."
            ),
            "items": {"type": "string"},
            "minItems": 1,
        },
        "communicate_info": {
            "type": "array",
            "description": (
                "Optional: values the agent MUST state to the customer, each as the exact "
                "substring to look for in what it says (a count, a price, a material, a date). "
                "Matching is a plain case-insensitive substring test, so give the bare value "
                "('10', 'polyester'), never a sentence. Every value must be one your reference "
                "actions actually surfaced — the grounding replay checks that it appears in "
                "their output. Recorded and reported, but it does NOT decide the grade on its "
                "own (a bare substring is too loose: 'pending' is satisfied by an agent "
                "discussing the wrong order). Whenever part of what the customer wants is to "
                "be TOLD something, put the graded requirement in `nl_assertions` and use this "
                "alongside it to pin the exact values."
            ),
            "items": {"type": "string"},
        },
        "nl_assertions": {
            "type": "array",
            "description": (
                "Claims about what the agent should have conveyed, in plain English ('Agent "
                "should tell the user that the gift-card refund cannot be redirected'). Judged "
                "by a language model, and this is the channel that GRADES what was said — a "
                "task whose correct outcome leaves the database unchanged is only gradable if "
                "it has these, and is rejected without them. Name the values in "
                "`communicate_info` too when they are literals."
            ),
            "items": {"type": "string"},
        },
        "tags": TAGS_PROP,
    },
    "required": list(SPEC_REQUIRED_KEYS),
}


def multi_item_exchange_reason(spec: dict[str, Any]) -> str:
    """Reject reason if the reference solution exchanges 2+ items in one call, else "".

    `modify_pending_order_items` is broken upstream for multi-item calls: the write
    loop in tau2's retail tools.py never recomputes `variant`, so every modified item
    receives the LAST pair's price and options. Both modified items therefore end up
    identical, and *which* one is corrupted depends on the order the agent happens to
    list its arguments in — an ordering the tool docs leave free. The graded DB end
    state thus differs between two equally correct solutions, so such a task is scored
    by coin flip.

    Single-item exchanges are unaffected (with one pair the leftover `variant` is that
    pair's own variant), and the tool is used by 38% of the retail test split, so it is
    the call pattern that is quarantined here, never the tool.
    """
    for i, action in enumerate(spec["actions"]):
        if action["name"] != "modify_pending_order_items":
            continue
        item_ids = action["arguments"].get("item_ids") or []
        if len(item_ids) >= 2:
            return (
                f"actions[{i}] modify_pending_order_items exchanges {len(item_ids)} items "
                "in a single call. That call is graded unreliably in this environment, so "
                "the task would be scored by luck. Redesign the task so the reference "
                "solution exchanges exactly ONE item per order (a single-element "
                "`item_ids`), or build the difficulty from something other than the "
                "number of exchanged items — e.g. the customer describing the "
                "replacement by attributes instead of an id, or a constraint that makes "
                "only one variant acceptable."
            )
    return ""


_DISCOVERABLE_IDS: dict[str, frozenset[str]] = {}

# Tools that only read. Used to check `_ID_CLASSES` against the domain's real tool
# signatures (see tests/test_tau2_identifier_gate.py) — nothing calls a tool here any more.
_READ_PREFIXES = ("get_", "find_", "search_", "list_", "calculate")

# Two shapes carry a digit without being record identifiers, in any domain:
# enumerated schema fields (`address1`, `account_number_last_4`) and calendar dates
# used as keys (airline has 30). A counterparty legitimately states a date, and a
# field name appearing in scenario prose would be a confusing rejection.
_NOT_AN_ID = _re.compile(r"[a-z][a-z_]*\d$|^\d{4}-\d{2}-\d{2}$")


def _harvest(db: dict[str, Any]) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    """(identifier CLASSES -> their instances, flattened entity records) from a state dump.

    Classes, not bare strings, because reachability is a property of the kind of
    identifier rather than of one value: if any single `order_id` turns up in a tool
    output then every order id is discoverable. The class name is the field the value
    was found under (`order_id`, `payment_method_id`), which is also the parameter name
    tools use for it — so a tool's signature says directly which classes it consumes.

    Instances come from two places, because domains store entities differently: mapping
    **keys** containing a digit (retail and airline index orders and users by id) and
    **values of `*_id` fields** (telecom stores entities as lists of records with the id
    as a field, so it has no such keys at all — 0 identifiers under a keys-only rule).

    The flattened records are the argument source for probing: each is one entity's leaf
    scalars merged into a flat dict, so a tool's parameters can be filled from a single
    real entity and the call resolves. Filling parameters independently would produce
    incoherent tuples — one user's first name with another's zip — and every lookup
    would fail.
    """
    classes: dict[str, set[str]] = {}
    records: list[dict[str, str]] = []

    def add(cls: str, value: str) -> None:
        classes.setdefault(cls, set()).add(value)

    def leaves(node: Any, into: dict[str, str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    into[k] = str(v)
                    if (k == "id" or k.endswith("_id")) and isinstance(v, str):
                        add(k, v)
                else:
                    leaves(v, into)
        elif isinstance(node, list):
            for v in node:
                leaves(v, into)

    for container, entities in (db or {}).items():
        items = entities.items() if isinstance(entities, dict) else None
        key_class = f"{container.rstrip('s')}_id"
        seq = items if items is not None else enumerate(entities or [])
        for key, entity in seq:
            if items is not None and isinstance(key, str):
                if any(c.isdigit() for c in key) and not _NOT_AN_ID.match(key):
                    add(key_class, key)
            if not isinstance(entity, (dict, list)):
                continue
            flat: dict[str, str] = {}
            leaves(entity, flat)
            if items is not None and isinstance(key, str):
                # A dict container is keyed BY the identifier, so the key is itself a
                # legal argument value for the matching `<singular>_id` parameter.
                flat.setdefault(key_class, key)
            if flat:
                records.append(flat)
    return classes, records


# Which identifier CLASSES a domain treats as look-up-able ("gated": some READ tool's output
# produces it, so handing it over in the customer's script deletes the discovery step) and
# which the customer is the only possible source of ("roots": never gated). Declared from
# the domain's tool signatures and policy, never from the human task set;
# tests/test_tau2_identifier_gate.py checks every class in the database is classified.
_ID_CLASSES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # find_user_id_by_name_zip turns a name and a zip into a user_id, and every other id
    # follows from it, so retail has no roots.
    "retail": (
        frozenset(
            {"id", "item_id", "order_id", "payment_method_id", "product_id", "user_id"}
        ),
        frozenset(),
    ),
}


def id_classes(domain: str) -> tuple[frozenset[str], frozenset[str]]:
    """`(gated, roots)` for `domain`, or two empty sets (gate nothing) when unclassified."""
    return _ID_CLASSES.get(domain, (frozenset(), frozenset()))


def _discoverable_ids(domain: str) -> frozenset[str]:
    """The identifier VALUES a generated task must not hand the agent, for this domain.

    Every instance of a gated class in the domain's initial database (see `_harvest`),
    minus the roots. Pure data: it reads the seeded state and consults `_ID_CLASSES`,
    making no tool calls and mutating nothing.
    """
    if domain in _DISCOVERABLE_IDS:
        return _DISCOVERABLE_IDS[domain]
    from tau2.runner import build_environment

    gated, roots = id_classes(domain)
    classes, _ = _harvest(build_environment(domain).tools.db.model_dump(mode="json"))
    out = {
        value
        for cls, values in classes.items()
        if cls in gated and cls not in roots
        for value in values
    }
    _DISCOVERABLE_IDS[domain] = frozenset(out)
    return _DISCOVERABLE_IDS[domain]


def leaked_identifier_reason(spec: dict[str, Any], domain: str) -> str:
    """Reject reason if the customer volunteers an id the agent could look up, else "".

    `ground_spec` requires every reference-action literal to be either revealable by
    the customer or produced by an earlier tool output. That rule is one-sided: the
    cheapest way to satisfy it is to paste the id into `known_info`, and a designer
    that found the entity by reading the database always has the id to hand. The
    result is a task whose discovery step has been deleted — the agent is told which
    order to act on instead of having to find it. Measured on tau2 retail: 94% of
    generated tasks hand over an order id against 11% of the human-authored ones, and
    the generated memory pool correspondingly contains ~0 of the "look it up yourself
    instead of stalling on the customer" heuristics that make up a fifth of the human
    pool.

    Requiring the id to be *derived* rather than *given* forces the reference solution
    to include the lookup calls, which both proves a discovery path exists and puts
    the id into `seen_outputs` so the existing anchoring rule is satisfied.
    """
    # Every field, not just the revealable ones: an id written into `unknown_info`
    # still reaches the user simulator's prompt, and "you do not remember your order
    # number" needs no id in it anyway.
    scenario = spec["scenario"]
    text = " ".join(str(v) for v in scenario.values() if isinstance(v, str))
    # Boundary-checked, not a bare substring test: telecom identifiers are as short as
    # `C1001`, which would otherwise match inside longer tokens. Cheap `in` filter
    # first, since the identifier universe runs to thousands.
    leaked = sorted(
        i
        for i in _discoverable_ids(domain)
        if i in text
        and _re.search(rf"(?<![A-Za-z0-9_]){_re.escape(i)}(?![A-Za-z0-9_])", text)
    )
    if not leaked:
        return ""
    return (
        f"the customer volunteers identifier(s) {leaked}, which the agent could have "
        "looked up from the environment. Handing them over deletes the search step "
        "that makes the task worth solving. Rewrite the scenario so the customer "
        "identifies the target the way a real one would — by what it contains, when "
        "it happened, where it went, what it cost — and move the id itself into "
        "`unknown_info` ('you do not remember your order number'). Then extend your "
        "reference solution with the lookup calls that recover the id (authenticate, "
        "list the account's records, read the matching one) so it is derived rather "
        "than assumed. Keep the description specific enough that exactly one record "
        "matches; credentials the customer really would know (name, zip, email) stay "
        "in `known_info`."
    )


def parse_spec(obj: Any) -> tuple[dict[str, Any] | None, str]:
    """Structural validation of a submitted spec. Returns (spec, "") or (None, reason)."""
    if not isinstance(obj, dict):
        return None, "spec must be a JSON object"
    for key in SPEC_REQUIRED_KEYS:
        if key not in obj:
            return None, f"spec is missing required key {key!r}"
    scenario = obj["scenario"]
    if not isinstance(scenario, dict):
        return None, "scenario must be a JSON object"
    for key in SCENARIO_REQUIRED_KEYS:
        value = scenario.get(key)
        if not isinstance(value, str) or not value.strip():
            return (
                None,
                f"scenario.{key} is missing or empty — describe the customer's situation",
            )
    actions = obj["actions"]
    if not isinstance(actions, list) or not actions:
        return None, "actions is empty — give the tool calls you executed"
    for i, a in enumerate(actions):
        if (
            not isinstance(a, dict)
            or "name" not in a
            or not isinstance(a.get("arguments"), dict)
        ):
            return None, f"actions[{i}] must be {{'name': str, 'arguments': dict}}"
    conds = obj["success_conditions"]
    if not isinstance(conds, list) or not conds:
        return None, "success_conditions is empty — state the outcome checks on the end state"
    if not all(isinstance(c, str) and c.strip() for c in conds):
        return None, "each success_condition must be a non-empty string"
    for key in ("communicate_info", "nl_assertions"):
        val = obj.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            return None, f"{key} must be a list of strings"
        if not all(isinstance(x, str) and x.strip() for x in val):
            return None, f"each entry in {key} must be a non-empty string"
    return obj, ""


def _nl_criteria(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The spec's (communicate_info, nl_assertions), each cleaned of blanks."""
    out = []
    for key in ("communicate_info", "nl_assertions"):
        out.append([x.strip() for x in (spec.get(key) or []) if x and x.strip()])
    return out[0], out[1]


def _anchor_literals(arguments: dict[str, Any]) -> list[str]:
    """Argument values that must be traceable to the revealable scenario text or
    earlier outputs."""
    literals: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str) and len(v.strip()) >= 3:
            literals.append(v.strip())
        elif isinstance(v, (int, float)) and v not in (0, 1):
            literals.append(str(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(arguments)
    return literals


def _revealable_text(scenario: dict[str, Any]) -> str:
    """The scenario text a role-playing customer can convey — everything the
    solver could learn by asking. Excludes unknown_info (the customer does not
    know it; the agent must look it up via tools)."""
    # No `persona`: SPEC_SCHEMA does not define one and spec_to_task never sets tau2's,
    # so listing it here only suggested a field that is always absent.
    return " ".join(
        str(scenario.get(k) or "")
        for k in ("reason_for_call", "known_info", "behavior")
    )


def ground_spec(spec: dict[str, Any], domain: str) -> tuple[bool, str]:
    """Replay the spec's actions on a fresh environment and check coherence.

    Returns (True, "") or (False, reason). Rejections carry the exact error so
    the explorer can self-correct in-session.
    """
    from tau2.data_model.message import ToolCall
    from tau2.runner import build_environment

    env = build_environment(domain)
    h0 = env.get_db_hash()
    revealable_lower = _revealable_text(spec["scenario"]).lower()
    seen_outputs = ""

    for i, action in enumerate(spec["actions"]):
        name, arguments = action["name"], action["arguments"]
        # Utility tools (GenericToolKit) take derived free-text arguments — the
        # solver computes those itself, so they need no anchoring in the ticket.
        utility = name in ("think", "calculate")
        missing = (
            []
            if utility
            else [
                lit
                for lit in _anchor_literals(arguments)
                if lit.lower() not in revealable_lower
                and lit.lower() not in seen_outputs
            ]
        )
        if missing:
            return False, (
                f"actions[{i}] {name} uses value(s) {missing} that the customer never "
                f"conveys (not in reason_for_call/known_info/behavior) and no earlier tool "
                f"output produces — a solver could not know them. Add them to known_info or "
                f"reason_for_call, or derive them from earlier calls."
            )
        result = env.get_response(
            ToolCall(
                id=f"ground_{i}", name=name, arguments=arguments, requestor="assistant"
            )
        )
        if result.error:
            return (
                False,
                f"actions[{i}] {name}({json.dumps(arguments)}) failed: {result.content}",
            )
        seen_outputs += (result.content or "").lower()

    communicate_info, nl_assertions = _nl_criteria(spec)

    # Every value the agent is required to say must be one the reference solution actually
    # surfaced, or the task asks it to report something it cannot know. Checked against the
    # concatenated tool output of the replay above.
    missing = [c for c in communicate_info if c.lower() not in seen_outputs]
    if missing:
        return False, (
            f"communicate_info {missing} never appears in the output of your reference "
            "actions, so the agent has no way to learn those values. Either add the calls "
            "that surface them or drop them. Give the bare value as it appears in the tool "
            "output (a count, a price, a material), not a sentence about it."
        )

    if env.get_db_hash() == h0 and not nl_assertions:
        return False, (
            "replaying the actions leaves the database unchanged and the spec has no "
            "`nl_assertions`, so nothing about this task is graded and a do-nothing agent "
            "would pass. Either design a task whose solution changes the database, or — if "
            "the point of the task is that the customer is told something (including being "
            "told why their request cannot be done) — write what the agent must convey as "
            "`nl_assertions`. `communicate_info` alone is not enough: it is recorded and "
            "reported but does not decide the grade, because a bare substring match is too "
            "loose to stand on its own."
        )
    return True, ""


def spec_to_task(spec: dict[str, Any], domain: str) -> Any:
    """Convert a grounded spec into a native tau2 Task.

    The tau2-specific adapter: maps the generic scenario onto tau2's
    StructuredUserInstructions (`behavior` → `task_instructions`) and carries any
    said-out-loud criteria across, so reward_basis is [DB] plus COMMUNICATE and/or
    NL_ASSERTION exactly when the spec supplies them — the shape the retail train split
    uses on 112 of its 114 tasks.

    `persona` is deliberately not set: retail's own 114 tasks leave it empty and put the
    customer's demeanour in `task_instructions`, so filling it would put our tasks in a
    field the split we imitate never uses while starving the one it does."""
    from tau2.data_model.tasks import (
        Action,
        Description,
        EvaluationCriteria,
        RewardType,
        StructuredUserInstructions,
        Task,
        UserScenario,
    )

    scenario = spec["scenario"]
    communicate_info, nl_assertions = _nl_criteria(spec)
    # COMMUNICATE is deliberately NOT added, matching the retail train split: 36 of its 114
    # tasks carry `communicate_info` yet not one puts COMMUNICATE in `reward_basis`, so the
    # substring check is recorded and reported but does not gate the reward. That looks
    # deliberate rather than accidental — the check is a bare case-insensitive substring
    # test, so a value like "pending", "delivered" or "paypal" is satisfied by an agent
    # talking about the wrong record entirely. NL_ASSERTION carries the grade instead.
    reward_basis = [RewardType.DB]
    if nl_assertions:
        reward_basis.append(RewardType.NL_ASSERTION)
    # Deterministic id: the task graded during generation and the one banked
    # for downstream use are the same object.
    digest = uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(spec, sort_keys=True)).hex[:8]
    task_id = f"gen_{domain}_{digest}"
    return Task(
        id=task_id,
        description=Description(purpose=spec.get("purpose")),
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain=domain,
                reason_for_call=scenario["reason_for_call"],
                known_info=scenario["known_info"],
                unknown_info=scenario.get("unknown_info"),
                task_instructions=scenario["behavior"],
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id=f"{task_id}_{i}",
                    requestor="assistant",
                    name=a["name"],
                    arguments=a["arguments"],
                )
                for i, a in enumerate(spec["actions"])
            ],
            communicate_info=communicate_info or None,
            nl_assertions=nl_assertions or None,
            reward_basis=reward_basis,
        ),
    )
