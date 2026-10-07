"""ACE's playbook mechanics: parsing, the ADD operation, and the bridge to a MemoryPool.

Offline and free — no LLM calls, no benchmark checkout. This is the ONLY CI gate the
`references/ace/` code gets: CI's `compileall` covers `daedalus tests plots` and its import
sweep walks `daedalus.*`, so neither reaches `references/`.

The curator's output path is the part worth pinning. It is the one place where a model's
free-form reply turns into a durable artifact, and every failure mode there is silent: a
mis-sectioned bullet still renders, an unparsed reply still leaves a valid playbook, and a
duplicated id still injects.
"""

from __future__ import annotations

import inspect
import json

import pytest

from references.ace.curation import full_trajectory, parse_operations
from references.ace.domains import CLAUSE_KEYS, domain
from references.ace.playbook import (
    ALLOWED_SECTIONS,
    apply_curator_operations,
    bullets,
    empty_playbook,
    extract_json_from_text,
    get_next_global_id,
    get_playbook_stats,
    parse_playbook_line,
    to_pool,
    validate_operations,
)


def add(section: str, content: str) -> dict:
    return {"type": "ADD", "section": section, "content": content}


# ── the playbook file format ──────────────────────────────────────────────────


def test_empty_playbook_has_the_eight_sections_and_no_bullets():
    text = empty_playbook()
    assert text.count("##") == 8
    assert bullets(text) == []
    assert get_playbook_stats(text)["total_bullets"] == 0
    assert get_next_global_id(text) == 1


def test_parse_line_accepts_both_upstream_formats():
    # The agentic path writes `[id] content`; the finance path in the main ace repo also
    # emits counters. Both must parse, and the counters must not end up inside the text.
    plain = parse_playbook_line("[shr-00001] always check the docs")
    assert plain["id"] == "shr-00001" and plain["content"] == "always check the docs"
    counted = parse_playbook_line("[shr-00002] helpful=5 harmful=1 :: paginate fully")
    assert counted["id"] == "shr-00002" and counted["content"] == "paginate fully"
    assert counted["helpful"] == 5 and counted["harmful"] == 1
    assert parse_playbook_line("## STRATEGIES AND HARD RULES") is None
    assert parse_playbook_line("") is None


# ── the ADD operation ─────────────────────────────────────────────────────────


def test_add_lands_in_its_section_with_a_sequential_id():
    text, next_id = apply_curator_operations(
        empty_playbook(),
        [add("strategies_and_hard_rules", "first"), add("verification_checklist", "second")],
        1,
    )
    assert next_id == 3
    placed = {section: (bid, content) for section, bid, content in bullets(text)}
    assert placed["STRATEGIES AND HARD RULES"] == ("shr-00001", "first")
    assert placed["VERIFICATION CHECKLIST"] == ("vc-00002", "second")


def test_ids_keep_climbing_across_calls_and_never_repeat():
    text = empty_playbook()
    seen = []
    next_id = 1
    for i in range(5):
        text, next_id = apply_curator_operations(
            text, [add("strategies_and_hard_rules", f"rule {i}")], next_id
        )
        seen.append(bullets(text)[-1][1])
    assert seen == ["shr-00001", "shr-00002", "shr-00003", "shr-00004", "shr-00005"]
    assert len(set(seen)) == 5
    # A resumed run recovers the counter from the file alone.
    assert get_next_global_id(text) == 6


def test_unknown_section_falls_through_to_others():
    text, _ = apply_curator_operations(
        empty_playbook(), [add("a_section_that_does_not_exist", "stray")], 1
    )
    assert [s for s, _, c in bullets(text) if c == "stray"] == ["OTHERS"]


def test_problem_solving_section_routes_to_others_upstream_quirk():
    """The whitelist spells it with an underscore, the header with a hyphen.

    Reproduced deliberately (see references/ace/playbook.py). If someone "fixes" the
    mismatch, this test fails and the fix has to be a stated deviation rather than a
    silent one — it changes where every such bullet lands, and our playbook would stop
    matching ACE's.
    """
    assert "problem_solving_heuristics_and_workflows" in ALLOWED_SECTIONS
    text, _ = apply_curator_operations(
        empty_playbook(), [add("problem_solving_heuristics_and_workflows", "plan first")], 1
    )
    assert [s for s, _, c in bullets(text) if c == "plan first"] == ["OTHERS"]


def test_existing_bullets_survive_a_later_add():
    text, next_id = apply_curator_operations(
        empty_playbook(), [add("strategies_and_hard_rules", "old")], 1
    )
    text, _ = apply_curator_operations(text, [add("strategies_and_hard_rules", "new")], next_id)
    contents = [c for _, _, c in bullets(text)]
    assert contents == ["old", "new"]


# ── validation ────────────────────────────────────────────────────────────────


def test_only_add_is_accepted():
    for bad in ("UPDATE", "DELETE", "MERGE", "CREATE_META"):
        with pytest.raises(ValueError, match="only ADD"):
            validate_operations([{"type": bad, "section": "others", "content": "x"}])


def test_operation_outside_the_whitelist_is_dropped_not_fatal():
    kept = validate_operations(
        [add("strategies_and_hard_rules", "keep"), add("invented_section", "drop")]
    )
    assert [op["content"] for op in kept] == ["keep"]


def test_malformed_operation_raises():
    with pytest.raises(ValueError, match="missing fields"):
        validate_operations([{"type": "ADD", "section": "others"}])
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_operations(["not a dict"])


# ── reading the curator's reply ───────────────────────────────────────────────


REPLY = {"reasoning": "because", "operations": [add("verification_checklist", "read it back")]}


def test_curator_reply_parses_bare_fenced_and_embedded():
    bare = json.dumps(REPLY)
    for raw in (
        bare,
        f"```json\n{bare}\n```",
        f"Sure, here you go:\n{bare}\nHope that helps!",
    ):
        operations, error = parse_operations(raw)
        assert error is None
        assert operations == REPLY["operations"]


def test_unparseable_reply_is_reported_not_raised():
    # Upstream keeps the playbook and continues; a long adaptation run must not be lost to
    # one bad reply.
    operations, error = parse_operations("I'm afraid I can't do that.")
    assert operations == [] and error
    operations, error = parse_operations(json.dumps({"operations": []}))
    assert operations == [] and "reasoning" in error


def test_braces_inside_strings_do_not_confuse_the_scanner():
    raw = 'noise {"reasoning": "a } brace", "operations": []} trailing'
    assert extract_json_from_text(raw) == {"reasoning": "a } brace", "operations": []}


# ── the pool bridge ───────────────────────────────────────────────────────────


def test_pool_round_trip_preserves_text_order_and_provenance():
    text, _ = apply_curator_operations(
        empty_playbook(),
        [add("strategies_and_hard_rules", "alpha"), add("verification_checklist", "beta")],
        1,
    )
    pool = to_pool(text, source="ace_smoke")
    assert pool.texts() == ["alpha", "beta"]
    assert [i.tags["ace_section"] for i in pool.items] == [
        "STRATEGIES AND HARD RULES",
        "VERIFICATION CHECKLIST",
    ]
    assert [i.tags["ace_id"] for i in pool.items] == ["shr-00001", "vc-00002"]
    assert {i.source_task_id for i in pool.items} == {"ace_smoke"}


# ── the trajectory the reflector reads ────────────────────────────────────────


def test_appworld_trajectory_is_not_truncated():
    """daedalus's shared renderer cuts execution output at 500 chars; ACE's must not.

    See references/ace/curation.py — handing the reflector the truncated text would starve
    exactly the material it is asked to diagnose.
    """
    long_output = "x" * 5000
    text = full_trajectory({"turns": [{"thought": "t", "code": "c", "execution_output": long_output}]})
    assert long_output in text
    assert "..." not in text


# ── per-benchmark clauses and agent wiring ────────────────────────────────────


def test_every_wired_benchmark_has_a_full_clause_set():
    for benchmark in ("appworld", "tau2", "automationbench"):
        clauses = domain(benchmark)
        assert set(clauses) == set(CLAUSE_KEYS)
        assert all(clauses[key].strip() for key in CLAUSE_KEYS)


def test_unknown_benchmark_names_the_file_to_edit():
    with pytest.raises(ValueError, match="references/ace/domains.py"):
        domain("nope")


def test_agent_registry_covers_the_three_table_benchmarks():
    from references.ace.agents import _AGENTS

    assert set(_AGENTS) == {"appworld", "tau2", "automationbench"}


@pytest.mark.parametrize("benchmark", ["appworld", "tau2", "automationbench"])
def test_agent_subclasses_keep_the_taskagent_contract(benchmark):
    """Signature-check without constructing — building one needs the sibling checkouts."""
    import importlib

    from references.ace.agents import _AGENTS

    module_path, cls_name = _AGENTS[benchmark].rsplit(".", 1)
    try:
        cls = getattr(importlib.import_module(module_path), cls_name)
    except ImportError as e:  # the benchmark extra is not installed here
        pytest.skip(f"{benchmark} not installed: {e}")
    inspect.signature(cls.__init__).bind(None, object(), run_idx=0)
    inspect.signature(cls.solve_task).bind(None, "task-1", heuristics=[], trace_suffix="x")


# ── the inference guard ───────────────────────────────────────────────────────


def test_inference_refuses_a_missing_or_bullet_less_playbook(tmp_path):
    """An empty playbook injects nothing and scores as the baseline, under ACE's name.

    The bullet-less case is the one that actually happens: accumulation writes the bare
    section skeleton before its first wave, so the path exists and looks plausible from the
    moment an adaptation run starts. Accumulation itself must tolerate both — it is the
    process building the thing.
    """
    from references.ace.agents.base import load_playbook

    skeleton = tmp_path / "playbook.txt"
    skeleton.write_text(empty_playbook(), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_playbook(str(tmp_path / "nope.txt"), required=True)
    with pytest.raises(ValueError, match="no bullets"):
        load_playbook(str(skeleton), required=True)

    grown, _ = apply_curator_operations(empty_playbook(), [add("others", "something")], 1)
    skeleton.write_text(grown, encoding="utf-8")
    assert load_playbook(str(skeleton), required=True) == grown

    # accumulation is building it, so neither case is an error there
    assert load_playbook(None, required=False) == empty_playbook()
    assert load_playbook(str(tmp_path / "nope.txt"), required=False) == empty_playbook()
