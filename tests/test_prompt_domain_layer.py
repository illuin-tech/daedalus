"""The τ² per-domain prompt layer (prompts/<domain>/ shadows the shared prompts/).

The retail expectations here are byte-exact snapshots of what these prompts rendered
BEFORE the domain layer existed, captured from the commit that introduced it. They are
the contract that lets the published retail results stand and old retail runs reproduce:
moving retail's text into prompts/retail/ must not change a single character of what the
solver or the explorer is shown. If one of these fails, retail's prompt has moved — either
fix the change or re-run the retail arms deliberately.
"""

from __future__ import annotations

from pathlib import Path


from daedalus.benchmarks.tau2 import PROMPTS, prompt_dirs
from daedalus.core.resources import render_prompt, task_guidelines

DOMAINS = ("retail",)
POLICY = "<<<DOMAIN POLICY PLACEHOLDER>>>"

# The retail identification paragraph, exactly as solver_protocol.txt carried it before
# the split. Kept inline rather than read from the file so the test cannot pass by
# agreeing with a file someone edited.
RETAIL_IDENTIFICATION = """\
Identify the user first, and only from a value they actually gave you. The lookup tools match exact
field values — a real email address, or a first name, last name and zip. Passing anything else
(their message text, a name where an email goes, a plausible-looking guess, a placeholder) returns
`User not found`: it never partially matches, so there is nothing to be learned from trying. If what
you have does not contain an email or a name-plus-zip, ask for one and wait for the reply before
calling any lookup tool — asking costs one turn, guessing costs a turn and tells you nothing."""

RETAIL_SOLVER_TAIL = f"""\
You are a customer-service agent. In each turn you may EITHER send a message to the user OR make a tool call — never both at once. Always follow the policy below.

{RETAIL_IDENTIFICATION}

<policy>
{POLICY}
</policy>"""


def _solver(domain: str, *, ticket: bool = False, env: bool = True) -> str:
    return render_prompt(
        "solver",
        prompt_dirs(domain),
        add_env_knowledge=env,
        domain_policy=POLICY,
        heuristics=[],
        ticket_mode=ticket,
    )


# ── the backwards-compatibility contract ─────────────────────────────────────


def test_retail_solver_prompt_is_unchanged():
    """Retail's composed solver prompt still ends exactly as it did before the split."""
    assert _solver("retail").endswith(RETAIL_SOLVER_TAIL + "\n")


def test_retail_ticket_mode_still_slots_between_identification_and_policy():
    prompt = _solver("retail", ticket=True)
    assert RETAIL_IDENTIFICATION in prompt
    assert prompt.index(RETAIL_IDENTIFICATION) < prompt.index("submitted their request in writing")
    assert prompt.index("submitted their request in writing") < prompt.index("<policy>")


def test_retail_task_guidelines_resolve_through_the_domain_dir():
    """Moving the file into prompts/retail/ left its rendered text identical."""
    rendered = task_guidelines(prompt_dirs("retail"), "explorer")
    assert rendered.startswith("## What a good task looks like in this environment")
    assert "the boots I bought last month" in rendered
    # the shared doctrine macros still resolve from core/prompts
    assert "**Write a request, not a specification.**" in rendered


# ── the domain layer itself ──────────────────────────────────────────────────


def test_shared_protocol_text_reaches_every_domain():
    """The half of solver_protocol.txt that is genuinely shared is not duplicated away."""
    for domain in DOMAINS:
        prompt = _solver(domain)
        assert "EITHER send a message to the user OR make a tool call" in prompt
        assert f"<policy>\n{POLICY}\n</policy>" in prompt


# ── the search-path mechanism ────────────────────────────────────────────────


def test_a_domain_without_a_directory_falls_back_to_the_shared_prompts():
    """Adding a domain costs nothing until it needs to say something different."""
    assert prompt_dirs("banking_knowledge") == [PROMPTS]
    assert prompt_dirs(None) == [PROMPTS]
    prompt = _solver("banking_knowledge")
    assert "EITHER send a message to the user OR make a tool call" in prompt
    # no domain block exists, so the slot collapses rather than leaking retail's
    assert "name, last name and zip" not in prompt


def test_single_path_callers_are_unaffected():
    """Benchmarks with one environment keep passing a bare Path."""
    appworld = Path("daedalus/benchmarks/appworld/prompts").resolve()
    one = render_prompt("solver", appworld, app_descriptions="", heuristics=[])
    many = render_prompt("solver", [appworld], app_descriptions="", heuristics=[])
    assert one == many


def test_benchmark_hook_matches_the_module_helper():
    """pipeline._task_guidelines and the explorer must read the same search path."""
    from daedalus.benchmarks.tau2.runner import Tau2Benchmark
    from daedalus.core.config import config_from_dict

    cfg = config_from_dict({"benchmark": "tau2", "tau2": {"domain": "retail"}})
    assert Tau2Benchmark().prompt_dirs(cfg) == prompt_dirs("retail")


# ── the playbook slot ─────────────────────────────────────────────────────────


def test_playbook_slot_is_inert_when_unset():
    """Adding the slot must not move a single character of any existing run's prompt.

    ACE injects its playbook verbatim (sections and bullet ids intact) through a slot of
    its own rather than through `heuristics`, which renumbers its entries. Every other
    method and every baseline leaves it unset, and this is the guard that it costs them
    nothing — the retail snapshots above are the byte-exact contract for the rest.
    """
    appworld = Path("daedalus/benchmarks/appworld/prompts").resolve()
    unset = render_prompt("solver", appworld, app_descriptions="", heuristics=[])
    empty = render_prompt("solver", appworld, app_descriptions="", heuristics=[], playbook="")
    assert unset == empty
    assert _solver("retail") == render_prompt(
        "solver",
        prompt_dirs("retail"),
        add_env_knowledge=True,
        domain_policy=POLICY,
        heuristics=[],
        ticket_mode=False,
        playbook="",
    )


def test_playbook_is_rendered_verbatim_not_renumbered():
    """Section headers and bullet ids reach the solver exactly as the playbook holds them."""
    appworld = Path("daedalus/benchmarks/appworld/prompts").resolve()
    playbook = "## STRATEGIES AND HARD RULES\n[shr-00001] alpha\n[shr-00002] beta"
    prompt = render_prompt(
        "solver", appworld, app_descriptions="", heuristics=[], playbook=playbook
    )
    assert playbook in prompt
    # The `heuristics` loop would have rewritten these as "1. alpha" / "2. beta".
    assert "1. alpha" not in prompt
