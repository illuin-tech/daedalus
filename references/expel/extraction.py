"""ExpeL's insight extraction (paper §4.2, Alg. 2) — the two critique calls.

    critique_compare      a failed and a successful trial of the SAME task (Fig. 2A)
    critique_all_success  a batch of L successful trials from DIFFERENT tasks (Fig. 2B)

Each call is handed the current insight list and answers with operations on it
(AGREE/REMOVE/EDIT/ADD), which `insights.apply_operations` folds in. The prompt text is
ExpeL's own, from its released implementation (`prompts/templates/human.py`,
`prompts/<domain>.py`); the paper's Figures 2 and 3 are elided images.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from daedalus.core.llm.client import LLMClient

from references.common.usage import Usage
from references.expel.insights import Insight, apply_operations, parse_operations

PROMPTS = Path(__file__).parent / "prompts"

# ExpeL's fixed persona for the extraction agent (`agent/expel.py`), which is also the
# first line of the paper's Figure 2.
AI_NAME = (
    "an advanced reasoning agent that can add, edit or remove rules from your existing "
    "rule set, based on forming new critiques of past task trajectories"
)

# The two domain-bound clauses of ExpeL's critique instruction. Every one of its own
# domains fills them differently — compare its alfworld ("were placed in a household
# environment and a task to complete" / "because you reached the maximum allowed number of
# steps without completing the task"), hotpotQA and webshop prompt files — so each
# benchmark here gets the same treatment. `task_kind` is the matching clause of its
# inference-time rules wrapper (`RULE_TEMPLATE`).
_DOMAIN_CLAUSES = {
    "tau2": {
        "environment": (
            "were placed in a customer-service conversation with a user, with access to the "
            "domain's tools and policy, and a request to resolve"
        ),
        "failure_reason": (
            "either because you reached the maximum allowed number of steps, or because the "
            "database did not end up in the state the user's request required"
        ),
        "task_kind": (
            "resolving a user's request in a customer-service conversation by interacting with "
            "the domain's tools"
        ),
    },
    "appworld": {
        "environment": (
            "were given access to an environment of app APIs and a task to complete by writing "
            "Python code"
        ),
        "failure_reason": (
            "either because you reached the maximum allowed number of steps, or because your "
            "solution did not satisfy the task's checks"
        ),
        "task_kind": "completing a task by calling app APIs from Python code",
    },
    "automationbench": {
        "environment": (
            "were given a workplace request and access to the REST APIs of the company's "
            "SaaS applications, only some of which this situation connects"
        ),
        "failure_reason": (
            "either because you reached the maximum allowed number of steps, or because the "
            "applications did not end up in the state the request required"
        ),
        "task_kind": (
            "carrying out a workplace request by reading the company's records through app "
            "APIs and making the writes it asks for"
        ),
    },
}


def domain_clauses(benchmark: str) -> dict[str, str]:
    """The environment / failure / task-kind clauses for a benchmark."""
    try:
        return _DOMAIN_CLAUSES[benchmark]
    except KeyError:
        raise ValueError(
            f"ExpeL has no domain clauses for benchmark {benchmark!r}; known: "
            f"{sorted(_DOMAIN_CLAUSES)}. Add one (three short phrases) in "
            f"references/expel/extraction.py."
        ) from None


def _render(name: str, **variables: object) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).render(**variables).strip()


def rules_block(benchmark: str, insights: list[Insight]) -> str:
    """ExpeL's inference-time wrapper around the insight list (its `RULE_TEMPLATE`)."""
    rules = "\n".join(f"{i}. {insight.text}" for i, insight in enumerate(insights, 1))
    return _render(
        "rules_injection.txt", task_kind=domain_clauses(benchmark)["task_kind"], rules=rules
    )


def fewshots_block(fewshots: list[str]) -> str:
    """ExpeL's few-shot wrapper (`human_instruction_fewshots_template`)."""
    return _render("fewshots_injection.txt", fewshots="\n\n".join(fewshots))


def _existing_rules(insights: list[Insight]) -> str:
    """The numbered rule list the critique prompt shows. Empty list → one blank line,
    which is what ExpeL passes so the section is present but empty."""
    if not insights:
        return ""
    return "\n".join(f"{i}. {insight.text}" for i, insight in enumerate(insights, 1))


def _suffix(insights: list[Insight], max_num_rules: int) -> str:
    """ExpeL switches to its 'focus on REMOVE' suffix once the list is at budget."""
    name = "full" if max_num_rules <= len(insights) else "not_full"
    return _render(f"critique_summary_suffix_{name}.txt")


def _critique(
    llm: LLMClient,
    instruction_file: str,
    human_file: str,
    benchmark: str,
    insights: list[Insight],
    max_num_rules: int,
    usage: Usage | None,
    effort: str | None,
    **human_vars: object,
) -> list[Insight]:
    clauses = domain_clauses(benchmark)
    system = _render(
        "system.txt",
        ai_name=AI_NAME,
        instruction=_render(
            instruction_file,
            environment=clauses["environment"],
            failure_reason=clauses["failure_reason"],
        ),
    )
    human = _render(
        human_file,
        existing_rules=_existing_rules(insights),
        format_rules_operation=_render("format_rules_operation.txt"),
        critique_summary_suffix=_suffix(insights, max_num_rules),
        **human_vars,
    )
    response = llm.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": human}],
        reasoning_effort=effort,
    )
    if usage is not None:
        usage.add(llm.model, response)
    operations = parse_operations(response.content or "")
    # ExpeL's own condition for weighting REMOVE more heavily.
    return apply_operations(insights, operations, list_full=max_num_rules + 5 <= len(insights))


def critique_compare(
    llm: LLMClient,
    benchmark: str,
    insights: list[Insight],
    task: str,
    success_history: str,
    fail_history: str,
    max_num_rules: int = 20,
    usage: Usage | None = None,
    effort: str | None = None,
) -> list[Insight]:
    """Contrast one failed trial with a successful one of the same task."""
    return _critique(
        llm,
        "system_critique_compare.txt",
        "critique_compare.txt",
        benchmark,
        insights,
        max_num_rules,
        usage,
        effort,
        task=task,
        success_history=success_history,
        fail_history=fail_history,
    )


def critique_all_success(
    llm: LLMClient,
    benchmark: str,
    insights: list[Insight],
    success_histories: list[str],
    max_num_rules: int = 20,
    usage: Usage | None = None,
    effort: str | None = None,
) -> list[Insight]:
    """Read patterns off a batch of L successful trials from different tasks."""
    return _critique(
        llm,
        "system_critique_all_success.txt",
        "critique_all_success.txt",
        benchmark,
        insights,
        max_num_rules,
        usage,
        effort,
        success_history="\n\n".join(success_histories),
    )
