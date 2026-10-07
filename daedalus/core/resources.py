"""Prompt composition for the benchmark agents (solver / explorer / refine / coverage survey)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

CORE_PROMPTS = Path(__file__).parent / "prompts"

# `benchmark_prompts_dir` throughout this module is a SEARCH PATH: either one directory
# or several, most specific first. A benchmark whose environments differ enough to need
# their own wording contributes an extra directory rather than a flag — τ² passes
# [benchmarks/tau2/prompts/<domain>, benchmarks/tau2/prompts], so a file present under
# the domain wins and everything else falls through to the shared copy. Benchmarks with
# a single environment keep passing one Path and are unaffected; nothing in core knows
# what a "domain" is, which is what keeps this from being a τ²-shaped special case.
PromptDirs = Path | str | Sequence[Path | str]


def _dirs(benchmark_prompts_dir: PromptDirs) -> list[Path]:
    """Normalise the search path to a list of directories, most specific first."""
    if isinstance(benchmark_prompts_dir, (str, Path)):
        return [Path(benchmark_prompts_dir)]
    return [Path(d) for d in benchmark_prompts_dir]


def _find(benchmark_prompts_dir: PromptDirs, name: str) -> Path | None:
    """First existing `name` along the search path, or None when no directory has it."""
    for d in _dirs(benchmark_prompts_dir):
        candidate = d / name
        if candidate.exists():
            return candidate
    return None


def _jinja(benchmark_prompts_dir):
    """Jinja environment that can resolve {% from %} against the benchmark dirs then core.

    Needed so a benchmark's prompt can import shared macros — the doctrine every
    benchmark's task guidelines repeat verbatim lives once in
    core/prompts/generation/task_doctrine.txt and is called from each benchmark's file,
    which keeps only what is genuinely environment-bound. The benchmark directories are
    searched first and in order, so a benchmark can shadow a core template by name if it
    ever needs to, and a more specific directory can shadow a less specific one.
    """
    from jinja2 import Environment, FileSystemLoader

    # Defaults deliberately left as jinja2.Template's (which builds a bare Environment),
    # so switching to this loader-backed environment cannot change any rendered prompt.
    search = [str(d) for d in _dirs(benchmark_prompts_dir)] + [str(CORE_PROMPTS)]
    return Environment(loader=FileSystemLoader(search))

# ── Prompt composition ───────────────────────────────────────────────────────
# Paths written as benchmarks/<name>/prompts/ below are resolved along the search path
# described above, so a benchmark that contributes a more specific directory can supply
# any one of these blocks per environment without the shared templates knowing.
# Every agent prompt is composed from up to three blocks:
#   general   — a benchmark-neutral template in CORE_PROMPTS (the "what/why").
#               Always present. Carries an {{ env_knowledge }} slot and an
#               {{ action_protocol }} slot.
#   env       — the {{ env_knowledge }} block. When add_env_knowledge is True this is
#               the benchmark's benchmarks/<name>/prompts/<kind>_env.txt; when False it
#               is the benchmark-AGNOSTIC CORE_PROMPTS/<sub>/<kind>_general_env.txt, if
#               one exists for this kind (only the explorer ships one). Absent → empty.
#   protocol  — benchmarks/<name>/prompts/<kind>_protocol.txt (how to act/emit here).
#               Always present.
#   task      — the {{ task_guidelines }} slot: benchmarks/<name>/prompts/
#               <kind>_task_guidelines.txt, what a GOOD TASK looks like in this
#               environment. UNCONDITIONAL — unlike the env block it is not gated on
#               add_env_knowledge or on whether a coverage goal exists, because the
#               explorer and the guideline updater must never disagree about it.
#               Absent → the slot collapses. These files import the shared doctrine
#               (core/prompts/generation/task_doctrine.txt) and keep only what is
#               environment-bound: the framing, the leak examples, where difficulty
#               comes from, and the genre list.
#   deliv     — the {{ deliverable_partition }} slot: benchmarks/<name>/prompts/
#               <kind>_deliverable_partition.txt. Same mechanism as `task`, for the one
#               rule the coverage survey cannot state benchmark-neutrally: which split of
#               "what the task delivers" is real HERE. It depends on what the benchmark can
#               grade, so the shared template must not assert one (see the per-benchmark
#               files for why). Absent → the slot collapses.
# The env and protocol blocks are themselves Jinja templates over the caller's
# runtime vars (app_descriptions / domain_policy / tool_list / ...), so we render
# inner-first: env and protocol with the vars, then the general template with the
# two rendered blocks slotted in.
_CORE_PROMPT = {
    "solver": ("agent", "solver_system.txt"),
    "explorer": ("generation", "explorer_system.txt"),
    "naive_explorer": ("generation", "naive_explorer_system.txt"),
    "explorer_refine": ("generation", "explorer_refine.txt"),
    "coverage_survey": ("generation", "coverage_survey_system.txt"),
}


def render_prompt(kind, benchmark_prompts_dir, add_env_knowledge=True, protocol=None, **variables):
    """Compose a benchmark agent prompt from general + [env] + protocol.

    kind: "solver" | "explorer" | "naive_explorer" | "explorer_refine" |
        "coverage_survey".
    benchmark_prompts_dir: the benchmark's prompts/ search path (one directory, or
        several most-specific-first; holds the env/protocol files).
    add_env_knowledge: include the benchmark's <kind>_env.txt block when present.
    protocol: protocol filename override; defaults to "<kind>_protocol.txt".
    variables: runtime + benchmark vars used by the env, protocol, and general templates.
    """
    env = _jinja(benchmark_prompts_dir)

    sub, core_name = _CORE_PROMPT[kind]
    general = (CORE_PROMPTS / sub / core_name).read_text(encoding="utf-8")

    protocol_name = protocol or f"{kind}_protocol.txt"
    protocol_file = _find(benchmark_prompts_dir, protocol_name)
    if protocol_file is None:
        searched = ", ".join(str(d) for d in _dirs(benchmark_prompts_dir))
        raise FileNotFoundError(f"no {protocol_name} on the prompt search path: {searched}")
    protocol_text = env.from_string(protocol_file.read_text(encoding="utf-8")).render(**variables)

    # env block: the benchmark's own hints when add_env_knowledge is on, else a
    # benchmark-agnostic fallback (<kind>_general_env.txt in the core prompts) when one
    # exists for this kind. Either way it fills the general template's {{ env_knowledge }}
    # slot; if the chosen file is absent the slot collapses cleanly.
    env_file = (
        _find(benchmark_prompts_dir, f"{kind}_env.txt")
        if add_env_knowledge
        else CORE_PROMPTS / sub / f"{kind}_general_env.txt"
    )
    env_text = ""
    if env_file is not None and env_file.exists():
        rendered = env.from_string(env_file.read_text(encoding="utf-8")).render(**variables).strip()
        if rendered:
            # Surrounding blank lines set the env block off from the general body
            # above and the protocol below; when absent the slot collapses cleanly.
            env_text = "\n" + rendered + "\n"

    return env.from_string(general).render(
        env_knowledge=env_text,
        action_protocol=protocol_text,
        task_guidelines=task_guidelines(benchmark_prompts_dir, kind, **variables),
        deliverable_partition=benchmark_block(
            benchmark_prompts_dir, f"{kind}_deliverable_partition", **variables
        ),
        **variables,
    )


def benchmark_block(benchmark_prompts_dir, name, **variables) -> str:
    """Render <name>.txt from the prompt search path as a template, or "" if absent.

    The generic loader behind the benchmark-specific slots in the shared templates.
    A more specific directory on the search path shadows a less specific one, so a
    benchmark whose environments need different wording supplies the block per
    environment without the shared template knowing about it.
    """
    f = _find(benchmark_prompts_dir, f"{name}.txt")
    if f is None:
        return ""
    return (
        _jinja(benchmark_prompts_dir)
        .from_string(f.read_text(encoding="utf-8"))
        .render(**variables)
        .strip()
    )


# The refiner rewrites a task, so it is bound by the same "what a good task is" block as the
# designer — it reads the explorer's file rather than one of its own.
_TASK_GUIDELINES_KIND = {"explorer_refine": "explorer"}


def task_guidelines(benchmark_prompts_dir, kind="explorer", **variables) -> str:
    """The benchmark's "what a good task looks like here" block, or "" if it has none.

    Exposed separately because two callers need the SAME text: render_prompt fills the
    explorer's {{ task_guidelines }} slot with it, and the guideline updater
    (core/generation/guidelines.py) shows it the same block so the guidelines it writes
    cannot drift into contradicting it — which is exactly what happened in the 50/90-session
    runs, where the evolved guidelines ended up instructing the explorer to "anchor each task
    to exact live source records" and to avoid "process every current ..." phrasing.
    """
    kind = _TASK_GUIDELINES_KIND.get(kind, kind)
    return benchmark_block(benchmark_prompts_dir, f"{kind}_task_guidelines", **variables)
