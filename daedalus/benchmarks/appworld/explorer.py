"""AppWorld explorer (and surveyor) for self-play generation.

One explorer session plays inside a sandbox world: it explores the live data, designs a
single grounded task, solves it in code to prove it feasible, then emits a spec
`{task, expected_path, success_conditions, tags}`. The spec is accepted only if its
expected path was actually executed. The same agent runs the one-time coverage survey
and, for the row-B ablation, explorer-only sessions (`explore_direct`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


from appworld import AppWorld

from daedalus.benchmarks.appworld.agent import _appworld_safe_name, parse_thought_and_code
from daedalus.benchmarks.appworld.direct_explore import app_catalog, explore_directly
from daedalus.benchmarks.appworld.hidden_apps import HiddenApps
from daedalus.core.config import ExperimentConfig
from daedalus.core.generation import coverage
from daedalus.benchmarks.appworld.paths import path_from_code, predicted_path
from daedalus.core.generation.survey_guardrails import guard_python_code
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging import console

_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)

# Give up a session after too many rejected specs (a sign the explorer isn't
# converging on a valid one), so it can't loop until the context window overflows.
_MAX_REJECTED_SPECS = 10


def parse_final_artifact(text: str) -> dict[str, Any] | None:
    """Return the explorer's final spec if the message is a ```json block.

    Returns None when the message has no json fence (i.e. it's still an
    exploration step), or when the block doesn't parse / lacks required keys.

    Optional keys ride along untouched — notably `tags`, the coverage tags the explorer
    declares for this task (see core/generation/coverage.py); they are filtered against
    the manifest when the task is banked, not here.
    """
    m = _JSON_FENCE_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except ValueError:  # JSONDecodeError is a ValueError subclass
        return None
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("task", "expected_path", "success_conditions")):
        return None
    return obj


class ExplorerAgent:
    """Runs one exploration→design→solve→emit session in a borrowed world."""

    def __init__(
        self, cfg: ExperimentConfig, accounting: CostAccounting | None = None
    ):
        self.cfg = cfg
        self.gen = cfg.generation
        self.verbose = cfg.logging.verbose
        self.log_prefix = ""  # set by parallel workers to disambiguate interleaved output
        # An explorer built outside a generation launch (the coverage-survey script)
        # opens its own ledger, so exploration is never untracked LLM work.
        self.accounting = accounting or CostAccounting.for_worker(cfg)
        self.llm = self.accounting.client(
            self.gen.explorer_model, "explorer", temperature=1.0
        )
        # Apps held out for evaluation: hidden from the catalog, from every runtime listing,
        # and from the channel itself, so no agent here learns they exist.
        self._hidden = HiddenApps(self.gen.excluded_apps or [])

    def _log(self, msg: str) -> None:
        console.detail(msg, verbose=self.verbose, prefix=self.log_prefix)

    def _app_descriptions(self, world: AppWorld) -> str:
        """App catalog shown to the explorer, minus the held-out apps (gmail/amazon appear
        only in the test_challenge split). They are also hidden from runtime listings and
        refused on the channel — see hidden_apps.py."""
        return app_catalog(world, self._hidden)

    def _render_explorer_system(
        self, world, prior_tasks, guidelines, coverage_goal="", coverage_tally=""
    ) -> str:
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "explorer",
            Path(__file__).parent / "prompts",
            add_env_knowledge=True,  # env facts always on; task style lives in task_guidelines
            protocol="explorer_protocol.txt",
            app_descriptions=self._app_descriptions(world),
            main_user=world.task.supervisor,
            prior_tasks=prior_tasks,
            guidelines=guidelines,
            coverage_goal=coverage_goal,
            coverage_tally=coverage_tally,
        )

    def _render_refine_user(
        self, difficulty, direction, exploration, current_task, variant_history,
        coverage_tally="",
    ) -> str:
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "explorer_refine",
            Path(__file__).parent / "prompts",
            add_env_knowledge=True,  # env facts always on; task style lives in task_guidelines
            protocol="explorer_refine_protocol.txt",
            difficulty=difficulty,
            direction=direction,
            exploration=_format_transcript(exploration),
            current_task=current_task,
            variant_history=variant_history,
            coverage_tally=coverage_tally,
        )

    def _kickoff_messages(
        self, world: AppWorld, prior_tasks: list[str], guidelines: list[str],
        coverage_goal: str = "", coverage_tally: str = "",
    ) -> list[dict[str, str]]:
        system = self._render_explorer_system(
            world, prior_tasks, guidelines, coverage_goal, coverage_tally
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                "Begin by inspecting the environment with code. Your first message must be a "
                "Thought and a ```python block (e.g. discover APIs and read some real data). "
                "Do NOT emit a ```json spec until you have explored the data and actually "
                "executed a working solution."
            )},
        ]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        return self.llm.generate(
            messages, reasoning_effort=self.gen.explorer_reasoning_effort
        ).content

    def _steer(self, turn: int, budget: int) -> str:
        """The remaining-turn reminder appended to one execution result, or "".

        Off unless `generation.explorer_turn_steering`. The wording escalates, because a
        bare count did not stop deepseek-v4-pro-0813 from opening a new app on its final
        turn: past the halfway mark it is told to pick ONE task, and in the last turns to
        stop exploring and emit.
        """
        left = budget - turn - 1
        if left <= 0:
            return ""
        if left <= 3:
            return (
                f"{left} turn(s) left. Stop exploring. EXECUTE the full solution to one "
                f"task now, then emit the ```json spec."
            )
        if left <= budget // 2:
            return (
                f"{left} turns left. Pick ONE task and execute its full solution. "
                f"Do not start on a new app."
            )
        return f"{left} turns left."

    def _run_loop(
        self, world: AppWorld, messages: list[dict[str, str]], tag: str,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        """Shared explore/refine turn loop: act in the world until a grounded spec
        is emitted (feasibility-by-construction) or the turn ceiling is hit."""
        session_log: list[dict] = []
        executed: list[str] = []  # realized `app.tool` calls across the session
        rejected = 0  # emitted specs that failed the grounding guard

        # Budget can be extended once: a spec rejected on the last turn reopens the loop
        # (see explorer_last_turn_grace) so the explorer can execute + resubmit rather than
        # ending the session unbanked.
        budget = self.gen.explorer_max_turns
        grace = self.gen.explorer_last_turn_grace
        grace_used = False
        turn = 0
        while turn < budget:
            content = self._generate(messages)
            messages.append({"role": "assistant", "content": content})
            thought, code = parse_thought_and_code(content)

            # Code-first: if the message has a python block, RUN it — even when a
            # premature json spec rides along in the same message (otherwise that
            # exploration code would be silently discarded and the session would
            # never satisfy the feasibility guard, looping until the ceiling).
            if code:
                # Held-out apps do not exist for this agent: refuse the call before it runs
                # (see hidden_apps.py) so no task can be designed around one.
                hidden = self._hidden.blocked(code)
                if hidden:
                    session_log.append({"turn": turn + 1, "kind": "blocked", "thought": thought,
                                        "code": code, "reason": hidden})
                    self._log(f"    [{tag}] turn {turn + 1}: BLOCKED — {_oneline(hidden, 90)}")
                    messages.append({"role": "user", "content": f"Output:\n```\n{hidden}\n```"})
                    turn += 1
                    continue
                try:
                    output = world.execute(code)
                except Exception as e:  # noqa: BLE001 — surface error back to the model
                    output = str(e)
                output = self._hidden.filter(output)
                exec_error = output.strip().startswith("Execution failed")
                executed.extend(path_from_code(code))
                session_log.append({"turn": turn + 1, "kind": "exec", "thought": thought,
                                    "code": code, "output": output, "exec_error": exec_error})
                self._log(
                    f"    [{tag}] turn {turn + 1}:{' [ERROR]' if exec_error else ''} "
                    f"{_oneline(thought, 90)}\n"
                    f"               code: {_oneline(code, 90)}\n"
                    f"               out : {_oneline(output, 90)}"
                )
                steer = (
                    self._steer(turn, budget)
                    if self.gen.explorer_turn_steering
                    else ""
                )
                messages.append({"role": "user", "content": (
                    f"Output:\n```\n{output}\n```" + (f"\n\n{steer}" if steer else "")
                )})
                turn += 1
                continue

            # No code in this message: a json spec means the explorer is done.
            artifact = parse_final_artifact(content)
            if artifact is not None:
                ok, reason = _grounded(artifact, executed, session_log)
                if ok:
                    self._log(f"    [{tag}] ✓ emitted spec after {turn + 1} turns")
                    return artifact, session_log
                rejected += 1
                self._log(f"    [{tag}] turn {turn + 1}: rejected spec ({rejected}/{_MAX_REJECTED_SPECS}) — {reason}")
                session_log.append({"turn": turn + 1, "kind": "rejected_spec",
                                    "content": content, "reason": reason})
                if rejected >= _MAX_REJECTED_SPECS:
                    self._log(f"    [{tag}] gave up after {rejected} rejected specs")
                    return None, session_log
                if turn >= budget - 1 and grace and not grace_used:
                    # Last-turn rejection: reopen the budget once so the explorer can run
                    # its solution and resubmit instead of ending the session unbanked.
                    grace_used = True
                    budget += grace
                    self._log(f"    [{tag}] last-turn spec rejected — granting {grace} more turns to converge")
                messages.append({"role": "user", "content": (
                    f"Not accepted: {reason}. Explore and actually EXECUTE the full "
                    f"solution in a ```python block — do NOT include a json spec until "
                    f"you have run the solution."
                )})
                turn += 1
                continue

            self._log(f"    [{tag}] turn {turn + 1}: no code — nudging for format")
            session_log.append({"turn": turn + 1, "kind": "no_code", "content": content})
            messages.append({"role": "user", "content": (
                "Respond with a Thought then a ```python block, or the final "
                "```json spec once the task is solved."
            )})
            turn += 1
            continue

        # Ceiling reached — ask once for the spec.
        self._log(f"    [{tag}] turn ceiling ({budget}) — requesting spec")
        messages.append({"role": "user", "content": (
            "Exploration limit reached. If you have solved a task, output the final "
            "```json spec now; otherwise output nothing parseable."
        )})
        artifact = parse_final_artifact(self._generate(messages))
        if artifact is not None and _grounded(artifact, executed, session_log)[0]:
            return artifact, session_log
        return None, session_log

    # ── coverage survey (one-time: the run's coverage goal + tag vocabulary) ──
    def _survey_system_prompt(self, world: AppWorld) -> str:
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "coverage_survey",
            Path(__file__).parent / "prompts",
            add_env_knowledge=False,  # the survey exists to discover the priors
            app_descriptions=self._app_descriptions(world),
            main_user=world.task.supervisor,
        )

    def _survey_loop(
        self, world: AppWorld, messages: list[dict[str, str]]
    ) -> tuple[coverage.CoverageGoal, list[dict]]:
        """Exploration loop terminating on the <coverage_goal> + <tag_manifest> message.

        The survey may change the world freely (it runs in its own borrowed copy, discarded
        when the survey ends) — calling a write API is how it learns what that API demands.
        AppWorld's channel is arbitrary Python, so every block goes through the shared
        guardrail first: that blocks escapes (host/filesystem/package internals/any task or
        eval set), not app mutations. Held-out apps are hidden on top of that."""
        max_turns = self.gen.coverage_tags_max_turns
        transcript: list[dict] = []
        for turn in range(max_turns):
            content = self._generate(messages)
            messages.append({"role": "assistant", "content": content})
            thought, code = parse_thought_and_code(content)
            if code:
                # An app the generator may not build tasks around must not shape the
                # coverage goal or earn a tag either, so hide it here as well.
                blocked = self._hidden.blocked(code) or guard_python_code(code)
                if blocked:
                    transcript.append({"turn": turn + 1, "kind": "blocked", "thought": thought,
                                       "code": code, "reason": blocked})
                    self._log(f"    [survey] turn {turn + 1}: BLOCKED — {_oneline(blocked, 90)}")
                    messages.append({"role": "user", "content": f"Output:\n```\nBLOCKED: {blocked}\n```"})
                    continue
                try:
                    output = world.execute(code)
                except Exception as e:  # noqa: BLE001 — surface error back to the model
                    output = str(e)
                output = self._hidden.filter(output)
                transcript.append({"turn": turn + 1, "kind": "exec", "thought": thought,
                                   "code": code, "output": output})
                self._log(f"    [survey] turn {turn + 1}: {_oneline(thought, 90)}\n"
                          f"               code: {_oneline(code, 90)}")
                messages.append({"role": "user", "content": f"Output:\n```\n{output}\n```"})
                continue
            goal = coverage.CoverageGoal.from_text(content)
            if goal.report and goal.tagged:
                transcript.append({"turn": turn + 1, "kind": "coverage_goal", "goal": goal.to_dict()})
                self._log(f"    [survey] coverage goal + {len(goal.tags)} tag(s) after {turn + 1} turns")
                return goal, transcript
            transcript.append({"turn": turn + 1, "kind": "text", "content": content})
            self._log(f"    [survey] turn {turn + 1}: no code / incomplete goal — nudging")
            messages.append({"role": "user", "content": (
                "Reply with a Thought and a single ```python block to keep exploring, or — once "
                "you have surveyed broadly — the final message carrying BOTH a "
                "<coverage_goal>…</coverage_goal> block and a <tag_manifest>…</tag_manifest> "
                "JSON block."
            )})
        self._log(f"    [survey] turn ceiling ({max_turns}) — requesting the coverage goal")
        messages.append({"role": "user", "content": (
            "Survey budget reached. Output your final <coverage_goal> and <tag_manifest> blocks "
            "now, based on everything you observed."
        )})
        content = self._generate(messages)
        transcript.append({"turn": max_turns + 1, "kind": "forced_goal", "content": content})
        return coverage.CoverageGoal.from_text(content), transcript

    def survey_coverage(self, sandbox_task_id: str) -> tuple[coverage.CoverageGoal, list[dict]]:
        """One-time survey of a borrowed world → the run's coverage goal + tag vocabulary."""
        exp_name = _appworld_safe_name(f"{self.cfg.appworld.experiment_name}_survey")
        with AppWorld(task_id=sandbox_task_id, experiment_name=exp_name) as world:
            messages = [
                {"role": "system", "content": self._survey_system_prompt(world)},
                {"role": "user", "content": (
                    "Begin surveying. Your first message must be a Thought and a single ```python "
                    "block — e.g. `apis.api_docs.show_app_descriptions()` and "
                    "`apis.api_docs.show_api_descriptions(app_name=...)` to size each app's surface. "
                    "Then, over later turns, log in and read real data in several DIFFERENT apps, "
                    "and actually CALL the interesting operations — including the ones that create, "
                    "update or delete — to see what each really demands and does. Emit the final "
                    "<coverage_goal> and <tag_manifest> blocks only once you have surveyed broadly."
                )},
            ]
            self._log(f"    [survey] world {sandbox_task_id} | coverage survey | "
                      f"≤{self.gen.coverage_tags_max_turns} turns"
                      + (f" | hidden apps: {', '.join(self._hidden.names)}"
                         if self._hidden else ""))
            return self._survey_loop(world, messages)

    def explore(
        self, sandbox_task_id: str, prior_tasks: list[str], guidelines: list[str],
        coverage_goal: str = "", coverage_tally: str = "", **_ignored: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        """Play one session in `sandbox_task_id`'s world.

        Returns (artifact_or_None, session_log). `prior_tasks` are the banked-task
        descriptions in T (novelty steering); `guidelines` is the distilled explorer
        memory (the only difficulty signal shown to the explorer); `coverage_goal` /
        `coverage_tally` are the survey's spread plan and the live per-tag tally.
        """
        exp_name = _appworld_safe_name(f"{self.cfg.appworld.experiment_name}_explore")
        with AppWorld(task_id=sandbox_task_id, experiment_name=exp_name) as world:
            messages = self._kickoff_messages(
                world, prior_tasks, guidelines, coverage_goal, coverage_tally
            )
            self._log(f"    [explorer] world {sandbox_task_id} | {len(prior_tasks)} prior tasks | "
                      f"{len(guidelines)} guidelines | ≤{self.gen.explorer_max_turns} turns")
            return self._run_loop(world, messages, tag="explorer")

    def explore_direct(
        self, sandbox_task_id: str, previous_heuristics: list[str]
    ) -> tuple[list[str], list[dict]]:
        """Table 3, row B: explore until diminishing returns or the turn ceiling, then emit
        the heuristics not already in `previous_heuristics` (the bank so far)."""
        from daedalus.core.resources import render_prompt

        exp_name = _appworld_safe_name(f"{self.cfg.appworld.experiment_name}_direct_explore")
        with AppWorld(task_id=sandbox_task_id, experiment_name=exp_name) as world:
            system = render_prompt(
                "naive_explorer",
                Path(__file__).parent / "prompts",
                add_env_knowledge=True,
                max_turns=self.gen.explorer_max_turns,
                allow_early_stop=True,
                previous_heuristics="\n".join(previous_heuristics),
                app_descriptions=self._app_descriptions(world),
                main_user=world.task.supervisor,
            )
            return explore_directly(
                world, self._generate, system, self._hidden, self.gen.explorer_max_turns,
                has_bank=bool(previous_heuristics), allow_early_stop=True,
                log=lambda msg: self._log(f"    [direct]{msg}"),
            )

    def refine(
        self, sandbox_task_id: str, prior_tasks: list[str], guidelines: list[str],
        exploration: list[dict], current_task: str,
        difficulty: str, variant_history: list[dict],
        coverage_goal: str = "", coverage_tally: str = "", **_ignored: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        """A refinement phase: re-open the world and adjust the task given the prior
        exploration and the verdict (too_easy → harder, too_hard → easier). Re-solves the
        adjusted task before emitting."""
        direction = {"too_easy": "harder", "too_hard": "easier"}.get(difficulty, "different")
        exp_name = _appworld_safe_name(f"{self.cfg.appworld.experiment_name}_explore")
        with AppWorld(task_id=sandbox_task_id, experiment_name=exp_name) as world:
            system = self._render_explorer_system(
                world, prior_tasks, guidelines, coverage_goal, coverage_tally
            )
            refine_user = self._render_refine_user(
                difficulty, direction, exploration, current_task, variant_history,
                coverage_tally,
            )
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": refine_user}]
            self._log(f"    [refine] {difficulty} → make it {direction} in {sandbox_task_id} ...")
            return self._run_loop(world, messages, tag="refine")


def _grounded(
    artifact: dict[str, Any], executed: list[str], session_log: list[dict]
) -> tuple[bool, str]:
    """Feasibility-by-construction: a spec is valid only if it is a real,
    non-empty task whose solution the explorer actually executed."""
    if len(str(artifact.get("task", "")).strip()) < 15:
        return False, "the task is empty or a placeholder — design a real, specific task"
    if not artifact.get("success_conditions"):
        return False, "success_conditions is empty — state the outcome checks"
    expected = predicted_path(artifact.get("expected_path", []))
    if not expected:
        return False, "expected_path is empty — give the minimal solution calls you executed"
    if not any(e.get("kind") == "exec" for e in session_log):
        return False, "no code was executed — you must explore and solve first"
    if not (set(expected) & set(executed)):
        return False, "none of the expected_path calls were actually executed"
    return True, ""


def _oneline(text: str, n: int) -> str:
    """First line of `text`, collapsed and truncated to `n` chars."""
    line = " ".join((text or "").split())
    return line[:n] + ("…" if len(line) > n else "")


def _format_transcript(session_log: list[dict]) -> str:
    """Render an explorer session's executed turns as readable Thought/Code/Output
    text (for feeding a prior exploration back into a refinement phase)."""
    parts = []
    for e in session_log:
        if e.get("kind") != "exec":
            continue
        out = e.get("output", "")
        if len(out) > 600:
            out = out[:600] + " …"
        parts.append(f"Thought: {e.get('thought','')}\nCode:\n{e.get('code','')}\nOutput:\n{out}")
    return "\n\n".join(parts) if parts else "(no executed turns)"
