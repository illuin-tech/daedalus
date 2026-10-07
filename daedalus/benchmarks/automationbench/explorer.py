"""AutomationBench explorer: design one task inside a borrowed world, and prove it.

One session is a single function-calling loop in a throwaway copy of one of the
benchmark's own worlds. The explorer reads real records with the same `api_search` /
`api_fetch` surface the solver gets, executes a candidate solution for real, and then
calls `submit_task` with the statement, the path it actually ran, and the outcome
conditions a judge can check.

Four guards sit on that terminal call. The spec has to parse (see `spec.py`); at least
one WRITE has to have succeeded, since an explorer that only read data has not proved its
task is feasible; **every write the spec declares has to appear among the calls actually
executed**, which is AppWorld's test and the one that catches a described-but-unperformed
solution; and no write it attempted may be left failing. That last
guard exists because a spec whose conditions depend on an endpoint that errors in this
world can never be satisfied by the solver either — the first run of this backend spent
111 solver attempts on tasks that required a Drive permission call the world refuses.

The same loop serves three jobs, differing only in the prompt and the terminal tool:
`explore` (design a task), `refine` (rewrite one that landed too easy or too hard), and
`survey_coverage` (the run's one-time coverage goal + tag vocabulary).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from daedalus.core.config import ExperimentConfig
from daedalus.core.generation import coverage
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.resources import render_prompt

from .env import AutomationBenchSession
from .spec import SPEC_SCHEMA, parse_spec, spec_to_task
from .task_loader import Task

_OUTPUT_LIMIT = 4000  # api_search returns whole endpoint schemas; cap what re-enters context

_RESET_TOOL = {
    "type": "function",
    "function": {
        "name": "reset_environment",
        "description": (
            "Discard every change you have made and restore the world to its initial "
            "state. Use it after a trial run, so your reference solution is executed "
            "once from a clean world."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_task",
        "description": (
            "Submit your final task once you have executed a working solution. Give the "
            "task statement, the expected_path you actually ran, and outcome-based "
            "success_conditions. A rejection explains what to fix."
        ),
        "parameters": SPEC_SCHEMA,
    },
}

_COVERAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_coverage_goal",
        "description": (
            "Submit the run's coverage goal: a short report on how tasks should be "
            "spread across this environment, plus the closed tag vocabulary that "
            "measures it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "The coverage goal, in prose."},
                "tags": {
                    "type": "array",
                    "description": "The closed tag set; target_share values sum to 1.",
                    "items": {
                        "type": "object",
                        "properties": {
                            # "tag", not "name": coverage.parse_tags reads this key and
                            # silently DROPS any entry that lacks it, which turned a
                            # perfectly good four-tag manifest into no manifest at all.
                            "tag": {"type": "string"},
                            "definition": {"type": "string"},
                            "target_share": {"type": "number"},
                        },
                        "required": ["tag", "definition", "target_share"],
                    },
                },
            },
            "required": ["report", "tags"],
        },
    },
}


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


class AutomationBenchExplorer:
    """Runs one exploration → design → ground → emit session in a borrowed world."""

    def __init__(self, cfg: ExperimentConfig, accounting: CostAccounting | None = None):
        self.cfg = cfg
        self.gen = cfg.generation
        self.verbose = cfg.logging.verbose
        self.log_prefix = ""
        self.accounting = accounting or CostAccounting.for_worker(cfg)
        self.llm = self.accounting.client(
            self.gen.explorer_model, "explorer", temperature=1.0
        )
        self._contexts: dict[str, Task] | None = None
        self._write_log: dict[str, Any] = {"ok": set(), "failed": {}, "executed": set()}

    # ----- context + session -----

    def _ctx_task(self, ctx: str) -> Task:
        if self._contexts is None:
            from .spec import build_contexts

            self._contexts = build_contexts(self.cfg)
        return self._contexts[ctx]

    def _session(self, ctx: str) -> AutomationBenchSession:
        return AutomationBenchSession(self._ctx_task(ctx), self.cfg.automationbench)

    def _log(self, msg: str) -> None:
        console.detail(msg, verbose=self.verbose, prefix=self.log_prefix)

    def _prompt_dirs(self) -> list[Path]:
        return [Path(__file__).parent / "prompts"]

    # ----- prompts -----

    def _system_prompt(
        self,
        kind: str,
        ctx: str,
        session: AutomationBenchSession,
        prior_tasks: list[str],
        guidelines: list[str],
        coverage_goal: str = "",
        coverage_tally: str = "",
        **extra: Any,
    ) -> str:
        return render_prompt(
            kind,
            self._prompt_dirs(),
            add_env_knowledge=True,
            domain=self._ctx_task(ctx).domain,
            services=", ".join(session.world.meta.allowed_services or []),
            prior_tasks=prior_tasks,
            guidelines=guidelines,
            coverage_goal=coverage_goal,
            coverage_tally=coverage_tally,
            **extra,
        )

    # ----- terminal tools -----

    def _task_terminal(self, session: AutomationBenchSession) -> dict[str, Any]:
        """submit_task: parse the spec, require a successful write, and refuse a spec that
        rests on a write this world rejects.

        The "did anything happen" test is the WRITE LOG, not a diff of the world against a
        fresh copy of its initial state. That diff cannot work: `WorldState` fills unset
        `meta.current_time`, `created_at` and row `id` fields from the clock and from
        `uuid4`, so two constructions of the SAME initial state differ on every task
        measured (15 of 15), and a world-diff guard would pass every submission — including
        one from an explorer that only read data.
        """

        def handler(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
            spec, reason = parse_spec(args)
            if spec is None:
                return None, reason
            if not self._write_log.get("ok"):
                return None, (
                    "no write of yours has succeeded in this world, so nothing you "
                    "describe has been executed. Run your reference solution for real "
                    "first, then submit the path you actually ran."
                )
            declared = self._declared_writes(spec)
            if not declared:
                return None, (
                    "expected_path contains no write (only GETs). A task is defined by "
                    "what it changes, and a read-only task cannot be checked against an "
                    "end state — design one that writes."
                )
            missing = sorted(declared - self._write_log.get("executed", set()))
            if missing:
                return None, (
                    f"you declared calls you never ran here: {missing[:3]}. Either "
                    "execute your reference solution for real from the current state, or "
                    "correct expected_path to the calls you actually made."
                )
            broken = self._unresolved_writes()
            if broken:
                return None, (
                    f"these writes never succeeded in this world: {'; '.join(broken[:3])}. "
                    "A condition that depends on one of them is unsatisfiable for the "
                    "solver too. Either get the call working and re-run your solution, or "
                    "design a task that does not need it."
                )
            return spec, ""

        return {"name": "submit_task", "schema": _SUBMIT_TOOL, "handler": handler}

    @staticmethod
    def _norm_call(method: str, url: str) -> str:
        """A call as "METHOD service /path" with ids masked — the comparable form.

        Both sides of the grounding test pass through here: the calls the explorer ran,
        and the `expected_path` entries it declares. The service comes from the schema's
        baseUrl table, never from the host string (`sheets.googleapis.com` is
        google_sheets, not "googleapis").
        """
        from .spec import _host_to_service

        url = str(url).split("?")[0]
        host = url.split("//")[-1].split("/")[0]
        service = _host_to_service().get(host) or host or "?"
        path = "/" + "/".join(url.split("//")[-1].split("/")[1:])
        # mask anything that looks like a record id: long, or digit-bearing
        path = re.sub(r"/(?=[^/]*\d)[A-Za-z0-9_.\-]{4,}", "/{id}", path)
        path = re.sub(r"/[A-Za-z0-9_\-]{16,}", "/{id}", path)
        return f"{str(method).upper()} {service} {path.rstrip('/')}"

    def _declared_writes(self, spec: dict[str, Any]) -> set[str]:
        """The non-GET calls the spec claims its reference solution made."""
        out = set()
        for entry in spec.get("expected_path") or []:
            parts = str(entry).split()
            if len(parts) < 2:
                continue
            method = parts[0].upper()
            if method in ("GET", "HEAD"):
                continue
            out.add(self._norm_call(method, parts[1]))
        return out

    def _record_call(self, name: str, args: dict[str, Any], observation: str) -> None:
        """Track which api_fetch WRITES worked, so the submit guard can refuse a spec
        that depends on one this world rejects. An error arrives as a JSON body with an
        `error` key, not as a raised exception, so the observation is what to read.

        The log's shape is self-healing: a caller that built it before `executed` existed
        (or with only the two write buckets) must not crash here, because the bookkeeping
        is an internal detail, not a contract.
        """
        self._write_log.setdefault("ok", set())
        self._write_log.setdefault("failed", {})
        self._write_log.setdefault("executed", set())
        failed_marker = '"error"' in observation or "'error'" in observation
        if name == "api_fetch" and not failed_marker:
            self._write_log["executed"].add(
                self._norm_call(args.get("method") or "", args.get("url") or "")
            )
        if name != "api_fetch":
            return
        method = str(args.get("method") or "").upper()
        if method in ("", "GET", "HEAD"):
            return
        key = f"{method} {str(args.get('url') or '')}"
        if failed_marker:
            self._write_log["failed"].setdefault(key, observation[:160])
        else:
            self._write_log["ok"].add(key)

    def _unresolved_writes(self) -> list[str]:
        """Writes that errored and never later succeeded, as "METHOD url — message"."""
        return [
            f"{k} — {v}"
            for k, v in self._write_log.get("failed", {}).items()
            if k not in self._write_log.get("ok", set())
        ]

    def _coverage_terminal(self) -> dict[str, Any]:
        def handler(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
            goal = coverage.CoverageGoal(
                report=str(args.get("report") or "").strip(),
                tags=coverage.parse_tags(args.get("tags")),
            )
            if not goal.report:
                return None, "report is empty; state how tasks should be spread here"
            return {"goal": goal}, ""

        return {"name": "submit_coverage_goal", "schema": _COVERAGE_TOOL, "handler": handler}

    # ----- the loop -----

    def _run_loop(
        self,
        messages: list[dict[str, Any]],
        session: AutomationBenchSession,
        tag: str,
        terminal: dict[str, Any],
        steer: Callable[[int, int, bool], str | None],
        max_turns: int,
        grace: int = 0,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        """Generic function-calling loop, shared by explore / refine / survey.

        `terminal` = {name, schema, handler}: the emit tool that ends the loop
        (handler(args) -> (obj, reason); obj is not None = accepted). `steer` nudges per
        turn. On the final turn only the terminal tool is offered, with tool_choice
        required, so a session cannot end without a submission attempt; `grace` reopens
        the budget once if that forced attempt is rejected.
        """
        tools = list(session.openai_tools) + [_RESET_TOOL, terminal["schema"]]
        allowed = {t["function"]["name"] for t in tools}
        self._write_log = {"ok": set(), "failed": {}, "executed": set()}
        session_log: list[dict] = []
        rejections = 0
        budget, grace_used, turn = max_turns, False, 0

        while turn < budget:
            remaining = budget - turn
            is_last = remaining == 1
            call_tools, tool_choice = tools, None
            nudge = steer(turn, remaining, is_last)
            if nudge and messages[-1].get("role") != "user":
                messages.append({"role": "user", "content": nudge})
            if is_last:
                call_tools = [terminal["schema"]]
                tool_choice = "required"

            resp = self.llm.generate(
                messages,
                tools=call_tools,
                tool_choice=tool_choice,
                reasoning_effort=self.gen.explorer_reasoning_effort,
            )

            if not resp.tool_calls:
                self._log(f"    [{tag}] turn {turn + 1}: text — nudging to act")
                session_log.append(
                    {"turn": turn + 1, "kind": "text", "content": resp.content}
                )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Use tool calls to do your work, then call {terminal['name']}. "
                            "Do not answer in plain text."
                        ),
                    }
                )
                turn += 1
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )
            if resp.content:
                session_log.append(
                    {"turn": turn + 1, "kind": "assistant", "content": resp.content}
                )

            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc.arguments or "{}")
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                if tc.name == terminal["name"]:
                    obj, reason = terminal["handler"](args)
                    session_log.append(
                        {
                            "turn": turn + 1,
                            "kind": "submit",
                            "accepted": obj is not None,
                            "arguments": args,
                            "reason": reason,
                        }
                    )
                    if obj is not None:
                        self._log(f"    [{tag}] turn {turn + 1}: {terminal['name']} accepted")
                        return obj, session_log
                    rejections += 1
                    self._log(f"    [{tag}] turn {turn + 1}: rejected — {reason[:90]}")
                    if rejections >= self.gen.spec_max_rejections:
                        self._log(f"    [{tag}] too many rejections — giving up")
                        return None, session_log
                    if is_last and grace and not grace_used:
                        grace_used = True
                        budget += grace
                        reason += (
                            f" You now have {grace} more turns and the full toolset back: "
                            "reset_environment, execute your reference solution from the "
                            "clean state, then submit what you actually ran."
                        )
                    result = f"REJECTED: {reason}"
                elif tc.name == "reset_environment":
                    # The world no longer holds what was written, so the evidence of what
                    # ran must go with it — otherwise a reset-then-submit would pass the
                    # grounding guard on stale history.
                    self._write_log = {"ok": set(), "failed": {}, "executed": set()}
                    session = self._session_like(session)
                    tools = list(session.openai_tools) + [_RESET_TOOL, terminal["schema"]]
                    allowed = {t["function"]["name"] for t in tools}
                    terminal = self._retarget(terminal, session)
                    result = "Environment reset to its initial state."
                    session_log.append({"turn": turn + 1, "kind": "reset"})
                elif tc.name not in allowed:
                    result = f"Tool {tc.name!r} is not available. Available: {', '.join(sorted(allowed))}"
                else:
                    res = session.call(tc.name, args)
                    from .env import render_observation

                    result = _truncate(render_observation(res))
                    self._record_call(tc.name, args, result)
                    session_log.append(
                        {
                            "turn": turn + 1,
                            "kind": "tool",
                            "name": tc.name,
                            "arguments": args,
                            "output": _truncate(result, 600),
                            "error": not res.success,
                        }
                    )
                    self._log(
                        f"    [{tag}] turn {turn + 1}: {tc.name}"
                        f"{' [ERROR]' if not res.success else ''}"
                    )

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            turn += 1

        self._log(f"    [{tag}] turn ceiling ({budget}) reached — no {terminal['name']}")
        return None, session_log

    def _session_like(self, session: AutomationBenchSession) -> AutomationBenchSession:
        """A fresh session on the same world (the reset_environment tool)."""
        return AutomationBenchSession(session.task, self.cfg.automationbench)

    def _retarget(self, terminal: dict[str, Any], session: AutomationBenchSession) -> dict[str, Any]:
        """Rebind a task terminal to the new session after a reset (the coverage terminal
        holds no session state, so it is returned unchanged)."""
        if terminal["name"] != "submit_task":
            return terminal
        return self._task_terminal(session)

    # ----- steering -----

    def _steer(self, terminal_name: str) -> Callable[[int, int, bool], str | None]:
        def steer(turn: int, remaining: int, is_last: bool) -> str | None:
            if is_last:
                return (
                    f"This is your LAST turn. Call {terminal_name} now with what you have "
                    "already executed."
                )
            if remaining <= 5:
                return f"{remaining} turns left — finish executing and call {terminal_name}."
            if turn and turn % 10 == 0:
                return f"{remaining} turns left."
            return None

        return steer

    # ----- the three entry points -----

    def explore(
        self,
        ctx: str,
        prior_tasks: list[str],
        guidelines: list[str],
        coverage_goal: str = "",
        coverage_tally: str = "",
        **_ignored: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        session = self._session(ctx)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    "explorer", ctx, session, prior_tasks, guidelines,
                    coverage_goal, coverage_tally,
                ),
            },
            {
                "role": "user",
                "content": (
                    "Explore this company's data with api_search and api_fetch until you "
                    "know what is really there. Design one task, EXECUTE your solution "
                    "for real, then call submit_task with the path you ran."
                ),
            },
        ]
        self._log(
            f"    [explorer] world {ctx} | ≤{self.gen.explorer_max_turns} turns"
        )
        return self._run_loop(
            messages,
            session,
            tag="explorer",
            terminal=self._task_terminal(session),
            steer=self._steer("submit_task"),
            max_turns=self.gen.explorer_max_turns,
            grace=self.gen.explorer_last_turn_grace,
        )

    def refine(
        self,
        ctx: str,
        prior_tasks: list[str],
        guidelines: list[str],
        exploration: list[dict],
        current_task: str,
        difficulty: str,
        variant_history: list[str],
        coverage_goal: str = "",
        coverage_tally: str = "",
        **extra: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        session = self._session(ctx)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    "explorer_refine", ctx, session, prior_tasks, guidelines,
                    coverage_goal, coverage_tally,
                    current_task=current_task,
                    difficulty=difficulty,
                    variant_history=variant_history,
                    nearest_task=extra.get("nearest_task", ""),
                    nearest_path=extra.get("nearest_path", []),
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The task above came out {difficulty}. Rewrite it, execute the new "
                    "solution for real in this world, then call submit_task."
                ),
            },
        ]
        self._log(f"    [refine:{difficulty}] world {ctx}")
        return self._run_loop(
            messages,
            session,
            tag=f"refine:{difficulty}",
            terminal=self._task_terminal(session),
            steer=self._steer("submit_task"),
            max_turns=self.gen.explorer_max_turns,
            grace=self.gen.explorer_last_turn_grace,
        )

    def survey_coverage(self, ctx: str) -> tuple[coverage.CoverageGoal, list[dict]]:
        """One-time survey of a throwaway world → the run's coverage goal + tags."""
        session = self._session(ctx)
        messages = [
            {
                "role": "system",
                # The survey never takes an env-knowledge block: it exists to discover
                # the priors, so it is always general + protocol.
                "content": render_prompt(
                    "coverage_survey",
                    self._prompt_dirs(),
                    add_env_knowledge=False,
                    domain=self._ctx_task(ctx).domain,
                    services=", ".join(session.world.meta.allowed_services or []),
                ),
            },
            {
                "role": "user",
                "content": (
                    "Begin surveying: read real records across several apps, then CALL the "
                    "write endpoints too — that is how you learn what they demand. Call "
                    "submit_coverage_goal once you have surveyed broadly."
                ),
            },
        ]
        self._log(f"    [survey] world {ctx} | ≤{self.gen.coverage_tags_max_turns} turns")
        result, log = self._run_loop(
            messages,
            session,
            tag="survey",
            terminal=self._coverage_terminal(),
            steer=self._steer("submit_coverage_goal"),
            max_turns=self.gen.coverage_tags_max_turns,
        )
        return (result or {}).get("goal", coverage.CoverageGoal()), log


__all__ = ["AutomationBenchExplorer", "spec_to_task"]
