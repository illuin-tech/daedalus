"""tau2 explorer — function-calling self-play task generation.

One session plays in a throwaway domain environment with the domain's own tools
plus three meta-tools:
  reset_environment()   rebuild the env (rehearse the reference solution from a
                        clean state after dirtying the DB during exploration)
  read_db(path)         read-only dot-path navigation of the raw domain DB
                        (finds real entity ids fast; NOT given to the solver)
  submit_task_spec(...) emit the spec; structural + grounding validation runs
                        inside the tool, so rejections come back as tool results
                        and the explorer self-corrects in-session.

Emits the ticket-only spec of tau2_spec.py; grounding = replaying the reference
actions on a fresh env (must succeed and change the DB hash).

The one-time coverage survey (`survey_coverage`) runs in the same loop with the same full
channel — it is expected to CALL the write tools (in its own throwaway env) to learn what
they demand, and can reset_environment() for a clean slate.
"""

from __future__ import annotations

import json
from typing import Any

from daedalus.benchmarks.tau2 import prompt_dirs
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.config import ExperimentConfig
from daedalus.core.logging import console
from daedalus.core.generation import coverage
from daedalus.core.generation.survey_guardrails import guard_tool_call
from daedalus.benchmarks.tau2.spec import (
    SPEC_SCHEMA,
    ground_spec,
    leaked_identifier_reason,
    multi_item_exchange_reason,
    parse_spec,
)

_OUTPUT_LIMIT = 3000  # chars of tool output fed back to the explorer
_EXPLORE_BUDGET = 5  # turns of free exploration before design is enforced

# Meta-tools shared by every explorer loop (coverage survey and task design alike).
_BASE_META: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "reset_environment",
            "description": (
                "Rebuild the environment from its pristine initial state, undoing every "
                "change your tool calls made. Use this before executing your final "
                "reference solution: the task will be graded from the clean initial state."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_db",
            "description": (
                "Read-only access to the raw domain database (you have this for task "
                "design; the agent solving your task will NOT). `path` is a dot path, "
                "e.g. '' lists top-level keys, 'users' lists user ids, "
                "'users.sara_doe_496.orders' reads a nested field. List indices are "
                "numeric, e.g. 'orders.#W0000001.items.0'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Dot path into the DB."}
                },
                "required": ["path"],
            },
        },
    },
]

# Terminal tool for a task-design loop (explore / refine).
_SUBMIT_SPEC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_task_spec",
        "description": (
            "Submit the final task spec. Only call this AFTER you have called "
            "reset_environment() and then executed your exact minimal reference "
            "solution from the clean state. The spec is validated by replaying "
            "your actions on a fresh environment; a rejection explains what to fix."
        ),
        "parameters": SPEC_SCHEMA,
    },
}

def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _read_db(db_dump: dict[str, Any], path: str) -> str:
    """Navigate the DB dump by dot path; dict levels list their keys when the
    value is large."""
    node: Any = db_dump
    if path.strip():
        for part in path.strip().split("."):
            if isinstance(node, dict):
                if part not in node:
                    return (
                        f"Error: key {part!r} not found; available: {list(node)[:50]}"
                    )
                node = node[part]
            elif isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError):
                    return f"Error: bad list index {part!r} (len={len(node)})"
            else:
                return f"Error: cannot descend into {type(node).__name__} with {part!r}"
    rendered = json.dumps(node, ensure_ascii=False, default=str)
    if len(rendered) > _OUTPUT_LIMIT and isinstance(node, dict):
        return f"(large object; keys) {json.dumps(list(node), default=str)[:_OUTPUT_LIMIT]}"
    return _truncate(rendered)


class Tau2ExplorerAgent:
    """Runs one exploration→design→ground→emit session in a throwaway tau2 env."""

    def __init__(
        self, cfg: ExperimentConfig, accounting: CostAccounting | None = None
    ):
        self.cfg = cfg
        self.gen = cfg.generation
        self.domain = cfg.tau2.domain
        self.verbose = cfg.logging.verbose
        self.log_prefix = ""
        console.silence_third_party(self.verbose)  # drop tau2's colorized loguru sink
        # An explorer built outside a generation launch (the coverage-survey script)
        # opens its own ledger, so exploration is never untracked LLM work.
        self.accounting = accounting or CostAccounting.for_worker(cfg)
        self.llm = self.accounting.client(
            self.gen.explorer_model, "explorer", temperature=1.0
        )

    def _log(self, msg: str) -> None:
        console.detail(msg, verbose=self.verbose, prefix=self.log_prefix)

    def _build_env(self):
        from tau2.runner import build_environment

        return build_environment(self.domain)

    def _system_prompt(
        self,
        env,
        prior_tasks: list[str],
        guidelines: list[str],
        coverage_goal: str = "",
        coverage_tally: str = "",
    ) -> str:
        v = dict(
            domain=self.domain,
            policy=env.get_policy(),
            tool_list=self._tool_lines(env),
            prior_tasks=prior_tasks,
            guidelines=guidelines,
            coverage_goal=coverage_goal,
            coverage_tally=coverage_tally,
            # Tell the explorer up front, so it never spends a design on a spec the
            # submit-time guard would reject.
            reject_multi_item_exchange=self.gen.reject_multi_item_exchange,
            withhold_discoverable_ids=self.gen.withhold_discoverable_ids,
        )
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "explorer",
            prompt_dirs(self.domain),
            add_env_knowledge=True,  # env facts always on; task style lives in task_guidelines
            **v,
        )

    def _tool_lines(self, env, exclude: set[str] | None = None) -> str:
        excluded = {x.lower() for x in (exclude or set())}
        return "\n".join(
            f"- {t.name}: {(t.short_desc or '').strip()}"
            for t in env.get_tools() if t.name.lower() not in excluded
        )

    def _spec_terminal(self) -> dict[str, Any]:
        """Terminal tool for a task-design loop: parse + ground the submitted spec."""

        def handler(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
            spec, reason = parse_spec(args)
            if spec is None:
                return None, reason
            if self.gen.reject_multi_item_exchange and (
                reason := multi_item_exchange_reason(spec)
            ):
                return None, reason
            if self.gen.withhold_discoverable_ids and (
                reason := leaked_identifier_reason(spec, self.domain)
            ):
                return None, reason
            ok, reason = ground_spec(spec, self.domain)
            return (spec, "") if ok else (None, reason)

        return {
            "name": "submit_task_spec",
            "schema": _SUBMIT_SPEC_TOOL,
            "handler": handler,
        }

    def _design_steer(self, turn: int, remaining: int, is_last: bool) -> str | None:
        """Per-turn budget nudge for a task-design loop (None during free exploration)."""
        if is_last:
            return (
                "This is your FINAL turn. Do not explore further — call "
                "submit_task_spec now with your best task."
            )
        if turn == _EXPLORE_BUDGET:
            return (
                f"You have used your ~{_EXPLORE_BUDGET}-turn exploration budget. "
                "Stop exploring now and focus on the design: decide on the task, "
                "call reset_environment(), execute your minimal reference solution "
                f"from the clean state, then call submit_task_spec. {remaining} turns left."
            )
        if turn > _EXPLORE_BUDGET:
            return (
                f"{remaining} turns left. Finish rehearsing your reference solution "
                "and call submit_task_spec before the budget runs out."
            )
        return None

    def _loop_tools(self, env, terminal: dict[str, Any]) -> list[dict[str, Any]]:
        """OpenAI tool schemas offered inside `_run_loop`: the env's domain tools, the
        meta tools (reset_environment / read_db), and the loop's terminal tool."""
        return [t.openai_schema for t in env.get_tools()] + _BASE_META + [terminal["schema"]]

    def _dispatch_tool_call(
        self, tc, env, terminal: dict[str, Any], tag: str, turn: int, session_log: list[dict],
        allowed: set[str] | None = None,
    ) -> tuple[str, Any]:
        """Handle one tool call from a `_run_loop` turn and log it. Returns (kind, value):
          ("accept", obj)   terminal tool accepted — obj is the grounded spec or coverage goal
          ("reject", reason) terminal tool rejected — reason is fed back to the model
          ("reset", None)    reset_environment — the caller rebuilds env + tool schemas
          ("tool", result)   a domain/read_db call — result is its tool-message content
        The caller owns control flow (early return, rejection counting, env rebuild)."""
        from tau2.data_model.message import ToolCall

        args = json.loads(tc.arguments or "{}")

        # Restricted channel (the read-only coverage survey): refuse anything off the
        # allowlist instead of dispatching it to the environment.
        if allowed is not None and (blocked := guard_tool_call(tc.name, allowed)):
            session_log.append(
                {"turn": turn + 1, "kind": "blocked", "name": tc.name, "reason": blocked}
            )
            self._log(f"    [{tag}] turn {turn + 1}: BLOCKED {tc.name}")
            return "tool", f"BLOCKED: {blocked}"

        if tc.name == terminal["name"]:
            obj, reason = terminal["handler"](args)
            if obj is not None:
                self._log(f"    [{tag}] ✓ {terminal['name']} accepted after {turn + 1} turns")
                session_log.append({"turn": turn + 1, "kind": "submit", "name": terminal["name"]})
                return "accept", obj
            self._log(f"    [{tag}] turn {turn + 1}: {terminal['name']} rejected — {reason}")
            session_log.append(
                {"turn": turn + 1, "kind": "rejected", "name": terminal["name"],
                 "args": args, "reason": reason}
            )
            return "reject", reason

        if tc.name == "reset_environment":
            session_log.append({"turn": turn + 1, "kind": "reset"})
            return "reset", None

        if tc.name == "read_db":
            result = _read_db(env.tools.db.model_dump(mode="json"), args.get("path", ""))
            session_log.append(
                {"turn": turn + 1, "kind": "read_db", "path": args.get("path", ""),
                 "output": _truncate(result, 600)}
            )
            return "tool", result

        tool_msg = env.get_response(
            ToolCall(id=tc.id, name=tc.name, arguments=args, requestor="assistant")
        )
        result = _truncate(tool_msg.content or "")
        session_log.append(
            {"turn": turn + 1, "kind": "tool", "name": tc.name, "arguments": args,
             "output": _truncate(result, 600), "error": bool(tool_msg.error)}
        )
        self._log(
            f"    [{tag}] turn {turn + 1}: {tc.name}{' [ERROR]' if tool_msg.error else ''}"
        )
        return "tool", result

    def _run_loop(
        self,
        messages: list[dict[str, Any]],
        tag: str,
        terminal: dict[str, Any],
        steer,
        max_turns: int,
        grace: int = 0,
        exclude: set[str] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        """Generic function-calling loop. `terminal` = {name, schema, handler}: the
        emit tool that ends the loop (handler(args) -> (obj, reason); obj is not None
        = accepted, returned; else the rejection is fed back). `steer` is a per-turn
        nudge callable (turn, remaining, is_last) -> str | None. `grace` > 0: if the
        forced final-turn terminal is rejected, extend the budget once by this many
        turns (full toolset restored) so the model can converge and resubmit instead
        of wasting the session. Whatever the channel offers is also the allowlist, so a
        call to a tool that is not on it is refused rather than dispatched. `exclude`
        drops tool names from the channel entirely — the coverage survey holds out what
        generation is not allowed to build tasks around."""
        env = self._build_env()
        excluded = {x.lower() for x in (exclude or set())}

        def _offer(e) -> list[dict[str, Any]]:
            return [s for s in self._loop_tools(e, terminal)
                    if s["function"]["name"].lower() not in excluded]

        tools = _offer(env)
        allowed = {t["function"]["name"] for t in tools}
        session_log: list[dict] = []
        rejections = 0

        # Budget can be extended once (see `grace`): a rejected forced final-turn submit
        # reopens the loop instead of ending it as a wasted session.
        budget = max_turns
        grace_used = False
        turn = 0
        while turn < budget:
            # Turn-budget steering: the model otherwise explores until the ceiling
            # and never submits. Push toward the terminal tool, count the turns down,
            # and on the final turn force the terminal call so it is not wasted.
            remaining = budget - turn
            is_last = remaining == 1
            tool_choice: str | dict[str, Any] | None = None
            call_tools = tools
            nudge = steer(turn, remaining, is_last)
            if nudge and messages[-1].get("role") != "user":
                messages.append({"role": "user", "content": nudge})
            if is_last:
                # Force the terminal tool on the final turn by offering ONLY it and requiring
                # a call. `tool_choice="required"` is model-agnostic; forcing a *specific*
                # function via {"function": {"name": …}} is rejected by gpt-5 reasoning models
                # through litellm (neither the nested nor flat shape validates), and this is
                # exactly the intent anyway — commit the spec now.
                call_tools = [t for t in tools if t["function"]["name"] == terminal["name"]]
                tool_choice = "required"

            resp = self.llm.generate(
                messages,
                tools=call_tools,
                tool_choice=tool_choice,
                reasoning_effort=self.gen.explorer_reasoning_effort,
            )

            # `resp.reasoning` is whatever the provider exposed as `reasoning_content`.
            # OpenAI's chat-completions API returns none for gpt-5.*, so the recorded
            # `thought` is empty for those models and the assistant's visible text in
            # `content` is the transcript. (This loop used to pass reasoning_summary="auto",
            # which LLMClient.generate discards as a legacy no-op — it never did anything.)
            if not resp.tool_calls:
                self._log(f"    [{tag}] turn {turn + 1}: text — nudging to act")
                session_log.append(
                    {
                        "turn": turn + 1,
                        "kind": "text",
                        "thought": resp.reasoning,
                        "content": resp.content,
                    }
                )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Use tool calls to do your work, then call "
                            f"{terminal['name']}. Do not answer in plain text."
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

            if resp.reasoning or resp.content:
                session_log.append(
                    {
                        "turn": turn + 1,
                        "kind": "assistant",
                        "thought": resp.reasoning,
                        "content": resp.content,
                    }
                )

            for tc in resp.tool_calls:
                kind, value = self._dispatch_tool_call(
                    tc, env, terminal, tag, turn, session_log, allowed
                )
                if kind == "accept":
                    return value, session_log
                if kind == "reject":
                    rejections += 1
                    if rejections >= self.gen.spec_max_rejections:
                        self._log(f"    [{tag}] too many rejections — giving up")
                        return None, session_log
                    if is_last and grace and not grace_used:
                        # Forced final-turn terminal rejected: reopen the budget once so the
                        # model can converge (full toolset returns next turn, is_last False now)
                        # and resubmit rather than wasting the session.
                        grace_used = True
                        budget += grace
                        self._log(
                            f"    [{tag}] last-turn {terminal['name']} rejected — "
                            f"granting {grace} more turns to converge"
                        )
                        value = (
                            value + f" You now have {grace} more turns and the full toolset back: "
                            "reset_environment, execute your reference solution from the clean "
                            f"state, then call {terminal['name']} with what you actually ran."
                        )
                    result = f"REJECTED: {value}"
                elif kind == "reset":
                    env = self._build_env()  # caller owns the loop-local env + schemas
                    tools = _offer(env)
                    allowed = {t["function"]["name"] for t in tools}
                    result = "Environment reset to its pristine initial state."
                else:  # "tool"
                    result = value

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            turn += 1

        self._log(
            f"    [{tag}] turn ceiling ({budget}) reached — no {terminal['name']}"
        )
        return None, session_log

    # ── coverage survey (one-time: the run's coverage goal + tag vocabulary) ──
    def _held_out(self) -> set[str]:
        """Tool names the coverage survey must not see or call.

        A tool generation may not build tasks around must not shape the coverage goal or
        earn a tag either, or the tally would chase a target no task can satisfy."""
        return {x.lower() for x in self.gen.excluded_apps}

    def _survey_system_prompt(self, env) -> str:
        # The survey never takes an env-knowledge block — it exists to discover the
        # priors, so it is always general + protocol.
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "coverage_survey",
            prompt_dirs(self.domain),
            add_env_knowledge=False,
            domain=self.domain,
            policy=env.get_policy(),
            tool_list=self._tool_lines(env, exclude=self._held_out()),
        )

    def _coverage_terminal(self) -> dict[str, Any]:
        def handler(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
            goal = coverage.CoverageGoal(
                report=str(args.get("report") or "").strip(),
                tags=coverage.parse_tags(args.get("tags")),
            )
            if not goal.report:
                return None, "report is empty — write the four report sections"
            if not goal.tagged:
                return None, (
                    "tags is empty or malformed — give a non-empty list of {tag: '#Label', "
                    "definition: <one line>, target_share: <number in (0,1]>}"
                )
            return {"goal": goal}, ""

        return {
            "name": "submit_coverage_goal",
            "schema": coverage.SUBMIT_COVERAGE_GOAL_TOOL,
            "handler": handler,
        }

    def _survey_steer(self, turn: int, remaining: int, is_last: bool) -> str | None:
        if is_last:
            return (
                "This is your FINAL turn. Call submit_coverage_goal now with your report "
                "and tag manifest."
            )
        half = self.gen.coverage_tags_max_turns // 2
        if turn == half:
            return (
                "About halfway through the survey. Make sure you have read real data AND "
                "exercised the interesting write operations across several different areas, then "
                f"call submit_coverage_goal. {remaining} turns left."
            )
        if turn > half:
            return f"{remaining} turns left. Wrap up the survey and call submit_coverage_goal."
        return None

    def survey_coverage(self, ctx: str) -> tuple[coverage.CoverageGoal, list[dict]]:
        """One-time survey of a throwaway env → the run's coverage goal + tag vocabulary."""
        messages = [
            {"role": "system", "content": self._survey_system_prompt(self._build_env())},
            {
                "role": "user",
                "content": (
                    "Begin surveying: use read_db to inspect real data across several areas, "
                    "then actually CALL the interesting operations — including the ones that "
                    "change state — to see what they demand and what they do. Call "
                    "submit_coverage_goal with your report and tag manifest once you have "
                    "surveyed broadly."
                ),
            },
        ]
        self._log(
            f"    [survey] domain {ctx} | coverage survey | "
            f"≤{self.gen.coverage_tags_max_turns} turns"
        )
        result, log = self._run_loop(
            messages,
            tag="survey",
            terminal=self._coverage_terminal(),
            steer=self._survey_steer,
            max_turns=self.gen.coverage_tags_max_turns,
            exclude=self._held_out(),
        )
        return (result or {}).get("goal", coverage.CoverageGoal()), log

    def explore(
        self,
        ctx: str,
        prior_tasks: list[str],
        guidelines: list[str],
        coverage_goal: str = "",
        coverage_tally: str = "",
        **_ignored: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        env = self._build_env()
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    env, prior_tasks, guidelines, coverage_goal, coverage_tally,
                ),
            },
            {
                "role": "user",
                "content": (
                    "Begin by exploring the environment (read_db and read-only tools) to find "
                    "real entities to build a task around. Then design the task, call "
                    "reset_environment(), execute your exact minimal reference solution from "
                    "the clean state, and finally call submit_task_spec."
                ),
            },
        ]
        self._log(
            f"    [explorer] domain {ctx} | {len(prior_tasks)} prior tasks | "
            f"{len(guidelines)} guidelines | ≤{self.gen.explorer_max_turns} turns"
        )
        return self._run_loop(
            messages,
            tag="explorer",
            terminal=self._spec_terminal(),
            steer=self._design_steer,
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
        variant_history: list[dict],
        coverage_goal: str = "",
        coverage_tally: str = "",
        **_ignored: Any,
    ) -> tuple[dict[str, Any] | None, list[dict]]:
        env = self._build_env()
        from daedalus.core.resources import render_prompt

        direction = {"too_easy": "harder", "too_hard": "easier"}.get(difficulty, "different")
        refine_user = render_prompt(
            "explorer_refine",
            prompt_dirs(self.domain),
            add_env_knowledge=True,  # env facts always on; task style lives in task_guidelines
            difficulty=difficulty,
            direction=direction,
            current_task=current_task,
            variant_history=variant_history,
            exploration=_format_transcript(exploration),
            coverage_tally=coverage_tally,
        )
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    env, prior_tasks, guidelines, coverage_goal, coverage_tally,
                ),
            },
            {"role": "user", "content": refine_user},
        ]
        self._log(f"    [refine] {difficulty} → adjust in {ctx} ...")
        return self._run_loop(
            messages,
            tag="refine",
            terminal=self._spec_terminal(),
            steer=self._design_steer,
            max_turns=self.gen.explorer_max_turns,
            grace=self.gen.explorer_last_turn_grace,
        )


def _format_transcript(session_log: list[dict]) -> str:
    """Render a prior session's tool activity for the refinement prompt."""
    parts = []
    for e in session_log:
        kind = e.get("kind")
        if kind == "tool":
            parts.append(
                f"{e['name']}({json.dumps(e.get('arguments', {}))}) -> {e.get('output', '')}"
            )
        elif kind == "read_db":
            parts.append(f"read_db({e.get('path', '')!r}) -> {e.get('output', '')}")
        elif kind == "reset":
            parts.append("reset_environment()")
    return "\n".join(parts) if parts else "(no tool activity)"
