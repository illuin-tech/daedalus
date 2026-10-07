"""AutomationBench solver TaskAgent.

A native tool-calling ReAct loop routed through the harness ``LLMClient`` (so
cost/retries/reasoning-effort and the memory-injection grid are shared with the rest of
the project). Tools come from the in-process AutomationBench world; grading is the
benchmark's own assertion sweep over the final world.

Note this is a *different scaffold* from the benchmark's shipped runner, which is a
`verifiers` StatefulToolEnv with its own search-result compression: same tasks, same
tool surface, same rubric, our loop. Scores are therefore comparable in kind to the
published table but are not a reproduction of it (see this benchmark's README).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from daedalus.core.agents.memory import SolverMemory
from daedalus.core.agents.task_pool import solve_tasks_parallel
from daedalus.core.config import ExperimentConfig, resolve_trace_dir
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.trace_logger import EvaluationResult, TraceLogger, Turn
from daedalus.core.resources import render_prompt

from .env import AutomationBenchSession, Grade, render_observation
from .task_loader import Task, load_tasks


class AutomationBenchTaskAgent:
    """TaskAgent for AutomationBench (see daedalus/core/benchmark.py for the contract)."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        self.cfg = cfg
        self.run_idx = run_idx
        self.accounting = CostAccounting.for_worker(cfg, run_idx=run_idx)
        self.llm = self.accounting.client(
            cfg.agent.model,
            "solver",
            temperature=cfg.agent.temperature,
            reasoning_effort=cfg.agent.reasoning_effort,
        )
        self.trace_dir = resolve_trace_dir(cfg, run_idx=run_idx)
        self._tasks_by_id = load_tasks(cfg)
        # A memory method may put a pre-formatted block here (sections, ids and all) to be
        # rendered verbatim into the prompt's `playbook` slot, instead of going through the
        # `heuristics` list, which renumbers its entries. Empty for every other run.
        self.solver_playbook: str = ""
        # Pool + retriever + injection timing (see core/agents/memory.py).
        self.memory = SolverMemory(cfg, "automationbench", self.accounting)

    def _build_system_prompt(self, task: Task, heuristics: list[str]) -> str:
        """general behaviour + [environment knowledge] + the tool-call protocol.

        The protocol carries the benchmark's OWN system message verbatim: it is identical
        across all 800 tasks and states scoring-relevant convention (the ~50-turn budget,
        and not enumerating skipped items — which the negative assertions punish).

        The tool list itself is NOT in the prompt: the native tool schemas carry the
        signatures and docstrings, so rendering them again would only duplicate them.
        """
        return render_prompt(
            "solver",
            Path(__file__).parent / "prompts",
            add_env_knowledge=True,
            benchmark_system_prompt=task.system_prompt,
            heuristics=heuristics,
            playbook=self.solver_playbook,
        )

    # ----- solving -----

    def solve_task(
        self,
        task_id: str,
        heuristics: list[str] | None = None,
        trace_suffix: str = "",
        instruction_override: str | None = None,
        evaluator: Any | None = None,
        task_override: Any | None = None,
    ) -> dict[str, Any]:
        """Solve one task and return its normalized trace dict.

        `heuristics` is the accumulation loop's mined note (injected instead of
        retrieving). The other three are the self-play generation hooks:
        `task_override` is a generated Task (its own instruction, run in the world of the
        context it was designed in), `instruction_override` replaces the instruction of
        whichever task is in hand, and `evaluator(trace) -> (success, details)` grades the
        attempt with the LLM judge in place of the benchmark's assertion sweep — a
        generated task has no assertions.
        """
        task: Task = task_override if task_override is not None else self._tasks_by_id[task_id]
        if instruction_override is not None:
            task = replace(task, instruction=instruction_override)
        accumulation_mode = heuristics is not None
        save_accum = accumulation_mode and self.cfg.accumulation.save_traces

        logger = TraceLogger(
            task.task_id,
            experiment_name=self.cfg.name,
            config=self.cfg.to_dict(),
            trace_dir=self.trace_dir,
            model=self.cfg.agent.model,
            save_every_turn=False,
            trace_suffix=trace_suffix,
        )
        if not accumulation_mode and not self.cfg.run.force and logger.trace_exists():
            # Say so: a generated task's id is a hash of its text, so a resumed
            # generation run CAN collide with an earlier attempt's trace, and a silent
            # {} here reads downstream as a failed attempt.
            console.detail(f"Skipping already completed task: {task.task_id}")
            return {}

        self.llm.reset_usage()
        self.llm.scope = self.accounting.scope("solver", task_id=task.task_id)
        try:
            self._solve(task, heuristics, logger, evaluator)
        except Exception as e:
            # Infra/API failure, or a raising assertion (strict mode is the benchmark's
            # default): record a failed trace so the task still counts in the
            # denominator. `extra.error` is the accumulation loop's abort signal, so it
            # is reserved for exactly this — never for an agent that merely gave up.
            logger.trace.extra = {
                "benchmark": "automationbench",
                "domain": task.domain,
                "error": str(e),
            }
            logger.log_evaluation(
                EvaluationResult(success=False, details=f"error: {e}", reward=0.0)
            )

        if (not accumulation_mode) or save_accum:
            logger.save()
        return logger.trace.to_dict()

    def _solve(
        self,
        task: Task,
        heuristics: list[str] | None,
        logger: TraceLogger,
        evaluator: Any | None = None,
    ) -> None:
        logger.log_task_instruction(task.instruction)
        heuristic_texts = self.memory.start(
            task.instruction, heuristics, task_id=task.task_id
        )

        turns: list[Turn] = []
        tools_used: list[str] = []
        totals = {"prompt": 0, "completion": 0, "cached": 0}
        total_cost = 0.0

        session = AutomationBenchSession(task, self.cfg.automationbench)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(task, heuristic_texts)},
            {"role": "user", "content": task.instruction},
        ]

        finished = False
        for i in range(self.cfg.automationbench.max_steps):
            # Persistent injections are appended to `messages` by SolverMemory; an
            # ephemeral one is handed back to add to this call only.
            tm = self.memory.turn(i, messages)

            resp = self.llm.generate(messages + tm.ephemeral, tools=session.openai_tools)
            totals["prompt"] += resp.prompt_tokens
            totals["completion"] += resp.completion_tokens
            totals["cached"] += resp.cached_tokens
            # The per-call figure from the call's own usage event — priced with its
            # effective service tier and cache breakdown, not re-derived here.
            if resp.estimated_cost_usd is None:
                logger.trace.cost_complete = False
            else:
                total_cost += resp.estimated_cost_usd
            if resp.usage_event_id:
                logger.trace.usage_event_ids.append(resp.usage_event_id)

            assistant: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
            if resp.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant)

            turn = Turn(
                turn_idx=i,
                thought=resp.content or "",
                query=tm.query[:200],
                retrieved_memories=tm.retrieved,
                token_usage={
                    "prompt": resp.prompt_tokens,
                    "completion": resp.completion_tokens,
                },
            )

            if not resp.tool_calls:
                turns.append(turn)
                finished = True
                break

            outputs: list[str] = []
            rendered: list[str] = []
            ok = True
            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc.arguments or "{}")
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                res = session.call(tc.name, args)
                observation = render_observation(res)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": observation}
                )
                rendered.append(
                    f"{tc.name}({json.dumps(args, ensure_ascii=False, default=str)})"
                )
                outputs.append(observation)
                ok = ok and res.success
                if tc.name not in tools_used:
                    tools_used.append(tc.name)

            turn.code = "\n".join(rendered)
            turn.execution_output = "\n".join(outputs)
            turn.execution_success = ok
            turns.append(turn)

        # The benchmark scores the final world whatever ended the loop — hitting the
        # step cap is not itself a failure here, so grade either way.
        # A generated task carries no assertions: the judge grades it from the trace, so
        # the trace has to be filled in before the evaluator sees it.
        grade = None if evaluator is not None else session.grade()
        self._grade(
            logger, task, turns, tools_used, totals, total_cost, grade, finished, evaluator
        )

    def _grade(
        self,
        logger: TraceLogger,
        task: Task,
        turns: list[Turn],
        tools_used: list[str],
        totals: dict[str, int],
        total_cost: float,
        grade: Grade | None,
        finished: bool,
        evaluator: Any | None = None,
    ) -> None:
        """Fill the trace and log the verdict on it.

        For a benchmark task, `evaluation.success` is `task_completed_correctly` — the
        official strict pass — so every shared consumer (metrics, pass^k, the run
        browser) keeps its usual meaning, and `partial_credit` rides in `extra` for
        `evaluate_run` to average. For a GENERATED task there are no assertions: the
        trace is filled first and the judge (`evaluator`) returns the verdict.
        """
        logger.trace.turns = turns
        logger.trace.num_turns = len(turns)
        logger.trace.total_tokens = totals
        logger.trace.total_cost_usd = total_cost
        logger.trace.extra = {
            "benchmark": "automationbench",
            "domain": task.domain,
            "example_id": task.example_id,
            "task_contract_sha256": task.contract_sha256,
            "toolset": "api",
            "tools_used": tools_used,
            "hit_step_cap": not finished,
        }
        if grade is None:  # generated task: the judge decides, from the filled trace
            success, details = evaluator(logger.trace.to_dict())
            logger.trace.extra["generated"] = True
            logger.trace.extra["judge_details"] = details
            logger.log_evaluation(
                EvaluationResult(
                    success=bool(success),
                    details=details,
                    reward=1.0 if success else 0.0,
                )
            )
            return

        logger.trace.extra["partial_credit"] = grade.partial_credit
        logger.trace.extra["assertions"] = grade.assertions
        scored = [a for a in grade.assertions if not a["excluded"]]
        logger.log_evaluation(
            EvaluationResult(
                success=grade.success,
                tests_passed=sum(1 for a in scored if a["passed"]),
                tests_total=len(scored),
                details=json.dumps(
                    {
                        "task_completed_correctly": grade.success,
                        "partial_credit": grade.partial_credit,
                        "hit_step_cap": not finished,
                        "assertions": grade.assertions,
                    },
                    default=str,
                ),
                reward=grade.partial_credit,
            )
        )

    def solve_tasks(self, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if task_ids is None:
            task_ids = list(self._tasks_by_id.keys())
        if self.cfg.run.max_tasks:
            task_ids = task_ids[: self.cfg.run.max_tasks]
        if self.cfg.run.parallel <= 1:
            return [
                self.solve_task(tid) for tid in console.track(task_ids, desc="solving")
            ]
        return solve_tasks_parallel(self, task_ids, self.cfg.run.parallel)
