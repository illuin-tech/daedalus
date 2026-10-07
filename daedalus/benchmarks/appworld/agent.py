"""ReAct-style coding agent for AppWorld tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


from appworld import AppWorld
from appworld.task import load_task_ids

from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.agents.task_pool import solve_tasks_parallel
from daedalus.core.logging import console
from daedalus.core.config import ExperimentConfig, resolve_trace_dir
from daedalus.core.logging.trace_logger import (
    EvaluationResult,
    RetrievedMemoryRecord,
    TraceLogger,
)

# Regex to extract code from ```python ... ``` blocks
CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
# Regex to extract Thought: ... before code block
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n```python)", re.DOTALL)

# Shown when a turn produces no code, restating the exact required format.
_NO_CODE_NUDGE = (
    "Please provide a code block to execute. Always use this exact format:\n\n"
    "Thought: <your reasoning>\n\n```python\n<your code>\n```"
)


def _score_with_world_evaluate(world: AppWorld) -> EvaluationResult:
    """Score a solved world with AppWorld's programmatic evaluator (the dev/test path).

    A harness/infra failure inside `world.evaluate()` is recorded as a non-success with
    an "Evaluation error" detail rather than crashing the run — the message keeps it
    distinguishable from a task that legitimately failed its checks.
    """
    try:
        tracker = world.evaluate()
        try:
            details = tracker.report(print_it=False, colorize=False) or ""
        except Exception:
            details = f"pass_count={tracker.pass_count} fail_count={tracker.fail_count}"
        return EvaluationResult(
            success=tracker.success,
            tests_passed=tracker.pass_count,
            tests_total=tracker.num_tests,
            details=details,
        )
    except Exception as e:
        return EvaluationResult(success=False, details=f"Evaluation error: {e}")


def _appworld_safe_name(experiment_name: str) -> str:
    """Strip the 'memory' substring from experiment_name before handing it to
    appworld. appworld's apply_db_changes() rejects any path containing the
    substring 'memory' (db.py:346) and the experiment_name is used to build
    on-disk paths under experiments/outputs/<name>/tasks/<id>/dbs/. We keep
    the original name for trace logging."""
    return experiment_name.replace("memory", "mem")


def parse_thought_and_code(text: str) -> tuple[str, str]:
    """Parse the agent's output into thought and code components."""
    thought = ""
    code = ""

    thought_match = THOUGHT_RE.search(text)
    if thought_match:
        thought = thought_match.group(1).strip()

    code_match = CODE_BLOCK_RE.search(text)
    if code_match:
        code = code_match.group(1).strip()

    # Fallback: if no thought marker but there's text before code
    if not thought and code_match:
        pre_code = text[: code_match.start()].strip()
        if pre_code:
            thought = pre_code

    return thought, code


class ReActAgent:
    """A ReAct agent that solves AppWorld tasks via code generation and execution."""

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
        # A memory method may put a pre-formatted block here (sections, ids and all) to be
        # rendered verbatim into the prompt's `playbook` slot, instead of going through the
        # `heuristics` list, which renumbers its entries. Empty for every other run.
        self.solver_playbook: str = ""
        # Set by MemoryReActAgent when a retriever is built; None => no memory.
        self.retriever = None
        self._injection_tracker = None

    def _inject_memories(
        self, messages: list[dict[str, str]], memories: list
    ) -> list[dict[str, str]]:
        """Place this turn's retrieved memories and return the messages to generate on.

        Persistent retrievers (everything except the whole-pool-every-turn
        all_every_turn) append the newly-surfaced heuristics into the REAL
        `messages`, so they stay in the conversation for the whole task. Ephemeral
        retrievers inject the full pool into a per-call COPY only (not kept), to
        avoid re-injecting the whole bank every turn (context rot).
        """
        if not memories:
            return messages
        tracker = self._injection_tracker
        to_inject, persist = tracker.select(memories) if tracker else (memories, True)
        if not to_inject:
            return messages
        msg = {
            "role": "user",
            "content": self._format_memory_injection(to_inject).strip(),
        }
        if persist:
            messages.append(msg)
            return messages
        return messages + [msg]

    def _build_initial_messages(
        self,
        world: AppWorld,
        heuristics: list[str] | None = None,
        instruction_override: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the initial prompt messages from the task.

        `instruction_override` lets a generated task be solved inside a borrowed
        world (the world's own `task.instruction` is ignored).
        """
        app_descriptions = json.dumps(
            [
                {"name": k, "description": v}
                for k, v in world.task.app_descriptions.items()
            ],
            indent=1,
        )
        instruction = instruction_override or world.task.instruction
        from daedalus.core.resources import render_prompt

        system = render_prompt(
            "solver",
            Path(__file__).parent / "prompts",
            app_descriptions=app_descriptions,
            main_user=world.task.supervisor,
            heuristics=heuristics or [],
            playbook=self.solver_playbook,
        )
        return [
            {"role": "system", "content": system.strip()},
            {
                "role": "user",
                "content": (
                    f"Task: {instruction}\n\n"
                    "Begin by thinking about your approach, then write your first code block."
                ),
            },
        ]

    def _format_observation(self, output: str) -> str:
        """Format execution output as a user message."""
        if not output.endswith("\n"):
            output += "\n"
        return f"Output:\n```\n{output}```"

    def _get_retrieved_memories(
        self, query: str, messages: list[dict[str, str]], turn_idx: int
    ) -> list[RetrievedMemoryRecord]:
        """Override point for memory-augmented agent. Base returns empty."""
        return []

    def _build_retrieval_query(
        self, messages: list[dict[str, str]], turn_idx: int
    ) -> str:
        """Build a query for memory retrieval from accumulated context.

        Uses the last assistant thought (if any) or the task instruction.
        Override in subclass for different strategies.
        """
        # Look for the last assistant message's thought
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                thought, _ = parse_thought_and_code(msg["content"])
                if thought:
                    return thought
        # Fallback: use the task instruction (first user message)
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"][:500]
        return ""

    def _format_memory_injection(self, memories: list[RetrievedMemoryRecord]) -> str:
        """Format retrieved memories for injection into prompt."""
        return self._format_heuristics_block([m.text for m in memories])

    def _format_heuristics_block(self, texts: list[str]) -> str:
        """The per-turn injection wording, over plain strings."""
        memories = texts
        if not memories:
            return ""
        lines = [
            "",
            "The following memories from previous tasks are available. "
            "They may or may not be relevant to your current situation. "
            "Use them if they help you avoid repeating past mistakes or "
            "inform a better approach — otherwise, ignore them.",
            "",
        ]
        for i, text in enumerate(memories, 1):
            lines.append(f"  {i}. {text}")
        lines.append("")
        return "\n".join(lines)

    def _reset_per_task_state(
        self, task_instruction: str = "", task_id: str = ""
    ) -> None:
        """Reset per-task retrieval state at the start of each task.

        No-op for the base agent; memory agents use it to reset the retriever and the
        injection tracker's dedup set so each task starts clean.
        """
        if self.retriever is not None:
            self.retriever.on_task_start(task_instruction, task_id=task_id)
        if self._injection_tracker is not None:
            self._injection_tracker.reset()

    def _generate_turn(
        self, messages: list[dict[str, str]], logger: TraceLogger, turn_idx: int
    ) -> tuple[str, str]:
        """Produce one turn: retrieve + inject memory, generate once, log, and return
        (assistant_message, code). `code` is "" when the model emitted none.

        This is the single override point for alternative turn strategies (see
        MemoryReActAgent's reason-then-retrieve); the surrounding loop in solve_task —
        execution, observation, completion, scoring — is shared by every variant.
        """
        query = self._build_retrieval_query(messages, turn_idx)
        memories = self._get_retrieved_memories(query, messages, turn_idx)
        logger.log_query(query[:200])
        logger.log_retrieved_memories(memories)
        gen_messages = self._inject_memories(messages, memories)
        response = self.llm.generate(gen_messages)
        thought, code = parse_thought_and_code(response.content)
        logger.log_thought(thought)
        logger.log_llm_call(response)
        return response.content, code

    def solve_task(
        self,
        task_id: str,
        heuristics: list[str] | None = None,
        trace_suffix: str = "",
        instruction_override: str | None = None,
        evaluator: "Callable[[dict[str, Any]], tuple[bool, str]] | None" = None,
        task_override: Any | None = None,
    ) -> dict[str, Any]:
        """Run the agent on a single task. Returns the trace dict.

        For generated tasks, pass `instruction_override` to run a custom instruction inside
        `task_id`'s seeded world, and `evaluator` to grade the trace against success
        conditions (the LLM judge) instead of `world.evaluate()`. `task_override` is for
        benchmarks with structured tasks; AppWorld rejects it.
        """
        if task_override is not None:
            raise ValueError(
                "AppWorld agents take instruction_override, not task_override"
            )
        config_dict = self.cfg.to_dict()
        # In accumulation mode (heuristics provided, or a generated task with its own
        # evaluator), don't skip on existing traces — every attempt re-runs.
        accumulation_mode = heuristics is not None or evaluator is not None
        save_accum_traces = accumulation_mode and self.cfg.accumulation.save_traces
        logger = TraceLogger(
            task_id=task_id,
            experiment_name=self.cfg.name,
            config=config_dict,
            trace_dir=self.trace_dir,
            model=self.cfg.agent.model,
            save_every_turn=self.cfg.logging.save_every_turn
            if not accumulation_mode
            else save_accum_traces,
            trace_suffix=trace_suffix,
        )

        # Skip if already completed (only in normal mode, unless force=True)
        if not accumulation_mode and not self.cfg.run.force and logger.trace_exists():
            console.detail(
                f"Skipping already completed task: {task_id}",
                verbose=self.cfg.logging.verbose,
            )
            return {}

        appworld_name = _appworld_safe_name(self.cfg.name)
        if self.run_idx is not None:
            appworld_name = f"{appworld_name}_run{self.run_idx}"
        with AppWorld(
            task_id=task_id,
            experiment_name=appworld_name,
        ) as world:
            messages = self._build_initial_messages(
                world, heuristics=heuristics, instruction_override=instruction_override
            )
            logger.log_task_instruction(instruction_override or world.task.instruction)
            self.llm.reset_usage()
            # Every solver call from here on is billed to THIS task.
            self.llm.scope = self.accounting.scope("solver", task_id=task_id)
            self._reset_per_task_state(
                instruction_override or world.task.instruction, task_id=task_id
            )

            for turn_idx in range(self.cfg.agent.max_turns):
                logger.start_turn(turn_idx)
                assistant_content, code = self._generate_turn(
                    messages, logger, turn_idx
                )
                messages.append({"role": "assistant", "content": assistant_content})

                if not code:
                    # Thinking-only turn: nudge back toward the required format.
                    messages.append({"role": "user", "content": _NO_CODE_NUDGE})
                    logger.log_code("")
                    logger.log_execution("", True)
                    logger.end_turn()
                    continue

                logger.log_code(code)
                try:
                    output = world.execute(code)
                    logger.log_execution(output, True)
                except Exception as e:
                    output = str(e)
                    logger.log_execution(output, False)
                logger.end_turn()
                messages.append(
                    {"role": "user", "content": self._format_observation(output)}
                )

                if world.task_completed():
                    break

            # Score: generated tasks use the judge over success conditions; real tasks
            # use AppWorld's own evaluator.
            if evaluator is not None:
                success, details = evaluator(logger.trace.to_dict())
                eval_result = EvaluationResult(success=success, details=details)
            else:
                eval_result = _score_with_world_evaluate(world)
            logger.log_evaluation(eval_result)
            # Witness that memory really was injected. None on the plain ReActAgent (which
            # has none); MemoryReActAgent sets `memory_record`. Without it an at-start
            # run's trace is indistinguishable from a baseline's, because nothing reaches
            # turns[*].retrieved_memories in that mode.
            logger.trace.extra["memory"] = getattr(self, "memory_record", None)
            if not accumulation_mode or save_accum_traces:
                logger.save()

        return logger.trace.to_dict()

    def solve_tasks(self, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Run the agent on multiple tasks, optionally in parallel."""
        if task_ids is None:
            task_ids = load_task_ids(self.cfg.appworld.dataset)
            if self.cfg.appworld.task_ids:
                task_ids = [t for t in task_ids if t in self.cfg.appworld.task_ids]

        # Apply max_tasks limit
        if self.cfg.run.max_tasks is not None:
            task_ids = task_ids[: self.cfg.run.max_tasks]

        parallel = self.cfg.run.parallel

        if parallel <= 1:
            # Sequential execution
            results = []
            for task_id in console.track(task_ids, desc="solving"):
                results.append(self.solve_task(task_id))
            return results
        else:
            # Parallel execution
            return self._solve_tasks_parallel(task_ids, parallel)

    def _solve_tasks_parallel(
        self, task_ids: list[str], num_workers: int
    ) -> list[dict[str, Any]]:
        """Run tasks in parallel using ProcessPoolExecutor."""
        return solve_tasks_parallel(self, task_ids, num_workers)
