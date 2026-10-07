"""tau2 task runner (TaskAgent) and Benchmark implementation.

Two solver modes share one agent:
  conversational — tau2's official setup: LLM user simulator plays the scenario.
  ticket         — scripted TicketUser delivers the task upfront, then stops
                   (deterministic, no user-sim LLM; used for learning phases).

Rewards come from tau2's own evaluator at run time (attached by run_simulation),
so evaluate_run only aggregates traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from daedalus.core.logging.accounting import CostAccounting
from daedalus.benchmarks.tau2 import prompt_dirs as tau2_prompt_dirs
from daedalus.core.agents.task_pool import solve_tasks_parallel
from daedalus.core.benchmark import Benchmark
from daedalus.core.logging import console
from daedalus.core.config import ExperimentConfig, resolve_trace_dir
from daedalus.core.logging.trace_logger import (
    EvaluationResult,
    TraceLogger,
    Turn,
)


def _select_tasks(cfg: ExperimentConfig) -> list:
    """The tau2 Task objects in scope: domain + split, narrowed by `task_ids`. Single
    source of truth for the runner and the benchmark."""
    from tau2.runner import get_tasks

    tasks = get_tasks(cfg.tau2.domain, task_split_name=cfg.tau2.task_split)
    if cfg.tau2.task_ids:
        want = set(cfg.tau2.task_ids)
        tasks = [t for t in tasks if t.id in want]
    return tasks


def _select_task_ids(cfg: ExperimentConfig) -> list[str]:
    """The in-scope task ids (see `_select_tasks`)."""
    return [t.id for t in _select_tasks(cfg)]


def _solver_mode(cfg: ExperimentConfig) -> str:
    """conversational for eval-time inference, ticket for learning phases."""
    if cfg.tau2.solver_mode:
        return cfg.tau2.solver_mode
    return "conversational" if cfg.kind == "inference" else "ticket"


def _render_tool_calls(msg) -> str:
    return "\n".join(
        f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
        for tc in msg.tool_calls
    )


class Tau2TaskRunner:
    """TaskAgent for tau2: runs one simulation per task via tau2's Orchestrator."""

    def __init__(self, cfg: ExperimentConfig, run_idx: int | None = None):
        from tau2.runner import build_environment

        # tau2-bench logs through loguru (colorized) — drop that sink so only
        # daedalus's own clean lines show. Runs in every process a runner lives in.
        console.silence_third_party(cfg.logging.verbose)
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
        self._build_environment = build_environment

        self._tasks_by_id = {t.id: t for t in _select_tasks(cfg)}
        self._policy = build_environment(cfg.tau2.domain).get_policy()

        self.memory_pool = None
        self.retriever = None
        # heuristics_at_start: inject the ENTIRE bank into the system prompt once at task
        # start (kept there for the whole conversation, sent as part of the cached system
        # message) and build NO per-turn retriever — so nothing is re-injected each turn.
        # A memory method may put a pre-formatted block here (sections, ids and all) to be
        # rendered verbatim into the prompt's `playbook` slot, instead of going through the
        # `heuristics` list, which renumbers its entries. Empty for every other run.
        self.solver_playbook: str = ""
        self._start_heuristics: list[str] = []
        # Witness that memory really was injected, stamped onto every trace's `extra`.
        # heuristics_at_start builds no retriever, so nothing reaches
        # turns[*].retrieved_memories and such a trace was indistinguishable from a
        # baseline one; the digest is what makes it auditable after the fact.
        self._memory_record: dict[str, Any] | None = None
        if cfg.memory.enabled:
            from daedalus.core.agents.memory import (
                build_retriever,
                injected_heuristics_record,
                load_solver_pool,
            )

            if cfg.memory.retrieval_mode != "pre_generation":
                raise ValueError(
                    f"tau2 supports retrieval_mode=pre_generation only "
                    f"(got {cfg.memory.retrieval_mode!r})"
                )
            self.memory_pool = load_solver_pool(cfg, "tau2")
            self._memory_record = injected_heuristics_record(
                self.memory_pool, cfg.memory.pool_path
            )
            if cfg.memory.heuristics_at_start:
                self._start_heuristics = list(self.memory_pool.texts())
            elif len(self.memory_pool) > 0:
                self.retriever, _ = build_retriever(cfg, self.memory_pool)

    def _retrieve(self, query: str):
        from daedalus.core.logging.trace_logger import RetrievedMemoryRecord

        if self.retriever is None or not query:
            return []
        retrieved = self.retriever.retrieve(
            query=query, reasoning_trace=query, top_k=self.cfg.memory.retriever.top_k
        )
        return [
            RetrievedMemoryRecord(
                memory_id=r.memory_id, text=r.text, score=r.score, source=r.source
            )
            for r in retrieved
        ]

    # ── solving ────────────────────────────────────────────────────────────

    def _render_system_prompt(self, mode: str, heuristics: list[str] | None) -> str:
        """The solver system prompt: general behaviour + [tau2 env knowledge] + tau2's
        action protocol."""
        from daedalus.core.resources import render_prompt

        return render_prompt(
            "solver",
            tau2_prompt_dirs(self.cfg.tau2.domain),
            add_env_knowledge=True,
            domain_policy=self._policy,
            heuristics=heuristics or [],
            playbook=self.solver_playbook,
            ticket_mode=(mode == "ticket"),
        )

    def _build_user(self, mode: str, task, env):
        """The simulated user: a scripted TicketUser (ticket mode) or tau2's LLM
        user-simulator (conversational mode)."""
        from daedalus.benchmarks.tau2.agent import TicketUser

        if mode == "ticket":
            return TicketUser(str(task.user_scenario))
        if mode == "conversational":
            from tau2.runner import build_user

            return build_user(
                "user_simulator",
                env,
                task,
                llm=self.cfg.tau2.user_llm,
                llm_args=dict(self.cfg.tau2.user_llm_args),
            )
        raise ValueError(f"Unknown tau2 solver_mode: {mode!r}")

    def solve_task(
        self,
        task_id: str,
        heuristics: list[str] | None = None,
        trace_suffix: str = "",
        instruction_override: str | None = None,
        evaluator: Any | None = None,
        task_override: Any | None = None,
    ) -> dict[str, Any]:
        """Run one tau2 simulation. Returns the normalized trace dict.

        `task_override` (a tau2 Task) runs a generated task; its own
        evaluation_criteria are scored natively. `evaluator` (trace callback)
        replaces the native reward if given. `instruction_override` is not
        meaningful for tau2 (tasks are structured) — pass task_override.
        """
        from tau2.data_model.tasks import Task
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner import run_simulation

        if instruction_override is not None:
            raise ValueError(
                "tau2 tasks are structured; pass task_override, not instruction_override"
            )

        if task_override is not None:
            task: Task = task_override
            task_id = task.id
        else:
            task = self._tasks_by_id[task_id]

        config_dict = self.cfg.to_dict()
        accumulation_mode = (
            heuristics is not None or evaluator is not None or task_override is not None
        )
        save_accum_traces = accumulation_mode and self.cfg.accumulation.save_traces
        logger = TraceLogger(
            task_id=task_id,
            experiment_name=self.cfg.name,
            config=config_dict,
            trace_dir=self.trace_dir,
            model=self.cfg.agent.model,
            save_every_turn=False,  # turns are built post-hoc from the simulation
            trace_suffix=trace_suffix,
        )
        if not accumulation_mode and not self.cfg.run.force and logger.trace_exists():
            console.detail(
                f"Skipping already completed task: {task_id}",
                verbose=self.cfg.logging.verbose,
            )
            return {}

        # Reset per-task retriever state (random draws, no-replacement exclusions).
        if self.retriever is not None:
            self.retriever.on_task_start(str(task.user_scenario), task_id=task_id)
        mode = _solver_mode(self.cfg)
        env = self._build_environment(self.cfg.tau2.domain)
        # heuristics_at_start puts the whole bank in the system prompt (self._start_heuristics);
        # otherwise heuristics is the accumulation-time evolving item (usually empty at inference).
        system_prompt = self._render_system_prompt(
            mode, (heuristics or []) + self._start_heuristics
        )

        from daedalus.benchmarks.tau2.agent import HarnessTau2Agent

        agent = HarnessTau2Agent(
            tools=env.get_tools(),
            domain_policy=self._policy,
            llm=self.llm,
            system_prompt=system_prompt,
            retrieve=self._retrieve if self.retriever is not None else None,
            retriever_ephemeral=(
                self.retriever.ephemeral if self.retriever is not None else False
            ),
            whole_trace_context=(
                self.retriever is not None and self.retriever.whole_trace_query
            ),
        )
        user = self._build_user(mode, task, env)

        self.llm.reset_usage()
        self.llm.scope = self.accounting.scope("solver", task_id=task_id)
        orch = Orchestrator(
            domain=self.cfg.tau2.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.cfg.tau2.max_steps,
            max_errors=self.cfg.tau2.max_errors,
            seed=self.cfg.tau2.seed,
            validate_communication=False,
        )
        sim = run_simulation(
            orch, evaluation_type=EvaluationType("all")
        )

        self._fill_trace(logger, task, sim, agent)

        if evaluator is not None:
            success, details = evaluator(logger.trace.to_dict())
            reward = 1.0 if success else 0.0
            logger.log_evaluation(
                EvaluationResult(success=success, details=details, reward=reward)
            )
        else:
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            details = {
                "termination_reason": str(sim.termination_reason),
                "reward_breakdown": {
                    str(k): v
                    for k, v in (sim.reward_info.reward_breakdown or {}).items()
                }
                if sim.reward_info and sim.reward_info.reward_breakdown
                else {},
            }
            logger.log_evaluation(
                EvaluationResult(
                    success=reward >= 0.999,
                    details=json.dumps(details),
                    reward=reward,
                )
            )

        if not accumulation_mode or save_accum_traces:
            logger.save()
            # tau2's own SimulationRun, next to the trace.
            sims_dir = self.trace_dir / "tau2_sims"
            sims_dir.mkdir(parents=True, exist_ok=True)
            stem = task_id if not trace_suffix else f"{task_id}_{trace_suffix}"
            (sims_dir / f"{stem}.json").write_text(
                sim.model_dump_json(indent=2), encoding="utf-8"
            )

        return logger.trace.to_dict()

    def _fill_trace(self, logger: TraceLogger, task, sim, agent) -> None:
        """Map the tau2 SimulationRun onto the normalized trace format."""
        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

        logger.log_task_instruction(str(task.user_scenario))

        messages = sim.messages or []
        call_idx = 0
        turn_idx = 0
        for i, msg in enumerate(messages):
            if not isinstance(msg, AssistantMessage):
                continue
            turn = Turn(turn_idx=turn_idx)
            record = None
            # The default greeting is injected by the orchestrator, not the LLM;
            # agent.call_records only covers real LLM calls.
            if msg.usage is not None and call_idx < len(agent.call_records):
                record = agent.call_records[call_idx]
                call_idx += 1

            if msg.is_tool_call():
                turn.thought = (record.dropped_text if record else "") or ""
                turn.code = _render_tool_calls(msg)
                outputs, ok = [], True
                for later in messages[i + 1 :]:
                    if (
                        isinstance(later, ToolMessage)
                        and later.requestor == "assistant"
                    ):
                        outputs.append(later.content or "")
                        ok = ok and not later.error
                        if len(outputs) == len(msg.tool_calls):
                            break
                    elif isinstance(later, (AssistantMessage, UserMessage)):
                        break
                turn.execution_output = "\n".join(outputs)
                turn.execution_success = ok
            else:
                turn.thought = msg.content or ""
                for later in messages[i + 1 :]:
                    if isinstance(later, UserMessage):
                        turn.execution_output = f"User: {later.content or ''}"
                        break
                    if isinstance(later, AssistantMessage):
                        break

            if record is not None:
                turn.query = record.query
                turn.retrieved_memories = record.retrieved_memories
                logger.trace.total_tokens["prompt"] += record.prompt_tokens
                logger.trace.total_tokens["completion"] += record.completion_tokens
                logger.trace.total_tokens["cached"] += record.cached_tokens
                logger.trace.usage_event_ids.extend(record.usage_event_ids)
                if not record.cost_complete:
                    logger.trace.cost_complete = False
                turn.token_usage = {
                    "prompt": record.prompt_tokens,
                    "completion": record.completion_tokens,
                    "cached": record.cached_tokens,
                }
            logger.trace.turns.append(turn)
            turn_idx += 1

        logger.trace.num_turns = len(logger.trace.turns)
        # The solver projection: the per-call estimates the agent's own usage events
        # carry, summed. (tau2 puts the same figure on each AssistantMessage.cost.)
        logger.trace.total_cost_usd = sum(
            r.estimated_cost_usd for r in agent.call_records
        )
        logger.trace.user_sim_cost_usd = float(sim.user_cost or 0.0)
        self._record_user_simulator_usage(logger, sim, messages)
        logger.trace.extra = {
            "benchmark": "tau2",
            "memory": self._memory_record,
            "domain": self.cfg.tau2.domain,
            "solver_mode": _solver_mode(self.cfg),
            "termination_reason": str(sim.termination_reason),
            "num_messages": len(messages),
            "xor_fixups": sum(1 for r in agent.call_records if r.dropped_text),
        }

    def _record_user_simulator_usage(self, logger: TraceLogger, sim, messages) -> None:
        """Emit `user_simulator` usage events for tau2's LLM user.

        The simulated user is a real, billed LLM in conversational mode, and it was
        missing from every headline total: the trace kept `user_sim_cost_usd` beside
        `total_cost_usd`, and each consumer decided for itself whether to add it (most
        did not). It is a cost role like any other now.

        tau2 reports per-UserMessage `usage` (prompt/completion tokens only — no cache
        detail) plus its own litellm-computed `cost`. The tokens are priced HERE, on this
        run's price snapshot, so the user simulator is costed on the same basis as the
        solver; tau2's own figure rides along, labelled with its source. When tau2 exposes
        no per-message usage at all, one cost-only event is written instead and the run
        reports its token detail as unavailable.
        """
        from tau2.data_model.message import UserMessage

        from daedalus.core.logging.usage import TokenUsage

        model = self.cfg.tau2.user_llm
        task_id = logger.task_id
        billed = [
            m
            for m in messages
            if isinstance(m, UserMessage) and getattr(m, "usage", None)
        ]
        if billed:
            for msg in billed:
                usage = msg.usage or {}
                logger.trace.usage_event_ids.append(
                    self.accounting.record_external(
                        "user_simulator",
                        model,
                        tokens=TokenUsage(
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                        ),
                        third_party_cost_usd=(
                            float(msg.cost) if getattr(msg, "cost", None) else None
                        ),
                        third_party_cost_source="tau2_completion_cost",
                        component="tau2_user_simulator",
                        task_id=task_id,
                    )
                )
            return
        if sim.user_cost:
            logger.trace.cost_complete = False
            logger.trace.usage_event_ids.append(
                self.accounting.record_external(
                    "user_simulator",
                    model,
                    tokens=None,
                    third_party_cost_usd=float(sim.user_cost),
                    third_party_cost_source="tau2_completion_cost",
                    component="tau2_user_simulator",
                    task_id=task_id,
                )
            )

    def solve_tasks(self, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if task_ids is None:
            # _tasks_by_id is already domain+split+task_ids+exclude filtered.
            task_ids = list(self._tasks_by_id)
        if self.cfg.run.max_tasks is not None:
            task_ids = task_ids[: self.cfg.run.max_tasks]

        if self.cfg.run.parallel <= 1:
            return [
                self.solve_task(task_id)
                for task_id in console.track(task_ids, desc="solving")
            ]
        return solve_tasks_parallel(self, task_ids, self.cfg.run.parallel)


class Tau2Benchmark(Benchmark):
    name = "tau2"

    def list_task_ids(self, cfg: ExperimentConfig) -> list[str]:
        return _select_task_ids(cfg)

    def prompt_dirs(self, cfg: ExperimentConfig) -> list[Path]:
        """The domain's prompts first, then tau2's shared ones (see tau2.prompt_dirs)."""
        return tau2_prompt_dirs(cfg.tau2.domain)

    def build_agent(
        self, cfg: ExperimentConfig, run_idx: int | None = None
    ) -> Tau2TaskRunner:
        return Tau2TaskRunner(cfg, run_idx=run_idx)

    def evaluate_run(
        self,
        cfg: ExperimentConfig,
        trace_dir: Path,
        task_ids: list[str] | None,
        run_idx: int | None,
    ) -> dict[str, Any]:
        """Aggregate rewards computed at run time from the trace files."""
        trace_files = sorted(p for p in trace_dir.glob("*.json"))
        results: dict[str, dict[str, Any]] = {}
        for path in trace_files:
            trace = json.loads(path.read_text(encoding="utf-8"))
            # Bookkeeping files (run_meta.json, evaluation.json, usage_migrated.json) sit
            # beside the traces in the flat layout; only a trace carries `success`.
            if "success" not in trace:
                continue
            tid = trace.get("task_id", path.stem)
            if task_ids and tid not in task_ids:
                continue
            evaluation = trace.get("evaluation") or {}
            results[tid] = {
                "success": bool(trace.get("success", False)),
                "reward": evaluation.get("reward"),
            }
        num_tasks = len(results)
        num_successes = sum(1 for r in results.values() if r["success"])
        rewards = [r["reward"] for r in results.values() if r["reward"] is not None]
        return {
            "num_tasks": num_tasks,
            "num_successes": num_successes,
            "success_rate": num_successes / num_tasks if num_tasks else 0.0,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "task_success": {tid: r["success"] for tid, r in sorted(results.items())},
        }

    def format_trajectory(self, trace: dict[str, Any]) -> str:
        """Render a tau2 conversation trace for memory-extraction prompts."""
        lines: list[str] = []
        for turn in trace.get("turns", []):
            if turn.get("code"):
                if turn.get("thought"):
                    lines.append(f"Agent (thinking): {turn['thought']}")
                lines.append(f"Tool call: {turn['code']}")
                status = "ok" if turn.get("execution_success", True) else "ERROR"
                lines.append(
                    f"Tool result ({status}): {turn.get('execution_output', '')}"
                )
            else:
                lines.append(f"Agent: {turn.get('thought', '')}")
                if turn.get("execution_output"):
                    lines.append(turn["execution_output"])  # already "User: ..."
        return "\n".join(lines)
