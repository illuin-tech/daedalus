"""Reflexion-style experience gathering — the offline data both AutoGuide and ExpeL learn from.

Both papers build their memory from **paired outcomes on the same training task**: ExpeL
contrasts a failed trial with a successful one (and reads patterns off batches of
successes), AutoGuide contrasts a desired with an undesired trajectory to locate the step
where they diverge. Neither can do anything with a single attempt, so both gather
experience the same way — run the task, and on failure self-reflect and retry, up to Z
retries (Reflexion, Shinn et al. 2023):

    ExpeL     Alg. 1 "Experience Gathering", `max_reflection_depth: 3` in its repo config.
    AutoGuide §B.1.3, "collect (τ+, τ−) pairs with ReAct+Reflexion".

That shared step lives here so the two methods differ only in what they *extract*, which is
the actual contribution of each paper. Gathering is method-neutral: the per-task records are
the same shape for both, and are checkpointed under `<experiment_dir>/experiences/`.

Contrast with DAEDALUS's accumulation loop, which also retries but rewrites a memory item
between attempts and reads task difficulty off the retry outcome; here the retries exist
only to produce contrasting trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from daedalus.core.config import ExperimentConfig, experiment_dir
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging import console
from daedalus.core.memory.extraction import generate_self_reflection
from daedalus.core.registry import get_benchmark

from references.common.accumulation import restore, run_tasks, worker_config, write_json

# Reflexion's own reflection prompt belongs to a third paper (Shinn et al.), and neither
# AutoGuide nor ExpeL ships one; daedalus's post-attempt reflection prompt
# (core/prompts/accumulation/self_reflection.txt, per-benchmark overrides included) plays
# exactly that role here, and is the same text DAEDALUS's accumulation uses — so the two
# methods' experience collection is not quietly better or worse than DAEDALUS's.


@dataclass
class Trial:
    """One attempt at a task."""

    index: int  # 0 = the first attempt, 1.. = retries after a reflection
    outcome: str  # "success" | "failure"
    trajectory: str  # the trace rendered by benchmark.format_trajectory
    actions: list[str] = field(default_factory=list)  # per-turn action text
    turn_texts: list[str] = field(
        default_factory=list
    )  # per-turn rendering, for prefixes
    reflection: str = ""  # what was reflected AFTER this trial (empty on the last one)
    cost_usd: float = 0.0
    tokens: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "cached": 0}
    )

    @property
    def success(self) -> bool:
        return self.outcome == "success"


@dataclass
class Experience:
    """Every trial run on one training task, with its outcome."""

    task_id: str
    task: str = ""
    trials: list[Trial] = field(default_factory=list)
    reflection_cost_usd: float = 0.0

    @property
    def successes(self) -> list[Trial]:
        return [t for t in self.trials if t.success]

    @property
    def failures(self) -> list[Trial]:
        return [t for t in self.trials if not t.success]

    @property
    def solved(self) -> bool:
        return bool(self.successes)

    @property
    def contrastive(self) -> bool:
        """Has both outcomes — the only case either paper can contrast."""
        return bool(self.successes and self.failures)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.trials) + self.reflection_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        trials = [Trial(**t) for t in data.get("trials", [])]
        return cls(
            task_id=data["task_id"],
            task=data.get("task", ""),
            trials=trials,
            reflection_cost_usd=data.get("reflection_cost_usd", 0.0),
        )


def experience_path(cfg: ExperimentConfig, task_id: str) -> Path:
    """Per-task checkpoint: `<experiment_dir>/experiences/<task>.json`."""
    return experiment_dir(cfg) / "experiences" / f"{task_id}.json"


def _actions(trace: dict[str, Any]) -> list[str]:
    """The action taken at each turn: the tool call / code, else the utterance.

    This is what AutoGuide compares to find the timestep where two trajectories diverge.
    """
    out = []
    for turn in trace.get("turns") or []:
        out.append((turn.get("code") or turn.get("thought") or "").strip())
    return out


def _turn_texts(trace: dict[str, Any]) -> list[str]:
    """Each turn rendered on its own, so a PREFIX of a trajectory can be reconstructed.

    AutoGuide identifies the context of the shared prefix τ:t of two trajectories, which
    needs turn-level granularity; the same layout as `format_trajectory_text`.
    """
    out = []
    for turn in trace.get("turns") or []:
        lines = []
        if turn.get("thought"):
            lines.append(f"Thought: {turn['thought']}")
        if turn.get("code"):
            lines.append(f"Code: {turn['code']}")
        output = turn.get("execution_output") or ""
        if output:
            lines.append(
                f"Output: {output[:500] + '...' if len(output) > 500 else output}"
            )
        out.append("\n".join(lines))
    return out


def task_summary(experience: "Experience", final_memory: str) -> dict[str, Any]:
    """The per-task record the `serve` viewer's accumulation tab reads.

    Uses DAEDALUS's accumulation field names (`attempts`, `final_outcome`, `final_memory`,
    `solved_without_memory`) so a method's run shows up there without viewer changes: one
    attempt per Reflexion trial, and whatever the method extracted as the memory.
    """
    trials = experience.trials
    streak = 0
    attempts = []
    for trial in trials:
        streak = streak + 1 if trial.success else 0
        attempts.append(
            {
                "attempt": trial.index + 1,
                "outcome": trial.outcome,
                "consecutive_successes": streak,
                "memory": trial.reflection or None,
                "memory_diff": None,
                "skipped_generation": not trial.reflection,
            }
        )
    tokens = {"prompt": 0, "completion": 0, "cached": 0}
    for trial in trials:
        for key in tokens:
            tokens[key] += int((trial.tokens or {}).get(key, 0) or 0)
    return {
        "task_id": experience.task_id,
        "task": experience.task,
        "attempts": attempts,
        "final_outcome": "success" if experience.solved else "failure",
        "final_memory": final_memory or None,
        # True when the FIRST, reflection-free trial already solved it.
        "solved_without_memory": bool(trials and trials[0].success),
        "cost_usd": experience.cost_usd,
        "tokens": tokens,
        "last_trace_text": trials[-1].trajectory if trials else "",
    }


def gather_experience(
    task_id: str,
    cfg: ExperimentConfig,
    reflection_llm: LLMClient,
    max_retries: int = 3,
) -> Experience:
    """Run one training task up to `max_retries` + 1 times, reflecting after each failure.

    Stops at the first success (Reflexion's loop, and both papers' setup): a solved task
    yields at most one success and the failures that preceded it, which is exactly the
    contrastive material the extraction steps want.
    """
    benchmark = get_benchmark(cfg)
    experience = Experience(task_id=task_id)
    reflections: list[str] = []

    for index in range(max_retries + 1):
        agent = benchmark.build_agent(cfg)
        # The accumulated reflections are injected the way this harness injects any
        # memory — as the solver prompt's `heuristics` block — which is Reflexion's
        # "reflection in context" for the next trial. An empty list (not None) keeps the
        # runner on its accumulation path (re-run the task, save the attempt trace).
        result = agent.solve_task(
            task_id,
            heuristics=list(reflections),
            trace_suffix=f"trial{index}",
        )
        success = bool(result.get("success", False))
        experience.task = result.get("task_instruction") or task_id
        tokens = result.get("total_tokens") or {}
        trial = Trial(
            index=index,
            outcome="success" if success else "failure",
            trajectory=benchmark.format_trajectory(result),
            actions=_actions(result),
            turn_texts=_turn_texts(result),
            cost_usd=float(result.get("total_cost_usd", 0.0) or 0.0),
            tokens={
                k: int(tokens.get(k, 0) or 0)
                for k in ("prompt", "completion", "cached")
            },
        )
        experience.trials.append(trial)
        console.info(f"    trial {index}: {trial.outcome}")

        if success or index == max_retries:
            break

        mark = (
            reflection_llm.ledger.tally.snapshot()
            if reflection_llm.ledger is not None
            else None
        )
        trial.reflection = generate_self_reflection(
            llm=reflection_llm,
            instruction=experience.task,
            trajectory_text=trial.trajectory,
            outcome=trial.outcome,
            benchmark=cfg.benchmark,
        )
        reflections.append(trial.reflection)
        # generate_self_reflection returns only text, so the reflection's cost comes from
        # the usage events it wrote (a ledger delta), not from re-pricing the client's
        # cumulative token counters — which could not see the service tier or the cache.
        if reflection_llm.ledger is not None:
            experience.reflection_cost_usd += reflection_llm.ledger.tally.since(mark)[
                "priced_calls_estimated_cost_usd"
            ]

    return experience


def _worker(
    task_id: str,
    config_path: str,
    setup: str,
    max_retries: int,
    reflection_model: str,
    reflection_reasoning_effort: str | None,
    experiment_name: str | None,
    index: int,
    n_tasks: int,
) -> tuple[int, dict[str, Any]]:
    """Top-level ProcessPoolExecutor worker (must be importable and picklable)."""
    # memory.enabled is off there: the trials carry reflections, never a retrieval pool.
    cfg = worker_config(config_path, setup, experiment_name)

    accounting = CostAccounting.for_worker(cfg)
    reflection_llm = accounting.client(
        reflection_model,
        "reflection",
        temperature=0.0,
        reasoning_effort=reflection_reasoning_effort,
    )
    console.info(f"[{index + 1}/{n_tasks}] Task: {task_id} (start)", flush=True)
    experience = gather_experience(
        task_id, cfg, reflection_llm, max_retries=max_retries
    )
    console.info(
        f"[{index + 1}/{n_tasks}] Task: {task_id} → "
        f"{len(experience.successes)}S/{len(experience.failures)}F"
        f"{' [contrastive]' if experience.contrastive else ''}",
        flush=True,
    )

    write_json(experience_path(cfg, task_id), experience.to_dict())
    return index, experience.to_dict()


def gather_experiences(
    cfg: ExperimentConfig,
    config_path: str,
    task_ids: list[str],
    setup: str,
    max_retries: int,
    reflection_model: str,
    reflection_reasoning_effort: str | None = None,
) -> list[Experience]:
    """Gather (and checkpoint) experience for every task, in parallel. Resumes.

    A task that dies is reported and dropped rather than fatal: it leaves no checkpoint,
    so a re-run picks it up again, and the other tasks' experience is not lost with it.
    """
    n_tasks = len(task_ids)
    ordered, pending = restore(task_ids, lambda t: experience_path(cfg, t))
    args = (
        config_path,
        setup,
        max_retries,
        reflection_model,
        reflection_reasoning_effort,
        cfg.name,
    )
    for index, data in run_tasks(
        _worker,
        pending,
        lambda i, tid: (tid, *args, i, n_tasks),
        max(1, cfg.run.parallel or 1),
    ):
        ordered[index] = data

    return [Experience.from_dict(d) for d in ordered if d is not None]
