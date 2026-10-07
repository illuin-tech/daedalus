"""The inference driver shared by every method under `references/`.

Each method's `run.py` is a thin `InferenceMethod` description handed to `main`: its config
block, how to build its solver, which of its fields the CLI may override, which extra models
the cost preflight must price, and the one-line summary it prints at the end. Everything
else — argument parsing, config loading, the run loop over pending runs, and scoring — is
the same for all of them and lives here, so an ERL, ExpeL, AutoGuide, ReasoningBank or ACE
run is driven exactly like a DAEDALUS one (`daedalus.scripts.inference`).

Artifacts land in `outputs/baselines/<setup>/inference/<name>/` (the paper's configs name the run
after its benchmark), in the same shapes as daedalus's own
`outputs/inference/<benchmark>/<category>/<name>/`.
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from daedalus.core.config import (
    ExperimentConfig,
    load_config,
    resolve_experiment_name,
    resolve_trace_dir,
    save_run_config,
)
from daedalus.core.env import load_dotenv
from daedalus.core.evaluation.evaluate import evaluate_experiment
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.registry import get_benchmark

from references.common.config import MethodConfig, require_kind


@dataclass(frozen=True)
class Override:
    """A CLI flag that overrides one field of the method's config block."""

    flag: str  # e.g. "--pool" or "-k"
    field: str  # e.g. "pool_path"
    type: type = str
    help: str = ""


@dataclass(frozen=True)
class InferenceMethod:
    """What a method contributes to the shared inference driver."""

    setup: str  # "erl", "expel", ... — the config block name and `cfg.setup`
    name: str  # "ERL", "ExpeL", ... — for messages
    config_cls: type[MethodConfig]
    build_agent: Callable[..., Any]  # (cfg, run_idx=...) -> a benchmark TaskAgent
    # Why daedalus's own retrieval is switched off, completing
    # "[<setup>] memory.enabled is ignored: <memory_note>".
    memory_note: str
    # Rows of the run header specific to this method, placed after `model`.
    header: Callable[[ExperimentConfig, Any], dict[str, Any]]
    overrides: tuple[Override, ...] = ()
    # Models this method calls at inference time, priced before the run starts.
    extra_models: Callable[[ExperimentConfig, Any], list[ModelRequirement]] = field(
        default=lambda cfg, mcfg: []
    )
    # Checks run after the CLI overrides and before anything is written.
    validate: Callable[[Any], None] | None = None
    # Prints the end-of-run retrieval/injection summary (only when runs were executed).
    summarize: Callable[[ExperimentConfig], None] | None = None


def resolve_task_ids(cfg: ExperimentConfig) -> list[str]:
    task_ids = get_benchmark(cfg).list_task_ids(cfg)
    if cfg.run.max_tasks is not None:
        task_ids = task_ids[: cfg.run.max_tasks]
    return task_ids


def is_run_complete(trace_dir: Path, task_ids: list[str]) -> bool:
    if not trace_dir.exists():
        return False
    existing = {p.stem for p in trace_dir.glob("*.json")}
    return all(tid in existing for tid in task_ids)


def run_experiment(method: InferenceMethod, cfg: ExperimentConfig, mcfg: MethodConfig) -> None:
    """Run the method's agent over the configured tasks (multi-run), then score."""
    console.configure(cfg.logging.verbose)
    save_run_config(cfg)  # config.yaml + run_meta.json
    # <setup>.yaml — also how the spawned solver workers read these settings.
    mcfg.save(cfg)
    CostAccounting.start(cfg, extra_models=method.extra_models(cfg, mcfg) or None)

    task_ids = resolve_task_ids(cfg)
    console.header(
        f"{method.setup} · inference",
        {
            "experiment": cfg.name,
            "benchmark": cfg.benchmark,
            "model": cfg.agent.model,
            **method.header(cfg, mcfg),
            "tasks": len(task_ids),
            "runs": cfg.run.num_runs,
            "parallel": cfg.run.parallel,
            "output": resolve_trace_dir(cfg, run_idx=0).parent,
        },
    )

    pending, completed = [], []
    for run_idx in range(cfg.run.num_runs):
        trace_dir = resolve_trace_dir(cfg, run_idx=run_idx)
        done = not cfg.run.force and is_run_complete(trace_dir, task_ids)
        (completed if done else pending).append(run_idx)
    if completed:
        console.info(f"Runs already complete: {completed}")

    for run_idx in pending:
        console.rule(f"run {run_idx}/{cfg.run.num_runs - 1}")
        agent = method.build_agent(cfg, run_idx=run_idx)
        console.info(f"output: {agent.trace_dir}")
        agent.solve_tasks(task_ids)
        console.info(f"run {run_idx} done")

    if pending and method.summarize is not None:
        method.summarize(cfg)

    console.rule()
    try:
        evaluate_experiment(cfg, task_ids)
    except Exception as e:  # noqa: BLE001 — a scoring failure must not discard the run
        console.info(f"evaluation skipped: {type(e).__name__}: {e}")
        traceback.print_exc()


def main(method: InferenceMethod) -> None:
    """`python -m references.<setup>.run --config <yaml>`."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=f"{method.name} inference on a benchmark")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=f"Override experiment_name → outputs/baselines/{method.setup}/inference/<name>/. "
        "Supports {benchmark}/{domain}/{model}/{date} tokens.",
    )
    parser.add_argument("--task-id", type=str, nargs="+", default=None, help="Task ID(s) to run")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max number of tasks")
    parser.add_argument("--parallel", type=int, default=None, help="Parallel workers")
    parser.add_argument("--num-runs", type=int, default=None, help="Independent runs")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    for o in method.overrides:
        parser.add_argument(
            o.flag,
            dest=f"override_{o.field}",
            metavar=o.flag.lstrip("-").replace("-", "_").upper(),
            type=o.type,
            default=None,
            help=o.help,
        )
    args = parser.parse_args()

    require_kind(args.config, "inference", method.setup)
    cfg = load_config(args.config)
    # → outputs/baselines/<setup>/inference/<name>/
    cfg.kind, cfg.setup = "inference", method.setup
    mcfg = method.config_cls.load(args.config)

    if args.experiment_name:
        cfg.experiment_name = resolve_experiment_name(args.experiment_name, cfg)
    if args.task_id:
        cfg.benchmark_config.task_ids = args.task_id
    if args.max_tasks is not None:
        cfg.run.max_tasks = args.max_tasks
    if args.parallel is not None:
        cfg.run.parallel = args.parallel
    if args.num_runs is not None:
        cfg.run.num_runs = args.num_runs
    if args.force:
        cfg.run.force = True
    for o in method.overrides:
        value = getattr(args, f"override_{o.field}")
        # Paths override when non-empty, numbers whenever given (as the flags always did).
        if value is not None and (o.type is not str or value):
            setattr(mcfg, o.field, value)

    if method.validate is not None:
        method.validate(mcfg)  # before save_run_config: a bad path writes nothing

    if cfg.memory.enabled:
        # Leaving daedalus's retrieval on would ALSO build a per-turn retriever and inject a
        # second, differently-selected set of memories — not this method any more.
        console.info(f"[{method.setup}] memory.enabled is ignored: {method.memory_note}")
        cfg.memory.enabled = False

    run_experiment(method, cfg, mcfg)
