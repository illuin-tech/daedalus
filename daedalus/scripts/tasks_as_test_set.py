#!/usr/bin/env python3
"""Evaluate models on the tasks a generation run invented — the generated tasks AS A TEST SET.

    uv run python -m daedalus.scripts.tasks_as_test_set \
        --generation outputs/generation/appworld_90sessions_taskrealism_exclude_apps \
        --model gpt-5.4-mini --model gpt-5.4 --num-runs 3

Elsewhere in this repo a generation run is a source of *heuristics*: its tasks exist to be
solved so that a lesson can be mined from the attempt. This script asks the other question —
are those tasks a **benchmark**? Each banked task carries an instruction and natural-language
success conditions, so a model can be run on it and graded, and the resulting ranking can be
compared against the same models' scores on the real held-out split.

Per (test set × model): the banked tasks are frozen into `test_set.json` with stable ids,
each task is run `--num-runs` times in the *starting world of the sandbox it was designed in*
with the generated instruction in place of the real one, and every trajectory is graded by one
LLM judge against the task's success conditions — condition by condition, so a task with
three conditions of which two hold scores 2/3 (partial credit) rather than just "failed".

Artifacts land in `outputs/evaluation-proxy/<generation_run>/<model>/` in the same shapes an inference
run writes (`run_<i>/<task>.json` traces, `evaluation.json` with `success_rate_mean`,
`partial_rate_mean` and `pass_hat_k`), plus `judgements/run_<i>/<task>.json` holding the
judge's per-condition verdicts.

Three things about the METHOD that matter when reading the numbers, all recorded in
`run_meta.json` / `evaluation.json`:

* Memory is forced OFF. This measures the model on the task, not a memory method.
* The solver settings (max turns, env knowledge, the split the sandbox worlds come from) are
  inherited from the generation run's own config, so the environment matches the one the
  tasks were designed in. `--config` overrides the base if you want something else.
* One fixed judge for every model (`--judge-model`, default gpt-5.4 at high effort,
  temperature 0). Never compare models judged by different judges — and note that a model
  which is also the judge is grading its own work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from daedalus.core.config import (
    ExperimentConfig,
    ProviderRoutingConfig,
    config_from_dict,
    config_from_snapshot,
    experiment_dir,
    kind_root,
    save_run_config,
)
from daedalus.core.env import load_dotenv
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import ModelRequirement
from daedalus.core.logging import console
from daedalus.core.memory.extraction import format_trajectory_text
from daedalus.core.registry import get_benchmark
from daedalus.core.testset.judge import judge_conditions
from daedalus.core.testset.manifest import MANIFEST_NAME, TestSet, TestTask, from_generation_run
from daedalus.core.testset.score import build_payload, judgement_dir

KIND = "testset"


# ── configuration ───────────────────────────────────────────────────────────────────────
def model_tag(model: str) -> str:
    """Folder-safe short name for a model string ('openrouter/qwen/x' → 'x')."""
    return model.split("/")[-1]


def base_config(generation_dir: Path, config_path: str | None) -> dict[str, Any]:
    """The solver config to evaluate with: the generation run's own, unless overridden.

    Inheriting it is what keeps the comparison honest — same max_turns, same env knowledge,
    same dataset split for the sandbox worlds as when the tasks were designed.
    """
    path = Path(config_path) if config_path else generation_dir / "config.yaml"
    if not path.exists():
        raise SystemExit(f"No config to build on: {path} does not exist.")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_cfg(
    raw: dict[str, Any], test_set: TestSet, model: str, args: argparse.Namespace
) -> ExperimentConfig:
    """Turn the inherited config into a test-set run config for one model."""
    cfg = config_from_snapshot(raw)
    cfg.kind, cfg.setup = KIND, "daedalus"
    # outputs/evaluation-proxy/<generation_run>/<model>/ — one folder per model, all under the test
    # set they were measured on, which is what makes a cross-model comparison greppable.
    suffix = "_own_heuristic" if args.own_heuristic else ""
    cfg.experiment_name = f"{test_set.generation_run}/{model_tag(model)}{suffix}"
    cfg.agent.model = model
    # Provider routing is per MODEL: an OpenRouter tag like "parasail/fp8" names an upstream
    # of the model it was pinned for, and no other. Inheriting the generation run's pin for a
    # DIFFERENT solver would send a filter no upstream of that model satisfies, failing every
    # call. Keep it only when this run evaluates the model the pin was written for.
    generator_model = test_set.generator_model or (raw.get("agent") or {}).get("model", "")
    if args.provider_order:
        # An explicit pin for the model being evaluated. Needed because the reset below fires
        # for every cross-model evaluation, which is most of them: an unpinned OpenRouter
        # model can land on an upstream that fails every call and scores it near 0%.
        cfg.provider_routing = ProviderRoutingConfig(
            order=list(args.provider_order), allow_fallbacks=False
        )
    elif model != generator_model:
        cfg.provider_routing = ProviderRoutingConfig()
    if args.max_turns is not None:
        cfg.agent.max_turns = args.max_turns
    if args.reasoning_effort is not None:
        cfg.agent.reasoning_effort = None if args.reasoning_effort == "null" else args.reasoning_effort
    if args.temperature is not None:
        cfg.agent.temperature = args.temperature
    cfg.run.num_runs = args.num_runs
    cfg.run.parallel = args.parallel if args.parallel is not None else (cfg.run.parallel or 1)
    cfg.run.force = args.force
    cfg.run.max_tasks = None  # task selection happens on the manifest, not the benchmark
    # A model is measured bare: no heuristics, no retrieval. Otherwise this would score a
    # memory method rather than the model.
    cfg.memory.enabled = False
    cfg.memory.heuristics_at_start = False
    cfg.accumulation.save_traces = True  # the judge path counts as accumulation mode
    cfg.logging.trace_dir = "outputs/traces/{experiment_name}"  # sentinel → experiment folder
    return cfg


# ── running one task ────────────────────────────────────────────────────────────────────
def _worker(
    cfg_dict: dict[str, Any],
    task_dict: dict[str, Any],
    run_idx: int,
    judge_model: str,
    judge_effort: str | None,
    position: str,
    heuristic: str | None = None,
) -> dict[str, Any]:
    """Run one test task once and judge it. Top-level for ProcessPoolExecutor."""
    cfg = config_from_dict(cfg_dict)
    console.silence_third_party(cfg.logging.verbose)
    task = TestTask(**task_dict)
    benchmark = get_benchmark(cfg)
    benchmark.prepare_accumulation(cfg)  # AppWorld: bind its on-disk experiment name
    agent = benchmark.build_agent(cfg, run_idx=run_idx)

    # The judge shares the launch ledger with the solver but bills to the `judge` role, so
    # the test-set total covers both instead of solver-plus-a-separate-judge-field.
    accounting = CostAccounting.for_worker(cfg, run_idx=run_idx)
    judge_llm = accounting.client(judge_model, "judge", component="testset_judge",
                                  temperature=0.0)
    judge_llm.scope = accounting.scope("judge", component="testset_judge", task_id=task.tid)
    verdict: dict[str, Any] = {}

    def evaluator(trace: dict[str, Any]) -> tuple[bool, str]:
        """Grade in place of the benchmark's own checker (there is no ground truth here)."""
        v = judge_conditions(
            judge_llm,
            instruction=task.instruction,
            success_conditions=task.success_conditions,
            trajectory_text=format_trajectory_text(trace),
            reasoning_effort=judge_effort,
        )
        verdict["v"] = v
        return v.success, v.details

    console.info(f"[{position}] {task.tid} ({task.sandbox_id}) start", flush=True)
    usage_mark = accounting.tally_snapshot()
    trace = agent.solve_task(
        task.sandbox_id,
        heuristics=[heuristic] if heuristic else None,
        instruction_override=task.instruction,
        evaluator=evaluator,
        trace_suffix=task.tid,  # trace file: <sandbox>_<tid>.json
    )
    v = verdict.get("v")
    if v is None:  # the solve loop never reached scoring (a crashed rollout)
        raise RuntimeError(f"{task.tid}: the rollout produced no judgement")

    record = {
        "tid": task.tid,
        "sandbox_id": task.sandbox_id,
        "instruction": task.instruction,
        "success_conditions": task.success_conditions,
        "tags": task.tags,
        "run_idx": run_idx,
        "model": cfg.agent.model,
        "num_turns": trace.get("num_turns"),
        "solver_cost_usd": float(trace.get("total_cost_usd", 0.0) or 0.0),
        "solver_tokens": trace.get("total_tokens") or {},
        # Every role this task paid for - solver, judge, and (tau2) the user simulator or an
        # LLM retriever. `solver_cost_usd + judge_cost_usd` was never the whole bill.
        "usage": accounting.usage_since(usage_mark),
        **v.to_dict(),
    }
    console.info(
        f"[{position}] {task.tid} → {'PASS' if v.success else 'fail'} "
        f"({v.num_met}/{len(v.met)} conditions)",
        flush=True,
    )
    path = judgement_dir(experiment_dir(cfg), run_idx) / f"{task.tid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def run_one_model(
    cfg: ExperimentConfig,
    test_set: TestSet,
    tasks: list[TestTask],
    args: argparse.Namespace,
    heuristics: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every repeat of every task for one model, then score."""
    base = experiment_dir(cfg)
    save_run_config(cfg)  # config.yaml + run_meta.json
    console.header(
        f"tasks-as-test-set · {cfg.agent.model}",
        {
            "test set": f"{test_set.generation_run} ({len(tasks)} tasks)",
            "benchmark": f"{test_set.benchmark} · {test_set.dataset}",
            "judge": f"{args.judge_model} ({args.judge_reasoning_effort})",
            "repeats": cfg.run.num_runs,
            "parallel": cfg.run.parallel,
            "memory": "each task's own banked heuristic" if heuristics else "off (a model is measured bare)",
            "output": base,
        },
    )

    # Price the solver AND the judge before either runs; the launch id then travels to the
    # task workers inside `cfg_dict`.
    accounting = CostAccounting.start(
        cfg, extra_models=[ModelRequirement(args.judge_model, "judge", "--judge-model")])
    cfg_dict = cfg.to_dict()
    for run_idx in range(cfg.run.num_runs):
        done = {p.stem for p in judgement_dir(base, run_idx).glob("*.json")}
        pending = [t for t in tasks if cfg.run.force or t.tid not in done]
        console.rule(f"run {run_idx}/{cfg.run.num_runs - 1} · {len(pending)} task(s) to do")
        if not pending:
            continue
        n_workers = max(1, cfg.run.parallel or 1)
        total = len(pending)
        if n_workers == 1:
            for i, task in enumerate(pending):
                _safe(_worker, cfg_dict, asdict(task), run_idx, args.judge_model,
                      args.judge_reasoning_effort, f"{i + 1}/{total}",
                      (heuristics or {}).get(task.tid))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_worker, cfg_dict, asdict(task), run_idx, args.judge_model,
                                args.judge_reasoning_effort, f"{i + 1}/{total}",
                                (heuristics or {}).get(task.tid)): task
                    for i, task in enumerate(pending)
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:  # noqa: BLE001 — one bad task must not sink the run
                        console.info(f"  [error] {futures[future].tid}: {type(e).__name__}: {e}")

    payload = build_payload(
        base,
        num_runs=cfg.run.num_runs,
        experiment_name=cfg.name,
        benchmark=test_set.benchmark,
        test_set={
            "generation_run": test_set.generation_run,
            "dataset": test_set.dataset,
            "generator_model": test_set.generator_model,
            "num_tasks": len(tasks),
            "judge_model": args.judge_model,
            "judge_reasoning_effort": args.judge_reasoning_effort,
        },
    )
    payload["usage"] = accounting.summary()
    (base / "evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _report(payload, cfg)
    return payload


def banked_heuristics(generation_dir: Path, tasks: list[TestTask]) -> dict[str, str]:
    """Map each test-set task to the heuristic its session banked, matched by instruction."""
    by_instruction = {}
    for path in sorted((generation_dir / "sessions").glob("session_*.json")):
        session = json.loads(path.read_text(encoding="utf-8"))
        if session.get("result") == "banked":
            by_instruction[session["trajectory"][-1]["task"].strip()] = session["heuristic"]
    missing = [t.tid for t in tasks if t.instruction.strip() not in by_instruction]
    if missing:
        raise SystemExit(f"No banked session matches test-set task(s): {missing}")
    return {t.tid: by_instruction[t.instruction.strip()] for t in tasks}


def _safe(fn, *fn_args) -> None:
    try:
        fn(*fn_args)
    except Exception as e:  # noqa: BLE001
        console.info(f"  [error] {type(e).__name__}: {e}")


def _report(payload: dict[str, Any], cfg: ExperimentConfig) -> None:
    agg = payload["aggregate"]
    console.rule(f"{cfg.agent.model} on {payload['test_set']['generation_run']}")
    for run in payload["per_run"]:
        console.info(
            f"  run_{run['run_idx']}: {run['num_successes']}/{run['num_tasks']} "
            f"({run['success_rate']:.1%})  partial {run['partial_rate']:.1%}"
        )
    console.info(f"success rate: {agg.get('success_rate_summary', 'n/a')}")
    if agg.get("partial_rate_mean") is not None:
        console.info(
            f"partial rate: {agg['partial_rate_mean']:.3f} ± {agg.get('partial_rate_std', 0):.3f}"
        )
    for key, value in (agg.get("pass_hat_k") or {}).items():
        console.info(f"  {key}: {value:.3f}")
    if agg.get("num_unparsed"):
        console.info(f"  unreadable judge replies (counted as failures): {agg['num_unparsed']}")
    console.info(
        f"cost: solver ${agg.get('solver_cost_usd', 0):.2f} + judge "
        f"${agg.get('judge_cost_usd', 0):.2f}"
    )
    console.info(f"evaluation → {experiment_dir(cfg) / 'evaluation.json'}")


# ── CLI ─────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Evaluate models on a generation run's tasks, judged against their "
                    "success conditions"
    )
    ap.add_argument("--generation", type=Path, required=True,
                    help="a generation run folder (outputs/daedalus/<name>)")
    ap.add_argument("--model", action="append", required=True,
                    help="model to evaluate; repeat the flag to sweep several, in order")
    ap.add_argument("--num-runs", type=int, default=3, help="repeats per task (pass^k needs >1)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="concurrent tasks (default: the generation run's run.parallel)")
    ap.add_argument("--judge-model", type=str, default="gpt-5.4",
                    help="ONE judge for every model, or the comparison is not a comparison")
    ap.add_argument("--judge-reasoning-effort", type=str, default="high")
    ap.add_argument("--max-tasks", type=int, default=None, help="first N tasks (smoke runs)")
    ap.add_argument("--task-id", type=str, nargs="+", default=None,
                    help="test-set ids to run (g000 g007 …)")
    ap.add_argument("--config", type=str, default=None,
                    help="base solver config (default: the generation run's own config.yaml)")
    ap.add_argument(
        "--provider-order", action="append", default=None, metavar="TAG",
        help="OpenRouter provider tag(s) to pin THIS evaluation to, e.g. "
             "--provider-order sail-research/fp4. Sets allow_fallbacks=false. Repeatable. "
             "Tags name upstreams of one model, so pass only tags of --model.",
    )
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    help="'null' for the provider default")
    ap.add_argument("--force", action="store_true", help="re-run and re-judge finished tasks")
    ap.add_argument("--own-heuristic", action="store_true",
                    help="give each task the heuristic its own session banked (paired with a "
                         "bare run, this measures what the heuristic adds on its task)")
    args = ap.parse_args()

    # Freeze once per test set, then reuse: extending the generation run must not silently
    # change what an already-measured model was measured on.
    root = kind_root(KIND) / args.generation.name
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        test_set = TestSet.load(manifest_path)
        console.info(f"test set: {manifest_path} ({len(test_set.tasks)} tasks, frozen)")
    else:
        test_set = from_generation_run(args.generation)
        test_set.save(manifest_path)
        console.info(f"test set: froze {len(test_set.tasks)} banked task(s) → {manifest_path}")

    tasks = test_set.tasks
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [t for t in tasks if t.tid in wanted]
        missing = wanted - {t.tid for t in tasks}
        if missing:
            raise SystemExit(f"No such test-set task(s): {sorted(missing)}")
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise SystemExit("No tasks selected.")

    raw = base_config(args.generation, args.config)
    heuristics = banked_heuristics(args.generation, tasks) if args.own_heuristic else None
    summaries = []
    for model in args.model:
        cfg = build_cfg(raw, test_set, model, args)
        console.configure(cfg.logging.verbose)
        summaries.append((model, run_one_model(cfg, test_set, tasks, args, heuristics)))

    if len(summaries) > 1:
        console.rule("all models")
        for model, payload in summaries:
            agg = payload["aggregate"]
            console.info(
                f"  {model_tag(model):22.22s} success {agg.get('success_rate_mean', 0):6.1%} "
                f"± {agg.get('success_rate_std', 0):.1%}   partial "
                f"{agg.get('partial_rate_mean', 0):6.1%}"
            )
        console.info(f"runs under {kind_root(KIND) / args.generation.name}/")


if __name__ == "__main__":
    main()
