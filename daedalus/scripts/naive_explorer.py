"""Table 3, row A: one explorer, N turns in a live AppWorld world, then one list of heuristics.

    uv run python -m daedalus.scripts.naive_explorer --max-turns 100 \
        --exclude-apps gmail,amazon --name appworld_ablation-A

The explorer explores freely for the whole budget and is then asked once for everything
it learned. There is no task generation, no solver and no judge, so the gap between this
pool and a generation pool measures the self-play loop. Match `--model`,
`--reasoning-effort` and `--exclude-apps` to the generation run's explorer.

Writes, into `outputs/daedalus/<name>/`:

    pool.json           a MemoryPool — use it as `memory.pool_path`, or consolidate it
    run_summary.json    tokens and cost
    transcript.json     every turn: thought, code, output, and the final emission
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.memory.pool import MemoryItem, MemoryPool

APPWORLD_PROMPTS = Path(__file__).resolve().parents[1] / "benchmarks" / "appworld" / "prompts"
OUT_ROOT = Path("outputs/daedalus")


def explore(task_id: str, args: argparse.Namespace, out: Path,
            accounting: CostAccounting) -> dict[str, Any]:
    """Run one session in `task_id`'s world. Returns heuristics, transcript and usage."""
    from appworld import AppWorld

    from daedalus.benchmarks.appworld.agent import _appworld_safe_name
    from daedalus.benchmarks.appworld.direct_explore import app_catalog, explore_directly
    from daedalus.benchmarks.appworld.hidden_apps import HiddenApps
    from daedalus.core.resources import render_prompt

    hidden = HiddenApps([a for a in args.exclude_apps.split(",") if a.strip()])
    llm = accounting.client(
        args.model, "explorer", component="naive_explorer",
        temperature=args.temperature, reasoning_effort=args.reasoning_effort,
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0,
             "cost_usd": 0.0}

    def generate(messages: list[dict[str, str]]) -> str:
        r = llm.generate(messages)
        usage["prompt_tokens"] += r.prompt_tokens
        usage["completion_tokens"] += r.completion_tokens
        usage["cached_tokens"] += r.cached_tokens
        usage["calls"] += 1
        usage["cost_usd"] += r.estimated_cost_usd or 0.0
        return r.content

    def checkpoint(transcript: list[dict], turn: int) -> None:
        """Partial transcript every few turns, so a killed run keeps its work. No pool.json:
        the heuristics exist only after the final emission."""
        if args.checkpoint_every and turn % args.checkpoint_every == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "transcript.json").write_text(
                json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    with AppWorld(task_id=task_id,
                  experiment_name=_appworld_safe_name(f"{args.name}_naive")) as world:
        system = render_prompt(
            "naive_explorer", APPWORLD_PROMPTS,
            add_env_knowledge=True,
            max_turns=args.max_turns,
            allow_early_stop=False,
            app_descriptions=app_catalog(world, hidden),
            main_user=world.task.supervisor,
        )
        print(f"world {task_id} | model {args.model} | {args.max_turns} turns"
              + (f" | hidden apps: {', '.join(hidden.names)}" if hidden else ""), flush=True)
        heuristics, transcript = explore_directly(
            world, generate, system, hidden, args.max_turns,
            log=lambda msg: print(f"{msg}  ${usage['cost_usd']:.2f}", flush=True),
            checkpoint=checkpoint,
        )
    return {"heuristics": heuristics, "transcript": transcript, "usage": usage,
            "task_id": task_id}


def write(out: Path, res: dict[str, Any], args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    MemoryPool([
        MemoryItem(
            memory_id=f"m_{hashlib.sha1(h.encode()).hexdigest()[:8]}",
            type="reflection",
            text=h,
            source_task_id=f"naive_explore:{res['task_id']}",
            source_trajectory_success=True,
            extraction_timestamp=stamp,
            tags={"source": "naive_exploration"},
        )
        for h in res["heuristics"]
    ]).save(out / "pool.json", overwrite=True)
    (out / "transcript.json").write_text(
        json.dumps(res["transcript"], indent=2, ensure_ascii=False), encoding="utf-8")
    transcript, usage = res["transcript"], res["usage"]
    (out / "run_summary.json").write_text(json.dumps({
        "kind": "naive_exploration",
        "name": args.name,
        "benchmark": "appworld",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sandbox_task_id": res["task_id"],
        "max_turns": args.max_turns,
        "turns_used": sum(1 for t in transcript if t["kind"] != "emission"),
        "exec_errors": sum(1 for t in transcript if t.get("exec_error")),
        "emission_attempts": sum(1 for t in transcript if t["kind"] == "emission"),
        "excluded_apps": [a for a in args.exclude_apps.split(",") if a.strip()],
        "num_heuristics": len(res["heuristics"]),
        "total_cost_usd": round(usage["cost_usd"], 4),
        "tokens": usage,
        "timestamp": stamp,
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="appworld_naive",
                    help="output folder under outputs/daedalus/")
    ap.add_argument("--max-turns", type=int, default=100,
                    help="exploration turns before the heuristics are asked for")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--task-id", default=None,
                    help="the sandbox world to explore (default: first train task)")
    ap.add_argument("--exclude-apps", default="",
                    help="comma-separated apps to hide entirely, e.g. gmail,amazon")
    ap.add_argument("--checkpoint-every", type=int, default=5, metavar="N",
                    help="write transcript.json every N turns (0 disables)")
    args = ap.parse_args()
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be at least 1")

    task_id = args.task_id
    if task_id is None:
        from appworld.task import load_task_ids

        task_id = load_task_ids("train")[0]
    out = OUT_ROOT / args.name
    if (out / "pool.json").exists():
        raise SystemExit(f"{out / 'pool.json'} exists — pick another --name or remove it")

    accounting = CostAccounting.standalone(
        out, experiment_name=f"naive_explorer:{args.name}", benchmark="appworld"
    )
    res = explore(task_id, args, out, accounting)
    write(out, res, args)
    print(f"\n{len(res['heuristics'])} heuristics, ${res['usage']['cost_usd']:.4f} → {out}")


if __name__ == "__main__":
    main()
