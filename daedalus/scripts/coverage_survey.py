"""Run the Surveyor on its own and save its coverage goal (any benchmark).

Generation runs this survey itself when `generation.coverage_tags` is on, so this script
is only for PRECOMPUTING it: survey once, review the goal by hand, then point any number
of runs at the artifact with

    generation:
      coverage_tags: true
      coverage_tags_path: "outputs/environment-survey/<benchmark>/coverage_goal_<ts>.txt"

`--exclude-apps gmail amazon` holds AppWorld apps out of the survey completely, so the coverage goal it writes cannot depend on them and the manifest
cannot contain a tag generation would never be able to earn. On AppWorld the hold-out is
total — the apps are gone from the catalog, from every runtime listing and error, and calls
to them are refused, so the survey never learns they exist (see
benchmarks/appworld/hidden_apps.py). It sets the same `generation.excluded_apps` the
generation run itself reads, and the explorer hides that set the same way, so keep the two
in sync.

The survey explores through the benchmark's own channel (AppWorld: `apis.*` Python; tau2:
domain tools + read_db; AutomationBench: its REST tools) and is expected to
CHANGE the environment — it works in a throwaway copy, and calling a write operation is the
only way to learn what it demands. It stays sandboxed by core/generation/survey_guardrails.py:
it can never reach the host, the benchmark's package internals, or any task/evaluation set.

Usage:
    python -m daedalus.scripts.coverage_survey --config configs/tau2/generation/daedalus.yaml
    python -m daedalus.scripts.coverage_survey --config <cfg> --max-turns 60 --out my_goal.txt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from daedalus.core.config import load_config
from daedalus.core.env import load_dotenv
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.generation import coverage
from daedalus.core.generation.backend import get_generation_backend
from daedalus.core.logging import console

SURVEY_ROOT = Path("outputs") / "environment-survey"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Coverage survey → reusable coverage-goal artifact")
    ap.add_argument("--config", required=True, help="Config YAML (benchmark, explorer model, domain)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Override generation.coverage_tags_max_turns")
    ap.add_argument("--out", default=None,
                    help="Artifact path (.txt); default "
                         "outputs/environment-survey/<benchmark>/coverage_goal_<domain>_<ts>.txt")
    ap.add_argument("--exclude-apps", nargs="*", default=None, metavar="NAME",
                    help="Hold these AppWorld apps out of the survey "
                         "entirely, so they cannot shape the coverage goal or earn a tag. On "
                         "AppWorld they are also hidden from every runtime listing and calls to "
                         "them are refused, so the survey never learns they exist. Accepts spaces "
                         "or commas: --exclude-apps gmail amazon. Overrides the config's "
                         "generation.excluded_apps.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.kind = "generation"  # so cfg.name / paths resolve like a generation run
    if args.max_turns is not None:
        cfg.generation.coverage_tags_max_turns = args.max_turns
    if args.exclude_apps is not None:
        # Same knob the generation run uses, so a survey and the run it feeds hold out the
        # same set — a tag for an app generation may not touch would sit UNDER forever and
        # could capture the late escalation slot.
        cfg.generation.excluded_apps = [
            x.strip() for a in args.exclude_apps for x in a.split(",") if x.strip()
        ]
    console.configure(cfg.logging.verbose)

    backend = get_generation_backend(cfg)
    contexts = backend.session_contexts(cfg)
    if not contexts:
        raise SystemExit(
            f"No environment contexts for benchmark {cfg.benchmark!r} — check the config's "
            "dataset/domain/split."
        )
    # The survey's ledger belongs with the survey's artifact, NOT with the generation run
    # it will later feed. Without an accounting handle the explorer falls back to
    # CostAccounting.for_worker(cfg), whose path is experiment_dir(cfg) — and with
    # cfg.kind set to "generation" above, that is outputs/daedalus/<name>/usage/.
    # Creating it there made `generate()` refuse to start ("a generation run already
    # exists"), so precomputing a survey blocked the run it was precomputed for.
    accounting = CostAccounting.standalone(
        SURVEY_ROOT / cfg.benchmark,
        experiment_name=f"coverage_survey:{cfg.benchmark}",
        experiment_kind="coverage_survey",
        benchmark=cfg.benchmark,
    )
    explorer = backend.make_explorer(cfg, accounting)
    ctx = contexts[0]

    header = {
        "benchmark": cfg.benchmark,
        "context": str(ctx),
        "model": cfg.generation.explorer_model,
        "max_turns": cfg.generation.coverage_tags_max_turns,
    }
    held_out = cfg.generation.excluded_apps or []
    if held_out:
        header["held out"] = ", ".join(held_out)
    console.header("daedalus · coverage survey", header)
    goal, transcript = explorer.survey_coverage(ctx)
    if not goal.tagged:
        raise SystemExit(
            "The survey produced no valid tag manifest — nothing written. Re-run (optionally "
            "with a larger --max-turns); see the printed turns above for what happened."
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    domain = getattr(getattr(cfg, cfg.benchmark, None), "domain", None) or cfg.generation.sandbox_dataset
    out = Path(args.out) if args.out else SURVEY_ROOT / cfg.benchmark / f"coverage_goal_{domain}_{ts}.txt"
    coverage.save(out, goal)
    out.with_suffix(".transcript.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "benchmark": cfg.benchmark,
                "context": str(ctx),
                "model": cfg.generation.explorer_model,
                "tags": goal.tags,
                "turns": transcript,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    console.rule("coverage survey complete")
    console.info(f"{len(goal.tags)} tag(s) → {out}")
    console.info(f"  transcript → {out.with_suffix('.transcript.json')}")
    for t in goal.tags:
        console.info(f"    {t['tag']:24s} {round(t['target_share'] * 100):3d}%  {t['definition']}")
    console.info(f"Point a generation config at it with:  coverage_tags_path: \"{out}\"")


if __name__ == "__main__":
    main()
