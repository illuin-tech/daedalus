"""Report memory-retrieval turnover/diversity for completed experiments.

Measures how much the injected memory *changes across turns* (see
`metrics.retrieval_diversity`). Use it to test the turnover hypothesis:
do the conditions that help (random/colbert @turn) renew their context each
turn, while those that don't (bm25/qwen @turn) keep re-injecting the same items?

Reads only existing traces — no model calls, no AppWorld needed.

Usage:
    # one or more inference runs by id (their path under outputs/inference/)
    python -m daedalus.scripts.retrieval_diversity \
        appworld/memory-use/qwen3-emb_k5 appworld/memory-use/qwen3-emb_k5_transient \
        appworld/memory-use/qwen3-emb_k5_with-repetition appworld/memory-use/qwen3-emb_k5_no-replacement

    # default: every experiment found under outputs/inference/
    python -m daedalus.scripts.retrieval_diversity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daedalus.core.config import DEFAULT_SETUP, find_run_dir, run_dirs, run_id
from daedalus.core.evaluation.metrics import aggregate_retrieval_diversity, load_traces

TRACES_ROOT = Path("outputs") / "inference"


def _run_dirs(exp_dir: Path) -> list[Path]:
    """Run subdirs (run_0/, run_1/, ...) if present, else the dir itself."""
    runs = sorted(d for d in exp_dir.glob("run_*") if d.is_dir())
    return runs or [exp_dir]


def diversity_for_experiment(name: str) -> dict | None:
    """Aggregate diversity over all runs of one experiment, or None if no traces.

    `name` is a run id (`appworld/memory-use/bm25_k5`, its path under outputs/inference/) or a
    folder name that is unique among the runs (see daedalus.core.config.find_run_dir).
    """
    exp_dir = find_run_dir("inference", DEFAULT_SETUP, name) or TRACES_ROOT / name
    traces = [t for d in _run_dirs(exp_dir) for t in load_traces(d)]
    if not traces:
        return None
    return aggregate_retrieval_diversity(traces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-retrieval turnover/diversity report")
    parser.add_argument(
        "experiments", nargs="*",
        help="Inference run ids under outputs/inference/ (default: all of them)",
    )
    args = parser.parse_args()

    names = args.experiments or sorted(
        run_id("inference", setup, d) for setup, d in run_dirs("inference")
        if setup == DEFAULT_SETUP
    )

    rows = []
    for name in names:
        agg = diversity_for_experiment(name)
        if agg:
            rows.append((name, agg))

    if not rows:
        print("No traces found.")
        return

    header = f"{'run':<52}{'tasks':>6}{'ret_turns':>10}{'unique':>8}{'turnover':>10}{'repeat%':>9}"
    print(header)
    print("-" * len(header))
    for name, a in rows:
        turnover = f"{a['avg_turnover']:.2f}" if a["avg_turnover"] is not None else "  n/a"
        print(
            f"{name:<52}{a['num_tasks']:>6}{a['avg_retrieval_turns']:>10.1f}"
            f"{a['avg_unique_items']:>8.1f}{turnover:>10}{a['avg_repeat_ratio']*100:>8.0f}%"
        )

    out_dir = Path("outputs") / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_diversity.json"
    out_path.write_text(
        json.dumps({name: a for name, a in rows}, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
