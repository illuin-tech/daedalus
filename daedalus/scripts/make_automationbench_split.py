"""Build an AutomationBench train/test split (and, optionally, a smaller eval subset).

AutomationBench ships one flat list of tasks per domain — no train/test partition — so a
memory experiment has to make its own: accumulation mines heuristics from the train side,
inference is evaluated on the test side. The split is written as the JSON
`{"train":[ids], "test":[ids], …}` that `automationbench.split_file` /
`automationbench.split` read.

Sampling is stratified and deterministic. The strata are (domain × breadth), where
breadth is how many distinct services the task's assertions land in, banded as 1, 2 or
3+. Breadth is the axis that separates a one-app edit from a cross-app workflow, so
stratifying on it keeps a small train slice from being all single-app tasks and keeps a
subset of the test side distributionally faithful to the whole. Allocation inside each
stratum uses largest-remainder, so the requested totals are hit exactly.

    # 30/70 train/test over the six scored domains, plus a 100-task eval subset
    uv run python -m daedalus.scripts.make_automationbench_split \
        --train-fraction 0.3 --subset-size 100 \
        --out daedalus/benchmarks/automationbench/splits/public_30_70.json

The subset is written as a third list, `test_100` (named for its size), so one file
serves both the full evaluation and the cheap one; point `automationbench.split` at
whichever you mean.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from daedalus.benchmarks.automationbench.config import PUBLIC_DOMAINS
from daedalus.benchmarks.automationbench.mirrored import compute_allowed_services
from daedalus.core.config import config_from_dict


def load_rows(domains: list[str], repo_root: str | None) -> list[dict[str, Any]]:
    """Load the tasks and label each with the strata keys the split needs."""
    from daedalus.benchmarks.automationbench.task_loader import load_tasks

    cfg = config_from_dict(
        {
            "benchmark": "automationbench",
            "automationbench": {"domains": domains, "repo_root": repo_root},
        }
    )
    rows: list[dict[str, Any]] = []
    for task in load_tasks(cfg).values():
        # The services the task is GRADED on, which is what "how broad is this task"
        # means here — the initial state also seeds apps a task only reads from.
        graded = compute_allowed_services({}, task.assertions, [])
        breadth = len(graded)
        rows.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "num_assertions": len(task.assertions),
                "services": graded,
                "breadth": breadth,
                "band": "1" if breadth <= 1 else ("2" if breadth == 2 else "3+"),
            }
        )
    return rows


def _largest_remainder(sizes: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate `budget` slots across groups proportionally to their size.

    Floor each exact share, then hand the leftover slots to the largest remainders
    (bigger group first on a tie). Never allocates more than a group holds.
    """
    total = sum(sizes.values())
    if total == 0:
        return {k: 0 for k in sizes}
    exact = {k: (n * budget / total) for k, n in sizes.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    leftover = budget - sum(alloc.values())
    order = sorted(sizes, key=lambda k: (-(exact[k] - alloc[k]), -sizes[k], k))
    while leftover > 0:
        progressed = False
        for k in order:
            if leftover <= 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:  # every group is full
            break
    return alloc


def stratified_pick(
    rows: list[dict[str, Any]], budget: int, seed: int, tag: str
) -> list[str]:
    """Pick `budget` task ids, allocated across (domain × band) strata."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"{r['domain']}/{r['band']}"].append(r)
    for key, items in groups.items():
        # Deterministic order inside each stratum before selection.
        random.Random(f"{seed}:{tag}:{key}").shuffle(items)

    alloc = _largest_remainder({k: len(v) for k, v in groups.items()}, budget)
    picked: list[str] = []
    for key, n in alloc.items():
        picked += [r["task_id"] for r in groups[key][:n]]
    return sorted(picked)


def _mix(rows: list[dict[str, Any]], ids: list[str]) -> dict[str, dict[str, int]]:
    """Domain and breadth-band counts for a list of ids, so the file shows its own shape."""
    by_id = {r["task_id"]: r for r in rows}
    domain: dict[str, int] = defaultdict(int)
    band: dict[str, int] = defaultdict(int)
    for i in ids:
        domain[by_id[i]["domain"]] += 1
        band[by_id[i]["band"]] += 1
    return {"by_domain": dict(sorted(domain.items())), "by_band": dict(sorted(band.items()))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--domains",
        nargs="+",
        default=list(PUBLIC_DOMAINS),
        help="domains to split (default: the six scored ones)",
    )
    ap.add_argument("--train-fraction", type=float, default=0.3)
    ap.add_argument(
        "--subset-size",
        type=int,
        default=0,
        help="also emit a stratified subset of the test side, as test_<n> (0 = none)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.domains, args.repo_root)
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be strictly between 0 and 1")

    train = stratified_pick(rows, round(len(rows) * args.train_fraction), args.seed, "train")
    train_set = set(train)
    test_rows = [r for r in rows if r["task_id"] not in train_set]
    test = sorted(r["task_id"] for r in test_rows)

    payload: dict[str, Any] = {
        "benchmark": "automationbench",
        "domains": args.domains,
        "split_by": "domain x assertion-breadth band",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "task_id_scheme": "info['task_name'], e.g. sales.multi_hop_lookup",
        "counts": {"all": len(rows), "train": len(train), "test": len(test)},
        "mix": {"train": _mix(rows, train), "test": _mix(rows, test)},
        "train": train,
        "test": test,
    }

    if args.subset_size:
        if args.subset_size > len(test):
            raise SystemExit(
                f"--subset-size={args.subset_size} exceeds the {len(test)}-task test side"
            )
        subset = stratified_pick(test_rows, args.subset_size, args.seed, "subset")
        name = f"test_{args.subset_size}"
        payload[name] = subset
        payload["counts"][name] = len(subset)
        payload["mix"][name] = _mix(rows, subset)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    c = payload["counts"]
    print(f"train={c['train']} test={c['test']} of {c['all']} tasks")
    for key in ("train", "test"):
        print(f"  {key}: {payload['mix'][key]['by_band']}")
    if args.subset_size:
        key = f"test_{args.subset_size}"
        print(f"  {key}: {payload['mix'][key]['by_domain']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
