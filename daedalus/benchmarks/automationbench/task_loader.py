"""Load and normalize AutomationBench tasks.

A task is one row of `automationbench.domains.get_domain_dataset(<domain>)`: four
columns — `example_id`, `prompt`, `answer`, `info` — where `info` is a JSON *string*
carrying `task_name`, `zapier_tools`, `initial_state` and `assertions`. The domain
module injects its noise records into the initial states as it builds that dataset, so
a task is only well-defined as the loader emits it; never rebuild one by hand.

Ids are `info["task_name"]` (e.g. "sales.multi_hop_lookup"), which is unique across all
800 public tasks and already domain-prefixed. `example_id` is NOT unique across domains
(770 distinct values over 800 tasks), so it is carried for reference only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from daedalus.core.config import ExperimentConfig

from . import ensure_automationbench_importable
from .config import PUBLIC_DOMAINS
from .mirrored import strip_none_values


@dataclass
class Task:
    task_id: str  # info["task_name"], e.g. "sales.multi_hop_lookup"
    domain: str
    example_id: int
    instruction: str  # the trigger message the workflow starts from
    system_prompt: str  # the benchmark's own system message for this task
    initial_state: dict
    assertions: list[dict] = field(default_factory=list)
    zapier_tools: list[str] = field(default_factory=list)
    # The benchmark's own fingerprint of the task inputs a rollout saw. A version bump
    # that changes a task changes this, which is what makes a stale run detectable
    # instead of quietly incomparable.
    contract_sha256: str = ""


def resolve_domains(cfg: ExperimentConfig) -> list[str]:
    """The domains to load; empty config value = the six scored ones."""
    ab = cfg.automationbench
    domains = list(ab.domains) if ab.domains else list(PUBLIC_DOMAINS)
    ensure_automationbench_importable(ab.repo_root)
    from automationbench.domains import get_available_domains

    available = get_available_domains()
    unknown = [d for d in domains if d not in available]
    if unknown:
        raise SystemExit(
            f"Unknown automationbench.domains {unknown} (available: {available})"
        )
    return domains


def load_tasks(cfg: ExperimentConfig) -> dict[str, Task]:
    """Return {task_id: Task}, honoring domains, split_file/split and task_ids."""
    ab = cfg.automationbench
    domains = resolve_domains(cfg)
    from automationbench.domains import get_domain_dataset
    from automationbench.task_contract import task_contract_sha256

    tasks: dict[str, Task] = {}
    for domain in domains:
        for row in get_domain_dataset(domain):
            info = json.loads(row["info"])
            prompt = list(row["prompt"])
            task_id = info["task_name"]
            if task_id in tasks:
                raise SystemExit(
                    f"duplicate AutomationBench task id {task_id!r} across domains "
                    f"{tasks[task_id].domain} and {domain}"
                )
            system = next(
                (m["content"] for m in prompt if m["role"] == "system"), ""
            )
            instruction = next(
                (m["content"] for m in prompt if m["role"] == "user"), ""
            )
            tasks[task_id] = Task(
                task_id=task_id,
                domain=domain,
                example_id=int(row["example_id"]),
                instruction=instruction,
                system_prompt=system,
                initial_state=strip_none_values(info["initial_state"]),
                assertions=[strip_none_values(a) for a in info["assertions"]],
                zapier_tools=list(info["zapier_tools"]),
                contract_sha256=task_contract_sha256(
                    example_id=row["example_id"], prompt=prompt, info=info
                ),
            )

    if ab.split_file and ab.split:
        tasks = _apply_split(tasks, ab.split_file, ab.split)

    if ab.task_ids:
        wanted = set(ab.task_ids)
        unknown = wanted - set(tasks)
        if unknown:
            raise SystemExit(
                f"automationbench.task_ids names {len(unknown)} id(s) absent from "
                f"domains {domains}, e.g. {sorted(unknown)[:3]}"
            )
        tasks = {tid: t for tid, t in tasks.items() if tid in wanted}
    return tasks


def _apply_split(tasks: dict[str, Task], split_file: str, split: str) -> dict[str, Task]:
    """Keep only the ids named by `split` in `split_file`.

    An id in the file that the loaded domains do not carry is an error, not a silent
    drop: it means the split was built over a different set of domains, and running on
    the surviving subset would quietly change the denominator.
    """
    split_path = Path(split_file).expanduser()
    try:
        split_map = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"could not read split_file {split_path}: {e}") from e
    ids = split_map.get(split)
    if not isinstance(ids, list):
        avail = sorted(k for k, v in split_map.items() if isinstance(v, list))
        raise SystemExit(
            f"split_file {split_path} has no split {split!r} (available: {avail})"
        )
    wanted = {str(i) for i in ids}
    unknown = wanted - set(tasks)
    if unknown:
        raise SystemExit(
            f"split_file {split_path} references {len(unknown)} id(s) absent from the "
            f"loaded domains, e.g. {sorted(unknown)[:3]}"
        )
    return {tid: t for tid, t in tasks.items() if tid in wanted}


def list_task_ids(cfg: ExperimentConfig) -> list[str]:
    return list(load_tasks(cfg).keys())
