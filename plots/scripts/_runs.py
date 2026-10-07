"""Reading inference runs — shared by the figure scripts.

A run is one folder under `outputs/inference/<benchmark>/<category>/` (daedalus's own runs) or
under `outputs/baselines/<setup>/inference/` (a baseline method implemented in references/),
selected by its `run_meta.json` (benchmark, domain, model) and scored by its `evaluation.json`.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path

import yaml

from daedalus.core.evaluation.evaluate import pass_hat_k_with_se  # noqa: E402


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def slug(s: str | None) -> str:
    """Filename-safe form of a CLI value ("openrouter/qwen" -> "openrouter-qwen")."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") if s else ""


def pass_hat_k(per_run: list[dict], agg: dict | None = None) -> tuple[dict[int, float],
                                                                     dict[int, float]]:
    """({k: pass^k}, {k: its SE}), int-keyed, exactly as the run browser reports them.

    Recomputed from `per_run[*].task_success` in preference to whatever the run stored, which
    is what `serve.py::_pass_metrics` does and for the same reason: a run scored before the
    variance estimator switched to Bessel (N-1) carries the OLD, narrower SE, so a figure
    trusting the stored field would mix two estimators across its own points. Every
    evaluation.json has always carried the per-task grid, so the current estimator is
    recoverable without re-scoring a single run. `agg` is the fallback for a run whose
    per-run grid is unreadable — value only, no SE.

    The ± is a TASK-sampling SE: the Bessel-corrected sd over the per-task values
    v_i = C(c_i, k)/C(n, k), divided by sqrt(N_tasks). pass^k is one figure computed over all
    n repeats jointly, so there is no across-repeat spread to take — see
    evaluation/evaluate.py::_mean_and_task_se for why no repeat-level estimate exists.
    """
    got = pass_hat_k_with_se(per_run or [])
    table = got.get("pass_hat_k")
    if not table:
        table, se = (agg or {}).get("pass_hat_k") or {}, {}
    else:
        se = got.get("pass_hat_k_se") or {}
    key = lambda s: int(s.split("^")[-1])  # noqa: E731
    return ({key(k): v for k, v in table.items()},
            {key(k): v for k, v in se.items()})


def is_original_prompt(run_dir: Path) -> bool:
    """Is this run a PROMPT control rather than a method arm?

    `agent.original_solver_prompt` swaps daedalus's solver prompt for the benchmark's own, so
    such a run measures our prompting, not our memory. It is memory-off and otherwise looks
    exactly like the baseline, which is how it silently became the baseline in one figure and
    understated a transferability gain (+4.3 pp reported where the real number was +14.8 pp).
    Every figure that picks "the baseline" excludes it through this one predicate.
    """
    return bool(config_block(run_dir, "agent").get("original_solver_prompt"))


def success_se(agg: dict) -> float:
    """The mean success rate's standard error over runs.

    `success_rate_se` since the estimator switch; derived from the stored SD for a run scored
    before it, so every figure shows one quantity across old and new runs.
    """
    se = agg.get("success_rate_se")
    if se is not None:
        return float(se)
    sd, n = agg.get("success_rate_std") or 0.0, agg.get("num_runs") or 0
    return float(sd) / math.sqrt(n) if n > 1 else 0.0


def mean_turns(run_dir: Path) -> float | None:
    """Mean number of solver turns per task-run, over every `run_*/<task>.json` trace."""
    turns = [
        t["num_turns"] for f in sorted(run_dir.glob("run_*/*.json"))
        if (t := load_json(f)) and t.get("num_turns") is not None
    ]
    return st.mean(turns) if turns else None


def config_block(run_dir: Path, key: str) -> dict:
    """One top-level block of the run's own `config.yaml`, or {} when it cannot be read."""
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
    return cfg.get(key) or {}


def memory_block(run_dir: Path) -> dict:
    """The run's own `memory:` config, or {} when it cannot be read."""
    return config_block(run_dir, "memory")


def same_pool(recorded: str | None, wanted: str) -> bool:
    """Is `recorded` (from a run's config) the pool the caller asked for?

    Compared as resolved paths when both exist on disk, so an absolute argument and the
    repo-relative string in the config still match. Falls back to a suffix match, which is
    what makes `--pool <generation folder>/consolidated_pool.json` work.
    """
    if not recorded:
        return False
    a, b = Path(recorded), Path(wanted)
    try:
        if a.exists() and b.exists() and a.resolve() == b.resolve():
            return True
    except OSError:
        pass
    return a.as_posix().endswith(b.as_posix()) or b.as_posix().endswith(a.as_posix())
