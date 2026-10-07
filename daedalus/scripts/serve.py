"""Local web UI for browsing daedalus runs under outputs/.

    uv run python -m daedalus.scripts.serve              # http://127.0.0.1:8000
    uv run python -m daedalus.scripts.serve --port 8080
    uv run python -m daedalus.scripts.serve --report     # print the inference table and exit
    uv run python -m daedalus.scripts.serve --outputs outputs-release   # browse another tree

Four tabs, each a sortable, filterable run table that drills down:

* **Inference** — every scored run under `outputs/inference/<benchmark>/<category>/` (and
  `outputs/baselines/<method>/inference/`): success rate ± SE, pass^k / pass@k, cost; then
  the task × repeat grid; then the turn-by-turn trace with the memories retrieved each turn.
* **DAEDALUS-curated** (accumulation) — per-task attempt timelines and the memory diff after each attempt.
* **DAEDALUS** (generation) — self-play sessions: outcome mix, refinements, guidelines, novelty,
  coverage tags, banked tasks, and each session's explorer transcript and solver retries.
* **Test sets** — models evaluated on a generated test set (`outputs/evaluation-proxy/`), down to each judge verdict.

Cost is the one the paper reports (`core/logging/paper_cost.py::cost_per_run`): every call
at standard-tier prices with each conversation's earlier prompt fully cached, divided by the
number of repeats. A `~` marks a figure reconstructed from incomplete records.

Read-only; everything is recomputed from the output files, cached per run by mtime. Stdlib
plus PyYAML on the server; charts use Plotly.js from a CDN and degrade to a note offline.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yaml

from daedalus.core.config import (
    DEFAULT_SETUP,
    find_run_dir,
    kind_root,
    run_dirs,
    set_outputs_root,
    run_id,
    run_slot,
)
from daedalus.core.evaluation.metrics import aggregate_metrics
from daedalus.core.logging.paper_cost import cost_per_run

# json files in an inference run dir that are not per-task traces
_EXCLUDE_STEMS = {"run_meta", "evaluation", "config", "run_summary", "pool",
                  "pool.log", "consolidated_pool", "tasks", "native_tasks"}


# ── small helpers ───────────────────────────────────────────────────────────

def _success_se(agg: dict) -> float:
    """Standard error of the mean success rate over repeats (derived from the SD if absent)."""
    se = agg.get("success_rate_se")
    if se is not None:
        return float(se)
    sd, n = agg.get("success_rate_std") or 0.0, agg.get("num_runs") or 0
    return float(sd) / math.sqrt(n) if n > 1 else 0.0


_PASS_KEYS = ("pass_hat_k", "pass_hat_k_se", "pass_at_k", "pass_at_k_se")


def _pass_metrics(agg: dict, per_run: list | None) -> dict:
    """pass^k and pass@k with their task-sampling SEs, recomputed from the per-run task grid
    so every run shows the same estimator regardless of when it was scored."""
    out = {k: agg.get(k) or {} for k in _PASS_KEYS}
    try:
        from daedalus.core.evaluation.evaluate import pass_at_k_with_se, pass_hat_k_with_se
        if per_run and len(per_run) > 1 and all("task_success" in r for r in per_run):
            out.update(pass_hat_k_with_se(per_run))
            out.update(pass_at_k_with_se(per_run))
    except Exception:  # noqa: BLE001 — a viewer must not fail over one missing metric
        pass
    return out


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _std(values: list[float], mean: float | None = None) -> float:
    """Sample std (n-1), matching evaluate.py's success-rate std."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    m = mean if mean is not None else sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5


def _reward_breakdown(t: dict[str, Any]) -> dict[str, Any]:
    """τ² per-task reward components (e.g. DB, NL_ASSERTION); {} for other benchmarks."""
    det = (t.get("evaluation") or {}).get("details")
    if isinstance(det, str):
        try:
            det = json.loads(det)
        except (ValueError, TypeError):
            return {}
    return (det.get("reward_breakdown") or {}) if isinstance(det, dict) else {}


def _task_score(t: dict[str, Any]) -> float:
    """Partial success in [0, 1]: AppWorld's test pass ratio, else the mean of τ²'s reward
    components, else a graded reward, else 0/1 success."""
    ev = t.get("evaluation") or {}
    if ev.get("tests_total"):
        return (ev.get("tests_passed") or 0) / ev["tests_total"]
    comps = [float(v) for v in _reward_breakdown(t).values() if isinstance(v, (int, float))]
    if comps:
        return sum(comps) / len(comps)
    try:
        if ev.get("reward") is not None:
            return float(ev["reward"])
    except (TypeError, ValueError):
        pass
    return 1.0 if t.get("success") else 0.0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _run_dirs(exp_dir: Path) -> list[Path]:
    """Per-repeat subdirs (run_0/, run_1/, …) if present, else the flat dir itself."""
    runs = sorted(d for d in exp_dir.glob("run_*") if d.is_dir())
    return runs or [exp_dir]


def _trace_files(d: Path) -> list[Path]:
    return [p for p in sorted(d.glob("*.json")) if p.stem not in _EXCLUDE_STEMS]


def _cost_fields(run_dir: Path) -> dict[str, Any]:
    """The paper's $ per run for one row, with whether it is exact and why not."""
    try:
        usd, info = cost_per_run(run_dir)
    except Exception as exc:  # noqa: BLE001 — a malformed ledger must not 500 the page
        usd, info = None, {"complete": False, "notes": [f"cost unavailable ({exc})"]}
    return {
        "cost_per_run_usd": round(usd, 6) if usd is not None else None,
        "cost_complete": bool(info.get("complete")),
        "cost_notes": list(info.get("notes") or [])[:4],
    }


def _memory_label(cfg: dict[str, Any]) -> str:
    mem = cfg.get("memory") or {}
    if not mem.get("enabled"):
        return "off"
    if mem.get("heuristics_at_start"):
        return "all-at-start"
    return ((mem.get("retriever") or {}).get("type")) or "on"


def _component(s: str | None) -> str | None:
    """Reject path traversal: a name must be a single plain path component."""
    if not s or "/" in s or "\\" in s or ".." in s:
        return None
    return s


def _run_ref(s: str | None) -> str | None:
    """A run id (`appworld/main-results/no-memory`): plain components joined by '/'."""
    if not s or not all(_component(part) for part in s.split("/")):
        return None
    return s


def _run_dir(kind: str, setup: str | None, name: str | None) -> Path | None:
    """The folder of one run, addressed by (setup, run id); None if either is unusable.

    Two setups may hold runs of the same name, and folder names repeat across benchmarks and
    sections, so every route carries the setup and the run's path under its kind root.
    """
    setup, name = _component(setup or DEFAULT_SETUP), _run_ref(name)
    if not setup or not name:
        return None
    d = find_run_dir(kind, setup, name)
    if d is not None:
        return d
    d = kind_root(kind, setup) / name
    return d if d.is_dir() else None


# ── per-run cache (keyed on a cheap dir signature) ───────────────────────────
_CACHE: dict[Any, tuple[Any, Any]] = {}


def _cached(key: Any, sig: Any, fn: Callable[[], Any]) -> Any:
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    val = fn()
    _CACHE[key] = (sig, val)
    return val


def _ledger_sig(d: Path) -> tuple:
    """One mtime per usage-ledger launch dir, so a resumed run's new spend is picked up."""
    root = d / "usage"
    if not root.is_dir():
        return ()
    return tuple(sorted((p.name, _mtime(p)) for p in root.iterdir() if p.is_dir()))


def _inf_sig(d: Path) -> tuple:
    return (_mtime(d), _mtime(d / "evaluation.json"),
            tuple(_mtime(x) for x in sorted(d.glob("run_*"))), _ledger_sig(d),
            _mtime(d / "retrieval"), _mtime(d / "embedding"))


def _acc_sig(d: Path) -> tuple:
    return (_mtime(d), _mtime(d / "task_summaries"), _mtime(d / "pool.json"),
            _mtime(d / "traces"), _ledger_sig(d))


def _gen_sig(d: Path) -> tuple:
    return (_mtime(d), _mtime(d / "sessions"), _mtime(d / "tasks.json"),
            _mtime(d / "traces"), _mtime(d / "run_summary.json"), _ledger_sig(d))


def _gather(kind: str, sig: Callable[[Path], tuple],
            collect: Callable[[Path, str], dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Every run of `kind` across every setup (daedalus + baselines/<method>), newest first."""
    out = []
    for setup, d in run_dirs(kind):
        r = _cached((kind, str(d)), sig(d), lambda d=d, s=setup: collect(d, s))
        if r:
            out.append(r)
    out.sort(key=lambda r: r["timestamp"], reverse=True)
    return out


# ── inference ────────────────────────────────────────────────────────────────
def collect_inference_run(exp_dir: Path, setup: str = DEFAULT_SETUP) -> dict[str, Any] | None:
    """One inference run's summary row, or None if not scored yet."""
    evaluation = _load_json(exp_dir / "evaluation.json")
    if not evaluation:
        return None
    meta = _load_json(exp_dir / "run_meta.json") or {}
    cfg = _load_yaml(exp_dir / "config.yaml")
    agg = evaluation.get("aggregate", {})

    per_run_traces = [
        [t for p in _trace_files(rd)
         if isinstance((t := _load_json(p)), dict) and "task_id" in t]
        for rd in _run_dirs(exp_dir)
    ]
    tm = aggregate_metrics([t for lst in per_run_traces for t in lst])
    run_scores = [s for lst in per_run_traces
                  if (s := _mean([_task_score(t) for t in lst])) is not None]
    score_mean = _mean(run_scores)

    return {
        "setup": setup,
        "name": exp_dir.name,
        "id": run_id("inference", setup, exp_dir),
        "category": run_slot("inference", setup, exp_dir)[1],
        "benchmark": meta.get("benchmark") or evaluation.get("benchmark") or "",
        "domain": meta.get("domain") or "",
        "model": meta.get("model") or "",
        "memory": _memory_label(cfg),
        "tasks": agg.get("num_tasks_per_run", tm.get("num_tasks", 0)),
        "runs": agg.get("num_runs", 1),
        "success_mean": agg.get("success_rate_mean"),
        "success_se": _success_se(agg),
        "score_mean": score_mean,
        "score_std": _std(run_scores, score_mean),
        **_pass_metrics(agg, evaluation.get("per_run")),
        "avg_turns": tm.get("avg_turns"),
        **_cost_fields(exp_dir),
        "timestamp": meta.get("timestamp") or "",
    }


def gather_inference_runs() -> list[dict[str, Any]]:
    return _gather("inference", _inf_sig, collect_inference_run)


def _inference_run_detail(d: Path, setup: str) -> dict[str, Any] | None:
    evaluation = _load_json(d / "evaluation.json")
    if not evaluation:
        return None
    meta = _load_json(d / "run_meta.json") or {}
    cfg = _load_yaml(d / "config.yaml")
    agg = evaluation.get("aggregate", {})

    rundirs = _run_dirs(d)
    run_names = [rd.name if rd != d else "." for rd in rundirs]
    tasks: dict[str, dict[str, Any]] = {}
    components: dict[str, list[float]] = {}  # τ² reward parts → values across task-runs
    for i, rd in enumerate(rundirs):
        for p in _trace_files(rd):
            t = _load_json(p)
            if not isinstance(t, dict) or "task_id" not in t:
                continue
            tid = str(t.get("task_id"))
            tasks.setdefault(tid, {"task_id": tid, "runs": []})["runs"].append({
                "run": run_names[i],
                "success": bool(t.get("success")),
                "turns": t.get("num_turns"),
                "score": _task_score(t),
                "term": (t.get("extra") or {}).get("termination_reason"),
            })
            for k, v in _reward_breakdown(t).items():
                if isinstance(v, (int, float)):
                    components.setdefault(k.split(".", 1)[-1], []).append(float(v))

    rows = []
    for tid, rec in tasks.items():
        rs = rec["runs"]
        rows.append({
            "task_id": tid, "runs": rs, "n": len(rs),
            "n_pass": sum(1 for r in rs if r["success"]),
            "score": _mean([r["score"] for r in rs]),
            "avg_turns": _mean([r["turns"] for r in rs]),
            "term": next((r["term"] for r in rs if r["term"]), ""),
        })
    rows.sort(key=lambda r: (r["n_pass"], r["task_id"]))
    consistency = Counter(r["n_pass"] for r in rows)

    return {
        "setup": setup, "name": d.name, "id": run_id("inference", setup, d),
        "benchmark": meta.get("benchmark", ""),
        "category": run_slot("inference", setup, d)[1],
        "domain": meta.get("domain", ""), "model": meta.get("model", ""),
        "memory": _memory_label(cfg), "timestamp": meta.get("timestamp", ""),
        "aggregate": agg, "run_names": run_names,
        "success_per_run": agg.get("success_rate_per_run") or [],
        **_pass_metrics(agg, evaluation.get("per_run")),
        "reward_components": {k: _mean(v) for k, v in sorted(components.items())},
        "num_runs": len(rundirs), "num_tasks": len(rows),
        "consistency": [{"passes": k, "count": consistency[k]} for k in sorted(consistency)],
        "rows": rows,
        **_cost_fields(d),
    }


def inference_run_detail(setup: str | None, name: str | None) -> dict[str, Any] | None:
    d = _run_dir("inference", setup, name)
    if d is None:
        return None
    return _cached(("inf_run", str(d)), _inf_sig(d),
                   lambda: _inference_run_detail(d, setup or DEFAULT_SETUP))


def inference_task_detail(setup: str | None, name: str | None, run: str | None,
                          task: str | None) -> dict[str, Any] | None:
    base = _run_dir("inference", setup, name)
    task = _component(task)
    if base is None or not task:
        return None
    if run and run != ".":
        run = _component(run)
        if not run:
            return None
        base = base / run
    return _load_json(base / f"{task}.json")


# ── accumulation ─────────────────────────────────────────────────────────────
def _acc_summaries(d: Path) -> list[dict[str, Any]]:
    return [s for p in sorted((d / "task_summaries").glob("*.json"))
            if isinstance((s := _load_json(p)), dict)]


def _fail_before_first_success(attempts: list[dict]) -> int:
    n = 0
    for a in attempts:
        if a.get("outcome") != "failure":
            break
        n += 1
    return n


def _is_banked(s: dict[str, Any]) -> bool:
    """Whether a task's heuristic entered the pool: it has one and the task was solved."""
    return bool(s.get("final_memory")) and s.get("final_outcome") == "success"


def _pool_size(d: Path) -> int | None:
    pool = _load_json(d / "pool.json")
    return len(pool.get("items") or []) if isinstance(pool, dict) else None


def collect_accumulation_run(d: Path, setup: str = DEFAULT_SETUP) -> dict[str, Any] | None:
    meta = _load_json(d / "run_meta.json") or {}
    sums = _acc_summaries(d)
    if not sums and not meta:
        return None
    attempts = [len(s.get("attempts") or []) for s in sums]
    return {
        "setup": setup, "name": d.name, "id": run_id("accumulation", setup, d),
        "benchmark": meta.get("benchmark", ""),
        "domain": meta.get("domain", ""), "model": meta.get("model", ""),
        "tasks": len(sums),
        "solved": sum(1 for s in sums if s.get("final_outcome") == "success"),
        "gave_up": sum(1 for s in sums if s.get("final_outcome") == "failure"),
        # pool.json is what inference reads; reference methods build it by their own rules
        "pool": _pool_size(d),
        "avg_attempts": _mean([float(a) for a in attempts]),
        **_cost_fields(d),
        "timestamp": meta.get("timestamp", ""),
    }


def gather_accumulation_runs() -> list[dict[str, Any]]:
    return _gather("accumulation", _acc_sig, collect_accumulation_run)


def _accumulation_run_detail(d: Path, setup: str) -> dict[str, Any] | None:
    meta = _load_json(d / "run_meta.json") or {}
    acc = _load_yaml(d / "config.yaml").get("accumulation") or {}
    rows, growth = [], []
    for s in _acc_summaries(d):
        attempts = s.get("attempts") or []
        banked = _is_banked(s)
        rows.append({
            "task_id": str(s.get("task_id", "?")),
            "outcome": s.get("final_outcome", "?"),
            "banked": banked,
            "attempts": len(attempts),
            "failures": sum(1 for a in attempts if a.get("outcome") == "failure"),
            "max_consec": max((a.get("consecutive_successes", 0) for a in attempts),
                              default=0),
            "fbf": _fail_before_first_success(attempts),
            "solved_wo_mem": bool(s.get("solved_without_memory")),
        })
        growth.append((growth[-1] if growth else 0) + banked)

    banked_rows = [r for r in rows if r["banked"]]
    return {
        "setup": setup, "name": d.name, "id": run_id("accumulation", setup, d),
        "benchmark": meta.get("benchmark", ""),
        "domain": meta.get("domain", ""), "model": meta.get("model", ""),
        "timestamp": meta.get("timestamp", ""),
        "extraction_model": acc.get("extraction_model", ""),
        "num_success_to_continue": acc.get("num_success_to_continue"),
        "max_failures": acc.get("max_failures"),
        "tasks": len(rows),
        "solved": sum(1 for r in rows if r["outcome"] == "success"),
        "banked": len(banked_rows),
        "pool": _pool_size(d),
        "gave_up": sum(1 for r in rows if r["outcome"] == "failure"),
        "solved_wo_mem": sum(1 for r in rows if r["solved_wo_mem"]),
        "attempts_to_bank": [r["attempts"] for r in banked_rows],
        "pool_growth": growth,
        "rows": rows,
        **_cost_fields(d),
    }


def accumulation_run_detail(setup: str | None, name: str | None) -> dict[str, Any] | None:
    d = _run_dir("accumulation", setup, name)
    if d is None:
        return None
    return _cached(("acc_run", str(d)), _acc_sig(d),
                   lambda: _accumulation_run_detail(d, setup or DEFAULT_SETUP))


def accumulation_task_detail(setup: str | None, name: str | None,
                             task: str | None) -> dict[str, Any] | None:
    d = _run_dir("accumulation", setup, name)
    task = _component(task)
    if d is None or not task:
        return None
    return _load_json(d / "task_summaries" / f"{task}.json")


# ── generation ───────────────────────────────────────────────────────────────
def _leading_wasted(transcript: list[dict]) -> int:
    """Explorer turns wasted before the first productive one (rejected spec / no code)."""
    n = 0
    for e in transcript:
        if e.get("kind") not in ("rejected_spec", "rejected", "no_code"):
            break
        n += 1
    return n


def _session_compact(rec: dict, stem: str) -> dict[str, Any]:
    expl = rec.get("explorer") or {}
    tr = expl.get("transcript") or []
    attempts = (rec.get("accumulation_summary") or {}).get("attempts") or []
    traj = rec.get("trajectory") or []  # every spec the session classified, in order
    # difficulty refinements, plus near-duplicate revisions recorded by some runs
    refines = (rec.get("refinements") or []) + (rec.get("novelty_refinements") or [])
    initial_turns = expl.get("num_turns") or len(tr)
    return {
        "idx": rec.get("session_index"),
        "file": stem,
        "result": rec.get("result") or rec.get("kind") or "unknown",
        "explorer_turns": initial_turns + sum(r.get("num_turns") or 0 for r in refines),
        "explorer_turns_initial": initial_turns,
        "wasted": _leading_wasted(tr),
        "too_easy_specs": sum(1 for e in traj if e.get("result") == "too_easy"),
        "too_hard_specs": sum(1 for e in traj if e.get("result") == "too_hard"),
        "n_refine": len(refines),
        "novelty": rec.get("novelty"),
        "guidelines_after": len(rec.get("guidelines_after") or []),
        "n_failures": sum(1 for a in attempts if a.get("outcome") == "failure"),
    }


def _gen_sessions_compact(d: Path) -> list[dict[str, Any]]:
    out = [_session_compact(rec, p.stem)
           for p in sorted((d / "sessions").glob("session_*.json"))
           if isinstance((rec := _load_json(p)), dict)]
    out.sort(key=lambda c: (c["idx"] if c["idx"] is not None else 1 << 30))
    return out


def _gen_compact_cached(d: Path) -> list[dict[str, Any]]:
    return _cached(("gen_comp", str(d)), _gen_sig(d), lambda: _gen_sessions_compact(d))


# session records that are not solver sessions: crashes and bookkeeping entries
_NON_SESSION = {"session_error", "survey", "coverage_survey", "discover", "plan", "unknown"}


def _gen_real_sessions(comp: list[dict]) -> list[dict]:
    return [c for c in comp
            if isinstance(c.get("idx"), int) and c["result"] not in _NON_SESSION]


def _tag_coverage(bank: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-tag realized-vs-target coverage of a generation bank ([] when not tagged)."""
    from daedalus.core.generation import coverage

    goal = coverage.CoverageGoal.from_dict(
        {"report": bank.get("coverage_goal") or "", "tags": bank.get("coverage_tags") or []}
    )
    return coverage.realized(goal, bank.get("tag_counts") or {}, len(bank.get("tasks") or []))


def _tag_gap(rows: list[dict[str, Any]]) -> float | None:
    """Mean |realized share − target share| over the tags (None when not tagged)."""
    if not rows:
        return None
    return sum(abs(r["share"] - r["target_share"]) for r in rows) / len(rows)


def _gen_yield(oc: Counter, n: int) -> float | None:
    """Banked share of the sessions that produced a novel task."""
    reached = n - oc.get("near_duplicate", 0) - oc.get("no_task", 0) - oc.get("excluded_app", 0)
    return oc.get("banked", 0) / reached if reached else None


def collect_generation_run(d: Path, setup: str = DEFAULT_SETUP) -> dict[str, Any] | None:
    meta = _load_json(d / "run_meta.json") or {}
    summary = _load_json(d / "run_summary.json") or {}
    bank = _load_json(d / "tasks.json") or {}
    comp = _gen_compact_cached(d)
    if not comp and not meta and not bank:
        return None
    sessions = _gen_real_sessions(comp)
    oc = Counter(c["result"] for c in sessions)
    return {
        "setup": setup, "name": d.name, "id": run_id("generation", setup, d),
        "benchmark": meta.get("benchmark", ""),
        "domain": meta.get("domain", ""), "model": meta.get("model", ""),
        "sessions": len(sessions),
        "errored": len(comp) - len(sessions),
        "banked": oc.get("banked", 0),
        # sessions whose final spec was off-target, and spec-level totals over refinements
        "bad_difficulty": oc.get("too_easy", 0) + oc.get("too_hard", 0),
        "too_easy": sum(c["too_easy_specs"] for c in sessions),
        "too_hard": sum(c["too_hard_specs"] for c in sessions),
        "near_dup": oc.get("near_duplicate", 0),
        "yield": _gen_yield(oc, len(sessions)),
        "guidelines": len(bank.get("guidelines") or []),
        "heuristics": summary.get("num_heuristics") or _pool_size(d),
        "coverage_tags": len(bank.get("coverage_tags") or []),
        "tag_gap": _tag_gap(_tag_coverage(bank)),
        **_cost_fields(d),
        "timestamp": meta.get("timestamp", ""),
    }


def gather_generation_runs() -> list[dict[str, Any]]:
    return _gather("generation", _gen_sig, collect_generation_run)


def _generation_run_detail(d: Path, setup: str) -> dict[str, Any] | None:
    meta = _load_json(d / "run_meta.json") or {}
    cfg = _load_yaml(d / "config.yaml")
    gen = cfg.get("generation") or {}
    acc = cfg.get("accumulation") or {}
    summary = _load_json(d / "run_summary.json") or {}
    bank = _load_json(d / "tasks.json") or {}
    comp = _gen_compact_cached(d)
    sessions = _gen_real_sessions(comp)
    banked_tasks = bank.get("tasks") or []
    oc = Counter(c["result"] for c in sessions)

    return {
        "setup": setup, "name": d.name, "id": run_id("generation", setup, d),
        "benchmark": meta.get("benchmark", ""),
        "domain": meta.get("domain", ""), "model": meta.get("model", ""),
        "timestamp": meta.get("timestamp", ""),
        "num_explorers": gen.get("num_explorers"),
        "max_refinements": gen.get("max_refinements"),
        "min_novelty": gen.get("min_novelty"),
        "num_success_to_continue": acc.get("num_success_to_continue"),
        "max_failures": acc.get("max_failures"),
        "sessions": sessions,
        "n_errors": len(comp) - len(sessions),
        "outcomes": dict(oc),
        "yield": _gen_yield(oc, len(sessions)),
        "guidelines": len(bank.get("guidelines") or []),
        "heuristics": summary.get("num_heuristics") or _pool_size(d),
        **_cost_fields(d),
        "bank": {
            "path_lens": [len(t.get("path") or []) for t in banked_tasks],
            "tool_freq": Counter(
                tok for t in banked_tasks for tok in (t.get("path") or [])
            ).most_common(14),
            "tasks": [
                {
                    "i": i,
                    "task": t.get("task") or "",
                    "tags": t.get("tags") or [],
                    "path": t.get("path") or [],
                    "novelty": t.get("novelty"),
                    "n_refine": t.get("refinements") or 0,  # the bank stores a count
                }
                for i, t in enumerate(banked_tasks)
            ],
            "tag_coverage": _tag_coverage(bank),
        },
        "coverage_goal": bank.get("coverage_goal") or "",
    }


def generation_run_detail(setup: str | None, name: str | None) -> dict[str, Any] | None:
    d = _run_dir("generation", setup, name)
    if d is None:
        return None
    return _cached(("gen_run", str(d)), _gen_sig(d),
                   lambda: _generation_run_detail(d, setup or DEFAULT_SETUP))


def generation_session_detail(setup: str | None, name: str | None,
                              session: str | None) -> dict[str, Any] | None:
    d = _run_dir("generation", setup, name)
    session = _component(session)
    if d is None or not session:
        return None
    return _load_json(d / "sessions" / f"{session}.json")


# ── generated test sets (outputs/evaluation-proxy/<test set>/<model>/) ────────────────
def _testset_dir(ts: str | None, model: str | None) -> Path | None:
    ts, model = _component(ts), _component(model)
    if not ts or not model:
        return None
    d = kind_root("testset") / ts / model
    return d if d.is_dir() else None


def _judgements(d: Path) -> list[dict[str, Any]]:
    """Every per-task judgement of every repeat."""
    return [r for run_dir in sorted((d / "judgements").glob("run_*"))
            for path in sorted(run_dir.glob("*.json"))
            if isinstance((r := _load_json(path)), dict)]


def collect_testset_run(ts: str, d: Path) -> dict[str, Any] | None:
    meta = _load_json(d / "run_meta.json") or {}
    ev = _load_json(d / "evaluation.json") or {}
    agg = ev.get("aggregate") or {}
    spec = ev.get("test_set") or {}
    if not meta and not agg:
        return None
    return {
        "testset": ts, "model_dir": d.name,
        "benchmark": meta.get("benchmark", ""), "model": meta.get("model", ""),
        "judge_model": spec.get("judge_model", ""),
        "tasks": agg.get("num_tasks_per_run") or spec.get("num_tasks"),
        "runs": agg.get("num_runs"),
        "success_mean": agg.get("success_rate_mean"), "success_se": _success_se(agg),
        "partial_mean": agg.get("partial_rate_mean"),
        **_pass_metrics(agg, ev.get("per_run")),
        "unparsed": agg.get("num_unparsed") or 0,
        "avg_turns": agg.get("avg_turns"),
        **_cost_fields(d),
        "timestamp": meta.get("timestamp", ""),
    }


def gather_testset_runs() -> list[dict[str, Any]]:
    """Every (test set, model) pair on disk."""
    root = kind_root("testset")
    if not root.is_dir():
        return []
    out = []
    for ts_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in ts_dir.iterdir() if p.is_dir()):
            row = _cached(("testset", str(model_dir)), _inf_sig(model_dir),
                          lambda t=ts_dir.name, m=model_dir: collect_testset_run(t, m))
            if row:
                out.append(row)
    return out


def testset_run_detail(ts: str | None, model: str | None) -> dict[str, Any] | None:
    """One model's results on one test set, per task across repeats."""
    d = _testset_dir(ts, model)
    if d is None:
        return None
    ev = _load_json(d / "evaluation.json") or {}
    meta = _load_json(d / "run_meta.json") or {}
    by_task: dict[str, dict[str, Any]] = {}
    for record in _judgements(d):
        tid = record.get("tid")
        if not tid:
            continue
        row = by_task.setdefault(tid, {
            "tid": tid, "sandbox_id": record.get("sandbox_id", ""),
            "task": record.get("instruction", ""),
            "num_conditions": record.get("num_conditions"),
            "runs": 0, "successes": 0, "partials": [], "turns": [],
        })
        row["runs"] += 1
        row["successes"] += 1 if record.get("success") else 0
        row["partials"].append(float(record.get("partial") or 0.0))
        if record.get("num_turns") is not None:
            row["turns"].append(float(record["num_turns"]))
    rows = [{
        **{k: v for k, v in row.items() if k not in ("partials", "turns")},
        "partial": _mean(row["partials"]),
        "avg_turns": _mean(row["turns"]),
        "pass_rate": (row["successes"] / row["runs"]) if row["runs"] else None,
    } for row in by_task.values()]
    rows.sort(key=lambda r: (r["pass_rate"] if r["pass_rate"] is not None else 0, r["tid"]))
    return {
        "testset": ts, "model_dir": model, "model": meta.get("model", ""),
        "benchmark": meta.get("benchmark", ""), "timestamp": meta.get("timestamp", ""),
        "aggregate": ev.get("aggregate") or {}, "per_run": ev.get("per_run") or [],
        "test_set": ev.get("test_set") or {},
        "condition_failures": ev.get("condition_failures") or {},
        "rows": rows,
        **_cost_fields(d),
    }


def testset_task_detail(ts: str | None, model: str | None,
                        task: str | None) -> dict[str, Any] | None:
    """One task: its conditions, every repeat's verdict, and the first repeat's trace."""
    d = _testset_dir(ts, model)
    task = _component(task)
    if d is None or not task:
        return None
    records = sorted((r for r in _judgements(d) if r.get("tid") == task),
                     key=lambda r: r.get("run_idx", 0))
    if not records:
        return None
    head = records[0]
    trace = None
    for run_dir in sorted(d.glob("run_*")):
        hit = next(iter(run_dir.glob(f"*_{task}.json")), None)
        if hit is not None:
            trace = _load_json(hit)
            break
    return {
        "testset": ts, "model_dir": model, "tid": task,
        "sandbox_id": head.get("sandbox_id", ""),
        "instruction": head.get("instruction", ""),
        "success_conditions": head.get("success_conditions") or [],
        "tags": head.get("tags") or [],
        "judgements": records,
        "trace": trace,
    }


# ── page (HTML shell + inline CSS + JS app) ──────────────────────────────────
_CSS = r"""
  :root { color-scheme: light dark;
    --bg:#fff; --fg:#141414; --muted:#6b7280; --line:#e5e7eb; --head:#f6f7f9;
    --row:#fafafa; --card:#fbfbfc; --accent:#2563eb;
    --ok:#16a34a; --bad:#dc2626; --amber:#d97706; --purple:#7c3aed; --teal:#0d9488; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1115; --fg:#e7e7e7; --muted:#9aa0aa; --line:#262a31; --head:#171a21;
      --row:#141821; --card:#141821; --accent:#6ea8fe;
      --ok:#4ade80; --bad:#f87171; --amber:#fbbf24; --purple:#c084fc; --teal:#2dd4bf; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  a { color:var(--accent); text-decoration:none; }
  #top { position:sticky; top:0; z-index:6; background:var(--bg); }
  header { display:flex; gap:14px; align-items:center; padding:12px 20px;
    border-bottom:1px solid var(--line); }
  h1 { font-size:15px; margin:0; font-weight:700; letter-spacing:.2px; }
  h3 { font-size:13px; margin:22px 0 8px; font-weight:650; text-transform:uppercase;
    letter-spacing:.6px; color:var(--muted); }
  h4 { font-size:12px; margin:0 0 8px; font-weight:600; color:var(--muted); }
  .muted { color:var(--muted); }
  /* a cost reconstructed from incomplete records (the cell also prefixes a ~) */
  .partial { border-bottom:1px dotted var(--muted); cursor:help; }
  .tabs { display:flex; gap:4px; }
  .tab { padding:6px 14px; border:1px solid transparent; border-radius:8px; cursor:pointer;
    font-size:13px; font-weight:550; color:var(--muted); background:none; }
  .tab:hover { color:var(--fg); }
  .tab.on { color:var(--fg); background:var(--head); border-color:var(--line); }
  button { font:inherit; }
  .btn { padding:5px 11px; border:1px solid var(--line); border-radius:8px;
    background:var(--head); color:var(--fg); cursor:pointer; font-size:13px; }
  .btn:hover { border-color:var(--accent); }
  input.filter { padding:6px 10px; border:1px solid var(--line); border-radius:8px;
    background:var(--bg); color:var(--fg); font-size:13px; min-width:240px; }
  #crumb { padding:8px 20px; font-size:13px; border-bottom:1px solid var(--line);
    display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #crumb .cr { cursor:pointer; color:var(--accent); }
  #crumb .cr:last-child { color:var(--fg); cursor:default; }
  #crumb .sep { color:var(--muted); }
  #app { padding:6px 20px 60px; }
  .listbar { display:flex; gap:8px; align-items:center; padding:10px 0; flex-wrap:wrap; }
  .facets { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  /* one multi-select dropdown per string column; <details> holds the open state */
  .fac { position:relative; }
  .fac > summary { list-style:none; cursor:pointer; user-select:none; font-size:13px;
    padding:6px 10px; border:1px solid var(--line); border-radius:8px; background:var(--head);
    color:var(--muted); white-space:nowrap; display:flex; gap:6px; align-items:center; }
  .fac > summary::-webkit-details-marker { display:none; }
  .fac > summary::after { content:"▾"; font-size:10px; opacity:.6; }
  .fac > summary:hover { border-color:var(--accent); }
  .fac[open] > summary { border-color:var(--accent); }
  /* an active filter is legible without opening it: value in the chip, accent border */
  .fac.on > summary { color:var(--fg); border-color:var(--accent);
    box-shadow:inset 0 0 0 1px var(--accent); }
  .fac > summary b { font-weight:600; color:var(--fg); }
  .fmenu { position:absolute; z-index:30; top:calc(100% + 4px); left:0; min-width:170px;
    max-height:min(320px,60vh); overflow:auto; padding:6px; background:var(--bg);
    border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.18); }
  .fmenu .fhead { display:flex; gap:8px; padding:2px 6px 6px; border-bottom:1px solid var(--line);
    margin-bottom:4px; font-size:12px; }
  .fmenu .fhead span { cursor:pointer; color:var(--accent); }
  .facet { display:flex; gap:7px; align-items:center; font-size:13px; cursor:pointer;
    user-select:none; padding:4px 6px; border-radius:6px; white-space:nowrap; }
  .facet:hover { background:var(--head); }
  .facet input { cursor:pointer; margin:0; }
  .facet .fn { margin-left:auto; color:var(--muted); font-size:11px; }
  .fclear { font-size:12px; color:var(--accent); cursor:pointer; }
  /* own vertical scroller so the sticky header pins to the table top, not the window */
  .twrap { overflow:auto; max-height:calc(100vh - 150px); }
  table { border-collapse:collapse; width:100%; white-space:nowrap; }
  th,td { text-align:right; padding:6px 12px; border-bottom:1px solid var(--line); }
  th.l,td.l { text-align:left; }
  th { position:sticky; top:0; background:var(--head); cursor:pointer; font-weight:600;
    user-select:none; box-shadow:inset 0 -1px 0 var(--line); z-index:1; }
  th:hover { color:var(--accent); }
  th.sorted::after { content:" ▾"; color:var(--accent); }
  th.sorted.asc::after { content:" ▴"; }
  tbody tr:nth-child(even) { background:var(--row); }
  tbody tr:hover { background:var(--head); }
  td.name { font-weight:600; }
  .bar { display:inline-block; height:8px; border-radius:4px; background:var(--accent);
    vertical-align:middle; margin-left:6px; opacity:.55; }
  .empty { padding:48px 8px; color:var(--muted); }
  .note { margin:6px 0 2px; color:var(--muted); font-size:11px; line-height:1.5; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
  .chip { font-size:12px; padding:3px 9px; border:1px solid var(--line); border-radius:999px;
    background:var(--head); }
  .chip b { color:var(--muted); font-weight:600; margin-right:4px; }
  .tiles { display:flex; flex-wrap:wrap; gap:10px; margin:12px 0; }
  .tile { border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:96px;
    background:var(--card); }
  .tile .tv { font-size:20px; font-weight:700; }
  .tile .tl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.4px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:12px 14px; background:var(--card); }
  .plot { width:100%; min-height:120px; }
  .badge { display:inline-block; padding:1px 7px; border-radius:6px; font-size:11px; font-weight:600;
    border:1px solid var(--line); margin:1px; }
  .badge.ok { color:var(--ok); border-color:var(--ok); }
  .badge.bad { color:var(--bad); border-color:var(--bad); }
  .badge.muted { color:var(--muted); }
  .timeline { display:flex; flex-wrap:wrap; gap:3px; margin:8px 0; }
  .dot { width:20px; height:20px; border-radius:5px; display:flex; align-items:center;
    justify-content:center; font-size:10px; font-weight:700; color:#fff; }
  .dot.ok { background:var(--ok); }
  .dot.bad { background:var(--bad); }
  .section { margin-top:6px; }
  .turn,.attempt,.tr,.refine { border:1px solid var(--line); border-radius:8px; padding:8px 10px;
    margin:8px 0; background:var(--card); }
  .turn-h,.attempt-h { font-weight:600; font-size:12px; margin-bottom:4px; }
  .thought { font-style:italic; color:var(--muted); margin:4px 0; white-space:pre-wrap; }
  .reason { margin:4px 0; white-space:pre-wrap; }
  pre.code,pre.out,pre.diff { margin:4px 0; padding:8px 10px; border-radius:6px; overflow-x:auto;
    background:var(--head); font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
    white-space:pre-wrap; word-break:break-word; }
  pre.out.err { border-left:3px solid var(--bad); }
  .mono { font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin-top:4px;
    overflow-wrap:anywhere; }
  pre.diff .add { color:var(--ok); }
  pre.diff .del { color:var(--bad); }
  pre.diff .hunk { color:var(--muted); }
  .mem { border-left:3px solid var(--accent); padding:4px 10px; margin:4px 0; background:var(--head);
    border-radius:0 6px 6px 0; }
  .mem-list { margin:4px 0; padding-left:18px; }
  ul.mem-list li { margin:2px 0; }
  details { margin:4px 0; }
  summary { cursor:pointer; color:var(--muted); font-size:12px; }
  .kbadge { display:inline-block; padding:1px 6px; border-radius:5px; font-size:11px; font-weight:600;
    color:#fff; margin-right:6px; background:var(--muted); }
  .k-read_db { background:var(--accent); }
  .k-tool,.k-exec { background:var(--teal); }
  .k-submit { background:var(--ok); }
  .k-rejected_spec,.k-rejected,.k-error,.k-blocked { background:var(--bad); }
  .k-no_code { background:var(--amber); }
  .k-reset { background:var(--muted); }
  .tr.err { border-color:var(--bad); }
  .nodiff { color:var(--muted); font-size:12px; }
"""

_SHELL = r"""
<div id="top">
  <header>
    <h1>daedalus</h1>
    <div class="tabs" id="tabs">
      <button class="tab" data-t="inference">inference</button>
      <button class="tab" data-t="accumulation">accumulation</button>
      <button class="tab" data-t="generation">generation</button>
      <button class="tab" data-t="testset">test sets</button>
    </div>
    <span style="flex:1"></span>
    <button class="btn" id="refresh">Refresh</button>
  </header>
  <div id="crumb"></div>
</div>
<div id="app"></div>
"""

_JS_CORE = r"""
const app=document.getElementById('app'), crumbEl=document.getElementById('crumb');
const S={tab:'inference'};

// formatters
const pct=(v,d=1)=>v==null?'':(100*v).toFixed(d)+'%';
const n1=v=>v==null?'':(+v).toFixed(1);
const n3=v=>v==null?'':(+v).toFixed(3);
const usd=v=>v==null?'':'$'+(+v).toFixed(4);
const when=v=>v?String(v).slice(0,16).replace('T',' '):'';
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtMax=v=>v==null?'':(v>=1000?(v/1000).toFixed(1)+'k':(Number.isInteger(+v)?String(v):(+v).toFixed(2)));
const avg=a=>{const x=a.filter(v=>v!=null);return x.length?x.reduce((s,v)=>s+(+v),0)/x.length:null;};

async function api(p){ const r=await fetch(p); try{ return await r.json(); }catch(e){ return null; } }

// A run is addressed by the pair (setup, id): daedalus runs live in outputs/<kind dir>/, a
// baseline's in outputs/baselines/<setup>/<kind dir>/; the id is the run's path there.
function runQ(setup,name){
  return 'setup='+encodeURIComponent(setup||'daedalus')+'&name='+encodeURIComponent(name); }
function runLabel(setup,name){ return (setup&&setup!=='daedalus')?setup+' / '+name:name; }

// outcome colors (generation / accumulation)
const OC={banked:'var(--ok)',too_easy:'var(--amber)',too_hard:'var(--bad)',
  near_duplicate:'var(--muted)',no_task:'var(--purple)',invalid_spec:'var(--purple)',
  excluded_app:'var(--muted)',success:'var(--ok)',failure:'var(--bad)'};
const ocColor=r=>OC[r]||'var(--teal)';

// ── layout helpers ──
function tiles(items){ return '<div class="tiles">'+items.filter(t=>t.value!=null&&t.value!=='')
  .map(t=>`<div class="tile"><div class="tv">${esc(t.value)}</div><div class="tl">${esc(t.label)}</div></div>`).join('')+'</div>'; }
function chips(items){ return '<div class="chips">'+items.filter(c=>c&&c.v!=null&&c.v!=='')
  .map(c=>`<span class="chip"><b>${esc(c.k)}</b>${esc(c.v)}</span>`).join('')+'</div>'; }
function card(title,inner){ return `<div class="card"><h4>${esc(title)}</h4>${inner}</div>`; }
function grid(cards){ return '<div class="grid">'+cards.filter(Boolean).join('')+'</div>'; }
function section(title,inner){ return `<div class="section"><h3>${esc(title)}</h3>${inner}</div>`; }
function longtext(txt,label){ const t=String(txt==null?'':txt);
  if(t.length<=700) return `<pre class="out">${esc(t)}</pre>`;
  return `<details><summary>${esc(label||'show')} (${t.length} chars)</summary><pre class="out">${esc(t)}</pre></details>`; }
// The banked tasks as the solver reads them: where task quality and near-duplicates show.
function bankedTasks(rows){ if(!rows.length) return '<div class="nodiff">— none banked —</div>';
  return rows.map(t=>{
    const tags=(t.tags||[]).map(g=>`<span class="chip">${esc(g)}</span>`).join('');
    const path=(t.path||[]).join(' → ');
    const meta=[t.novelty!=null?`novelty ${n3(t.novelty)}`:'', t.n_refine?`${t.n_refine} refine`:'']
      .filter(Boolean).join(' · ');
    return `<div class="tr"><div class="attempt-h">#${t.i}${meta?` · <span class="muted">${esc(meta)}</span>`:''}</div>`
      +`<div class="reason">${esc(t.task)}</div>`
      +(tags?`<div class="chips">${tags}</div>`:'')
      +(path?`<div class="muted mono">${esc(path)}</div>`:'')+'</div>'; }).join(''); }

// Coverage tags: realized share per tag vs its target (multi-label, so shares need not sum
// to 100%). Bars share one scale so over/under reads at a glance.
function tagCoverage(rows){ if(!rows.length) return '<div class="nodiff">— not tagged —</div>';
  const scale=Math.max(...rows.map(r=>Math.max(r.share,r.target_share)),0.01);
  const cls={UNDER:'bad','on-track':'ok',OVER:'muted'};
  const tr=rows.map(r=>`<tr>
    <td class="l name">${esc(r.tag)}</td>
    <td>${r.count}/${r.num_banked}</td>
    <td class="l">${pct(r.share,0)}<span class="bar" style="width:${(120*r.share/scale).toFixed(0)}px"></span></td>
    <td>${pct(r.target_share,0)}</td>
    <td class="l"><span class="badge ${cls[r.flag]||'muted'}">${esc(r.flag)}</span></td>
    <td class="l muted" style="white-space:normal">${esc(r.definition)}</td></tr>`).join('');
  return `<div class="twrap"><table><thead><tr><th class="l">tag</th><th>banked</th>
    <th class="l">realized</th><th>target</th><th class="l">status</th>
    <th class="l">definition</th></tr></thead><tbody>${tr}</tbody></table></div>`; }
function memoryHtml(txt){ if(!txt) return '<div class="nodiff">— none —</div>';
  const items=String(txt).split(/\n/).map(l=>l.replace(/^\s*[-*]\s?/,'').trim()).filter(Boolean);
  return '<ul class="mem-list">'+items.map(i=>`<li>${esc(i)}</li>`).join('')+'</ul>'; }
function diffHtml(d){ if(d==null) return ''; if(d==='(no change)') return '<div class="nodiff">(no change)</div>';
  return '<pre class="diff">'+String(d).split(/\n/).map(l=>{const c=l.startsWith('+')?'add':l.startsWith('-')?'del':l.startsWith('@@')?'hunk':'';
    return `<span class="${c}">${esc(l)}</span>`;}).join('\n')+'</pre>'; }

// ── Plotly charts (rendered to SVG) ──
// Real hex colors (Plotly can't resolve CSS vars in SVG attrs), theme-aware.
const DARK=matchMedia('(prefers-color-scheme: dark)').matches;
const C=DARK
  ? {accent:'#6ea8fe',ok:'#4ade80',bad:'#f87171',amber:'#fbbf24',purple:'#c084fc',teal:'#2dd4bf',
     muted:'#9aa0aa',fg:'#e7e7e7',grid:'#262a31'}
  : {accent:'#2563eb',ok:'#16a34a',bad:'#dc2626',amber:'#d97706',purple:'#7c3aed',teal:'#0d9488',
     muted:'#6b7280',fg:'#141414',grid:'#e5e7eb'};
const OCH={banked:C.ok,too_easy:C.amber,too_hard:C.bad,near_duplicate:C.muted,no_task:C.purple,
  invalid_spec:C.purple,excluded_app:C.muted,success:C.ok,failure:C.bad};
const ocColorH=r=>OCH[r]||C.teal;

function baseLayout(extra){
  extra=extra||{};
  const ax={gridcolor:C.grid,zerolinecolor:C.grid,linecolor:C.grid,color:C.muted,
    tickfont:{size:10},automargin:true};
  const L=Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:C.muted,size:11,family:'ui-sans-serif,system-ui,sans-serif'},
    margin:{l:44,r:14,t:8,b:32}, height:210, bargap:0.25, showlegend:false,
  }, extra);
  L.xaxis=Object.assign({},ax,extra.xaxis||{});
  L.yaxis=Object.assign({},ax,extra.yaxis||{});
  if(extra.yaxis2) L.yaxis2=Object.assign({},ax,{showgrid:false},extra.yaxis2);
  return L;
}
const PLOT_CFG={displayModeBar:false, responsive:true, staticPlot:false};

// build a placeholder now, queue the figure; flushPlots() renders after DOM insertion
let PLOTQ=[], PLOTN=0;
function plot(build){ const id='plot'+(PLOTN++); PLOTQ.push({id,build}); return `<div class="plot" id="${id}"></div>`; }

// pass^k (all k attempts land) and pass@k (any one lands) on one axis; they meet at k=1,
// the mean success rate. Error bars are the task-sampling SE.
function kCurveCard(a){
  const pk=Object.entries(a.pass_hat_k||{}), pak=Object.entries(a.pass_at_k||{});
  if(pk.length<2 && pak.length<2) return '';
  const ser=(ent,se,name,color)=>({
    x:ent.map(e=>+e[0].split(/[\^@]/)[1]), y:ent.map(e=>100*e[1]), name:name,
    type:'scatter', mode:'lines+markers', line:{color:color}, marker:{color:color},
    error_y:{type:'data',array:ent.map(e=>100*((se||{})[e[0]]||0)),visible:true,
             color:color,thickness:1,width:3,opacity:0.45},
    hovertemplate:name+' %{x}: %{y:.1f}% ±%{error_y.array:.1f}<extra></extra>'});
  return card('k-curves — pass@k (any of k) vs pass^k (all of k)', plot(()=>({
    data:[ser(pak,a.pass_at_k_se,'pass@k',C.ok), ser(pk,a.pass_hat_k_se,'pass^k',C.accent)],
    layout:{yaxis:{range:[0,100]}, xaxis:{title:{text:'k',font:{size:10}},dtick:1},
            height:236, showlegend:true,
            legend:{orientation:'h',y:-0.32,font:{size:10}}}})));
}
function flushPlots(){
  const q=PLOTQ; PLOTQ=[];
  if(typeof Plotly==='undefined'){ q.forEach(({id})=>{ const el=document.getElementById(id);
    if(el) el.innerHTML='<div class="nodiff">Plotly.js failed to load (offline?)</div>'; }); return; }
  q.forEach(({id,build})=>{ const el=document.getElementById(id); if(!el) return;
    try{ const f=build(); Plotly.newPlot(el, f.data, baseLayout(f.layout), PLOT_CFG); }
    catch(e){ el.innerHTML='<div class="nodiff">no data</div>'; } });
}
const noData='<div class="nodiff">no data</div>';

// horizontal bars — rows:[{label,value,color?}]
function figHBar(rows,color){ if(!rows.length) return noData;
  return plot(()=>({ data:[{type:'bar',orientation:'h',
      x:rows.map(r=>r.value), y:rows.map(r=>r.label),
      marker:{color: rows.some(r=>r.color)?rows.map(r=>r.color||color||C.accent):(color||C.accent)},
      hovertemplate:'%{y}: %{x}<extra></extra>'}],
    layout:{margin:{l:150,r:24,t:8,b:28}, height:Math.max(120, rows.length*24+46)} })); }

// integer histogram — vertical bars of counts (cap+ bucket for the tail)
function figIntHist(values,color,cap=8){ values=values.filter(v=>v!=null);
  if(!values.length) return noData;
  const m=new Map(); let hi=0;
  values.forEach(v=>{ const k=Math.min(v,cap+1); m.set(k,(m.get(k)||0)+1); if(k>hi)hi=k; });
  const labels=[],vals=[]; for(let k=0;k<=hi;k++){ labels.push(k>cap?cap+'+':String(k)); vals.push(m.get(k)||0); }
  return plot(()=>({ data:[{type:'bar',x:labels,y:vals,marker:{color:color||C.accent},
      hovertemplate:'%{x}: %{y}<extra></extra>'}], layout:{} })); }

// float histogram over [0,1] — for the partial-credit score distribution
function figFloatHist(values,color,bins=10){ values=values.filter(v=>v!=null);
  if(!values.length) return noData;
  const counts=new Array(bins).fill(0), labels=[];
  for(let b=0;b<bins;b++) labels.push((b/bins).toFixed(1));
  values.forEach(v=>{ let b=Math.floor(v*bins); if(b>=bins)b=bins-1; if(b<0)b=0; counts[b]++; });
  return plot(()=>({ data:[{type:'bar',x:labels,y:counts,marker:{color:color||C.accent},
      hovertemplate:'[%{x}, +0.1): %{y}<extra></extra>'}],
    layout:{bargap:0.04, xaxis:{title:{text:'partial success',font:{size:10}}}} })); }

// multi-series lines over x — series:[{y,name,color,axis:'r'?,dash?}], thresholds:[{v,label}]
function figLines(x,series,{thresholds, y1title,y2title}={}){
  if(!x.length) return noData;
  const hasR=series.some(s=>s.axis==='r');
  const data=series.map(s=>({ x, y:s.y, name:s.name, type:'scatter', mode:'lines+markers',
    line:{color:s.color,width:2,dash:s.dash?'dot':'solid'}, marker:{color:s.color,size:5},
    yaxis:s.axis==='r'?'y2':'y', connectgaps:false,
    hovertemplate:(s.name||'')+' @%{x}: %{y}<extra></extra>' }));
  const layout={ showlegend:series.filter(s=>s.name).length>1,
    legend:{orientation:'h',y:-0.22,font:{size:10}}, xaxis:{title:{text:'session',font:{size:10}}} };
  if(y1title) layout.yaxis={title:{text:y1title,font:{size:10}}};
  if(hasR) layout.yaxis2={overlaying:'y',side:'right',showgrid:false,color:C.muted,
    title:{text:y2title||'',font:{size:10}}};
  layout.shapes=(thresholds||[]).map(t=>({type:'line',xref:'paper',x0:0,x1:1,y0:t.v,y1:t.v,
    line:{color:C.amber,width:1,dash:'dash'}}));
  layout.annotations=(thresholds||[]).map(t=>({xref:'paper',x:1,y:t.v,text:t.label,showarrow:false,
    font:{color:C.amber,size:9},xanchor:'right',yanchor:'bottom'}));
  return plot(()=>({data,layout})); }

// outcome timeline — one colored square per session along the real session index
function figTimeline(sessions){ if(!sessions.length) return noData;
  const cats=[...new Set(sessions.map(s=>s.result))];
  return plot(()=>({ data:cats.map(c=>{ const pts=sessions.filter(s=>s.result===c);
      return {x:pts.map(s=>s.idx), y:pts.map(()=>0), name:c, mode:'markers', type:'scatter',
        marker:{color:ocColorH(c),size:13,symbol:'square'},
        hovertemplate:'session %{x}: '+c+'<extra></extra>'}; }),
    layout:{showlegend:true, legend:{orientation:'h',y:-0.5,font:{size:10}}, height:130,
      margin:{l:24,r:14,t:8,b:20}, yaxis:{visible:false},
      xaxis:{title:{text:'session',font:{size:10}}}} })); }

// ── sortable table + filterable list ──
function buildTable(cols,rows,onClick){
  const numeric=new Set(['int','num1','num3','pct','pct0','pct_std','usd','cost','passk','when']);
  cols.forEach(c=>{ if(c.left==null) c.left=!numeric.has(c.kind||'text'); });
  const st={k:(cols.find(c=>c.k==='timestamp')?'timestamp':cols[0].k),asc:false};
  const el=document.createElement('div'); el.className='twrap';
  const table=document.createElement('table'); el.appendChild(table);
  const colOf=k=>cols.find(c=>c.k===k)||cols[0];
  function sv(row,c){
    if(c.num){ const x=c.num(row); return x==null?-Infinity:x; }
    const v=row[c.k];
    if(c.kind==='passk'){ const xs=Object.values(v||{}); return xs.length?Math.max(...xs):-1; }
    if(v==null) return -Infinity; return typeof v==='number'?v:String(v); }
  // natural sort: numbers numerically; strings with numeric-aware collation so
  // task ids "0,1,2,…,10" order correctly (and "0d8a4ee_1" vs "…_2" too).
  function cmp(x,y){
    if(typeof x==='number' && typeof y==='number') return x-y;
    if(typeof x==='number') return -1;
    if(typeof y==='number') return 1;
    return String(x).localeCompare(String(y), undefined, {numeric:true, sensitivity:'base'}); }
  function fmt(row,c){ const v=row[c.k],k=c.kind||'text';
    if(c.cell) return c.cell(row);
    switch(k){ case 'pct':return pct(v); case 'pct0':return pct(v,0);
      case 'pct_std':{ if(v==null)return''; const s=row.success_se?' ±'+(100*row.success_se).toFixed(1):'';
        return pct(v)+s; }
      case 'num1':return n1(v); case 'num3':return n3(v); case 'usd':return usd(v);
      case 'cost':return costCell(row);
      case 'int':return v==null?'':String(v);
      case 'passk':{ const ks=Object.entries(v||{}); return ks.length?ks.map(([kk,x])=>kk+'='+(100*x).toFixed(1)).join('  '):''; }
      case 'when':return when(v); default:return esc(v); } }
  function draw(){
    const rs=[...rows].sort((a,b)=>{const c=cmp(sv(a,colOf(st.k)),sv(b,colOf(st.k)));return st.asc?c:-c;});
    table.innerHTML='<thead><tr>'+cols.map(c=>`<th data-k="${c.k}" class="${c.left?'l':''} ${c.k===st.k?'sorted '+(st.asc?'asc':''):''}">${esc(c.h)}</th>`).join('')+'</tr></thead>'
      +'<tbody>'+rs.map(r=>`<tr data-i="${rows.indexOf(r)}">`+cols.map(c=>`<td class="${c.left?'l':''} ${c.k==='name'||c.k==='task_id'?'name':''}">${fmt(r,c)}</td>`).join('')+'</tr>').join('')+'</tbody>';
    table.querySelectorAll('th').forEach(th=>th.onclick=()=>{ const k=th.dataset.k; if(k===st.k)st.asc=!st.asc; else {st.k=k;st.asc=false;} draw(); });
    if(onClick) table.querySelectorAll('tbody tr').forEach(tr=>{ tr.style.cursor='pointer'; tr.onclick=()=>onClick(rows[+tr.dataset.i]); });
  }
  draw(); return el;
}

// Every plain string column ({k,h} with no kind/num/cell) with 2..FACET_MAX distinct values
// becomes a multi-select dropdown filter.
const FACET_MAX=14;
function facetsFor(cols,rows){
  return cols.filter(c=>!c.kind&&!c.num&&!c.cell&&c.k!=='name').map(c=>{
    const vals=[...new Set(rows.map(r=>r[c.k]).filter(v=>typeof v==='string'&&v!==''))].sort();
    return {key:c.k,label:c.h,vals};
  }).filter(f=>f.vals.length>1&&f.vals.length<=FACET_MAX);
}

function listView(cols,rows,onClick,searchKeys){
  const box=document.createElement('div');
  const bar=document.createElement('div'); bar.className='listbar';
  const input=document.createElement('input');
  input.className='filter'; input.placeholder='filter…'; input.autocomplete='off';
  bar.appendChild(input);

  const facets=facetsFor(cols,rows);
  const sel={};                                    // key -> Set of kept values (all = no filter)
  const fbar=document.createElement('div'); fbar.className='facets';
  facets.forEach(f=>{
    sel[f.key]=new Set(f.vals);
    const n=v=>rows.filter(r=>r[f.key]===v).length;
    const d=document.createElement('details'); d.className='fac';
    d.innerHTML=`<summary><span>${esc(f.label)}</span></summary><div class="fmenu">`
      +`<div class="fhead"><span data-all="1">all</span><span data-none="1">none</span></div>`
      +f.vals.map(v=>`<label class="facet"><input type="checkbox" data-v="${esc(v)}" checked>`
        +`<span>${esc(v)}</span><span class="fn">${n(v)}</span></label>`).join('')
      +`</div>`;
    const boxes=()=>[...d.querySelectorAll('.fmenu input')];
    d.querySelector('[data-all]').onclick=()=>{ boxes().forEach(cb=>cb.checked=true);
      sel[f.key]=new Set(f.vals); apply(); };
    d.querySelector('[data-none]').onclick=()=>{ boxes().forEach(cb=>cb.checked=false);
      sel[f.key]=new Set(); apply(); };
    boxes().forEach(cb=>cb.onchange=()=>{
      cb.checked?sel[f.key].add(cb.dataset.v):sel[f.key].delete(cb.dataset.v); apply(); });
    f.el=d; fbar.appendChild(d);
  });
  if(facets.length) bar.appendChild(fbar);

  const clear=document.createElement('span'); clear.className='fclear'; clear.textContent='clear';
  clear.onclick=()=>{ input.value='';
    facets.forEach(f=>{ sel[f.key]=new Set(f.vals);
      f.el.querySelectorAll('.fmenu input').forEach(cb=>cb.checked=true); f.el.open=false; });
    apply(); };
  bar.appendChild(clear);
  const count=document.createElement('span'); count.className='muted count'; bar.appendChild(count);
  const host=document.createElement('div');
  box.appendChild(bar); box.appendChild(host);

  function apply(){
    const q=input.value.trim().toLowerCase();
    let f=rows;
    facets.forEach(fc=>{ const s=sel[fc.key];
      // all selected = no filter, so a run missing the field is not dropped
      if(s.size!==fc.vals.length) f=f.filter(r=>s.has(r[fc.key])); });
    if(q) f=f.filter(r=>searchKeys.map(k=>r[k]).join(' ').toLowerCase().includes(q));
    facets.forEach(fc=>{ const s=sel[fc.key], all=s.size===fc.vals.length;
      const txt=all?'':(s.size===0?'none':(s.size===1?[...s][0]:s.size+' of '+fc.vals.length));
      fc.el.classList.toggle('on',!all);
      fc.el.querySelector('summary span').innerHTML=esc(fc.label)+(txt?': <b>'+esc(txt)+'</b>':'');
    });
    const filtered=f.length!==rows.length;
    clear.style.display=filtered?'':'none';
    host.innerHTML=''; host.appendChild(buildTable(cols,f,onClick));
    count.textContent=f.length+(filtered?' / '+rows.length:'')+' run'+(rows.length===1?'':'s'); }
  input.oninput=apply;
  apply(); return box;
}

// Close any open filter menu on an outside click or Escape (one menu open at a time).
document.addEventListener('click',e=>{
  document.querySelectorAll('details.fac[open]').forEach(d=>{ if(!d.contains(e.target)) d.open=false; }); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape')
  document.querySelectorAll('details.fac[open]').forEach(d=>d.open=false); });

// ── navigation ──
let VIEW=null;   // {load: fn} current view, re-run on refresh
function setApp(node){ app.innerHTML=''; if(typeof node==='string') app.innerHTML=node; else app.appendChild(node); }
function loading(){ setApp('<div class="empty">loading…</div>'); }
function setCrumbs(list){ crumbEl.innerHTML=list.map((c,i)=>`<span class="cr" data-i="${i}">${esc(c.label)}</span>`).join('<span class="sep">/</span>');
  crumbEl.querySelectorAll('.cr').forEach(e=>{ const c=list[+e.dataset.i]; if(c.go) e.onclick=c.go; }); }
function markTab(){ document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.t===S.tab)); }
"""


_JS_VIEWS = r"""
// ── shared renderers ──
// The paper's cost: standard-tier prices, full prompt caching, per repeat.
const COST_H='$ / run (std, cached)';
const costText=r=>r.cost_per_run_usd==null?'':(r.cost_complete?'':'~')+usd(r.cost_per_run_usd);
function costCell(r){
  if(r.cost_per_run_usd==null) return `<span class="muted" title="${esc((r.cost_notes||[]).join('\n')||'no token records')}">—</span>`;
  const tip=['standard-tier prices, full prompt caching, per repeat',...(r.cost_notes||[])].join('\n');
  return `<span class="${r.cost_complete?'':'partial'}" title="${esc(tip)}">${esc(costText(r))}</span>`; }
function costNote(d){ return (!d.cost_complete&&(d.cost_notes||[]).length)
  ?`<div class="note">~ cost reconstructed: ${esc(d.cost_notes.join(' · '))}</div>`:''; }

function attemptTimeline(attempts){ if(!attempts||!attempts.length) return '';
  return '<div class="timeline">'+attempts.map(a=>`<span class="dot ${a.outcome==='success'?'ok':'bad'}" title="attempt ${a.attempt}: ${a.outcome} · streak ${a.consecutive_successes||0}">${a.consecutive_successes||0}</span>`).join('')+'</div>'; }
function attemptsDetail(attempts){ if(!attempts||!attempts.length) return '<div class="nodiff">no attempts</div>';
  return attempts.map(a=>`<div class="attempt"><div class="attempt-h">attempt ${a.attempt} · <span class="badge ${a.outcome==='success'?'ok':'bad'}">${esc(a.outcome)}</span> ${a.skipped_generation?'<span class="badge muted">no gen</span>':''} <span class="muted">streak ${a.consecutive_successes||0}</span></div>`
    +(a.memory_diff!==undefined?diffHtml(a.memory_diff):'')
    +(a.memory?`<details><summary>memory snapshot</summary>${memoryHtml(a.memory)}</details>`:'')+'</div>').join(''); }
function transcriptEntry(e){ const k=e.kind||'?';
  const head=`<span class="kbadge k-${k}">${esc(k)}</span>${e.turn?'<span class="muted">t'+e.turn+'</span> ':''}${e.name?'<b>'+esc(e.name)+'</b>':(e.path?'<b>'+esc(e.path)+'</b>':'')}`;
  let body='';
  if(e.arguments) body+=`<pre class="code">${esc(JSON.stringify(e.arguments))}</pre>`;
  if(e.code) body+=`<pre class="code">${esc(e.code)}</pre>`;
  if(e.output!=null) body+=longtext(e.output,'output');
  if(e.reason) body+=`<div class="muted">${esc(e.reason)}</div>`;
  return `<div class="tr ${(e.error||e.exec_error)?'err':''}">${head}${body}</div>`; }
function turnCard(t){
  const mem=(t.retrieved_memories&&t.retrieved_memories.length)
    ?`<div class="mem"><b>retrieved memories (${t.retrieved_memories.length})</b><ul class="mem-list">`
      +t.retrieved_memories.map(m=>`<li>${esc(typeof m==='string'?m:JSON.stringify(m))}</li>`).join('')+'</ul>'
      +(t.reasoning_with_memory?`<div class="reason">${esc(t.reasoning_with_memory)}</div>`:'')+'</div>':'';
  const ok=t.execution_success;
  const badge=ok===false?'<span class="badge bad">exec error</span>':(ok?'<span class="badge ok">ok</span>':'');
  return `<div class="turn"><div class="turn-h">turn ${t.turn_idx!=null?t.turn_idx:''} ${badge}</div>`
    +mem+(t.thought?`<div class="thought">${esc(t.thought)}</div>`:'')
    +(t.code?`<pre class="code">${esc(t.code)}</pre>`:'')
    +(t.execution_output!=null?longtext(t.execution_output,'output'):'')+'</div>'; }

// pass^k / pass@k columns: percent ± task-sampling SE. p@1 is omitted (it equals p^1).
function kCols(prefix,sym,ks,withSe){
  const val=(r,k)=>(r[prefix]||{})[sym+k], se=(r,k)=>(r[prefix+'_se']||{})[sym+k];
  return ks.map(k=>({k:prefix+k,h:sym==='pass^'?'p^'+k:'p@'+k,left:false,num:r=>val(r,k),
    cell:r=>{const v=val(r,k); if(v==null) return ''; const e=withSe&&se(r,k);
      return (100*v).toFixed(1)+(e?' ±'+(100*e).toFixed(1):'');}})); }

// ════ INFERENCE ════
const INF_COLS=[{k:'setup',h:'setup'},{k:'category',h:'category'},{k:'name',h:'run'},{k:'benchmark',h:'benchmark'},{k:'domain',h:'domain'},
  {k:'model',h:'model'},{k:'memory',h:'memory'},{k:'tasks',h:'tasks',kind:'int'},
  {k:'runs',h:'runs',kind:'int'},{k:'success_mean',h:'success',kind:'pct_std'},
  {k:'score_mean',h:'partial success',left:false,num:r=>r.score_mean,
    cell:r=>r.score_mean==null?'':pct(r.score_mean)+(r.score_std?' ±'+(100*r.score_std).toFixed(1):'')},
  ...kCols('pass_hat_k','pass^',[1,2,3,4,5],true), ...kCols('pass_at_k','pass@',[2,3,4,5],true),
  {k:'avg_turns',h:'turns',kind:'num1'},
  {k:'cost_per_run_usd',h:COST_H,kind:'cost'},
  {k:'timestamp',h:'when',kind:'when'}];
let INF_RUN=null;

async function showInfList(){ S.tab='inference'; markTab(); VIEW={load:showInfList};
  setCrumbs([{label:'inference'}]); loading();
  const rows=await api('/api/inference');
  if(!rows||!rows.length){ setApp('<div class="empty">No scored inference runs under outputs/inference/ or outputs/baselines/*/inference/.</div>'); return; }
  setApp(listView(INF_COLS,rows,r=>showInfRun(r.setup,r.id),
    ['setup','category','name','benchmark','domain','model','memory'])); }

async function showInfRun(setup,name){ VIEW={load:()=>showInfRun(setup,name)}; loading();
  const d=await api('/api/inference/run?'+runQ(setup,name));
  if(!d){ setApp('<div class="empty">not found</div>'); return; }
  INF_RUN=d;
  setCrumbs([{label:'inference',go:showInfList},{label:runLabel(setup,name)}]);
  const a=d.aggregate||{}, rows=d.rows;
  let h=chips([{k:'setup',v:d.setup},{k:'category',v:d.category},{k:'benchmark',v:d.benchmark},{k:'domain',v:d.domain},
    {k:'model',v:d.model},{k:'memory',v:d.memory},{k:'when',v:when(d.timestamp)}]);
  h+=tiles([{label:'success ±se over repeats',value:a.success_rate_summary||pct(a.success_rate_mean)},
    {label:'partial success',value:pct(avg(rows.map(r=>r.score)))},
    {label:'runs',value:d.num_runs},{label:'tasks',value:d.num_tasks},
    {label:'avg turns',value:n1(avg(rows.map(r=>r.avg_turns)))},
    {label:COST_H,value:costText(d)},
    {label:'pass@'+d.num_runs+' (any of '+d.num_runs+')',
     value:pct((a.pass_at_k||{})['pass@'+d.num_runs])}]);
  h+=costNote(d);
  const perRun=(d.success_per_run||[]).map((v,i)=>({label:(d.run_names[i]||('run_'+i)),value:v}));
  const cons=(d.consistency||[]).map(c=>({label:c.passes+'/'+d.num_runs,value:c.count}));
  const comps=Object.entries(d.reward_components||{}).map(([k,v])=>({label:k,value:v}));
  h+=grid([
    card('partial success distribution', figFloatHist(rows.map(r=>r.score), C.accent)),
    comps.length?card('reward components (τ²)', figHBar(comps, C.teal)):'',
    perRun.length?card('success per run', figHBar(perRun, C.accent)):'',
    kCurveCard(a),
    cons.length?card('per-task consistency (runs passed)', figHBar(cons, C.teal)):'']);
  const tcols=[{k:'task_id',h:'task'},
    {k:'grid',h:'runs',left:true,cell:r=>r.runs.map(x=>`<span class="badge ${x.success?'ok':'bad'}" title="${esc(x.run)}">${x.success?'✓':'✗'}</span>`).join('')},
    {k:'n_pass',h:'passed',kind:'int',cell:r=>r.n_pass+'/'+r.n},
    {k:'score',h:'partial success',kind:'num3'},
    {k:'avg_turns',h:'turns',kind:'num1'},{k:'term',h:'termination'}];
  const tbl=buildTable(tcols,rows,r=>showInfTask(setup,name,r.runs[0]?r.runs[0].run:'.',r.task_id));
  const box=document.createElement('div'); box.innerHTML=h+section('tasks','');
  box.querySelector('.section').appendChild(tbl); setApp(box); flushPlots(); }

async function showInfTask(setup,name,run,task){ VIEW={load:()=>showInfTask(setup,name,run,task)}; loading();
  const q=runQ(setup,name)+'&run='+encodeURIComponent(run||'')+'&task='+encodeURIComponent(task);
  const t=await api('/api/inference/task?'+q);
  if(!t){ setApp('<div class="empty">trace not found</div>'); return; }
  setCrumbs([{label:'inference',go:showInfList},{label:runLabel(setup,name),go:()=>showInfRun(setup,name)},{label:'task '+task}]);
  const ev=t.evaluation||{};
  let h=chips([{k:'task',v:task},{k:'run',v:run},{k:'success',v:t.success?'✓ pass':'✗ fail'},
    {k:'reward',v:ev.reward!=null?n3(ev.reward):''},
    {k:'tests',v:ev.tests_total?(ev.tests_passed+'/'+ev.tests_total):''},
    {k:'turns',v:t.num_turns},{k:'termination',v:(t.extra||{}).termination_reason}]);
  const runs=(INF_RUN&&INF_RUN.run_names)||[];
  if(runs.length>1){ h+='<div class="chips">'+runs.map(rn=>`<button class="btn" data-run="${esc(rn)}" ${rn===run?'style="border-color:var(--accent)"':''}>${esc(rn)}</button>`).join('')+'</div>'; }
  if(t.task_instruction) h+=section('task',`<div class="mem">${esc(t.task_instruction)}</div>`);
  const turns=(t.turns||[]).map(turnCard).join('');
  h+=section('turns ('+(t.turns||[]).length+')',turns||'<div class="nodiff">no turns</div>');
  if(ev.details) h+=section('evaluation',longtext(ev.details,'details'));
  const box=document.createElement('div'); box.innerHTML=h;
  box.querySelectorAll('button[data-run]').forEach(b=>b.onclick=()=>showInfTask(setup,name,b.dataset.run,task));
  setApp(box); }

// ════ ACCUMULATION ════
const ACC_COLS=[{k:'setup',h:'setup'},{k:'name',h:'run'},{k:'benchmark',h:'benchmark'},{k:'domain',h:'domain'},
  {k:'model',h:'model'},{k:'tasks',h:'tasks',kind:'int'},
  {k:'solved',h:'solved',kind:'int'},{k:'gave_up',h:'gave up',kind:'int'},
  {k:'pool',h:'pool items',kind:'int'},
  {k:'avg_attempts',h:'avg attempts',kind:'num1'},
  {k:'cost_per_run_usd',h:COST_H,kind:'cost'},
  {k:'timestamp',h:'when',kind:'when'}];

async function showAccList(){ S.tab='accumulation'; markTab(); VIEW={load:showAccList};
  setCrumbs([{label:'accumulation'}]); loading();
  const rows=await api('/api/accumulation');
  if(!rows||!rows.length){ setApp('<div class="empty">No runs under outputs/daedalus-curated/ or outputs/baselines/*/memory/.</div>'); return; }
  setApp(listView(ACC_COLS,rows,r=>showAccRun(r.setup,r.id),
    ['setup','name','benchmark','domain','model'])); }

async function showAccRun(setup,name){ VIEW={load:()=>showAccRun(setup,name)}; loading();
  const d=await api('/api/accumulation/run?'+runQ(setup,name));
  if(!d){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'accumulation',go:showAccList},{label:runLabel(setup,name)}]);
  let h=chips([{k:'setup',v:d.setup},{k:'benchmark',v:d.benchmark},{k:'domain',v:d.domain},{k:'model',v:d.model},
    {k:'extractor',v:d.extraction_model},
    {k:'success streak',v:d.num_success_to_continue},{k:'max failures',v:d.max_failures},
    {k:'when',v:when(d.timestamp)}]);
  h+=tiles([{label:'tasks',value:d.tasks},{label:'solved',value:d.solved},
    {label:'banked',value:d.banked},{label:'pool items',value:d.pool},{label:'gave up',value:d.gave_up},
    {label:'solved w/o mem',value:d.tasks?pct(d.solved_wo_mem/d.tasks,0):''},
    {label:COST_H,value:costText(d)}]);
  h+=costNote(d);
  const growth=(d.pool_growth||[]);
  h+=grid([
    card('attempts to bank (banked tasks)', figIntHist(d.attempts_to_bank, C.accent)),
    card('solver failures before banking (banked tasks)',
      figIntHist(d.rows.filter(r=>r.banked).map(r=>r.failures), C.bad)),
    card('solved without memory', figHBar([
      {label:'solved w/o mem',value:d.solved_wo_mem,color:C.ok},
      {label:'needed memory',value:d.tasks-d.solved_wo_mem,color:C.accent}])),
    growth.length?card('banked heuristics (cumulative)', plot(()=>({data:[{
      x:growth.map((_,i)=>i+1), y:growth, type:'scatter', mode:'lines', line:{color:C.teal},
      hovertemplate:'task %{x}: %{y}<extra></extra>'}],
      layout:{xaxis:{title:{text:'task',font:{size:10}}}}}))):'']);
  const tcols=[{k:'task_id',h:'task'},
    {k:'outcome',h:'outcome',cell:r=>`<span class="badge ${r.outcome==='success'?'ok':'bad'}">${esc(r.outcome)}</span>`},
    {k:'banked',h:'banked',left:false,num:r=>r.banked?1:0,
      cell:r=>r.banked?'<span class="badge ok">✓</span>':'—'},
    {k:'attempts',h:'attempts',kind:'int'},{k:'failures',h:'failures',kind:'int'},
    {k:'max_consec',h:'max streak',kind:'int'},{k:'fbf',h:'fails→1st ✓',kind:'int'},
    {k:'solved_wo_mem',h:'w/o mem',cell:r=>r.solved_wo_mem?'✓':'—'}];
  const tbl=buildTable(tcols,d.rows,r=>showAccTask(setup,name,r.task_id));
  const box=document.createElement('div'); box.innerHTML=h+section('tasks','');
  box.querySelector('.section').appendChild(tbl); setApp(box); flushPlots(); }

async function showAccTask(setup,name,task){ VIEW={load:()=>showAccTask(setup,name,task)}; loading();
  const s=await api('/api/accumulation/task?'+runQ(setup,name)+'&task='+encodeURIComponent(task));
  if(!s){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'accumulation',go:showAccList},{label:runLabel(setup,name),go:()=>showAccRun(setup,name)},{label:'task '+task}]);
  const attempts=s.attempts||[];
  let h=chips([{k:'task',v:task},{k:'outcome',v:s.final_outcome},{k:'attempts',v:attempts.length},
    {k:'solved w/o mem',v:s.solved_without_memory?'yes':'no'}]);
  h+=section('attempt timeline',attemptTimeline(attempts));
  h+=section('final memory',memoryHtml(s.final_memory));
  h+=section('attempts',attemptsDetail(attempts));
  if(s.last_trace_text) h+=section('last trace',longtext(s.last_trace_text,'transcript'));
  setApp(h); }

// ════ GENERATION ════
const GEN_COLS=[{k:'setup',h:'setup'},{k:'name',h:'run'},{k:'benchmark',h:'benchmark'},{k:'domain',h:'domain'},
  {k:'model',h:'model'},{k:'sessions',h:'sessions',kind:'int'},{k:'banked',h:'banked',kind:'int'},
  {k:'bad_difficulty',h:'bad difficulty',kind:'int'},
  {k:'too_easy',h:'Σ too easy',kind:'int'},{k:'too_hard',h:'Σ too hard',kind:'int'},
  {k:'near_dup',h:'near dup',kind:'int'},{k:'errored',h:'errored',kind:'int'},{k:'yield',h:'yield',kind:'pct0'},
  {k:'coverage_tags',h:'tags',kind:'int'},{k:'tag_gap',h:'tag gap',kind:'pct0'},
  {k:'guidelines',h:'guidelines',kind:'int'},{k:'heuristics',h:'heuristics',kind:'int'},
  {k:'cost_per_run_usd',h:COST_H,kind:'cost'},
  {k:'timestamp',h:'when',kind:'when'}];

async function showGenList(){ S.tab='generation'; markTab(); VIEW={load:showGenList};
  setCrumbs([{label:'generation'}]); loading();
  const rows=await api('/api/generation');
  if(!rows||!rows.length){ setApp('<div class="empty">No runs under outputs/daedalus/.</div>'); return; }
  setApp(listView(GEN_COLS,rows,r=>showGenRun(r.setup,r.id),
    ['setup','name','benchmark','domain','model'])); }

async function showGenRun(setup,name){ VIEW={load:()=>showGenRun(setup,name)}; loading();
  const d=await api('/api/generation/run?'+runQ(setup,name));
  if(!d){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'generation',go:showGenList},{label:runLabel(setup,name)}]);
  const S2=d.sessions||[];          // solver sessions only, sorted by session index
  const X=S2.map(s=>s.idx);         // x-axis is the true session number (gaps = errored)
  let h=chips([{k:'setup',v:d.setup},{k:'benchmark',v:d.benchmark},{k:'domain',v:d.domain},
    {k:'model',v:d.model},
    {k:'explorers',v:d.num_explorers},{k:'max refine',v:d.max_refinements},
    {k:'min novelty',v:d.min_novelty},{k:'streak',v:d.num_success_to_continue},
    {k:'max failures',v:d.max_failures},
    {k:'errored',v:d.n_errors||null},{k:'when',v:when(d.timestamp)}]);
  h+=tiles([{label:'sessions',value:S2.length},{label:'banked',value:(d.outcomes||{}).banked||0},
    {label:'yield',value:pct(d.yield,0)},{label:'guidelines',value:d.guidelines},
    {label:'heuristics',value:d.heuristics},
    {label:COST_H,value:costText(d)}]);
  h+=costNote(d);
  const toolRows=(d.bank.tool_freq||[]).map(([t,c])=>({label:t,value:c}));
  h+=grid([
    card('outcome mix over sessions', figTimeline(S2)),
    card('refinements vs guidelines over time', figLines(X,[
      {name:'refinements', y:S2.map(s=>s.n_refine), color:C.accent},
      {name:'guidelines (cumulative)', y:S2.map(s=>s.guidelines_after), color:C.teal, axis:'r'}],
      {y1title:'refinements', y2title:'guidelines'})),
    card('exploration turns over time', figLines(X,[
      {name:'explore turns (all calls)', y:S2.map(s=>s.explorer_turns), color:C.accent},
      {name:'initial exploration', y:S2.map(s=>s.explorer_turns_initial), color:C.teal},
      {name:'wasted (leading)', y:S2.map(s=>s.wasted), color:C.amber}])),
    card('novelty of tasks over time', figLines(X,[{name:'novelty', y:S2.map(s=>s.novelty), color:C.accent}],
      {thresholds:d.min_novelty!=null?[{v:d.min_novelty,label:'min_novelty'}]:[]})),
    card('solver failures before banking (banked tasks)',
      figIntHist(S2.filter(s=>s.result==='banked').map(s=>s.n_failures), C.bad)),
    card('banked-task solution length (calls)', figIntHist(d.bank.path_lens, C.teal, 12)),
    toolRows.length?card('tools used by banked tasks', figHBar(toolRows)):'']);
  const tagRows=d.bank.tag_coverage||[];
  if(tagRows.length) h+=section('coverage tags — realized vs target',tagCoverage(tagRows)
    +(d.coverage_goal?`<details><summary>coverage goal</summary><pre class="out">${esc(d.coverage_goal)}</pre></details>`:''));
  const bt=d.bank.tasks||[];
  if(bt.length) h+=section(`banked tasks (${bt.length})`,bankedTasks(bt));
  const tcols=[{k:'idx',h:'session',kind:'int'},
    {k:'result',h:'result',cell:s=>`<span class="badge" style="border-color:${ocColor(s.result)};color:${ocColor(s.result)}">${esc(s.result)}</span>`},
    {k:'explorer_turns',h:'explore turns',kind:'int'},
    {k:'explorer_turns_initial',h:'initial',kind:'int'},{k:'wasted',h:'wasted',kind:'int'},
    {k:'n_refine',h:'refines',kind:'int'},{k:'novelty',h:'novelty',kind:'num3'},
    {k:'guidelines_after',h:'guidelines',kind:'int'},
    {k:'n_failures',h:'solver fails',kind:'int'}];
  const tbl=buildTable(tcols,S2,s=>showGenSession(setup,name,s.file));
  const box=document.createElement('div'); box.innerHTML=h+section('sessions','');
  box.querySelector('.section').appendChild(tbl); setApp(box); flushPlots(); }

async function showGenSession(setup,name,file){ VIEW={load:()=>showGenSession(setup,name,file)}; loading();
  const s=await api('/api/generation/session?'+runQ(setup,name)+'&session='+encodeURIComponent(file));
  if(!s){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'generation',go:showGenList},{label:runLabel(setup,name),go:()=>showGenRun(setup,name)},{label:'session '+(s.session_index!=null?s.session_index:file)}]);
  const allRefs=(s.refinements||[]).map(r=>({...r,kind:'difficulty'}))
    .concat((s.novelty_refinements||[]).map(r=>({...r,kind:'near-duplicate'})));
  let h=chips([{k:'session',v:s.session_index},{k:'result',v:s.result},{k:'sandbox',v:s.sandbox_id},
    {k:'novelty',v:s.novelty!=null?n3(s.novelty):''},{k:'refinements',v:allRefs.length}]);
  const pp=s.predicted_path||[];
  if(pp.length) h+=section('predicted path','<div class="chips">'+pp.map(t=>`<span class="chip">${esc(t)}</span>`).join('')+'</div>');
  // e.task is the request text; some runs stored a spec object whose `purpose` stands in for it
  const traj=(s.trajectory||[]).map((e,i)=>{ const r=e.result;
    const txt=(typeof e.task==='string')?e.task:((e.task&&e.task.purpose)||'');
    return `<div class="tr"><span class="badge" style="border-color:${ocColor(r)};color:${ocColor(r)}">${esc(r)}</span> <span class="muted">spec ${i}</span>${txt?`<div class="reason">${esc(txt)}</div>`:''}</div>`; }).join('');
  if(traj) h+=section('trajectory (spec → result)',traj);
  const expl=s.explorer||{};
  const tr=(expl.transcript||[]).map(transcriptEntry).join('');
  h+=section('explorer transcript ('+(expl.num_turns||0)+' turns)',tr?`<details open><summary>${(expl.transcript||[]).length} entries</summary>${tr}</details>`:'<div class="nodiff">none</div>');
  const refs=allRefs.map((r,i)=>`<div class="refine"><div class="attempt-h">refinement ${i+1} · ${r.kind} · ${r.num_turns||0} turns</div>`
    +(r.refinement?`<div class="reason">${esc(typeof r.refinement==='string'?r.refinement:JSON.stringify(r.refinement))}</div>`:'')
    +((r.artifact&&r.artifact.purpose)?`<div class="muted">→ ${esc(r.artifact.purpose)}</div>`:'')+'</div>').join('');
  if(refs) h+=section('refinements',refs);
  const acc=s.accumulation_summary||{};
  if(acc.attempts) h+=section('solver retry loop',attemptTimeline(acc.attempts)+attemptsDetail(acc.attempts));
  if(s.heuristic) h+=section('banked heuristic',memoryHtml(s.heuristic));
  h+=section('guidelines',chips([{k:'before',v:(s.guidelines_before||[]).length},{k:'after',v:(s.guidelines_after||[]).length}])
    +((s.guidelines_after||[]).length?`<details><summary>guidelines after (${s.guidelines_after.length})</summary>${memoryHtml(s.guidelines_after.map(g=>'- '+g).join('\n'))}</details>`:''));
  setApp(h); }

// ════ GENERATED TEST SETS ════
// One row per (test set, model): does a generated test set rank models like the real split?
const tsQ=(ts,model)=>'ts='+encodeURIComponent(ts)+'&model='+encodeURIComponent(model);
const TS_COLS=[{k:'testset',h:'test set'},{k:'model_dir',h:'model'},
  {k:'benchmark',h:'benchmark'},{k:'tasks',h:'tasks',kind:'int'},{k:'runs',h:'runs',kind:'int'},
  {k:'success_mean',h:'success',kind:'pct_std'},
  {k:'partial_mean',h:'partial',kind:'pct'},
  ...kCols('pass_hat_k','pass^',[1,2,3],false), ...kCols('pass_at_k','pass@',[2,3],false),
  {k:'avg_turns',h:'turns',kind:'num1'},
  {k:'judge_model',h:'judge'},
  {k:'unparsed',h:'unjudged',left:false,num:r=>r.unparsed,
    cell:r=>r.unparsed?`<span class="badge bad">${r.unparsed}</span>`:'—'},
  {k:'cost_per_run_usd',h:COST_H,kind:'cost'},{k:'timestamp',h:'when',kind:'when'}];

async function showTsList(){ S.tab='testset'; markTab(); VIEW={load:showTsList};
  setCrumbs([{label:'test sets'}]); loading();
  const rows=await api('/api/testset');
  if(!rows||!rows.length){ setApp('<div class="empty">No runs under outputs/evaluation-proxy/. '
    +'Build one with <code>python -m daedalus.scripts.tasks_as_test_set --generation '
    +'outputs/daedalus/&lt;run&gt; --model &lt;model&gt;</code>.</div>'); return; }
  setApp(listView(TS_COLS,rows,r=>showTsRun(r.testset,r.model_dir),
    ['testset','model_dir','benchmark','model','judge_model'])); }

async function showTsRun(ts,model){ VIEW={load:()=>showTsRun(ts,model)}; loading();
  const d=await api('/api/testset/run?'+tsQ(ts,model));
  if(!d){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'test sets',go:showTsList},{label:ts+' · '+model}]);
  const a=d.aggregate||{}, spec=d.test_set||{};
  let h=chips([{k:'test set',v:ts},{k:'model',v:d.model},{k:'benchmark',v:d.benchmark},
    {k:'split',v:spec.dataset},{k:'judge',v:spec.judge_model},
    {k:'generator',v:spec.generator_model},{k:'when',v:when(d.timestamp)}]);
  h+=tiles([{label:'success ±se over repeats',value:a.success_rate_summary||pct(a.success_rate_mean)},
    {label:'partial',value:pct(a.partial_rate_mean)},
    {label:'tasks',value:spec.num_tasks},{label:'repeats',value:a.num_runs},
    {label:'avg turns',value:n1(a.avg_turns)},
    {label:COST_H,value:costText(d)},
    {label:'pass@'+a.num_runs+' (any of '+a.num_runs+')',
     value:pct((a.pass_at_k||{})['pass@'+a.num_runs])}]);
  h+=costNote(d);
  const perRun=(d.per_run||[]).map(r=>({label:'run_'+r.run_idx,value:r.success_rate}));
  // how often each success condition held: the first thing to read when judging fairness
  const conds=Object.entries(d.condition_failures||{})
    .map(([k,v])=>({label:k.replace('condition_','cond '),value:v.met_rate}));
  h+=grid([
    perRun.length?card('success per repeat', figHBar(perRun, C.accent)):'',
    conds.length?card('condition met-rate', figHBar(conds, C.teal)):'',
    kCurveCard(a),
    card('partial credit per task', figFloatHist((d.rows||[]).map(r=>r.partial), C.purple))]);
  const tcols=[{k:'tid',h:'task'},
    {k:'pass_rate',h:'passed',left:false,num:r=>r.pass_rate,
      cell:r=>r.pass_rate==null?'':`<span class="badge ${r.pass_rate>0?'ok':'bad'}">`
        +`${r.successes}/${r.runs}</span>`},
    {k:'partial',h:'partial',kind:'pct'},
    {k:'num_conditions',h:'conds',kind:'int'},
    {k:'avg_turns',h:'turns',kind:'num1'},
    {k:'sandbox_id',h:'sandbox'},
    {k:'task',h:'request',cell:r=>`<span class="muted">${esc((r.task||'').slice(0,110))}</span>`}];
  const tbl=buildTable(tcols,d.rows,r=>showTsTask(ts,model,r.tid));
  const box=document.createElement('div'); box.innerHTML=h+section('tasks','');
  box.querySelector('.section').appendChild(tbl); setApp(box); flushPlots(); }

async function showTsTask(ts,model,tid){ VIEW={load:()=>showTsTask(ts,model,tid)}; loading();
  const t=await api('/api/testset/task?'+tsQ(ts,model)+'&task='+encodeURIComponent(tid));
  if(!t){ setApp('<div class="empty">not found</div>'); return; }
  setCrumbs([{label:'test sets',go:showTsList},{label:ts+' · '+model,go:()=>showTsRun(ts,model)},
    {label:tid}]);
  const js=t.judgements||[];
  let h=chips([{k:'task',v:tid},{k:'sandbox',v:t.sandbox_id},
    {k:'repeats',v:js.length},
    {k:'passed',v:js.filter(j=>j.success).length+'/'+js.length},
    {k:'tags',v:(t.tags||[]).join(' ')}]);
  h+=section('generated request',`<div class="mem">${esc(t.instruction)}</div>`);
  // one row per success condition, one column per repeat
  const conds=(t.success_conditions||[]).map((c,i)=>{
    const cells=js.map(j=>{const o=(j.outcomes||[])[i]||{};
      return `<td title="${esc(o.reason||'')}">${o.met?'<span class="badge ok">✓</span>'
        :'<span class="badge bad">✗</span>'}</td>`;}).join('');
    return `<tr><td class="muted">${i+1}</td>`
      +`<td class="l" style="white-space:normal;max-width:60em">${esc(c)}</td>${cells}</tr>`;});
  h+=section('success conditions × repeats',
    `<div class="twrap"><table><thead><tr><th>#</th><th class="l">condition</th>`
    +js.map(j=>`<th>run_${j.run_idx}</th>`).join('')+`</tr></thead><tbody>`
    +conds.join('')+`</tbody></table></div>`);
  h+=section('judge verdicts', js.map(j=>`<div class="turn"><div class="turn-h">run_${j.run_idx} `
    +`${j.success?'<span class="badge ok">pass</span>':'<span class="badge bad">fail</span>'} `
    +`<span class="muted">${j.num_met}/${j.num_conditions} conditions · ${j.num_turns} turns</span>`
    +`</div><div class="reason">${esc(j.reason||'')}</div>`
    +(j.parsed===false?'<div class="badge bad">judge reply unreadable</div>':'')+'</div>').join(''));
  const tr=t.trace||{};
  if((tr.turns||[]).length){
    h+=section('first repeat, turns ('+tr.turns.length+')',tr.turns.map(turnCard).join(''));
  }
  setApp(h); }

// ── bootstrap ──
const TABS={inference:showInfList,accumulation:showAccList,generation:showGenList,testset:showTsList};
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>TABS[t.dataset.t]());
document.getElementById('refresh').onclick=()=>{ if(VIEW&&VIEW.load) VIEW.load(); };
showInfList();
"""


# Plotly.js from CDN (needs network on first load; charts degrade to a note if absent).
_PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

_PAGE = (
    "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>daedalus · runs</title>\n<style>" + _CSS + "</style>\n" + _PLOTLY_CDN
    + "</head>\n<body>"
    + _SHELL + "\n<script>\n" + _JS_CORE + "\n" + _JS_VIEWS + "\n</script>\n</body></html>\n"
).encode("utf-8")


# ── HTTP server ──────────────────────────────────────────────────────────────
def _one(q: dict[str, list[str]], key: str) -> str | None:
    return (q.get(key) or [None])[0]


_ROUTES: dict[str, Callable[[dict[str, list[str]]], Any]] = {
    "/api/inference": lambda q: gather_inference_runs(),
    "/api/inference/run": lambda q: inference_run_detail(_one(q, "setup"), _one(q, "name")),
    "/api/inference/task": lambda q: inference_task_detail(
        _one(q, "setup"), _one(q, "name"), _one(q, "run"), _one(q, "task")),
    "/api/accumulation": lambda q: gather_accumulation_runs(),
    "/api/accumulation/run": lambda q: accumulation_run_detail(
        _one(q, "setup"), _one(q, "name")),
    "/api/accumulation/task": lambda q: accumulation_task_detail(
        _one(q, "setup"), _one(q, "name"), _one(q, "task")),
    "/api/generation": lambda q: gather_generation_runs(),
    "/api/generation/run": lambda q: generation_run_detail(_one(q, "setup"), _one(q, "name")),
    "/api/generation/session": lambda q: generation_session_detail(
        _one(q, "setup"), _one(q, "name"), _one(q, "session")),
    "/api/testset": lambda q: gather_testset_runs(),
    "/api/testset/run": lambda q: testset_run_detail(_one(q, "ts"), _one(q, "model")),
    "/api/testset/task": lambda q: testset_task_detail(
        _one(q, "ts"), _one(q, "model"), _one(q, "task")),
}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(_PAGE, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(b"", "image/x-icon", status=204)
        elif path in _ROUTES:
            value = _ROUTES[path](parse_qs(parsed.query))
            status = 404 if value is None else 200
            body = json.dumps(value, default=str).encode("utf-8")
            self._send(body, "application/json", status=status)
        else:
            self._send(b"not found", "text/plain", status=404)

    def log_message(self, *args: Any) -> None:  # keep the console quiet
        pass


# ── --report: the inference table as text ────────────────────────────────────
def print_report(rows: list[dict[str, Any]]) -> None:
    """One line per scored inference run: MSR ± SE, pass^k at k = #repeats, $ / run."""
    header = (f"{'setup':14s} {'benchmark':10s} {'category':22s} {'run':52s} "
              f"{'MSR ± SE':>13s} {'pass^k':>11s} {'$/run':>9s}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["setup"], r["benchmark"], r["category"], r["name"])):
        msr = ("" if r["success_mean"] is None
               else f"{100 * r['success_mean']:.1f} ± {100 * r['success_se']:.1f}")
        pk = r.get("pass_hat_k") or {}
        k = max((int(key.split("^")[1]) for key in pk), default=0)
        pass_k = f"p^{k} {100 * pk[f'pass^{k}']:.1f}" if k > 1 else ""
        usd = r["cost_per_run_usd"]
        cost = "" if usd is None else ("" if r["cost_complete"] else "~") + f"{usd:.3f}"
        print(f"{r['setup'][:14]:14s} {r['benchmark'][:10]:10s} {r['category'][:22]:22s} "
              f"{r['name'][:52]:52s} {msr:>13s} {pass_k:>11s} {cost:>9s}")
    print("\n$/run: standard-tier prices, full prompt caching, per repeat; "
          "~ = reconstructed from incomplete records.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default 8000)")
    parser.add_argument("--report", action="store_true",
                        help="print the inference run table to stdout and exit")
    parser.add_argument("--outputs", default="outputs",
                        help="the outputs folder to browse (default ./outputs)")
    args = parser.parse_args()
    set_outputs_root(args.outputs)

    if args.report:
        print_report(gather_inference_runs())
        return

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    counts = (
        f"{len(gather_inference_runs())} inference · "
        f"{len(gather_accumulation_runs())} accumulation · "
        f"{len(gather_generation_runs())} generation · "
        f"{len(gather_testset_runs())} test set"
    )
    print(f"daedalus run browser → http://{args.host}:{args.port}  ({counts}) from {args.outputs}/",
          flush=True)
    if not any((Path(args.outputs) / d).is_dir() for d in ("daedalus", "daedalus-curated", "baselines")) \
            and any((Path(args.outputs) / d).is_dir() for d in ("generation", "accumulation", "references")):
        print(f"  note: {args.outputs}/ has the pre-release layout (generation/, accumulation/, "
              "references/); the browser reads daedalus/, daedalus-curated/ and baselines/. "
              "Download the released outputs, or pass --outputs <folder>.", flush=True)
    print("Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
