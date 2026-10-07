"""The Consolidator: merge a run's accepted heuristics into one bank (paper Section 3).

    uv run python -m daedalus.scripts.consolidate outputs/daedalus/<name>
    uv run python -m daedalus.scripts.consolidate outputs/daedalus/<name> --pool pool_first10.json

Reads `<run>/<pool>` (default `pool.json`) and writes a new pool in the same format, ready
for `memory.pool_path`. Three strategies:

  single        (default, the method) ONE call over every note with `consolidate.txt`,
                removing redundant or overlapping advice. Writes `consolidated_<pool>`.
  dedup         ONE call that only removes duplicates: notes are split into their bullets
                and nothing is rewritten (Table 6, "dedup only"). Writes `dedup_<pool>`.
  hierarchical  whole notes dealt at random into `--leaves` pools, then consolidated two by
                two with the single-call prompt (8 → 4 → 2 → 1; Table 7). Writes
                `iterative_consolidated_<pool>`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from daedalus.core.env import load_dotenv
from daedalus.core.llm.client import LLMClient
from daedalus.core.logging import console
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.memory.pool import MemoryItem, MemoryPool
from daedalus.core.resources import CORE_PROMPTS

_PROMPT = CORE_PROMPTS / "consolidate.txt"
_DEDUP_PROMPT = CORE_PROMPTS / "consolidate_dedup_single.txt"
_OUTPUT_NAME = {"single": "consolidated_{}", "dedup": "dedup_{}",
                "hierarchical": "iterative_consolidated_{}"}

_APP = re.compile(r"\bapis\.([a-z_]+)\.")
_RECEIVER = re.compile(r"\b([a-z][a-z0-9_]{2,})\.[a-z_]{3,}\s*\(")
_LIST_PREFIX = re.compile(r"^\s*(?:\d+\s*[.)]\s*|[-*•]\s+)")
# Real identifiers that say nothing about which area a note is about.
_GENERIC_TOKENS = {
    "access_token", "login", "logout", "password", "complete_task", "show_profile",
    "show_account_passwords", "api_docs", "show_api_doc", "show_api_descriptions",
    "search_api_docs", "supervisor", "apis",
}


def _numbered(texts: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))


def _parse_json_list(text: str, key: str) -> list:
    """Parse the model's strict-JSON reply, tolerating an accidental ```json fence."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s[: s.rfind("```")] if "```" in s else s
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"model did not return valid JSON: {e}")
    entries = data.get(key) if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise SystemExit(f"model reply has no {key!r} list")
    return entries


def _bullets(notes: list[str]) -> list[str]:
    """Notes split into their atomic bullets (a banked note is a short list of lessons)."""
    out = []
    for n in notes:
        for b in re.split(r"\n\s*[-*•]\s+|^\s*[-*•]\s+", n):
            b = " ".join(b.split())
            if len(b) > 25:
                out.append(b)
    return out


def _topic_for(text: str, vocab: set[str]) -> str:
    """A deterministic topic tag: the first area (app) the lesson names, else "general"."""
    low = text.lower()
    hits = [a for a in vocab if re.search(rf"(?<![\w.]){re.escape(a)}(?![\w])", low)]
    return sorted(hits, key=lambda a: (low.index(a), a))[0] if hits else "general"


def consolidate_single(llm: LLMClient, notes: list[str], effort: str) -> list[dict]:
    prompt = Template(_PROMPT.read_text(encoding="utf-8")).render(
        n=len(notes), notes=_numbered(notes)
    )
    resp = llm.generate([{"role": "user", "content": prompt}], reasoning_effort=effort)
    return [h for h in _parse_json_list(resp.content, "heuristics") if isinstance(h, dict)]


def consolidate_dedup(llm: LLMClient, notes: list[str], effort: str) -> list[dict]:
    bullets = _bullets(notes)
    if not bullets:
        raise SystemExit("pool has no heuristics to deduplicate")
    prompt = Template(_DEDUP_PROMPT.read_text(encoding="utf-8")).render(
        n=len(bullets), items=_numbered(bullets)
    )
    resp = llm.generate([{"role": "user", "content": prompt}], reasoning_effort=effort)
    # Strip a list number the model may echo from the numbered input.
    out = [t for t in (_LIST_PREFIX.sub("", str(x)).strip()
                       for x in _parse_json_list(resp.content, "items")) if t]
    # A reply that lost almost everything summarised instead of pruning: keep the input.
    if len(out) < max(1, len(bullets) // 10):
        console.info(f"  dedup returned {len(out)} of {len(bullets)} — keeping the input")
        out = bullets
    console.info(f"  {len(bullets)} → {len(out)} heuristics after dedup")
    vocab = set()
    for n in notes:
        vocab |= set(_APP.findall(n)) | {m.lower() for m in _RECEIVER.findall(n)}
    vocab -= _GENERIC_TOKENS
    return [{"topic": _topic_for(t, vocab), "text": t} for t in out]


def consolidate_hierarchical(llm: LLMClient, notes: list[str], effort: str, leaves: int,
                             seed: int, concurrency: int = 8) -> list[dict]:
    shuffled = list(notes)
    random.Random(seed).shuffle(shuffled)
    leaves = max(1, min(leaves, len(shuffled)))
    pools: list[list] = [shuffled[i::leaves] for i in range(leaves)]

    def texts(pool: list) -> list[str]:
        return [str(h.get("text", "")).strip() if isinstance(h, dict) else h for h in pool]

    while len(pools) > 1:
        pairs = [(pools[i], pools[i + 1]) for i in range(0, len(pools) - 1, 2)]
        with cf.ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(pairs)))) as ex:
            merged = list(ex.map(
                lambda ab: consolidate_single(llm, [t for t in texts(ab[0]) + texts(ab[1]) if t],
                                              effort), pairs))
        if len(pools) % 2:  # the odd one out rides to the next round untouched
            merged.append(pools[-1])
        console.info(f"  round: {len(pools)} pools → {len(merged)} · "
                     f"{sum(map(len, pools))} → {sum(map(len, merged))} items")
        pools = merged
    if pools[0] and not isinstance(pools[0][0], dict):  # leaves=1
        return consolidate_single(llm, pools[0], effort)
    return pools[0]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a generation or accumulation folder")
    parser.add_argument("--pool", default="pool.json", help="pool file inside the folder")
    parser.add_argument("--output", default=None, help="default: <run_dir>/<strategy name>")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--strategy", choices=tuple(_OUTPUT_NAME), default="single")
    parser.add_argument("--leaves", type=int, default=8, help="hierarchical: starting pools")
    parser.add_argument("--seed", type=int, default=0, help="hierarchical: random split seed")
    args = parser.parse_args()

    console.configure()
    run_dir = Path(args.run_dir)
    notes = [t.strip() for t in MemoryPool.load(run_dir / args.pool).texts() if t and t.strip()]
    if not notes:
        raise SystemExit(f"{run_dir / args.pool} has no heuristics to consolidate")
    console.header("daedalus · consolidate", {"source": run_dir / args.pool,
                                              "notes": len(notes), "strategy": args.strategy})

    # The calls land in the run's own usage ledger, under the `consolidation` role.
    accounting = CostAccounting.standalone(run_dir, experiment_name=f"consolidate:{run_dir.name}")
    llm = accounting.client(args.model, "consolidation", temperature=0.0)
    if args.strategy == "single":
        heuristics = consolidate_single(llm, notes, args.reasoning_effort)
    elif args.strategy == "dedup":
        heuristics = consolidate_dedup(llm, notes, args.reasoning_effort)
    else:
        heuristics = consolidate_hierarchical(llm, notes, args.reasoning_effort, args.leaves,
                                              args.seed)

    src = (f"consolidate:{run_dir.name}" if args.pool == "pool.json"
           else f"consolidate:{run_dir.name}/{Path(args.pool).stem}")
    stamp = datetime.now(timezone.utc).isoformat()
    pool = MemoryPool([
        MemoryItem(
            memory_id=f"m_{uuid.uuid4().hex[:8]}",
            type="reflection",
            text=str(h.get("text", "")).strip(),
            source_task_id=src,
            source_trajectory_success=True,
            extraction_timestamp=stamp,
            tags={"topic": str(h["topic"]).strip()} if h.get("topic") else {},
        )
        for h in heuristics if str(h.get("text", "")).strip()
    ])
    out = Path(args.output) if args.output else run_dir / _OUTPUT_NAME[args.strategy].format(
        Path(args.pool).name)
    actual = pool.save(out, overwrite=True)
    usage = accounting.summary()
    console.info(f"{len(notes)} notes → {len(pool)} heuristics "
                 f"({sum(len(t) for t in pool.texts())} chars) → {actual}")
    console.info(f"cost: ${usage['priced_calls_estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
