"""Convert a PREPING playbook into a daedalus memory pool.

    uv run python references/preping/playbook_to_pool.py   # the AppWorld playbook.json -> pool.json
    uv run python references/preping/playbook_to_pool.py --playbook <in.json> --out <out.json>

PREPING (Choi et al., 2026) builds a **Playbook**: bullets grouped into sections
(`strategies`, `code_snippets`, `pitfalls`, `apis`), each carrying a Curator effectiveness
`tag` and helpful/harmful counters. Daedalus stores heuristics as a flat `MemoryPool`. This
turns the former into the latter so a PREPING playbook can be evaluated on exactly the same
path as a daedalus pool — point an inference config's `memory.pool_path` at the output, or run
`daedalus.scripts.consolidate` on it.

Same convention as the other methods under `references/`: one pool item per bullet, the
originating task in `source_task_id`, and every PREPING-specific field duplicated into `tags`
so nothing is lost and nothing has to be re-parsed later.

Field mapping (see MAPPING_NOTES for the two judgement calls):

    content                 -> text
    metadata.created_at     -> extraction_timestamp
    metadata.source_task_id -> source_task_id, as `preping:<n>`
    id, section, tag,       -> tags
      version, counters

The bullets are emitted in section order, then in the order PREPING wrote them, so the output
is a stable function of the input — re-running it produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daedalus.core.memory.pool import MemoryItem, MemoryPool  # noqa: E402

# The AppWorld playbook and its conversion, as released (outputs/baselines/preping/memory/).
APPWORLD = Path("outputs/baselines/preping/memory/appworld")
# PREPING's PlaybookSection order (src/preping/core/memory/playbook/types.py). Unknown
# sections are kept and appended, so a playbook from a newer version is not silently cut.
SECTION_ORDER = ("strategies", "code_snippets", "pitfalls", "apis")

MAPPING_NOTES = """
Two fields do not map cleanly, and are handled deliberately rather than guessed:

`source_trajectory_success` is set True for every bullet. In daedalus it records whether the
trajectory a heuristic was mined from was ultimately solved. PREPING has no such flag: its
bullets come from trajectories its validator already gated, and its `tag`
(helpful/harmful/neutral) is a Curator effectiveness LABEL, not a trajectory outcome.
Mapping `harmful` -> False would read as "this came from a failed run", which is a different
claim. The tag is preserved verbatim in `tags.preping_tag` instead, so filter on that.

`memory_id` is daedalus's content hash (`m_<sha1[:8]>`), not PREPING's `strategies-001`, so an
item keeps the same id across pools that share text — which is what daedalus's text-level dedup
assumes. The original id is preserved in `tags.preping_id`; `--keep-ids` uses it directly.
""".strip()


def load_bullets(path: Path) -> list[dict[str, Any]]:
    """Every bullet in the playbook, section by section.

    Accepts the three shapes a playbook has been seen in: the section->list mapping
    `Playbook.save` writes, the same mapping nested under a `sections` key, and a bare list
    of bullets. Anything else is an error rather than a silent empty pool.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [b for b in data if isinstance(b, dict)]
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected an object or a list, got {type(data).__name__}")
    sections = data.get("sections") if isinstance(data.get("sections"), dict) else data
    buckets = {k: v for k, v in sections.items() if isinstance(v, list)}
    if not buckets:
        raise SystemExit(f"{path}: no section holds a list of bullets (keys: {list(sections)})")
    ordered = [s for s in SECTION_ORDER if s in buckets]
    ordered += [s for s in buckets if s not in SECTION_ORDER]  # keep unknown sections
    out = []
    for name in ordered:
        for b in buckets[name]:
            if isinstance(b, dict) and str(b.get("content") or "").strip():
                out.append({**b, "section": b.get("section") or name})
    return out


def to_item(b: dict[str, Any], keep_ids: bool) -> MemoryItem:
    """One playbook bullet as a daedalus MemoryItem."""
    text = " ".join(str(b["content"]).split())
    meta = b.get("metadata") or {}
    task = meta.get("source_task_id")
    return MemoryItem(
        memory_id=(str(b.get("id")) if keep_ids and b.get("id")
                   else f"m_{hashlib.sha1(text.encode()).hexdigest()[:8]}"),
        type="reflection",
        text=text,
        source_task_id=f"preping:{task}" if task not in (None, "") else "preping",
        # See MAPPING_NOTES: PREPING's tag is an effectiveness label, not a run outcome.
        source_trajectory_success=True,
        extraction_timestamp=str(meta.get("created_at") or ""),
        tags={
            "source": "preping",
            "preping_id": b.get("id"),
            "preping_section": b.get("section"),
            "preping_tag": b.get("tag"),
            "preping_version": b.get("version"),
            "preping_usage_count": meta.get("usage_count"),
            "preping_helpful_count": meta.get("helpful_count"),
            "preping_harmful_count": meta.get("harmful_count"),
        },
    )


def token_count(texts: list[str]) -> int | None:
    """o200k_base tokens across the pool — what an at-start injection actually costs."""
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(t)) for t in texts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--playbook", type=Path, default=APPWORLD / "playbook.json")
    ap.add_argument("--out", type=Path, default=APPWORLD / "pool.json")
    ap.add_argument("--sections", default="",
                    help="comma-separated sections to keep (default: all)")
    ap.add_argument("--drop-harmful", action="store_true",
                    help="drop bullets the Curator tagged 'harmful' (kept by default, so the "
                         "pool is PREPING's artifact rather than our edit of it)")
    ap.add_argument("--exclude-apps", default="",
                    help="comma-separated apps held out of the daedalus comparison, e.g. "
                         "gmail,amazon — bullets mentioning one are REPORTED, and dropped "
                         "only with --drop-excluded")
    ap.add_argument("--drop-excluded", action="store_true",
                    help="also drop the --exclude-apps bullets instead of just counting them")
    ap.add_argument("--keep-ids", action="store_true",
                    help="keep PREPING's bullet ids as memory_id (default: daedalus content hash)")
    ap.add_argument("--notes", action="store_true", help="print the mapping notes and exit")
    args = ap.parse_args()

    if args.notes:
        print(MAPPING_NOTES)
        return
    if not args.playbook.is_file():
        raise SystemExit(f"no playbook at {args.playbook}")

    bullets = load_bullets(args.playbook)
    print(f"{args.playbook}: {len(bullets)} bullet(s)")
    by_section: dict[str, int] = {}
    for b in bullets:
        by_section[b["section"]] = by_section.get(b["section"], 0) + 1
    for s, n in by_section.items():
        print(f"  {s:<15} {n}")

    keep = bullets
    wanted = [s.strip() for s in args.sections.split(",") if s.strip()]
    if wanted:
        keep = [b for b in keep if b["section"] in wanted]
        print(f"  --sections {','.join(wanted)}: {len(keep)} kept")
    tagged = {}
    for b in bullets:
        tagged[b.get("tag")] = tagged.get(b.get("tag"), 0) + 1
    print(f"  tags: {tagged}")
    if args.drop_harmful:
        before = len(keep)
        keep = [b for b in keep if b.get("tag") != "harmful"]
        print(f"  --drop-harmful: {before - len(keep)} dropped")

    apps = [a.strip().lower() for a in args.exclude_apps.split(",") if a.strip()]
    if apps:
        pats = {a: re.compile(rf"(?<![\w.@-]){re.escape(a)}(?![\w.@-])", re.I) for a in apps}
        hits = {a: [b for b in keep if p.search(b["content"])] for a, p in pats.items()}
        total = {b["id"] for v in hits.values() for b in v}
        for a, v in hits.items():
            print(f"  mentions {a!r}: {len(v)} bullet(s)")
        # Held-out apps are the one real confound in this comparison: a playbook built with
        # access to apps daedalus's explorer never saw is not measuring the same thing.
        if args.drop_excluded and total:
            keep = [b for b in keep if b["id"] not in total]
            print(f"  --drop-excluded: {len(total)} dropped")
        elif total:
            print(f"  (kept — pass --drop-excluded to remove all {len(total)})")

    # Text-level dedup, the same key daedalus's InjectionTracker uses, so two identical
    # bullets in different sections cannot be injected twice.
    items, seen = [], set()
    for b in keep:
        it = to_item(b, args.keep_ids)
        if it.text in seen:
            continue
        seen.add(it.text)
        items.append(it)
    dropped = len(keep) - len(items)
    if dropped:
        print(f"  {dropped} exact-duplicate text(s) merged")

    if not items:
        raise SystemExit("nothing left to write — loosen the filters")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    MemoryPool(items).save(args.out, overwrite=True)

    texts = [i.text for i in items]
    chars = sum(len(t) for t in texts)
    toks = token_count(texts)
    print(f"\nwrote {args.out}")
    print(f"  {len(items)} items, {chars} chars"
          + (f", {toks} tokens (o200k_base)" if toks is not None else ""))
    # Round-trip through the real loader: a file daedalus cannot read is not a conversion.
    back = MemoryPool.load(args.out)
    assert len(back) == len(items) and back.texts() == texts, "round-trip mismatch"
    print(f"  reloads as a MemoryPool: {len(back)} items, texts identical")
    # Repo-relative, because that is what goes in a config's memory.pool_path.
    try:
        shown = args.out.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        shown = args.out
    print(f"  use it: memory.pool_path: \"{shown}\"  (memory.enabled: true)")


if __name__ == "__main__":
    main()
