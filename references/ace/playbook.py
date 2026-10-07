"""ACE's playbook: its on-disk format, its ADD operation, and the bridge to a MemoryPool.

Ported from `ace-appworld/experiments/code/ace/playbook.py` and the `get_section_slug`
half of its `utils.py`. Behaviour is upstream's; what changed is listed under
"Deviations from upstream" in `references/ace/README.md`.

The format is a flat text file: fixed `## SECTION` headers, and under each a line per
bullet, `[<slug>-<00000>] <content>`. Note there are NO `helpful=/harmful=` counters on
this path — `format_playbook_line` drops them upstream, and none of the shipped AppWorld
playbooks carry any. (The finance path in the main `ace` repo keeps them; the agentic one
does not, and nothing reads them at evaluation time either way.)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from daedalus.core.memory.pool import MemoryItem, MemoryPool

# The eight sections, in order, with the header text upstream writes and the id slug it
# assigns. Kept verbatim on every benchmark, including where a section cannot apply (τ²
# has no code snippets) — inventing a per-benchmark taxonomy would be our design, not
# ACE's, and the curator simply leaves an inapplicable section empty.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("STRATEGIES AND HARD RULES", "shr"),
    ("APIs TO USE FOR SPECIFIC INFORMATION", "api"),
    ("USEFUL CODE SNIPPETS AND TEMPLATES", "code"),
    ("COMMON MISTAKES AND CORRECT STRATEGIES", "cms"),
    ("PROBLEM-SOLVING HEURISTICS AND WORKFLOWS", "psw"),
    ("VERIFICATION CHECKLIST", "vc"),
    ("TROUBLESHOOTING AND PITFALLS:", "ts"),
    ("OTHERS", "misc"),
)

# The sections a curator operation may name (`adaptation_react.py`). Anything else is
# dropped, op and all.
#
# UPSTREAM QUIRK, reproduced: the whitelist spells the fifth one with an underscore
# (`problem_solving_…`) while the header normalises to a HYPHEN
# (`problem-solving_…`). So an ADD naming that section passes the whitelist, then fails
# to match any header and is routed to OTHERS. It never fired in ACE's published run —
# their trained playbook has exactly the 2 seed bullets in each — but it is preserved
# rather than quietly fixed.
ALLOWED_SECTIONS: frozenset[str] = frozenset(
    {
        "strategies_and_hard_rules",
        "apis_to_use_for_specific_information",
        "useful_code_snippets_and_templates",
        "common_mistakes_and_correct_strategies",
        "problem_solving_heuristics_and_workflows",
        "verification_checklist",
        "troubleshooting_and_pitfalls",
        "others",
    }
)

_SLUGS: dict[str, str] = {
    _normalised: _slug
    for _normalised, _slug in (
        (header.lower().replace(" ", "_").replace("&", "and").rstrip(":"), slug)
        for header, slug in SECTIONS
    )
}
# Upstream's slug map also answers for the whitelist spelling of the fifth section, which
# no header produces; without it a bullet routed to OTHERS would still be slugged "psw".
_SLUGS.setdefault("problem_solving_heuristics_and_workflows", "psw")

_FULL_LINE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
_SIMPLE_LINE = re.compile(r"\[([^\]]+)\]\s*(.*)")
_TRAILING_NUMBER = re.compile(r"-(\d+)$")


def normalise_section(name: str) -> str:
    """Upstream's section-name normalisation, used for both headers and ADD operations."""
    return name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")


def section_slug(section: str) -> str:
    """The 2-4 letter id prefix for a section; upstream's fallback for unknown names."""
    clean = normalise_section(section)
    if clean in _SLUGS:
        return _SLUGS[clean]
    words = clean.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5] if w)


def empty_playbook() -> str:
    """The 8-section skeleton, no bullets."""
    return "\n\n".join(f"## {header}" for header, _ in SECTIONS)


def parse_playbook_line(line: str) -> dict[str, Any] | None:
    """Parse one bullet line. Returns None for headers, blanks and prose."""
    text = line.strip()
    match = _FULL_LINE.match(text)
    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
            "raw_line": line,
        }
    match = _SIMPLE_LINE.match(text)
    if match:
        return {
            "id": match.group(1),
            "helpful": 0,
            "harmful": 0,
            "content": match.group(2).strip(),
            "raw_line": line,
        }
    return None


def get_next_global_id(playbook_text: str) -> int:
    """One past the highest numeric suffix in use. Ids are global, not per-section."""
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        match = _TRAILING_NUMBER.search(parsed["id"])
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def format_playbook_line(bullet_id: str, content: str) -> str:
    """`[id] content` — upstream drops the counters here, so this path never writes them."""
    return f"[{bullet_id}] {content}"


def validate_operations(operations: Any) -> list[dict[str, Any]]:
    """Upstream's strict validation: ADD only, whitelisted sections only.

    Raises ValueError on a malformed op (the caller keeps the playbook unchanged and logs
    the failure); silently DROPS a well-formed op naming a section outside the whitelist,
    which is upstream's `continue`.
    """
    if not isinstance(operations, list):
        raise ValueError("'operations' field must be a list")
    kept: list[dict[str, Any]] = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError(f"Operation {i} must be a dictionary")
        if "type" not in op:
            raise ValueError(f"Operation {i} missing required 'type' field")
        if op["type"] != "ADD":
            raise ValueError(
                f"Operation {i} has invalid type {op['type']!r}. ACE implements only ADD."
            )
        missing = {"type", "section", "content"} - set(op)
        if missing:
            raise ValueError(f"ADD operation {i} missing fields: {sorted(missing)}")
        if normalise_section(str(op.get("section", ""))) not in ALLOWED_SECTIONS:
            continue
        kept.append(op)
    return kept


def apply_curator_operations(
    playbook_text: str, operations: list[dict[str, Any]], next_id: int
) -> tuple[str, int]:
    """Append each ADD to its section; unknown sections fall through to OTHERS.

    Upstream also carries UPDATE / MERGE / DELETE / CREATE_META as commented-out `TODO:
    not implemented yet` branches, and its curator prompt tells the model not to emit
    them. ADD is genuinely the whole operation set.
    """
    lines = playbook_text.strip().split("\n")

    known: set[str] = set()
    for line in lines:
        if line.strip().startswith("##"):
            known.add(normalise_section(line.strip()[2:]))

    pending: list[tuple[str, str]] = []
    for op in operations:
        section = normalise_section(str(op.get("section", "general")))
        if section not in known and section != "general":
            section = "others"
        bullet_id = f"{section_slug(section)}-{next_id:05d}"
        next_id += 1
        pending.append((section, format_playbook_line(bullet_id, op.get("content", ""))))

    out: list[str] = []
    current: str | None = None
    for line in lines:
        if line.strip().startswith("##"):
            if current is not None:
                out.extend(b for s, b in pending if s == current)
                pending = [(s, b) for s, b in pending if s != current]
            current = normalise_section(line.strip()[2:])
        out.append(line)
    if current is not None:
        out.extend(b for s, b in pending if s == current)
        pending = [(s, b) for s, b in pending if s != current]

    if pending:  # sections that do not exist in this file at all
        leftovers = [b for _, b in pending]
        try:
            index = next(i for i, line in enumerate(out) if line.strip() == "## OTHERS")
            for offset, bullet in enumerate(leftovers):
                out.insert(index + 1 + offset, bullet)
        except StopIteration:
            out.extend(leftovers)

    return "\n".join(out), next_id


def get_playbook_stats(playbook_text: str) -> dict[str, Any]:
    """`{total_bullets, by_section}` — rendered into the curator prompt each call."""
    stats: dict[str, Any] = {"total_bullets": 0, "by_section": {}}
    current = "general"
    for line in playbook_text.strip().split("\n"):
        if line.strip().startswith("##"):
            current = line.strip()[2:].strip()
            continue
        if parse_playbook_line(line):
            stats["total_bullets"] += 1
            stats["by_section"].setdefault(current, {"count": 0})
            stats["by_section"][current]["count"] += 1
    return stats


def _balanced_objects(text: str) -> list[str]:
    """Every balanced {...} span, skipping braces inside strings (upstream's scanner)."""
    found: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    if text[i] == "\\":
                        i += 1
                    i += 1
            i += 1
        if depth == 0:
            found.append(text[start:i])
    return found


def extract_json_from_text(text: str | None) -> dict[str, Any] | None:
    """Parse the curator's reply: bare JSON, a ```json fence, or an embedded object.

    Returns None rather than raising — a curator call that comes back unparseable leaves
    the playbook untouched and the run continues, which is upstream's behaviour.
    """
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for match in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    for candidate in _balanced_objects(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def bullets(playbook_text: str) -> list[tuple[str, str, str]]:
    """Every bullet as `(section header, id, content)`, in file order."""
    out: list[tuple[str, str, str]] = []
    current = "OTHERS"
    for line in playbook_text.strip().split("\n"):
        if line.strip().startswith("##"):
            current = line.strip()[2:].strip()
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            out.append((current, parsed["id"], parsed["content"]))
    return out


def to_pool(playbook_text: str, source: str = "") -> MemoryPool:
    """One `MemoryItem` per bullet, so `serve`, the plots and pass^k read an ACE run.

    The playbook itself stays the artifact inference injects (verbatim, sections intact);
    this is the comparable view of it. Section and bullet id are preserved in `tags` so
    nothing has to be re-parsed later — same convention as
    `references/preping/playbook_to_pool.py`.
    """
    stamp = datetime.now(timezone.utc).isoformat()
    items = [
        MemoryItem(
            memory_id=f"m_{hashlib.sha1(content.encode('utf-8')).hexdigest()[:8]}",
            type="reflection",
            text=content,
            source_task_id=source,
            source_trajectory_success=True,
            extraction_timestamp=stamp,
            tags={"ace_id": bullet_id, "ace_section": section, "method": "ace"},
        )
        for section, bullet_id, content in bullets(playbook_text)
    ]
    return MemoryPool(items=items)
