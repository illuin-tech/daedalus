"""Tag-based coverage steering for self-play generation (benchmark-agnostic).

A ONE-TIME survey of the environment (`generation.coverage_tags`) decides how the run's
tasks should be spread: a prose **coverage goal** (which axis to spread along and with what
distribution, argued from evidence) plus a CLOSED set of **tags** — short reusable labels a
task designer can assign — each with a definition and a `target_share`. The survey explores
a throwaway copy of the environment and is expected to CHANGE it (calling a write operation
is the only way to learn what it demands); its changes are discarded.

From then on every explorer session sees:
  - the coverage goal (verbatim), and
  - a live tally of the banked tasks per tag (`#Spotify 4/20 (target 18%) — UNDER`),
so it steers itself toward the under-covered tags. Tags are multi-label (a task carries
several), so `target_share` is the fraction of BANKED TASKS carrying that tag and the
shares need not sum to 1. Enforcement is soft (informational) until the run is
`coverage_escalate_after` through, when the single most-under tag becomes a directive.

The vocabulary is closed: a submitted tag outside the manifest is dropped, never counted,
and never creates a new tag. Tags are taken exactly as the explorer declares them at
submit time — never derived from the executed tool path, never judged.

This module owns the data format and the rendering; the per-benchmark explorers own the
survey channel (`survey_coverage`), and the pipeline owns the counting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COVERAGE_GOAL_RE = re.compile(r"<coverage_goal>(.*?)</coverage_goal>", re.DOTALL | re.IGNORECASE)
TAG_MANIFEST_RE = re.compile(r"<tag_manifest>(.*?)</tag_manifest>", re.DOTALL | re.IGNORECASE)

# Terminal tool for the function-calling survey channels (tau2, AutomationBench). AppWorld has no
# tool channel and delivers the same two payloads as <coverage_goal>/<tag_manifest> text.
SUBMIT_COVERAGE_GOAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_coverage_goal",
        "description": (
            "Submit your final coverage goal: the prose report plus the closed tag "
            "vocabulary a task designer will assign from."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "description": (
                        "The full report: four short subsections (Axis / Distribution / "
                        "Guidance / Pitfalls), each a few plain sentences, not an outline."
                    ),
                },
                "tags": {
                    "type": "array",
                    "description": (
                        "The CLOSED tag vocabulary spanning your chosen axis — as many tags as "
                        "that axis genuinely needs, no filler. Multi-label: a task carries "
                        "several, so the shares need not sum to 1."
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "tag": {
                                "type": "string",
                                "description": "Short, stable, '#'-prefixed label, e.g. '#Pagination'.",
                            },
                            "definition": {
                                "type": "string",
                                "description": "One line: when a task-designer should assign this tag.",
                            },
                            "target_share": {
                                "type": "number",
                                "description": (
                                    "Fraction of banked tasks that should carry this tag, in (0, 1]."
                                ),
                            },
                        },
                        "required": ["tag", "definition", "target_share"],
                    },
                },
            },
            "required": ["report", "tags"],
        },
    },
}


def _normalize_tag(raw: Any) -> str:
    """Canonical '#Label' form of a tag (tolerates a missing '#' and stray whitespace)."""
    tag = " ".join(str(raw or "").split())
    return "#" + tag.lstrip("#").strip() if tag.lstrip("#").strip() else ""


def parse_tags(raw: Any) -> list[dict[str, Any]]:
    """Validate a raw tag manifest into [{tag, definition, target_share}, ...].

    Invalid entries (missing fields, unparseable or out-of-range share, duplicates) are
    dropped rather than failing the whole survey — an empty result means "no manifest",
    which leaves generation running exactly as if coverage tagging were off.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tag = _normalize_tag(item.get("tag"))
        definition = " ".join(str(item.get("definition") or "").split())
        try:
            share = float(item.get("target_share"))
        except (TypeError, ValueError):
            continue
        if not tag or not definition or not 0.0 < share <= 1.0 or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        out.append({"tag": tag, "definition": definition, "target_share": share})
    return out


@dataclass
class CoverageGoal:
    """The survey's output: the prose goal + the closed tag vocabulary.

    The two halves degrade independently. `tagged` (has tags) drives the counting and the
    live tally; a report without tags still steers as prose. An `empty` goal means
    generation behaves exactly as it did before coverage steering existed.
    """

    report: str = ""
    tags: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tagged(self) -> bool:
        """True when tag counting + the live tally are available (a manifest is present)."""
        return bool(self.tags)

    @property
    def empty(self) -> bool:
        """True when there is nothing to inject at all."""
        return not (self.report or self.tags)

    @property
    def names(self) -> list[str]:
        return [t["tag"] for t in self.tags]

    def canonical(self, submitted: Any) -> str | None:
        """The manifest's own spelling of a submitted tag, or None if out of vocabulary."""
        key = _normalize_tag(submitted).lower()
        return next((t["tag"] for t in self.tags if t["tag"].lower() == key), None)

    def keep_known(self, submitted: Any) -> tuple[list[str], list[str]]:
        """Split submitted tags into (kept canonical tags, dropped out-of-vocabulary ones)."""
        kept: list[str] = []
        dropped: list[str] = []
        for raw in submitted if isinstance(submitted, list) else []:
            known = self.canonical(raw)
            if known is None:
                dropped.append(str(raw))
            elif known not in kept:
                kept.append(known)
        return kept, dropped

    # ── serialization ────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {"report": self.report, "tags": self.tags}

    @classmethod
    def from_dict(cls, data: Any) -> "CoverageGoal":
        data = data if isinstance(data, dict) else {}
        return cls(report=str(data.get("report") or "").strip(), tags=parse_tags(data.get("tags")))

    def to_text(self) -> str:
        """The self-contained artifact: the prose report + the manifest as a JSON block."""
        manifest = json.dumps({"tags": self.tags}, indent=2, ensure_ascii=False)
        return f"{self.report.strip()}\n\n<tag_manifest>\n{manifest}\n</tag_manifest>\n"

    @classmethod
    def from_text(cls, text: str) -> "CoverageGoal":
        """Parse an artifact (or a raw survey message) carrying either or both blocks."""
        text = text or ""
        manifest = TAG_MANIFEST_RE.search(text)
        tags: list[dict[str, Any]] = []
        if manifest:
            try:
                payload = json.loads(manifest.group(1))
            except ValueError:
                payload = None
            tags = parse_tags(payload.get("tags") if isinstance(payload, dict) else payload)
        goal = COVERAGE_GOAL_RE.search(text)
        if goal:
            report = goal.group(1)
        else:  # a saved artifact is the bare report followed by the manifest block
            report = TAG_MANIFEST_RE.sub("", text)
        return cls(report=report.strip(), tags=tags)


def merge(goals: "list[CoverageGoal]") -> CoverageGoal:
    """Fold several worlds' surveys into one goal.

    Needed where one world is NOT the environment. On τ² a domain IS the environment and
    on AppWorld every world exposes every app, so one survey sees everything; on
    AutomationBench a world seeds 2-7 of 22 services, and a single-world survey wrote a
    vocabulary that omitted slack — seeded in 16 of 30 worlds and graded in over half the
    test tasks — which left the steering unable to ask for it for a whole run.

    Tags are unioned on their canonical label; a tag seen in several worlds keeps the MEAN
    of its target shares, and the shares are renormalised to sum to 1. The report keeps the
    first world's prose in full (it is the richest by construction when the caller orders
    contexts by new-service coverage) and appends only the extra tags the later worlds
    contributed, because the prose is largely world-generic while the manifest is not.
    """
    goals = [g for g in goals if g and not g.empty]
    if not goals:
        return CoverageGoal()
    if len(goals) == 1:
        return goals[0]

    shares: dict[str, list[float]] = {}
    definition: dict[str, str] = {}
    order: list[str] = []
    for g in goals:
        for tag in g.tags:
            key = tag["tag"]
            if key not in shares:
                shares[key], order = [], order + [key]
                definition[key] = tag["definition"]
            shares[key].append(float(tag["target_share"]))
    total = sum(sum(v) / len(v) for v in shares.values()) or 1.0
    tags = [
        {
            "tag": k,
            "definition": definition[k],
            "target_share": round((sum(shares[k]) / len(shares[k])) / total, 4),
        }
        for k in order
    ]
    first = goals[0]
    extra = [k for k in order if k not in {t["tag"] for t in first.tags}]
    note = ""
    if extra:
        note = (
            "\n\nSurveyed " + str(len(goals)) + " worlds of this environment. The worlds "
            "beyond the first contributed these further tags: " + ", ".join(extra) + "."
        )
    return CoverageGoal(report=(first.report + note).strip(), tags=tags)


def load(path: str | Path) -> CoverageGoal:
    """Load a coverage goal from a precomputed artifact.

    Accepts the `.txt`/`.md` artifact written by the survey (prose + `<tag_manifest>`
    block) or a plain `.json` manifest (`{"report": ..., "tags": [...]}`).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return CoverageGoal.from_dict(json.loads(text))
    return CoverageGoal.from_text(text)


def save(path: str | Path, goal: CoverageGoal) -> Path:
    """Write the self-contained `.txt` artifact (re-loadable via `coverage_tags_path`)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(goal.to_text(), encoding="utf-8")
    return p


def realized(
    goal: CoverageGoal, tag_counts: dict[str, int], num_banked: int
) -> list[dict[str, Any]]:
    """Per-tag realized-vs-target coverage, most-under first.

    One row per manifest tag: `count` banked tasks carrying it, `share` of the banked
    tasks (multi-label, so the shares do not sum to 1), the `target_share` from the
    manifest, and an OVER/UNDER/on-track `flag`. Used by the run summary and the
    observability UI; empty when the run has no coverage goal."""
    rows = []
    for t in goal.tags:
        count = int((tag_counts or {}).get(t["tag"], 0))
        share = count / num_banked if num_banked else 0.0
        target = float(t["target_share"])
        rows.append({
            "tag": t["tag"],
            "definition": t["definition"],
            "count": count,
            "num_banked": num_banked,
            "share": share,
            "target_share": target,
            "flag": "OVER" if share > target else ("UNDER" if share < target else "on-track"),
        })
    rows.sort(key=lambda r: -(r["target_share"] - r["share"]))
    return rows


# ── prompt blocks ────────────────────────────────────────────────────────────
def render_tally(
    goal: CoverageGoal,
    tag_counts: dict[str, int],
    num_banked: int,
    progress: float,
    escalate_after: float,
    escalate_ratio: float,
    available: "list[str] | None" = None,
) -> str:
    """The live per-tag coverage block injected into the explorer's system prompt.

    Lines are sorted most-under first. Shares are computed against the number of BANKED
    tasks (not the number of tag assignments), because a task carries several tags. Once
    the run is `escalate_after` through, the single most-under tag that is also severely
    under (share < target * escalate_ratio) is escalated from feedback to a directive.

    `available` narrows the tally to the tags THIS session's world can actually serve
    (see GenerationBackend.tags_for_context). Without it, a benchmark whose worlds each
    hold a slice of the environment ends up ordering sessions to build around an app that
    is not there: the explorer declines, the tag stays under, and it captures the
    escalation slot for the rest of the run. None = no filtering, the historical
    behaviour and the right one where every world exposes everything.
    """
    if not goal.tagged:
        return ""
    rows = realized(goal, tag_counts, num_banked)  # already sorted most-under first
    if available is not None:
        keep = {str(x) for x in available}
        rows = [r for r in rows if r["tag"] in keep]
        if not rows:
            # Every tag needs something this world does not have. Steering here would
            # point the explorer at an app it cannot reach, so say nothing and let it
            # build what the world supports.
            return (
                "This world serves none of the run's coverage tags, so there is nothing "
                "to steer toward here: build the best task this world's data supports, "
                "and leave `tags` empty."
            )
    lines = [
        f"Running coverage of the {num_banked} banked task(s) against that goal. Tags are "
        "MULTI-LABEL — a task carries several — so these percentages are per-tag shares of "
        "the banked tasks and do not sum to 100%. Steer this task toward the UNDER tags:",
    ]
    lines += [
        f"- {r['tag']}: {r['count']}/{num_banked} (target {round(r['target_share'] * 100)}%)"
        f" — {r['flag']}  ·  {r['definition']}"
        for r in rows
    ]
    if progress >= escalate_after:
        severe = next(
            (
                r
                for r in rows
                if num_banked > 0
                and r["flag"] == "UNDER"
                and r["share"] < r["target_share"] * escalate_ratio
            ),
            None,
        )
        if severe is not None:
            lines.append(
                f">> You MUST build THIS task around {severe['tag']} ({severe['definition']}) "
                "unless it is genuinely infeasible here."
            )
    lines.append(
        "When you submit, set `tags` to the tags above that genuinely define your task (only "
        "those that apply). Use those exact labels — anything else is ignored and counts for nothing."
    )
    return "\n".join(lines)
