"""Draw the DAEDALUS pipeline (paper Figures 2 and 3, merged) as ASCII art.

Writes src/data/pipeline-diagram.json. Each line is a list of runs
[text, owner, role, fill], where `owner` is the component or arrow a character belongs to,
`role` is "stroke" (walls, arrows), "label" (a box's name) or "text" (an arrow's caption), and
`fill` names, for its four quarters (top-right, bottom-right, bottom-left, top-left), the box
whose interior they lie in, so boxes get a background that meets their walls exactly.

    python3 scripts/pipeline_diagram.py
"""
from __future__ import annotations

import json
from pathlib import Path

W, H = 86, 28
art = [[" "] * W for _ in range(H)]
owner = [[""] * W for _ in range(H)]
role = [[""] * W for _ in range(H)]
fill = [[""] * (2 * W) for _ in range(2 * H)]  # quarter grid: which box interior is here


def put(x: int, y: int, s: str, pid: str, r: str = "stroke") -> None:
    for i, ch in enumerate(s):
        art[y][x + i] = ch
        owner[y][x + i] = pid
        role[y][x + i] = r


def box(pid: str, x: int, y: int, w: int, h: int, lines: list[str], title: str = "",
        dashed: bool = False, filled: bool = True) -> None:
    hz, vt = ("┄", "┆") if dashed else ("─", "│")
    x1, y1 = x + w - 1, y + h - 1
    if filled:  # interior, from wall centre to wall centre
        for hy in range(2 * y + 1, 2 * y1 + 1):
            for hx in range(2 * x + 1, 2 * x1 + 1):
                fill[hy][hx] = pid
    put(x, y, "┌" + hz * (w - 2) + "┐", pid)
    if title:
        put(x + 2, y, " " + title + " ", pid, "label")
    for r in range(1, h - 1):
        put(x, y + r, vt, pid)
        put(x1, y + r, vt, pid)
    put(x, y1, "└" + hz * (w - 2) + "┘", pid)
    for i, text in enumerate(lines):
        put(x + (w - len(text)) // 2, y + 1 + i, text, pid, "label")


def vline(pid: str, x: int, y0: int, y1: int) -> None:
    for y in range(y0, y1 + 1):
        put(x, y, "│", pid)


def hline(pid: str, x0: int, x1: int, y: int) -> None:
    put(x0, y, "─" * (x1 - x0 + 1), pid)


def text(pid: str, x: int, y: int, s: str) -> None:
    put(x, y, s, pid, "text")


# ── frames and boxes (the loop before what it contains) ───────────────────
box("sessions", 14, 0, 72, 23, [], title="× n sessions", dashed=True, filled=False)
box("guidelines", 16, 1, 20, 3, ["guideline memory"])
box("surveyor", 0, 7, 12, 3, ["SURVEYOR"])
box("explorer", 18, 7, 16, 3, ["EXPLORER"])
box("loop", 42, 5, 42, 12, [], title="SOLVER LOOP")
box("solver", 46, 7, 12, 3, ["SOLVER"])
box("extractor", 69, 7, 13, 3, ["EXTRACTOR"])
box("judge", 46, 12, 12, 3, ["JUDGE"])
box("hbank", 46, 19, 20, 3, ["heuristic memory"])
box("consolidator", 50, 24, 14, 3, ["CONSOLIDATOR"])
box("memory", 19, 24, 23, 3, ["consolidated memory"])

# ── arrows ────────────────────────────────────────────────────────────────
# Surveyor → Explorer: the coverage goal.
put(12, 8, "─────▶", "a-tags")
text("a-tags", 2, 10, "coverage")
text("a-tags", 4, 11, "tags")

# Guideline memory ⇄ Explorer: guidelines in, lessons from each refinement out.
vline("a-guide", 22, 4, 5)
put(22, 6, "▼", "a-guide")
put(29, 7, "┴", "a-lesson")
vline("a-lesson", 29, 5, 6)
put(29, 4, "▲", "a-lesson")
text("a-lesson", 31, 5, "lessons")

# Explorer → Solver: the task. Explorer → Judge: its success conditions.
hline("a-task", 34, 44, 8)
put(45, 8, "▶", "a-task")
text("a-task", 36, 7, "task")
put(29, 9, "┬", "a-cond")
vline("a-cond", 29, 10, 12)
put(29, 13, "└", "a-cond")
hline("a-cond", 30, 44, 13)
put(45, 13, "▶", "a-cond")
text("a-cond", 31, 10, "success")
text("a-cond", 31, 11, "conditions")

# Inside the loop: Solver → Judge, Judge → Extractor, Extractor → Solver.
put(51, 9, "┬", "a-trace")
vline("a-trace", 51, 10, 10)
put(51, 11, "▼", "a-trace")
text("a-trace", 53, 10, "trace")
hline("a-fail", 58, 74, 13)
put(75, 13, "┘", "a-fail")
vline("a-fail", 75, 11, 12)
put(75, 10, "▲", "a-fail")
text("a-fail", 61, 14, "failure")
put(58, 8, "◀" + "─" * 10, "a-heur")
text("a-heur", 59, 7, "heuristic")

# Judge → Explorer (red): too easy or too hard, so the Explorer refines the task.
put(48, 14, "┬", "a-feedback")
vline("a-feedback", 48, 15, 17)
put(48, 18, "┘", "a-feedback")
hline("a-feedback", 25, 47, 18)
put(24, 18, "└", "a-feedback")
vline("a-feedback", 24, 11, 17)
put(24, 10, "▲", "a-feedback")
text("a-feedback", 16, 19, "too easy / too hard: refine")
text("a-feedback", 16, 20, "(dropped after 5 refinements)")

# Judge → heuristic memory (green): three successes in a row.
put(56, 14, "┬", "a-bank")
vline("a-bank", 56, 15, 17)
put(56, 18, "▼", "a-bank")
text("a-bank", 58, 15, "✓✓✓ 3 in a row")

# After the last session: heuristic memory → Consolidator → consolidated memory → the base agent.
put(56, 21, "┬", "a-consolidate")
vline("a-consolidate", 56, 22, 22)
put(56, 23, "▼", "a-consolidate")
text("a-consolidate", 58, 23, "after n sessions")
put(42, 25, "◀" + "─" * 7, "a-deploy")
text("a-deploy", 17, 27, "→ base agent, at test time")

# Where an arrow crosses a wall or a frame, draw the crossing (it belongs to the arrow).
for x, y, pid in [(14, 8, "a-tags"), (42, 8, "a-task"), (42, 13, "a-cond"),
                  (48, 16, "a-feedback"), (56, 16, "a-bank"), (56, 22, "a-consolidate")]:
    put(x, y, "┼", pid)

# ── output ────────────────────────────────────────────────────────────────
def quarters(x: int, y: int) -> str:
    q = [fill[2 * y][2 * x + 1], fill[2 * y + 1][2 * x + 1], fill[2 * y + 1][2 * x], fill[2 * y][2 * x]]
    return ",".join(q) if any(q) else ""


def uniform_rows(f: str) -> bool:
    """A run may span several characters only if its fill does not change left to right."""
    if not f:
        return True
    tr, br, bl, tl = f.split(",")
    return tr == tl and br == bl


runs = []
for y in range(H):
    line = []
    for x in range(W):
        cell = [art[y][x], owner[y][x], role[y][x], quarters(x, y)]
        last = line[-1] if line else None
        if last and last[1:] == cell[1:] and uniform_rows(cell[3]):
            last[0] += cell[0]
        else:
            line.append(cell)
    while line and line[-1][0].strip() == "" and not line[-1][1] and not line[-1][3]:
        line.pop()
    runs.append(line)

parts = sorted({p for row in owner for p in row if p} | {p for row in fill for p in row if p})
dst = Path(__file__).resolve().parents[1] / "src" / "data" / "pipeline-diagram.json"
dst.write_text(json.dumps({"width": W, "parts": parts, "runs": runs}, ensure_ascii=False) + "\n")
print("\n".join("".join(row).rstrip() for row in art))
