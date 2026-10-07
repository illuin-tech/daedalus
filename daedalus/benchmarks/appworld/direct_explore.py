"""Explorer-only sessions: explore a live world, then emit one list of heuristics.

The loop behind the two exploration-only ablations of the paper (Table 3):

  (A) one long session (`daedalus.scripts.naive_explorer`, 100 turns): the explorer
      spends its whole budget exploring and is asked for everything it learned;
  (B) many short sessions (`generation.stage: direct_exploration_with_bank`, 40 turns):
      each session also sees the heuristics banked by earlier sessions, may stop early
      once more exploration is unlikely to help, and reports only what is new.

No task, solver, judge or extractor is involved. The final message must hold one block,

    <heuristics>
    - first heuristic
    - second heuristic
    </heuristics>

which `parse_heuristics_block` reads. A missing block is asked for again, once.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from daedalus.benchmarks.appworld.agent import parse_thought_and_code
from daedalus.benchmarks.appworld.hidden_apps import HiddenApps
from daedalus.core.memory.extraction import has_empty_heuristics_block, parse_heuristics_block

_KEEP_EXPLORING = "Keep exploring: Thought, then one ```python block."
_RETRY_EMISSION = (
    "That did not contain a parseable <heuristics> block. Reply with ONLY the block: a line "
    "`<heuristics>`, then one `- ` bullet per heuristic, then a line `</heuristics>`. No "
    "code, no prose outside the block."
)


def app_catalog(world: Any, hidden: HiddenApps) -> str:
    """The app catalog shown to an explorer, minus the held-out apps."""
    return json.dumps(
        [{"name": k, "description": v}
         for k, v in world.task.app_descriptions.items()
         if k.lower() not in hidden.names],
        indent=1,
    )


def _emission_request(max_turns: int, has_bank: bool) -> str:
    scope = (
        "only the genuinely new facts, corrections or important qualifications that are "
        "absent from <previous_heuristics>"
        if has_bank else "everything you learned about this environment"
    )
    closing = (
        "Do not copy, paraphrase or generalize an existing heuristic. If you discovered "
        "nothing new, return an empty <heuristics></heuristics> block."
        if has_bank else
        "Include everything worth carrying forward — do not summarise it away, and do not "
        "keep anything back."
    )
    return (
        f"Your {max_turns} exploration turns are up. Write no more code.\n\n"
        f"Output {scope} as ONE list, in exactly this format and nothing else:\n\n"
        "<heuristics>\n- first heuristic\n- second heuristic\n</heuristics>\n\n"
        "One bullet per heuristic, one line each. Every bullet must be concrete, actionable "
        "and transferable: name the exact operation, the exact parameter and the exact error "
        "text, and state it so it is useful to an agent that has never seen this "
        f"environment. {closing}"
    )


def explore_directly(
    world: Any,
    generate: Callable[[list[dict[str, str]]], str],
    system_prompt: str,
    hidden: HiddenApps,
    max_turns: int,
    *,
    has_bank: bool = False,
    allow_early_stop: bool = False,
    log: Callable[[str], None] = print,
    checkpoint: Callable[[list[dict], int], None] | None = None,
) -> tuple[list[str], list[dict]]:
    """Explore `world` for up to `max_turns` turns, then return (heuristics, transcript).

    Every model message costs one turn, including ones without code. With
    `allow_early_stop`, a `<heuristics>` block sent instead of code ends the session, but
    only once at least one operation has run. With `has_bank`, an explicitly empty block is
    a valid answer ("nothing new").
    """
    messages = [
        {"role": "system", "content": hidden.filter(system_prompt)},
        {"role": "user", "content": "Begin exploring. Thought, then one ```python block."},
    ]
    transcript: list[dict] = []

    def emitted(content: str) -> tuple[list[str], bool]:
        heuristics = parse_heuristics_block(content)
        return heuristics, bool(heuristics) or (has_bank and has_empty_heuristics_block(content))

    for turn in range(1, max_turns + 1):
        content = generate(messages)
        messages.append({"role": "assistant", "content": content})
        thought, code = parse_thought_and_code(content)
        left = max_turns - turn

        if not code:
            heuristics, is_emission = emitted(content) if allow_early_stop else ([], False)
            explored = any(e["kind"] == "exec" for e in transcript)
            if is_emission and explored:
                transcript.append({"turn": turn, "kind": "emission", "attempt": 1,
                                   "parsed": len(heuristics), "early_stop": True,
                                   "content": content})
                log(f"  early stop after {turn - 1} turn(s): {len(heuristics)} heuristic(s)")
                return heuristics, transcript
            if is_emission:
                transcript.append({"turn": turn, "kind": "rejected_emission",
                                   "reason": "no environment operation executed yet",
                                   "content": content})
                nudge = "Do not finish before exercising the live environment. "
            else:
                transcript.append({"turn": turn, "kind": "no_code", "thought": thought,
                                   "content": content})
                nudge = "No python block in that message. "
            messages.append({"role": "user",
                             "content": f"{nudge}{_KEEP_EXPLORING} {left} turns left."})
            log(f"  turn {turn:>3}/{max_turns}  [no code]")
            continue

        blocked = hidden.blocked(code)
        if blocked:
            transcript.append({"turn": turn, "kind": "blocked", "thought": thought,
                               "code": code, "reason": blocked})
            messages.append({"role": "user", "content": f"Output:\n```\n{blocked}\n```"})
            log(f"  turn {turn:>3}/{max_turns}  [blocked]")
            continue

        try:
            output = world.execute(code)
        except Exception as e:  # noqa: BLE001 — surface the error back to the model
            output = str(e)
        output = hidden.filter(output)
        failed = output.strip().startswith("Execution failed")
        transcript.append({"turn": turn, "kind": "exec", "thought": thought, "code": code,
                           "output": output, "exec_error": failed})
        # A countdown every tenth turn: a model told its budget only once tends to wrap up
        # early.
        tail = f"\n\n{left} turns left." if left and left % 10 == 0 else ""
        messages.append({"role": "user", "content": f"Output:\n```\n{output}\n```{tail}"})
        log(f"  turn {turn:>3}/{max_turns}  {'ERR ' if failed else ''}{(thought or '')[:64]}")
        if checkpoint:
            checkpoint(transcript, turn)

    heuristics: list[str] = []
    for attempt in (1, 2):
        request = _emission_request(max_turns, has_bank) if attempt == 1 else _RETRY_EMISSION
        messages.append({"role": "user", "content": request})
        content = generate(messages)
        messages.append({"role": "assistant", "content": content})
        heuristics, is_emission = emitted(content)
        transcript.append({"turn": max_turns + attempt, "kind": "emission",
                           "attempt": attempt, "parsed": len(heuristics),
                           "early_stop": False, "content": content})
        log(f"  emission attempt {attempt}: {len(heuristics)} heuristic(s) parsed")
        if is_emission:
            break
    return heuristics, transcript
