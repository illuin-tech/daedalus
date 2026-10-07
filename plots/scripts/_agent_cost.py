"""Per-agent cost for a run whose usage ledger never recorded it.

A pre-ledger generation run kept no per-call usage, but it kept every explorer TRANSCRIPT:
`thought`, `code` and `output` for each turn. That is enough to rebuild the token bill,
because an explorer call is a conversation — turn n resends the system prompt plus every
earlier turn — so the prompt of each turn is the concatenation of things that ARE stored:

    prompt(n)      = base + Σ_{i<n} (assistant_i + user_i)
    completion(n)  = stored assistant text at turn n, scaled by `mult`

Two things are not stored and have to be calibrated from a run that does have a ledger:

  base   the per-turn fixed prefix — the explorer system prompt (app docs, guidelines,
         prior tasks) plus the opening user message. ~7.0-7.6k tokens; it is resent every
         turn, and is about 27% of the prompt bill.
  mult   the ratio of billed completion tokens to stored assistant text. ~3.4-3.6x, because
         these are reasoning models: `reasoning_tokens` are inside `completion_tokens` and
         the hidden reasoning is never written to the transcript. Only ~28% of the billed
         completion is recoverable as text, so this factor is doing real work.

`cache` (the observed cache-read share of prompt) is calibrated the same way.

**Priced per TURN, never on aggregates.** `gpt-5.4` carries a long-context surcharge chosen
by *that call's* prompt size (`cost.py::rates_for`), so summing 40M tokens into one notional
call triggers the surcharge for everything and overstates the bill by 60-78% — measured. The
turn is the unit the provider billed, so the turn is the unit priced here.

**Validated by leave-one-out** across the two runs that have a ledger: calibrate on one,
predict the other's explorer cost, compare against the ledger's own per-call estimates.

    fit no_survey                  -> no_survey_no_expl_guideline   $49.07 vs $46.39   +5.8%
    fit no_survey_no_expl_guideline -> no_survey                    $41.45 vs $43.91   -5.6%

So treat the explorer output as ±6%. A run with a real ledger always uses the ledger instead.

The JUDGE and the EXTRACTOR have no transcripts at all, so they are predicted from the thing
that drives each — chosen by testing candidate predictors on the two ledgered runs and keeping
whichever disagreed least between them:

  judge       one call per solver trace, EXACTLY (2044/2044 and 2114/2114, delta 0 in both),
              and its prompt is the trajectory it reads. So cost is carried by solver-trace
              TOKENS, not by call count: $/trace-token disagrees by 3.2% between the two runs
              against 13.6% for $/call.
  extraction  cost per FAILED attempt: $0.0317 vs $0.0319, 0.6% apart — the tightest constant
              here. (Call count is not the unit: 1456 and 1429 calls against 345 and 301
              failures, a relation that does not hold, so nothing is built on it.)

The three predictions are then normalised onto the bucket total, which is exact (session costs
minus the traced solver, an identity checked against both ledgered runs). Unnormalised they
over-predict it by ~14%, and the bar must not inherit that: the total is measured, the split
is modelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# The explorer's own model, not the solver's — they differ, and it matters: the solver runs
# on gpt-5.4-mini while the explorer, judge and extraction run on gpt-5.4. Pricing explorer
# turns at the solver's model understates them roughly threefold.
DEFAULT_EXPLORER_MODEL = "gpt-5.4"


def _encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def explorer_model(run: Path) -> str:
    """`generation.explorer_model` from the run's own config, or the default."""
    import yaml

    try:
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return DEFAULT_EXPLORER_MODEL
    return str((cfg.get("generation") or {}).get("explorer_model")
               or DEFAULT_EXPLORER_MODEL)


def explorer_turns(run: Path) -> list[tuple[int, int]]:
    """(history prompt tokens, stored completion tokens) for every explorer turn in the run.

    "History" is the accumulated conversation only — `base` is added at pricing time, so the
    same turn list can be priced under different calibrations. The final turn of each call is
    the one that emitted the accepted spec: the explorer returns on it WITHOUT logging it
    (`appworld/explorer.py`), so it is added here with the artifact as its completion. That
    +1 per call was confirmed against the ledger, which holds exactly 1570 explorer calls for
    each ledgered run against 1570 predicted.
    """
    enc = _encoder()
    tk = lambda s: len(enc.encode(str(s or ""), disallowed_special=()))  # noqa: E731
    out: list[tuple[int, int]] = []
    for path in sorted((run / "sessions").glob("session_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        calls = [rec.get("explorer") or {}]
        calls += (rec.get("refinements") or []) + (rec.get("novelty_refinements") or [])
        for call in calls:
            tr = call.get("transcript") or []
            if not tr and not call.get("artifact"):
                continue
            # `content` carries a rejected spec; `thought`+`code` an executed turn.
            asst = [tk(e.get("thought")) + tk(e.get("code")) + tk(e.get("content")) for e in tr]
            user = [tk(e.get("output")) for e in tr]
            emitted = tk(json.dumps(call.get("artifact") or {}))
            for n in range(1, len(tr) + 2):
                comp = asst[n - 1] if n <= len(tr) else emitted
                out.append((sum(asst[:n - 1]) + sum(user[:n - 1]), comp))
    return out


def trace_tokens(run: Path) -> int:
    """Total tokens of solver-trace text — what the judge reads, one trace at a time."""
    from daedalus.core.logging.paper_cost import _read_traces

    enc = _encoder()
    tk = lambda s: len(enc.encode(str(s or ""), disallowed_special=()))  # noqa: E731
    total = 0
    for trace in _read_traces(run):
        for turn in trace.get("turns") or []:
            total += (tk(turn.get("content")) + tk(turn.get("thought"))
                      + tk(turn.get("code")) + tk(turn.get("output")))
    return total


def failed_attempts(run: Path) -> int:
    """Solver attempts that failed — the unit the extractor is invoked over."""
    n = 0
    for path in sorted((run / "sessions").glob("session_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for att in (rec.get("accumulation_summary") or {}).get("attempts") or []:
            n += att.get("outcome") == "failure"
    return n


def calibrate_rest(run: Path, events: Iterable[Any]) -> tuple[float, float] | None:
    """(judge $/trace-token, extraction $/failed attempt) from a ledgered run."""
    judge = extraction = 0.0
    for e in events:
        if e.role == "judge":
            judge += float(e.estimated_cost_usd or 0.0)
        elif e.role == "extraction":
            extraction += float(e.estimated_cost_usd or 0.0)
    toks, fails = trace_tokens(run), failed_attempts(run)
    # A run whose ledger holds only the post-hoc consolidation call has judge == extraction == 0
    # while still having traces and failures, so it would calibrate to (0, 0) and, averaged in
    # with real calibrations, drag every later estimate toward zero. It is not a calibration.
    if not toks or not fails or judge <= 0 or extraction <= 0:
        return None
    return judge / toks, extraction / fails


def calibrate(run: Path, events: Iterable[Any]) -> tuple[float, float, float] | None:
    """(base, mult, cache) from a run whose ledger recorded its explorer calls."""
    prompt = completion = cached = 0
    for e in events:
        if e.role != "explorer":
            continue
        prompt += e.tokens.prompt_tokens
        completion += e.tokens.completion_tokens
        cached += e.tokens.cache_read_tokens
    turns = explorer_turns(run)
    stored = sum(c for _, c in turns)
    if not turns or not prompt or not stored:
        return None
    base = (prompt - sum(h for h, _ in turns)) / len(turns)
    return max(base, 0.0), completion / stored, cached / prompt


def estimated_cost(run: Path, cal: tuple[float, float, float], tier: str) -> float | None:
    """Price every reconstructed explorer turn under one calibration. Per turn — see module."""
    from daedalus.core.logging.cost import get_price_snapshot
    from daedalus.core.logging.usage import TokenUsage

    turns = explorer_turns(run)
    if not turns:
        return None
    base, mult, cache = cal
    snap = get_price_snapshot(None)
    model = explorer_model(run)
    total = 0.0
    for hist, stored in turns:
        prompt = int(hist + base)
        total += snap.price_call(model, TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=int(stored * mult),
            cache_read_tokens=int(prompt * cache),
        ), tier)
    return total
