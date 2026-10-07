"""Centralized, colorless console output for daedalus.

Every benchmark and the core pipeline route their user-facing progress/status
text through this one module, so a run reads the same clean way no matter which
benchmark drives it. No colors, no `rich`, no emoji — just plain text that reads
well in a terminal and copies cleanly into a log file. All output is flushed
immediately so interleaved parallel-worker lines are never lost.

Two layers:

  * module functions — ``rule`` / ``header`` / ``info`` / ``detail`` /
    ``progress`` — for run-level output (headers, summaries, per-task progress).
  * plain strings carry their own ``prefix`` (e.g. ``"[w0] "``) for the
    per-worker generation streams, so a caller whose prefix is assigned after
    construction just passes it in.

``silence_third_party`` removes the un-harmonized output that would otherwise
leak past this module: tau2-bench logs through **loguru**, whose default stderr
sink is ANSI-colorized (we drop that sink), and AppWorld's starlette deprecation
warnings.

It is idempotent and cheap; entry points call it once (via ``configure``) and
subprocess workers call it at start-up so their process is quiet too.
"""

from __future__ import annotations

import sys
from typing import Any

# Fixed rule width so every header/summary lines up regardless of benchmark.
_WIDTH = 64


# ── setup / third-party silencing ────────────────────────────────────────────
def configure(verbose: bool = False) -> None:
    """Prepare console output for a run: silence third-party colored/noisy logs.

    Call once at each entry point (run / accumulate / generate). Idempotent."""
    silence_third_party(verbose=verbose)


def silence_third_party(verbose: bool = False) -> None:
    """Quiet non-daedalus output so only our clean lines remain.

    Removes tau2-bench's colorized loguru sink. When ``verbose``, tau2 warnings are re-attached in
    plain (uncolored) text so genuine problems are still visible."""
    # loguru (tau2-bench): drop the colored sink; optionally re-add a plain one.
    try:
        from loguru import logger as _loguru  # type: ignore

        _loguru.remove()
        if verbose:
            _loguru.add(
                sys.stderr,
                level="WARNING",
                colorize=False,
                format="{time:HH:mm:ss} | {level} | {message}",
            )
    except Exception:  # loguru not installed / tau2 not in use — nothing to do.
        pass

    # warnings (AppWorld → starlette): building each AppWorld app re-emits a wall
    # of StarletteDeprecationWarnings (deprecated HTTP_422 status name, httpx
    # testclient). They are upstream and not actionable here, so drop the whole
    # category. Called before any world is built, so nothing leaks first.
    import warnings

    try:
        from starlette.exceptions import StarletteDeprecationWarning

        warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
    except Exception:  # starlette not installed (no AppWorld) — nothing to do.
        pass


# ── primitives ───────────────────────────────────────────────────────────────
def _emit(line: str = "") -> None:
    print(line, flush=True)


def info(*args: Any, prefix: str = "", sep: str = " ", **_ignore: Any) -> None:
    """Print one always-on line (accepts ``print``-style args; ``flush`` ignored)."""
    _emit(prefix + sep.join(str(a) for a in args))


def detail(msg: str, verbose: bool = False, prefix: str = "") -> None:
    """Print a line only when ``verbose`` — for per-turn / diagnostic streams."""
    if verbose:
        _emit(f"{prefix}{msg}")


def progress(index: int, total: int, label: str, note: str = "") -> None:
    """A ``[i/n] label`` progress line (e.g. per-task solving)."""
    _emit(f"[{index}/{total}] {label}{note}")


# ── progress bars (clean, monochrome tqdm) ───────────────────────────────────
# One place for progress bars so every task loop looks the same. Bars render to
# stderr (tqdm default), leaving the stdout header/summary lines untouched, and
# degrade to a no-op when tqdm isn't installed.
_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed}<{remaining}{postfix}"


class _NullBar:
    """No-op stand-in when tqdm is unavailable (keeps call sites unconditional)."""

    def update(self, n: int = 1) -> None: ...
    def set_postfix(self, *a: Any, **k: Any) -> None: ...
    def write(self, msg: str) -> None:
        _emit(msg)

    def close(self) -> None: ...
    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


def progress_bar(total: int, desc: str = "solving"):
    """A tqdm bar to drive manually — ``.update()`` / ``.set_postfix()`` / ``.write()``
    / ``.close()``. Use for pools whose items finish out of order (``as_completed``).
    Returns a no-op bar if tqdm isn't installed."""
    try:
        from tqdm import tqdm
    except Exception:
        return _NullBar()
    return tqdm(
        total=total, desc=desc, unit="task", dynamic_ncols=True, leave=True,
        bar_format=_BAR_FORMAT,
    )


def track(iterable, total: int | None = None, desc: str = "solving"):
    """Wrap a sequential iterable in a clean tqdm bar (passthrough without tqdm)."""
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(
        iterable, total=total, desc=desc, unit="task", dynamic_ncols=True, leave=True,
        bar_format=_BAR_FORMAT,
    )


# ── structured blocks ────────────────────────────────────────────────────────
def rule(title: str = "") -> None:
    """A full-width separator; with a title, a titled band (blank line above)."""
    if title:
        _emit()
        _emit("─" * _WIDTH)
        _emit(f"  {title}")
        _emit("─" * _WIDTH)
    else:
        _emit("─" * _WIDTH)


def header(title: str, fields: dict[str, Any] | None = None) -> None:
    """A titled band followed by aligned ``label   value`` lines.

    Used for run headers and end-of-run summaries so both read identically."""
    _emit()
    _emit("─" * _WIDTH)
    _emit(f"  {title}")
    if fields:
        _emit("─" * _WIDTH)
        width = max(len(k) for k in fields)
        for key, value in fields.items():
            _emit(f"  {key.ljust(width)}   {value}")
    _emit("─" * _WIDTH)
