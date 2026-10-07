"""Hiding held-out apps from AppWorld's generation agents.

`generation.excluded_apps` names apps that generated tasks may never build on — they are
held out so evaluation can test transfer to an app the memory has never seen. Dropping
them from the app catalog we paste into the prompt is not enough: AppWorld's channel is
arbitrary Python, so an agent can enumerate the surface at runtime
(`apis.api_docs.show_app_descriptions()`, `dir(apis)`), read a held-out app's docs, or see
its name in an error message. This module closes both directions:

  code    a block naming a hidden app is refused before it runs;
  output  every execution result is filtered, so a hidden app never appears in a catalog,
          a doc search, a listing, or an error.

The agent therefore explores an environment in which those apps are simply not there, and
the pipeline's `excluded_used` check becomes a backstop rather than the only defense.

WHAT COUNTS AS THE APP, AND WHAT DOES NOT
  Only the app *identity* is hidden: `apis.gmail...`, the bare token in a listing or an
  error, a `name` / `app_name` / `account_name` field equal to it, and the app segment of an
  API path (`/gmail/emails` — every api_doc carries one, and an agent that reshapes results
  down to `path` strings would otherwise walk straight past the record-level filter).
  An address like `joyce-weav@gmail.com` is left alone: the supervisor's own email is a gmail
  address, so scrubbing those would corrupt every contact, transaction and profile in the
  world, and generated tasks would carry mangled addresses. The residue is that an agent may
  infer from an address that a mail app exists somewhere; it still has no catalog entry, no
  docs and no way to call one.

KNOWN LIMIT: DERIVED COUNTS
  This filter sees the printed OUTPUT of a code block, not the values living inside it. Code
  that holds the live catalog and prints something derived from it —
  `print("total apps:", len(apis.api_docs.show_app_descriptions()))` — emits `total apps: 11`
  where only 9 are nameable, and there is nothing on that line to match. Closing it would mean
  filtering at the api boundary rather than the output. Left open deliberately: a count cannot
  be turned into a task, whereas every channel that could (the catalog, `dir(apis)`, the docs,
  doc search, stored credentials, error rosters, and the calls themselves) is closed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

_DECODER = json.JSONDecoder()

# Fields whose value names the app a record is about, so the whole record goes with it.
_IDENTITY_KEYS = ("name", "app", "app_name", "app_name_", "account_name")


class HiddenApps:
    """The apps an agent must not know exist, plus the two filters that hide them.

    Falsy (and every method a no-op) when nothing is held out, so callers can wire it in
    unconditionally.
    """

    def __init__(self, names: Sequence[str] = ()):
        self.names = tuple(sorted({n.strip().lower() for n in names if n and n.strip()}))
        self._pats = []
        for n in self.names:
            e = re.escape(n)
            self._pats.append(re.compile(
                # `apis.gmail...` — attribute access, the one identity form that sits behind
                # a dot — or the bare token. A URL path counts (`/gmail/emails` is how every
                # api_doc names its app), an address or longer identifier does not.
                rf"\bapis\s*\.\s*{e}\b|(?<![\w.@-]){e}(?![\w.@-])",
                re.IGNORECASE,
            ))

    def __bool__(self) -> bool:
        return bool(self.names)

    def mentions(self, text: str) -> bool:
        """Whether `text` names a hidden app (as the app, not as part of an address)."""
        return any(p.search(text) for p in self._pats)

    def blocked(self, code: str) -> str | None:
        """What to show instead of running `code`, or None if it names no hidden app.

        Verbatim the failure AppWorld itself raises for an app that does not exist, so a
        held-out app is indistinguishable from a typo. A refusal phrased as policy is its own
        signal: the survey that first ran against this module reported back that "guessed
        hidden app names were blocked", i.e. it had worked out that something was being kept
        from it. This says nothing to work out.
        """
        for name, pat in zip(self.names, self._pats):
            if pat.search(code):
                return ("Execution failed. Traceback:\n"
                        '  File "<python-input>", line 1, in <module>\n'
                        f"Exception: No app named '{name}' found.")
        return None

    def filter(self, text: str) -> str:
        """Execution output with every trace of a hidden app removed.

        AppWorld returns JSON (indent=1) for API results, sometimes inside a traceback or
        interleaved with the agent's own prints. So we prune each JSON value in place —
        dropping the records that are *about* a hidden app — and scrub what is left in the
        surrounding plain text (e.g. the app names a 422 lists back).
        """
        if not self or not self.mentions(text):
            return text
        return self._drop_lines(self._prune_json(text))

    # ── internals ────────────────────────────────────────────────────────────
    def _is_name(self, value: Any) -> bool:
        return isinstance(value, str) and value.strip().lower() in self.names

    def _prune_json(self, text: str) -> str:
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch in "[{":
                try:
                    value, end = _DECODER.raw_decode(text, i)
                except ValueError:
                    out.append(ch)
                    i += 1
                    continue
                out.append(json.dumps(self._prune(value), indent=1))
                i = end
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _prune(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._prune(v) for v in value if not self._drop(v)]
        if isinstance(value, dict):
            return {k: self._prune(v) for k, v in value.items() if not self._is_name(k)}
        if isinstance(value, str):
            return self._scrub(value)
        return value

    def _drop(self, value: Any) -> bool:
        """A list entry goes entirely when it is *about* a hidden app: the catalog record,
        the `dir(apis)` name, the doc-search hit, the stored credential, or a bare string
        naming it — an agent that reshapes results keeps only strings like `/gmail/emails`,
        so a scrubbed `//emails` would be both a leak and nonsense.

        Dict records that merely *mention* one (a payment memo, a note) are kept and scrubbed
        instead: deleting them would misreport what the world contains."""
        if isinstance(value, str):
            return self.mentions(value)
        return isinstance(value, dict) and any(
            self._is_name(value.get(k)) for k in _IDENTITY_KEYS
        )

    def _drop_lines(self, text: str) -> str:
        """Free text outside any JSON value: drop whole LINES that name a hidden app.

        Excising just the name is not enough, because agents format their own tables from
        results they already hold — `f"{a['name']}: {n} ops"` — and a scrubbed line reads
        `": 48 ops"`, which announces an app whose name was withheld. A survey run against the
        scrub-only version reported back that it was looking for "the blank shopping/email
        apps", so the residue leaked exactly what the filter exists to hide. The line is the
        unit an agent formats in, so the line is what has to go.

        JSON string values are still scrubbed in place by _scrub (via _prune), where there are
        no lines to speak of — that is what keeps a 422's app roster well-formed.
        """
        if "\n" not in text:
            return "" if self.mentions(text) else text
        kept = [ln for ln in text.split("\n") if not self.mentions(ln)]
        return "\n".join(kept)

    def _scrub(self, text: str) -> str:
        """Drop the name from free text, taking the list punctuation around it with it."""
        for pat in self._pats:
            if not pat.search(text):
                continue
            text = re.sub(rf"['\"]?(?:{pat.pattern})['\"]?\s*,\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(rf",\s*['\"]?(?:{pat.pattern})['\"]?", "", text, flags=re.IGNORECASE)
            text = pat.sub("", text)
        return text
