"""Shared guardrails for the coverage-survey explorers (the Surveyor).

THE CONTRACT (identical for every benchmark)
  A survey may interact with the environment ONLY through the same channel the solver uses,
  and that channel must never reach:
    (a) the host / process (filesystem, shell, arbitrary imports),
    (b) the benchmark's own package internals, or
    (c) the evaluation / test task set (task specs, verifiers, ground-truth answers).
  Changing the environment's own state is EXPECTED (the survey works in a throwaway copy and
  must call write operations to learn what they demand); these guards only stop escapes.

WHY THE ENFORCEMENT DIFFERS PER BENCHMARK
  The contract is the same; the escape surface is not, because the execution models differ:
    - AppWorld runs ARBITRARY PYTHON (`world.execute`). Its native guard blocks only
      destructive builtins (open/system/exit), NOT reads (pathlib/os.listdir) or imports, so
      a survey could read `data/tasks` and the package source. -> needs `guard_python_code`,
      which refuses host/package/task-set access while leaving every `apis.*` call (reads and
      writes alike) available.
    - tau2 is FUNCTION-CALLING only (reset_environment / read_db / domain tools / submit).
      read_db navigates the domain DB dump only. No code, no filesystem, no task-set access
      -> the guard is an allowlist of tool names (`guard_tool_call`); nothing else is possible.
    - AutomationBench is function-calling over its REST tools, also guarded by the allowlist.

  So the AppWorld code filter and the tool allowlist are two enforcements of ONE contract.
"""

from __future__ import annotations

import ast

# ── (a)+(b)+(c) for a Python-exec channel (AppWorld) ──────────────────────────
# Import roots that reach the host, the process, or package internals. Denied wholesale:
# a survey never needs them — it explores through the environment's own API object.
_DENY_IMPORT_ROOTS = {
    "os", "sys", "inspect", "pkgutil", "pathlib", "importlib", "subprocess", "shutil",
    "glob", "socket", "requests", "httpx", "urllib", "builtins", "sqlite3", "ctypes",
    "multiprocessing", "pickle", "marshal", "io", "fcntl", "resource", "ftplib",
    # benchmark packages themselves (reading their source / task loaders is an escape)
    "appworld", "daedalus", "tau2", "automationbench",
}
# Substrings that signal a filesystem read, a dynamic escape, or a reach for the task set.
_DENY_SUBSTR = (
    # filesystem / dynamic escape
    "open(", "read_text", "read_bytes", "iterdir", "listdir", "walk(", ".glob(",
    "__import__", "eval(", "exec(", "compile(", "globals(", "getattr(__", "__builtins__",
    "__file__", "os.environ", "getenv", "subprocess", "popen",
    # reaching for the eval / task set (all three benchmarks, plus their loaders)
    "load_task_ids", "appworld_root", "data/tasks", "api_docs/standard", "ground_truth",
    "load_dataset", "task_ids", "test_normal", "test_challenge", "specs.json",
)


def guard_python_code(code: str, extra_deny_substr: tuple[str, ...] = ()) -> str | None:
    """AppWorld channel: reject code that escapes the apis-only sandbox; else None.

    Allowed: `apis.*` calls (incl. `apis.api_docs.*`) and ordinary data-processing
    (json, re, collections, math, datetime, itertools, string). Denied: host/filesystem
    access, package introspection, dynamic-exec, and any reach for the task/eval set.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # let the environment surface the syntax error the normal way
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _DENY_IMPORT_ROOTS:
                    return _msg(f"import of '{a.name}'")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _DENY_IMPORT_ROOTS:
                return _msg(f"import from '{node.module}'")
    low = code.lower()
    for s in _DENY_SUBSTR + tuple(extra_deny_substr):
        if s in low:
            return _msg(f"use of '{s.strip()}'")
    return None


def _msg(what: str) -> str:
    return (f"{what} is not available — explore ONLY through the environment's own API "
            "surface (e.g. `apis.*`); the host filesystem, package internals, and any task/"
            "eval set are off-limits.")


# ── tool-call channel (tau2, AutomationBench) ─────────────────────────────────
def guard_tool_call(name: str, allowed: set[str]) -> str | None:
    """Function-calling channel: allow only the intended tools; else return a reason.

    `allowed` is the survey's whitelist — the environment's read/discovery tools plus its
    survey terminal (e.g. domain read tools + read_db/run_sql + submit). Anything else (a
    tool that could enumerate other worlds, read configs, or write) is refused.
    """
    if name in allowed:
        return None
    return (f"tool '{name}' is not available to the survey — use only the environment's own "
            f"read/discovery tools ({', '.join(sorted(allowed))}).")
