"""AutomationBench connector: locate the cloned repo and make `automationbench.*`
importable.

AutomationBench is Zapier's business-workflow benchmark (47 simulated SaaS apps in one
in-process pydantic `WorldState`, 800 public tasks, deterministic final-state
assertions). Its scoring is a pure function of the final world, and it needs no server,
container or account.

It is a *package*, but we do not install it: `automation-bench` depends on
`verifiers>=0.2.0`, which pulls prime-sandboxes, prime-tunnel, openai-agents, mcp, gepa
and more, and this connector uses none of it. Instead we put the checkout on `sys.path`
and import the four things we need — the world schema, the domain task loaders, the
tools, and the rubric — none of which import `verifiers`. The package's own data
(domains, endpoint schemas) resolves relative to its `__file__`.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_setup_lock = threading.Lock()
_resolved_root: Path | None = None


def find_automationbench_root(explicit: str | None = None) -> Path:
    """Locate the AutomationBench checkout.

    Order: explicit config value (`automationbench.repo_root`), then the
    `benchmarks/AutomationBench` submodule. A directory only counts if it carries the two
    markers we actually import from (`automationbench/domains` and
    `automationbench/rubric`), so an empty or unrelated folder fails instead of
    half-loading.
    """

    def is_root(p: Path) -> bool:
        pkg = p / "automationbench"
        return (pkg / "domains").is_dir() and (pkg / "rubric").is_dir()

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not is_root(p):
            raise FileNotFoundError(
                f"automationbench.repo_root={p} is not an AutomationBench checkout "
                f"(expected {p}/automationbench/domains and {p}/automationbench/rubric)"
            )
        return p

    submodule = Path(__file__).resolve().parents[3] / "benchmarks" / "AutomationBench"
    if is_root(submodule):
        return submodule

    raise FileNotFoundError(
        f"Could not locate the AutomationBench checkout at {submodule}. Set "
        "automationbench.repo_root in the config, or fetch the submodule:\n"
        "  git submodule update --init benchmarks/AutomationBench"
    )


def ensure_automationbench_importable(explicit: str | None = None) -> Path:
    """Idempotently put the AutomationBench checkout on `sys.path`.

    Called from every entry point into the connector (task loading, session setup), so a
    worker process that only ever touches one of them is still wired correctly. Cheap
    after the first call.
    """
    global _resolved_root
    with _setup_lock:
        if _resolved_root is not None:
            return _resolved_root
        root = find_automationbench_root(explicit)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _resolved_root = root
        return root
