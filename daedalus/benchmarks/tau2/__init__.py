"""τ²-bench integration.

Kept stdlib-only so importing the package stays cheap; the heavyweight tau2 imports
live in the submodules.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


def prompt_dirs(domain: str | None) -> list[Path]:
    """τ²'s prompt search path for `domain`, most specific first.

    τ² is one benchmark wrapping several customer-service environments, and a little of
    what a prompt has to say is bound to the environment rather than to τ² — which lookup
    tools exist and how they fail, what a customer of THIS service would ask for. That
    text lives in prompts/<domain>/ and shadows the shared copy in prompts/; everything
    the environments genuinely share stays in prompts/ and is written once.

    A domain with no directory of its own simply falls through to the shared prompts, so
    adding a domain costs nothing until it needs to say something different.
    """
    domain_dir = PROMPTS / domain if domain else None
    if domain_dir is not None and domain_dir.is_dir():
        return [domain_dir, PROMPTS]
    return [PROMPTS]
