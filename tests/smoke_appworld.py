"""AppWorld smoke test: one tiny inference + one tiny self-play generation.

Run:   uv run python tests/smoke_appworld.py
       (or)  uv run python -m tests.smoke_appworld

Requires the AppWorld extra + the benchmarks/appworld submodule (see README).
Raises a clear error if AppWorld isn't available.
"""

from __future__ import annotations

from tests._smoke import require, run_generation, run_inference


def main() -> None:
    require(
        "appworld", "appworld", "AppWorld",
        note="AppWorld 0.2.0 is not on PyPI — fetch the benchmarks/appworld submodule (see README).",
    )
    run_inference("smoke/appworld_inference.yaml")
    run_generation("smoke/appworld_generation.yaml")
    print("\n✅ AppWorld smoke OK (inference + generation)")


def test_smoke_appworld() -> None:  # pytest entry point
    main()


if __name__ == "__main__":
    main()
