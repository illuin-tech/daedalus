"""tau2-bench smoke test: one tiny inference + one tiny self-play generation.

Run:   uv run python tests/smoke_tau2.py
       (or)  uv run python -m tests.smoke_tau2

Requires the tau2 extra + the benchmarks/tau2-bench submodule (see README).
Raises a clear error if tau2 isn't available.
"""

from __future__ import annotations

from tests._smoke import require, run_generation, run_inference


def main() -> None:
    require(
        "tau2", "tau2", "tau2-bench",
        note="tau2-bench is not on PyPI — fetch the benchmarks/tau2-bench submodule (see README).",
    )
    run_inference("smoke/tau2_inference.yaml")
    run_generation("smoke/tau2_generation.yaml")
    print("\n✅ tau2 smoke OK (inference + generation)")


def test_smoke_tau2() -> None:  # pytest entry point
    main()


if __name__ == "__main__":
    main()
