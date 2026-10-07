"""AutomationBench smoke test: one tiny inference + one tiny accumulation.

Run:   uv run python tests/smoke_automationbench.py
       (or)  uv run python -m tests.smoke_automationbench

Needs the automationbench extra (`datasets`) and an AutomationBench checkout — no
server, no container, no account: the world is one in-process pydantic object. The
generation half runs one judge-graded self-play session: the explorer designs and executes
a task in a borrowed world, the solver attempts it, the judge grades it. Raises a clear
error if the checkout or the extra is missing.
"""

from __future__ import annotations

from tests._smoke import require, run_generation, run_inference


def main() -> None:
    require(
        "datasets", "automationbench", "AutomationBench",
        note="Its domain task builders return HuggingFace Datasets.",
    )
    try:
        from daedalus.benchmarks.automationbench import ensure_automationbench_importable

        root = ensure_automationbench_importable(None)
    except FileNotFoundError as e:
        raise RuntimeError(
            "[AutomationBench] no checkout found. Fetch the submodule:\n"
            "  git submodule update --init benchmarks/AutomationBench\n"
            "or set automationbench.repo_root in the config. "
            f"Underlying error: {e}"
        ) from e
    print(f"[smoke] AutomationBench checkout: {root}")

    run_inference("smoke/automationbench_inference.yaml")
    run_generation("smoke/automationbench_generation.yaml")
    print("\n✅ AutomationBench smoke OK (inference + generation)")


def test_smoke_automationbench() -> None:  # pytest entry point
    main()


if __name__ == "__main__":
    main()
