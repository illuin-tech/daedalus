"""Offline checks for the plumbing shared by the reference methods (`references/common`)."""

from __future__ import annotations

import json

import pytest

from references.common.accumulation import restore, run_tasks
from references.common.run import InferenceMethod, Override


def _square(task_id: str, index: int) -> tuple[int, dict]:
    """Top-level (picklable) worker; task "boom" fails like a crashed environment."""
    if task_id == "boom":
        raise RuntimeError("env died")
    return index, {"task_id": task_id, "value": index * index}


@pytest.mark.parametrize("max_workers", [1, 2])
def test_run_tasks_orders_results_and_drops_failures(max_workers):
    jobs = [(0, "a"), (1, "boom"), (2, "c"), (3, "d")]
    results = run_tasks(_square, jobs, lambda i, tid: (tid, i), max_workers)
    assert [i for i, _ in results] == [0, 2, 3]
    assert [r["task_id"] for _, r in results] == ["a", "c", "d"]


def test_restore_reads_checkpoints_and_lists_the_rest(tmp_path):
    (tmp_path / "t1.json").write_text(json.dumps({"task_id": "t1"}), encoding="utf-8")
    ordered, pending = restore(["t0", "t1", "t2"], lambda t: tmp_path / f"{t}.json")
    assert ordered == [None, {"task_id": "t1"}, None]
    assert pending == [(0, "t0"), (2, "t2")]


def test_every_method_declares_its_inference_hooks():
    import importlib

    for setup in ("erl", "expel", "autoguide", "reasoningbank", "ace"):
        method = importlib.import_module(f"references.{setup}.run").METHOD
        assert isinstance(method, InferenceMethod)
        assert method.setup == setup == method.config_cls.SETUP
        fields = {f for f in method.config_cls.__dataclass_fields__}
        for override in method.overrides:
            assert isinstance(override, Override)
            assert override.field in fields, (setup, override.field)
