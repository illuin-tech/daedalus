"""Every Retriever subclass must accept the base `on_task_start` keyword arguments.

Callers invoke it by keyword — `retriever.on_task_start(instruction, task_id=task_id)` in
core/agents/memory.py and in all three benchmark agents — so a subclass whose override
drops a parameter raises TypeError on every task. That is not a crash you notice: the
per-task error is swallowed into the task's result, the run "completes", and the arm
scores 0/0 with five green runs. AutoGuide shipped exactly that way when `task_id` was
added to the base signature and its override was missed.

Signature-only, so it needs no API key and no bank.
"""

from __future__ import annotations

import inspect

import pytest

from daedalus.core.memory.retrieval import Retriever


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _all_retrievers():
    # Import the reference methods so their subclasses register.
    import references.autoguide.retrieval  # noqa: F401
    import references.erl.retrieval  # noqa: F401

    return [c for c in _subclasses(Retriever) if "on_task_start" in c.__dict__]


BASE_PARAMS = set(inspect.signature(Retriever.on_task_start).parameters) - {"self"}


def test_base_signature_is_what_callers_use():
    """Guards the constant this test compares against."""
    assert BASE_PARAMS == {"task_instruction", "task_id"}


@pytest.mark.parametrize("cls", _all_retrievers(), ids=lambda c: c.__module__ + "." + c.__name__)
def test_override_accepts_every_base_parameter(cls):
    params = inspect.signature(cls.on_task_start).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # **kwargs absorbs anything
    missing = BASE_PARAMS - set(params)
    assert not missing, (
        f"{cls.__module__}.{cls.__name__}.on_task_start is missing {sorted(missing)}; "
        f"callers pass those by keyword, so every task would raise TypeError"
    )


def test_autoguide_override_is_callable_with_task_id():
    """The specific break: AutoGuide's override, called the way the agents call it."""
    from references.autoguide.retrieval import ContextAwareRetriever

    inspect.signature(ContextAwareRetriever.on_task_start).bind(
        None, "some instruction", task_id="42"
    )
