from argparse import Namespace

from daedalus.core.testset.manifest import TestSet as FrozenTestSet
from daedalus.scripts.tasks_as_test_set import build_cfg


def _args() -> Namespace:
    return Namespace(
        max_turns=None,
        reasoning_effort=None,
        temperature=None,
        num_runs=1,
        parallel=1,
        force=False,
        provider_order=None,
        own_heuristic=False,
    )


def _test_set(generator_model: str) -> FrozenTestSet:
    return FrozenTestSet(
        generation_run="generation",
        benchmark="appworld",
        dataset="train",
        generator_model=generator_model,
        tasks=[],
    )


def test_testset_eval_does_not_inherit_other_model_provider_routing():
    raw = {
        "agent": {"model": "openrouter/qwen/qwen3.6-35b-a3b"},
        "provider_routing": {"order": ["parasail/fp8"], "allow_fallbacks": False},
    }
    cfg = build_cfg(
        raw,
        _test_set("openrouter/qwen/qwen3.6-35b-a3b"),
        "openrouter/xiaomi/mimo-v2.5",
        _args(),
    )

    assert cfg.provider_routing.to_body() is None


def test_testset_eval_preserves_routing_for_the_generator_model():
    raw = {
        "agent": {"model": "openrouter/qwen/qwen3.6-35b-a3b"},
        "provider_routing": {"order": ["parasail/fp8"], "allow_fallbacks": False},
    }
    cfg = build_cfg(
        raw,
        _test_set("openrouter/qwen/qwen3.6-35b-a3b"),
        "openrouter/qwen/qwen3.6-35b-a3b",
        _args(),
    )

    assert cfg.provider_routing.to_body() == {
        "order": ["parasail/fp8"],
        "allow_fallbacks": False,
    }


def test_provider_order_pins_a_cross_model_evaluation():
    """--provider-order survives the reset that fires when the solver did not generate the bank.

    Without it every cross-model test-set run is unpinned, which is how a 0%-scoring upstream
    put three published test-set numbers about 30 points low.
    """
    raw = {
        "agent": {"model": "gpt-5.4"},
        "provider_routing": {"order": ["parasail/fp8"], "allow_fallbacks": False},
    }
    args = _args()
    args.provider_order = ["sail-research/fp4"]
    cfg = build_cfg(
        raw, _test_set("gpt-5.4"), "openrouter/deepseek/deepseek-v4-flash-0731", args
    )

    assert cfg.provider_routing.to_body() == {
        "order": ["sail-research/fp4"],
        "allow_fallbacks": False,
    }
