"""Every generation backend must be constructible (no API key, no benchmark data)."""

from __future__ import annotations

import pytest

from daedalus.core.config import BENCHMARKS, ExperimentConfig
from daedalus.core.generation.backend import get_generation_backend


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_generation_backend_is_instantiable(benchmark):
    backend = get_generation_backend(ExperimentConfig(benchmark=benchmark))
    assert backend.heuristic_admissible("any text", ExperimentConfig(benchmark=benchmark)) == (
        True, "")
