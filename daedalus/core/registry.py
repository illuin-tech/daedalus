"""Benchmark registry. Lazy imports so running one benchmark never imports the
others' (heavyweight) packages."""

from __future__ import annotations

import importlib

from daedalus.core.benchmark import Benchmark
from daedalus.core.config import ExperimentConfig

_REGISTRY = {
    "appworld": "daedalus.benchmarks.appworld.runner.AppWorldBenchmark",
    "tau2": "daedalus.benchmarks.tau2.runner.Tau2Benchmark",
    "automationbench": "daedalus.benchmarks.automationbench.runner.AutomationBenchBenchmark",
}


def get_benchmark(cfg: ExperimentConfig) -> Benchmark:
    """Instantiate the Benchmark implementation selected by cfg.benchmark."""
    try:
        cls_path = _REGISTRY[cfg.benchmark]
    except KeyError:
        raise ValueError(
            f"Unknown benchmark {cfg.benchmark!r}; known: {sorted(_REGISTRY)}"
        ) from None
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()
