"""The output layout: every config writes where the released outputs hold the paper's run.

The release names folders in the paper's terms (daedalus/, daedalus-curated/, baselines/,
evaluation-proxy/), and a config's `experiment_name` + `category` is that folder. These pin
the mapping so a re-run lands on the released run rather than beside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daedalus.core.config import (
    config_from_dict,
    experiment_dir,
    find_run_dir,
    load_config,
    run_id,
)

REPO = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    return config_from_dict({"benchmark": "appworld", "experiment_name": "x", **kw})


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"kind": "generation"}, "outputs/daedalus/x"),
        ({"kind": "accumulation"}, "outputs/daedalus-curated/x"),
        ({"kind": "testset"}, "outputs/evaluation-proxy/x"),
        ({"kind": "inference", "category": "main-results"},
         "outputs/inference/appworld/main-results/x"),
        ({"kind": "inference"}, "outputs/inference/appworld/misc/x"),
        ({"kind": "accumulation", "setup": "erl"}, "outputs/baselines/erl/memory/x"),
        # A baseline holds one evaluation per benchmark: no <benchmark>/<category> levels.
        ({"kind": "inference", "setup": "erl", "category": "ignored"},
         "outputs/baselines/erl/inference/x"),
    ],
)
def test_experiment_dir(kw, expected):
    assert experiment_dir(_cfg(**kw)) == Path(expected)


def _header_output(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Output: "):
            return line.removeprefix("# Output: ").split()[0].rstrip("/")
    raise AssertionError(f"{path} has no '# Output:' header")


_PAPER_CONFIGS = sorted(
    [p for b in ("appworld", "tau2", "automationbench")
     for p in (REPO / "configs" / b).rglob("*.yaml")]
    + list((REPO / "references").glob("*/configs/*.yaml"))
)
_KIND_BY_FOLDER = {"generation": "generation", "accumulation": "accumulation",
                   "inference": "inference"}


@pytest.mark.parametrize("path", _PAPER_CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_paper_config_writes_to_its_header_folder(path):
    cfg = load_config(path)
    if path.parts[-3] == "references" or path.parent.name == "configs":
        cfg.kind = "accumulation" if path.stem.endswith("accumulation") else "inference"
        cfg.setup = path.parts[-3]
    else:
        cfg.kind = _KIND_BY_FOLDER[path.parent.name if path.parent.name != "models"
                                   else "inference"]
    assert experiment_dir(cfg).as_posix() == _header_output(path)


def test_runs_are_addressed_by_their_path(tmp_path, monkeypatch):
    """Folder names repeat across benchmarks, so a run id is its path under the kind root."""
    monkeypatch.chdir(tmp_path)
    for bench in ("appworld", "tau2"):
        d = tmp_path / "outputs/inference" / bench / "main-results/no-memory"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text("{}")
    tau2 = Path("outputs/inference/tau2/main-results/no-memory")
    assert run_id("inference", "daedalus", tau2) == "tau2/main-results/no-memory"
    assert find_run_dir("inference", "daedalus", "tau2/main-results/no-memory") == tau2
