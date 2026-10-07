"""A task whose world directory is missing must be filtered out, not passed to the evaluator."""

from __future__ import annotations

from daedalus.benchmarks.appworld.runner import split_by_world_dir


def test_task_without_dbs_dir_is_separated(tmp_path):
    for tid in ("good", "broken"):
        (tmp_path / tid).mkdir()
    (tmp_path / "good" / "dbs").mkdir()  # "broken" was killed before it wrote its DBs

    gradable, skipped = split_by_world_dir(tmp_path, ["good", "broken"])

    assert gradable == ["good"]
    assert skipped == ["broken"]


def test_order_is_preserved_and_unknown_ids_count_as_missing(tmp_path):
    for tid in ("a", "c"):
        (tmp_path / tid / "dbs").mkdir(parents=True)

    gradable, skipped = split_by_world_dir(tmp_path, ["a", "b", "c", "d"])

    assert gradable == ["a", "c"]
    assert skipped == ["b", "d"]
