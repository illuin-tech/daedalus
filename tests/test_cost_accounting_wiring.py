"""End-to-end wiring: config → preflight → launch ledger → aggregated run summary.

These are the checks that the pieces are actually CONNECTED — the failure mode the plan
was written against was never a wrong formula, it was a real call that no summary counted.
Fake litellm responses throughout: no benchmark checkout and no network.
"""

from __future__ import annotations

import pytest

import daedalus.core.llm.client as client_mod
from daedalus.core.config import config_from_dict, experiment_dir
from daedalus.core.logging.accounting import CostAccounting
from daedalus.core.logging.preflight import (
    ModelRequirement,
    PricePreflightError,
    preflight_prices,
    required_models,
)
from daedalus.core.logging.usage import TokenUsage
from daedalus.core.logging.usage_aggregate import UsageAggregator


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _response(prompt=1000, completion=100, cached=0, content="hi", model="gpt-5.4"):
    return _Obj(
        id="req-1",
        model=model,
        usage=_Obj(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=_Obj(
                cached_tokens=cached, cache_creation_tokens=None
            ),
            completion_tokens_details=None,
        ),
        choices=[_Obj(message=_Obj(content=content, tool_calls=None))],
    )


@pytest.fixture
def fake_completion(monkeypatch):
    queue: list = []
    monkeypatch.setattr(
        client_mod.litellm,
        "completion",
        lambda **kwargs: queue.pop(0) if queue else _response(),
    )
    return queue


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Experiment folders are resolved relative to the cwd, so give each test its own."""
    monkeypatch.chdir(tmp_path)


def _cfg(tmp_path, **overrides):
    data = {
        "experiment_name": "acct-test",
        "benchmark": "appworld",
        "category": "tests",
        "agent": {"model": "gpt-5.4"},
        "logging": {"trace_dir": str(tmp_path / "traces")},
    }
    data.update(overrides)
    cfg = config_from_dict(data)
    return cfg


# ---- preflight (plan §6.3) -----------------------------------------------------------




def test_ticket_mode_does_not_require_a_user_simulator_price(tmp_path):
    cfg = _cfg(
        tmp_path, benchmark="tau2", kind="accumulation", tau2={"solver_mode": "ticket"}
    )
    assert "user_simulator" not in {r.role for r in required_models(cfg)}


def test_a_non_llm_retriever_needs_no_price(tmp_path):
    cfg = _cfg(tmp_path, memory={"enabled": True, "retriever": {"type": "bm25"}})
    assert "retriever" not in {r.role for r in required_models(cfg)}


def test_preflight_stops_before_paid_work_on_an_unknown_model(tmp_path):
    cfg = _cfg(tmp_path, agent={"model": "gpt-5.9-unreleased"})
    with pytest.raises(PricePreflightError) as excinfo:
        preflight_prices(cfg, announce=False)
    # The message has to name the model AND the config field that chose it.
    assert "gpt-5.9-unreleased" in str(excinfo.value)
    assert "agent.model" in str(excinfo.value)


def test_preflight_can_be_downgraded_to_a_warning(tmp_path):
    cfg = _cfg(tmp_path, agent={"model": "gpt-5.9-unreleased"})
    cfg.cost_accounting.require_complete_estimates = False
    result = preflight_prices(cfg, announce=False)
    assert result["unpriceable"] and not result["models"]


def test_preflight_reports_an_extra_model_the_config_never_mentions(tmp_path):
    cfg = _cfg(tmp_path)
    result = preflight_prices(
        cfg,
        extra=[ModelRequirement("gpt-4o-mini", "judge", "--judge-model")],
        announce=False,
    )
    assert {m["role"] for m in result["models"]} == {"solver", "judge"}


# ---- launch / worker / resume (plan §5, §13.8) ---------------------------------------


def test_start_mints_a_launch_id_that_workers_inherit_through_the_config(tmp_path):
    cfg = _cfg(tmp_path)
    parent = CostAccounting.start(cfg, announce=False)
    assert cfg.cost_accounting.launch_id
    # A spawned worker rebuilds the config from this dict and must land in the same launch.
    worker_cfg = config_from_dict(cfg.to_dict())
    worker = CostAccounting.for_worker(worker_cfg)
    assert worker.ledger.launch_id == parent.ledger.launch_id


def test_a_resume_opens_a_new_launch_and_both_launches_count(tmp_path, fake_completion):
    cfg = _cfg(tmp_path)
    for _ in range(2):
        cfg.cost_accounting.launch_id = ""  # what a fresh process starts from
        acc = CostAccounting.start(cfg, announce=False)
        acc.client("gpt-5.4", "solver").generate([{"role": "user", "content": "x"}])
    summary = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).summary()
    assert summary["num_launches"] == 2
    assert summary["num_completed_calls"] == 2


def test_a_completed_call_survives_a_worker_that_writes_no_summary(
    tmp_path, fake_completion
):
    """The ledger line is flushed inside generate(); nothing later is needed to keep it."""
    cfg = _cfg(tmp_path)
    acc = CostAccounting.start(cfg, announce=False)
    acc.client("gpt-5.4", "solver").generate([{"role": "user", "content": "x"}])
    del acc  # the "worker" dies here: no trace, no session record, no launch summary
    summary = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).summary()
    assert summary["num_completed_calls"] == 1
    assert summary["estimated_total_cost_usd"] > 0


def test_the_run_total_is_the_sum_of_every_role(tmp_path, fake_completion):
    cfg = _cfg(tmp_path)
    acc = CostAccounting.start(cfg, announce=False)
    for role in ("solver", "judge", "extraction", "verifier", "user_simulator"):
        acc.client("gpt-5.4", role).generate([{"role": "user", "content": "x"}])
    summary = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).summary()
    assert set(summary["cost_usd_by_role"]) == {
        "solver",
        "judge",
        "extraction",
        "verifier",
        "user_simulator",
    }
    assert summary["estimated_total_cost_usd"] == pytest.approx(
        sum(summary["cost_usd_by_role"].values())
    )


def test_disabled_accounting_is_refused_at_an_entry_point(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.cost_accounting.enabled = False
    with pytest.raises(RuntimeError):
        CostAccounting.start(cfg, announce=False)


# ---- the tally used for per-task / per-session summaries ------------------------------


def test_usage_since_reports_only_the_work_in_the_window(tmp_path, fake_completion):
    cfg = _cfg(tmp_path)
    acc = CostAccounting.start(cfg, announce=False)
    solver = acc.client("gpt-5.4", "solver")
    solver.generate([{"role": "user", "content": "before"}])
    mark = acc.tally_snapshot()
    solver.generate([{"role": "user", "content": "inside"}])
    acc.client("gpt-5.4", "judge").generate([{"role": "user", "content": "inside"}])
    window = acc.usage_since(mark)
    assert window["calls_by_role"] == {"judge": 1, "solver": 1}
    whole = acc.usage_since(None)
    assert whole["calls_by_role"] == {"judge": 1, "solver": 2}


# ---- external (third-party) calls: tau2's user simulator ------------------------------


def test_an_external_call_is_priced_locally_and_keeps_its_own_figure_apart(tmp_path):
    cfg = _cfg(tmp_path, benchmark="tau2")
    acc = CostAccounting.start(cfg, announce=False)
    acc.record_external(
        "user_simulator",
        "gpt-4.1",
        tokens=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0),
        third_party_cost_usd=1.75,
        third_party_cost_source="tau2_completion_cost",
    )
    (event,) = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).events
    assert event.estimated_cost_usd == pytest.approx(2.0)  # OUR snapshot's gpt-4.1 rate
    assert event.provider_reported_cost_usd == 1.75  # tau2's own number, kept apart
    assert event.provider_cost_source == "tau2_completion_cost"


def test_a_cost_only_external_call_makes_the_run_incomplete(tmp_path):
    """tau2 sometimes exposes `sim.user_cost` and no tokens: that cannot be re-priced."""
    cfg = _cfg(tmp_path, benchmark="tau2")
    acc = CostAccounting.start(cfg, announce=False)
    acc.record_external(
        "user_simulator",
        "gpt-4.1",
        tokens=None,
        third_party_cost_usd=0.5,
        third_party_cost_source="tau2_completion_cost",
    )
    summary = acc.summary()
    assert summary["estimated_total_cost_usd"] is None
    assert summary["cost_complete"] is False
    assert any("token detail" in r for r in summary["cost_incompleteness_reasons"])


# ---- LLM retrieval (plan §8.3) --------------------------------------------------------






# ---- the per-task trace projection (plan §8.1, §9.3) ---------------------------------


def test_a_trace_projects_the_ledger_rather_than_re_estimating(
    tmp_path, fake_completion
):
    from daedalus.core.logging.trace_logger import TraceLogger

    cfg = _cfg(tmp_path)
    acc = CostAccounting.start(cfg, announce=False)
    llm = acc.client("gpt-5.4", "solver")
    llm.scope = acc.scope("solver", task_id="t1")
    logger = TraceLogger("t1", cfg.name, {}, tmp_path / "traces", model="gpt-5.4")

    logger.start_turn(0)
    for _ in range(2):  # two calls in one turn (reason-then-retrieve)
        logger.log_llm_call(llm.generate([{"role": "user", "content": "x"}]))
    logger.end_turn()

    events = UsageAggregator.from_experiment_dir(experiment_dir(cfg)).events
    assert logger.trace.usage_event_ids == [e.event_id for e in events]
    assert logger.trace.total_cost_usd == pytest.approx(
        sum(e.estimated_cost_usd for e in events)
    )
    # Both calls counted on the turn, not just the last one.
    assert logger.trace.turns[0].token_usage["prompt"] == 2000
    assert logger.trace.cost_complete is True


def test_an_unpriceable_call_marks_the_trace_incomplete(tmp_path, fake_completion):
    from daedalus.core.logging.trace_logger import TraceLogger

    cfg = _cfg(tmp_path)
    cfg.cost_accounting.require_complete_estimates = False
    acc = CostAccounting.start(cfg, announce=False)
    llm = acc.client("mystery-9", "solver")
    logger = TraceLogger("t1", cfg.name, {}, tmp_path / "traces", model="mystery-9")

    fake_completion.append(_response(model="mystery-9"))
    logger.start_turn(0)
    logger.log_llm_call(llm.generate([{"role": "user", "content": "x"}]))
    logger.end_turn()

    assert logger.trace.cost_complete is False
    assert logger.trace.total_cost_usd == 0.0  # nothing invented for the unpriced call


# ---- run-cost resolution for readers (run browser + figures) --------------------------


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def test_a_live_ledger_summary_wins_over_everything(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(
        tmp_path / "evaluation.json",
        {
            "usage": {
                "estimated_total_cost_usd": 5.0,
                "cost_complete": True,
                "cost_usd_by_role": {"solver": 3.0, "user_simulator": 2.0},
                "cost_accounting_version": 2,
            }
        },
    )
    _write(tmp_path / "usage_migrated.json", {"recomputed_partial_cost_usd": 1.0})
    info = resolve_run_cost(tmp_path, legacy_total=9.0)
    assert (info["source"], info["total_usd"], info["complete"]) == ("ledger", 5.0, True)


def test_an_incomplete_ledger_total_is_still_used_but_flagged(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(
        tmp_path / "evaluation.json",
        {
            "usage": {
                "estimated_total_cost_usd": None,
                "priced_calls_estimated_cost_usd": 4.0,
                "cost_complete": False,
                "cost_incompleteness_reasons": ["1 call could not be priced"],
            }
        },
    )
    info = resolve_run_cost(tmp_path)
    assert info["total_usd"] == 4.0
    assert info["complete"] is False and "incomplete" in info["label"]


def test_a_zero_call_ledger_block_does_not_shadow_the_migration(tmp_path):
    """Re-scoring a pre-ledger run must not replace its cost with a confident $0.00.

    `UsageAggregator` finds no events in such a folder, so `evaluate_experiment` stamps a
    usage block reporting zero calls into evaluation.json. That block used to win over the
    `usage_migrated.json` beside it, so simply re-running a finished config wiped the run's
    cost — and reported it as complete.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(
        tmp_path / "evaluation.json",
        {
            "usage": {
                "estimated_total_cost_usd": None,
                "priced_calls_estimated_cost_usd": 0,
                "num_completed_calls": 0,
                "cost_complete": False,
                "cost_incompleteness_reasons": ["no usage ledger for this run"],
            }
        },
    )
    _write(
        tmp_path / "usage_migrated.json",
        {"recomputed_partial_cost_usd": 4.10, "migration_basis": "recomputed"},
    )
    info = resolve_run_cost(tmp_path)
    assert (info["source"], info["total_usd"]) == ("migrated", 4.10)


def test_an_empty_ledger_folder_with_work_reports_itself_incomplete(tmp_path):
    """The other half: the aggregator must not call an absent ledger a complete $0."""
    from daedalus.core.logging.usage_aggregate import UsageAggregator

    _write(tmp_path / "evaluation.json", {"aggregate": {}})
    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["cost_complete"] is False
    assert summary["estimated_total_cost_usd"] is None
    assert any("no usage ledger" in r for r in summary["cost_incompleteness_reasons"])


def test_a_genuinely_empty_folder_is_not_flagged(tmp_path):
    """A fresh launch that has spent nothing yet is not a pre-ledger run."""
    from daedalus.core.logging.usage_aggregate import UsageAggregator

    summary = UsageAggregator.from_experiment_dir(tmp_path).summary()
    assert summary["cost_complete"] is True


def test_a_migrated_recomputation_beats_the_legacy_sum(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(
        tmp_path / "usage_migrated.json",
        {
            "recomputed_partial_cost_usd": 21.68,
            "migration_basis": "recomputed",
            "cost_usd_by_role": {"solver": 9.0, "user_simulator": 12.68},
        },
    )
    info = resolve_run_cost(tmp_path, legacy_total=9.0)
    assert (info["source"], info["total_usd"]) == ("migrated", 21.68)


def test_a_solver_only_migration_loses_to_the_legacy_total(tmp_path):
    """The legacy figure covers explorer/judge/extraction; the migration could not.

    Preferring the migration here reported one generation run as $2.40 instead of $36.66.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(
        tmp_path / "usage_migrated.json",
        {
            "recomputed_partial_cost_usd": 2.40,
            "migration_basis": "solver-only",
            "cost_usd_by_role": {"solver": 2.40},
        },
    )
    info = resolve_run_cost(tmp_path, legacy_total=36.66)
    assert (info["source"], info["total_usd"]) == ("legacy", 36.66)
    assert info["recomputed_solver_share_usd"] == 2.40


def test_run_summary_supplies_the_legacy_total_when_the_caller_has_none(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(tmp_path / "run_summary.json", {"total_cost_usd": 43.47})
    _write(
        tmp_path / "usage_migrated.json",
        {"recomputed_partial_cost_usd": 6.26, "migration_basis": "solver-only"},
    )
    assert resolve_run_cost(tmp_path)["total_usd"] == 43.47


def test_a_run_with_no_cost_anywhere_reports_none(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    info = resolve_run_cost(tmp_path)
    assert info["total_usd"] is None and info["source"] == "legacy"
    assert info["notes"]  # it says to run the migration


def test_a_generation_worker_writes_its_ledger_into_the_shared_run_folder(tmp_path):
    """Every explorer worker's ledger must land under the RUN's experiment dir.

    Self-play generation renames the experiment per worker (`<name>_w0`, …) so their
    environments cannot collide. `CostAccounting.for_worker` derives its path from
    `experiment_dir(cfg)`, so without an override each worker wrote to
    `outputs/daedalus/<name>_w0/usage/` — five sibling folders holding nothing but
    ledgers, none of which `UsageAggregator.from_experiment_dir(<run>)` can see. One real
    run lost sight of all 4,868 of its own usage events that way.
    """
    from pathlib import Path

    from daedalus.core.logging.usage_ledger import usage_root

    cfg = config_from_dict(
        {
            "benchmark": "appworld",
            "experiment_name": "gen_ledger_probe",
            "kind": "generation",
            "logging": {"trace_dir": str(tmp_path)},
        }
    )
    shared = Path(experiment_dir(cfg))
    cfg.experiment_name = "gen_ledger_probe_w0"  # what the worker does
    per_worker = Path(experiment_dir(cfg))
    assert per_worker != shared, "precondition: the worker's own dir differs"

    # Set on the config, not passed per call: this process builds several accounting
    # objects (explorer, accumulation loop, solver agent) and each would otherwise make
    # its own sibling folder.
    cfg.cost_accounting.ledger_dir = str(shared)
    for built in (CostAccounting.for_worker(cfg),
                  CostAccounting.for_worker(cfg, run_idx=0)):
        assert usage_root(built.experiment_dir) == usage_root(shared), (
            "every ledger built in the worker must live under the run folder"
        )
    # provenance is kept: the event scope still names the worker
    assert CostAccounting.for_worker(cfg).base_scope.experiment_name == "gen_ledger_probe_w0"

    # unset, it still falls back to the experiment's own folder
    cfg.cost_accounting.ledger_dir = ""
    assert usage_root(CostAccounting.for_worker(cfg).experiment_dir) == usage_root(per_worker)


def test_a_ledger_block_reporting_zero_events_is_not_a_complete_zero(tmp_path):
    """`num_events: 0` means the aggregator saw no ledger, not that nothing was spent.

    A generation run finalised while its workers wrote to sibling folders and stored
    `{"num_events": 0, "cost_complete": True, "estimated_total_cost_usd": 0}`. Trusting it
    reported "$0.00, complete" for a $45 run — worse than reporting nothing, because it
    looked authoritative. Such a block is skipped so the session total is used instead.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(tmp_path / "run_summary.json", {
        "total_cost_usd": 45.39,
        "usage": {"estimated_total_cost_usd": 0, "cost_complete": True,
                  "num_events": 0, "num_completed_calls": 0,
                  "cost_accounting_version": 2},
    })
    r = resolve_run_cost(tmp_path)
    assert r["total_usd"] == 45.39, "the empty block must not win over the session total"
    assert r["source"] != "ledger"
    assert not r["complete"]

    # A block that simply omits the count is still trusted — only an explicit zero is not.
    _write(tmp_path / "run_summary.json", {
        "total_cost_usd": 45.39,
        "usage": {"estimated_total_cost_usd": 12.0, "cost_complete": True,
                  "cost_accounting_version": 2},
    })
    r = resolve_run_cost(tmp_path)
    assert r["total_usd"] == 12.0 and r["source"] == "ledger"


# ---- auditable multi-method cost resolution ------------------------------------------
# The run browser's `costs` tab and `scripts/cost_report` put five accounting methods side
# by side for a paper to choose from. These pin the two things that make that trustworthy:
# the LIVE ledger is what gets read (not the snapshot a run froze when it finished), and no
# method is ever silently substituted for another.


def _ledger_event(**kwargs):
    """One ledger event dict, with the fields the aggregator needs."""
    from daedalus.core.logging.usage import USAGE_SCHEMA_VERSION

    base = {
        "schema_version": USAGE_SCHEMA_VERSION,
        "role": "solver",
        "requested_model": "gpt-5.4",
        "status": "completed",
        "provider": "openai",
        "tokens": TokenUsage(prompt_tokens=10, completion_tokens=1).to_dict(),
        "estimated_cost_usd": 1.0,
    }
    base.update(kwargs)
    base.setdefault("event_id", __import__("uuid").uuid4().hex)
    return base


def _write_ledger(run_dir, events, launch="launch_a"):
    import json as _json

    path = run_dir / "usage" / launch / "worker_1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for e in events:
            fh.write(_json.dumps(e) + "\n")


def test_the_live_ledger_beats_a_stale_embedded_snapshot(tmp_path):
    """Calls billed AFTER a run wrote its summary are only on disk in the ledger.

    A resume, or `scripts/consolidate` run afterwards, appends a new launch directory and
    touches nothing else. Reading the embedded block first understated eight runs here.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(tmp_path / "evaluation.json", {
        "usage": {"estimated_total_cost_usd": 2.0, "cost_complete": True,
                  "num_completed_calls": 2, "cost_accounting_version": 2}})
    _write_ledger(tmp_path, [_ledger_event(), _ledger_event()], launch="launch_a")
    _write_ledger(tmp_path, [_ledger_event(role="consolidation")], launch="launch_b")

    info = resolve_run_cost(tmp_path)
    assert info["total_usd"] == 3.0, "the third, later call must be counted"
    assert info["source"] == "ledger_live"
    assert info["ledger_estimated_cost_usd"] == 3.0
    assert info["stored_usage_estimate_usd"] == 2.0
    assert info["stored_usage_is_stale"] is True
    assert any("stale" in n for n in info["notes"])


def test_a_post_processing_only_ledger_is_added_to_a_legacy_total(tmp_path):
    """A pre-ledger run that was consolidated later has two NON-overlapping records.

    The ledger holds only the consolidation pass; the legacy artifacts hold the execution
    that predates it. Preferring either alone loses the other, and taking a max loses the
    smaller one, so the scopes are summed and the result is never called complete.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(tmp_path / "run_summary.json", {"total_cost_usd": 66.47})
    _write_ledger(tmp_path, [_ledger_event(role="consolidation", estimated_cost_usd=0.2)])

    info = resolve_run_cost(tmp_path)
    assert info["source"] == "hybrid"
    assert info["total_usd"] == pytest.approx(66.67)
    assert info["complete"] is False
    assert info["legacy_cost_usd"] == 66.47
    assert info["ledger_estimated_cost_usd"] == pytest.approx(0.2)


def test_a_corrupt_ledger_degrades_to_the_snapshot_instead_of_raising(tmp_path):
    """One copied run folder must not take the whole run browser down with it.

    `UsageAggregator` refuses a ledger with duplicate event ids, which is what copying a
    run produces. Every row in the browser now reads the ledger, so that exception would
    have been a 500 on the list page rather than one bad row.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write(tmp_path / "evaluation.json", {
        "usage": {"estimated_total_cost_usd": 7.0, "cost_complete": True,
                  "num_completed_calls": 1, "cost_accounting_version": 2}})
    _write_ledger(tmp_path, [_ledger_event(event_id="dup"), _ledger_event(event_id="dup")])

    info = resolve_run_cost(tmp_path)
    assert info["total_usd"] == 7.0
    assert info["ledger_unreadable"] is True
    assert info["complete"] is False
    assert any("unreadable" in n for n in info["notes"])


def test_litellms_opinion_is_not_reported_as_a_provider_charge(tmp_path):
    """`provider_native_cost_usd` is money the provider named, not a second estimate.

    litellm's `response_cost` is its own price table applied to our tokens. Counting it as
    provider-observed spend would let a fully-priced OpenAI run claim an invoice-grade
    figure it does not have.
    """
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write_ledger(tmp_path, [
        _ledger_event(provider_reported_cost_usd=9.0,
                      provider_cost_source="litellm_response_cost"),
        _ledger_event(provider_reported_cost_usd=2.5,
                      provider_cost_source="provider_usage_cost"),
    ])
    info = resolve_run_cost(tmp_path)
    assert info["provider_native_cost_usd"] == 2.5, "only the router's own charge counts"
    assert info["provider_native_complete"] is False, "1 of 2 calls is not coverage"
    assert info["source"] == "ledger_live", "partial native coverage is never the headline"
    assert info["provider_native_coverage"]["covered_calls"] == 1


def test_a_fully_native_run_reports_the_provider_charge_as_the_headline(tmp_path):
    from daedalus.core.logging.run_cost import resolve_run_cost

    _write_ledger(tmp_path, [
        _ledger_event(provider_reported_cost_usd=2.0,
                      provider_cost_source="provider_usage_cost"),
        _ledger_event(provider_reported_cost_usd=4.0,
                      provider_cost_source="provider_usage_cost"),
    ])
    info = resolve_run_cost(tmp_path)
    assert (info["source"], info["total_usd"]) == ("provider_native", 6.0)
    assert info["complete"] is True
    # The reproducible estimate stays visible beside it — that gap is the finding.
    assert info["ledger_estimated_cost_usd"] == 2.0












def test_aggregate_metrics_totals_the_cached_prompt_tokens(tmp_path):
    """`total_cached_prompt_tokens` is a subset of `total_prompt_tokens`, never an addition."""
    from daedalus.core.evaluation.metrics import aggregate_metrics

    traces = [
        {"task_id": "a", "total_tokens": {"prompt": 100, "completion": 10, "cached": 80}},
        {"task_id": "b", "total_tokens": {"prompt": 200, "completion": 20, "cached": 150}},
    ]
    m = aggregate_metrics(traces)
    assert m["total_prompt_tokens"] == 300
    assert m["total_cached_prompt_tokens"] == 230
    assert m["total_cached_prompt_tokens"] <= m["total_prompt_tokens"]

    # A trace that never recorded the split contributes 0, not a crash.
    assert aggregate_metrics(
        [{"task_id": "a", "total_tokens": {"prompt": 5, "completion": 1}}]
    )["total_cached_prompt_tokens"] == 0


# ---- the uniform cost formula (inference tab's `cost` column) -------------------------


def _trace(prompt, completion, cached=None, per_turn=None, user_sim=None, model="gpt-5.4-mini"):
    """A pre-ledger task trace. `per_turn` is a list of (prompt, completion, cached?)."""
    turns = []
    for t in per_turn or []:
        usage = {"prompt": t[0], "completion": t[1]}
        if len(t) > 2:
            usage["cached"] = t[2]
        turns.append({"token_usage": usage})
    totals = {"prompt": prompt, "completion": completion}
    if cached is not None:
        totals["cached"] = cached
    out = {"task_id": "a", "total_tokens": totals, "turns": turns,
           "config": {"agent": {"model": model}}}
    if user_sim is not None:
        out["user_sim_cost_usd"] = user_sim
    return out




def test_full_cache_counterfactual_keeps_first_cache_then_prices_only_prompt_growth(tmp_path):
    from daedalus.core.logging.cost import get_price_snapshot
    from daedalus.core.logging.paper_cost import full_cache_cost

    trace = _trace(
        5000, 300, cached=1024,
        per_turn=[(2000, 100, 1024), (3000, 200, 0)],
        model="gpt-5.4-mini",
    )
    got = full_cache_cost(
        tmp_path, "gpt-5.4-mini", "flex", traces=[trace]
    )

    # First request preserves its observed 1,024 cached tokens. The second reuses the
    # previous 2,000-token prompt rounded down to 1,920 (= 15 * 128), leaving 1,080 fresh.
    snap = get_price_snapshot()
    expected = snap.price_call(
        "gpt-5.4-mini",
        TokenUsage(prompt_tokens=2000, completion_tokens=100,
                   cache_read_tokens=1024, cache_details_available=True),
        "flex",
    ) + snap.price_call(
        "gpt-5.4-mini",
        TokenUsage(prompt_tokens=3000, completion_tokens=200,
                   cache_read_tokens=1920, cache_details_available=True),
        "flex",
    )
    assert got["cost_usd"] == pytest.approx(round(expected, 6))
    assert got["cached_prompt_tokens"] == 1024 + 1920
    assert got["fresh_prompt_tokens"] == (2000 - 1024) + (3000 - 1920)
    assert got["complete"] is True

    undiscounted = full_cache_cost(
        tmp_path, "gpt-5.4-mini", "flex", traces=[trace],
        pricing_tier_override="standard",
    )
    assert undiscounted["cost_usd"] == pytest.approx(round(2 * expected, 6))
    assert undiscounted["pricing_tier"] == "standard"


def test_raw_standard_cost_reprices_historical_aux_roles_and_maximally_caches_explorer(
    tmp_path,
):
    """A standard-price column must never carry a flex-priced dollar residual.

    Pre-ledger generation saved role token totals but not the explorer's per-call history.
    The requested paper convention prices every explorer prompt token as cached (the
    absolute lower bound), while judge/extraction retain their observed cache split.
    """
    from daedalus.core.logging.cost import get_price_snapshot
    from daedalus.core.logging.paper_cost import full_cache_cost

    (tmp_path / "traces").mkdir()
    trace = _trace(
        2_000, 100, cached=1_000,
        per_turn=[(2_000, 100, 1_000)], model="gpt-5.4-mini",
    )
    trace["total_cost_usd"] = 1.0
    _write(tmp_path / "traces" / "task.json", trace)
    _write(tmp_path / "config.yaml", {
        "agent": {"model": "gpt-5.4-mini"},
        "generation": {"explorer_model": "gpt-5.4", "judge_model": "gpt-5.4"},
        "accumulation": {"extraction_model": "gpt-5.4"},
    })
    _write(tmp_path / "run_summary.json", {
        "total_cost_usd": 10.0,
        "cost_usd_by_role": {"solver": 1.0, "explorer": 9.0},
        "tokens_by_role": {
            "solver": {"prompt": 2_000, "completion": 100, "cached": 1_000},
            "explorer": {"prompt": 4_000, "completion": 200, "cached": 0},
        },
    })

    got = full_cache_cost(
        tmp_path, "gpt-5.4-mini", "flex", pricing_tier_override="standard"
    )
    snap = get_price_snapshot()
    expected_solver = snap.price_call(
        "gpt-5.4-mini",
        TokenUsage(prompt_tokens=2_000, completion_tokens=100,
                   cache_read_tokens=1_000),
        "standard",
    )
    expected_explorer = snap.price_call(
        "gpt-5.4",
        TokenUsage(prompt_tokens=4_000, completion_tokens=200,
                   cache_read_tokens=4_000),
        "standard",
    )
    assert got["cost_usd"] == pytest.approx(round(expected_solver + expected_explorer, 6))
    assert got["cached_prompt_tokens"] == 1_000 + 4_000
    assert got["fresh_prompt_tokens"] == 1_000
    assert got["complete"] is False
    assert any("maximal-cache lower bound" in note for note in got["notes"])


def test_raw_standard_cost_converts_only_generation_residual_not_user_simulator(tmp_path):
    """A tau2 simulator scalar is independent of the flex-priced generation residual."""
    from daedalus.core.logging.cost import get_price_snapshot
    from daedalus.core.logging.paper_cost import full_cache_cost

    (tmp_path / "traces").mkdir()
    trace = _trace(
        2_000, 100, cached=1_000,
        per_turn=[(2_000, 100, 1_000)], model="gpt-5.4-mini", user_sim=3.0,
    )
    trace["total_cost_usd"] = 1.0
    _write(tmp_path / "traces" / "task.json", trace)
    _write(tmp_path / "config.yaml", {
        "kind": "generation",
        "generation": {"explorer_model": "gpt-5.4", "judge_model": "gpt-5.4"},
        "accumulation": {"extraction_model": "gpt-5.4"},
    })
    _write(tmp_path / "run_summary.json", {"total_cost_usd": 10.0})

    got = full_cache_cost(
        tmp_path, "gpt-5.4-mini", "flex", pricing_tier_override="standard"
    )
    solver = get_price_snapshot().price_call(
        "gpt-5.4-mini",
        TokenUsage(prompt_tokens=2_000, completion_tokens=100,
                   cache_read_tokens=1_000),
        "standard",
    )
    # Historical total: $1 solver + $3 simulator + $6 generation-role residual.
    # Only the final term receives the 2x flex-to-standard conversion.
    assert got["cost_usd"] == pytest.approx(round(solver + 3.0 + 2 * 6.0, 6))
    assert any("converted from flex to standard" in note for note in got["notes"])
















