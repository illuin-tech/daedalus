"""Config schemas and YAML loader for daedalus experiments.

Uses plain dataclasses (not Pydantic) so the harness stays decoupled from the
benchmarks' own pydantic models.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Per-benchmark task-selection configs live with each connector so a benchmark's
# files stay together. They import only stdlib, so pulling them in here is cheap
# and doesn't drag in the benchmark libraries (appworld/tau2/gym).
from daedalus.benchmarks.appworld.config import AppWorldConfig
from daedalus.benchmarks.automationbench.config import AutomationBenchConfig
from daedalus.benchmarks.tau2.config import Tau2Config

# The setup ("which method") a run belongs to — see ExperimentConfig.setup and
# experiment_dir below. A baseline method's runs live under outputs/baselines/<setup>/.
DEFAULT_SETUP = "daedalus"
# The benchmarks with a config block (and a connector under daedalus/benchmarks/).
BENCHMARKS = ("appworld", "tau2", "automationbench")
BASELINES_DIRNAME = "baselines"


@dataclass
class AgentConfig:
    """The solver (and, at test time, the base agent)."""

    model: str = "gpt-5.4-mini"
    max_turns: int = 30  # step ceiling (AppWorld, AutomationBench)
    temperature: float = 1.0
    # Passed straight to the provider through litellm (e.g. none | low | medium | high).
    # None sends nothing (provider default); non-reasoning models ignore it.
    reasoning_effort: str | None = None


@dataclass
class RetrieverConfig:
    """Per-turn retrieval (paper Section 5.2); unused with `heuristics_at_start`."""

    # random | bm25 | dense | dense_remote | all_every_turn (core/memory/retrieval.py).
    type: str = "bm25"
    model_name: str | None = None  # dense / dense_remote embedder
    top_k: int = 5
    seed: int = 0  # random: each task's draw is seeded from (seed, task_id)
    # dense_remote: an OpenAI-compatible embeddings endpoint, e.g. "http://localhost:8000/v1".
    base_url: str | None = None


@dataclass
class InjectionConfig:
    """What happens to a retrieved note after its turn (paper Table 10):
    transient | cumulative_dedup | cumulative_no_replacement | cumulative_verbatim."""

    policy: str = "cumulative_dedup"


@dataclass
class MemoryConfig:
    """The heuristic bank at inference."""

    enabled: bool = False
    pool_path: str | None = None
    # True: inject the whole bank into the system prompt at task start (ALL@START, the
    # method) and build no retriever. False: retrieve per turn with `retriever`.
    heuristics_at_start: bool = False
    # AppWorld per-turn retrieval query: pre_generation (the previous turn's reasoning) |
    # reason_then_retrieve (a separate reasoning call writes it; R2R in the paper).
    retrieval_mode: str = "pre_generation"
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)


@dataclass
class LoggingConfig:
    """Trace logging settings."""

    # Where per-task traces go. "" = the experiment folder (see resolve_trace_dir); set
    # by generation workers so every worker writes into the run folder.
    trace_dir: str = ""
    save_every_turn: bool = True
    verbose: bool = False


@dataclass
class AccumulationConfig:
    """The Solver loop (core/generation/accumulate.py), shared by accumulation on labeled
    tasks and by self-play generation."""

    # Stop after `num_success_to_continue` consecutive successes (Ns) or `max_failures`
    # failed attempts (Nf). Only failures consume the budget.
    max_failures: int = 8
    num_success_to_continue: int = 3
    # The Extractor, which writes/revises the heuristic after each failure.
    extraction_model: str = "gpt-5.4"
    # none | low | medium | high | xhigh | max (model-dependent); null = provider default.
    # Read by accumulation on labeled tasks; the DAEDALUS generation runs of the paper sent
    # no effort to the extractor (see pipeline.py).
    extraction_reasoning_effort: str | None = None
    save_traces: bool = True


@dataclass
class GenerationConfig:
    """Self-play generation (core/generation/pipeline.py); no training labels are read."""

    # daedalus (the method) | direct_exploration_with_bank (Table 3, row B) |
    # task_generation_solver (Table 3, row C). The two ablations are AppWorld-only.
    stage: str = "daedalus"
    # Whose seeded worlds the explorer plays in (their instructions and labels are unused).
    sandbox_dataset: str = "train"
    explorer_model: str = "gpt-5.4"
    explorer_reasoning_effort: str = "high"
    explorer_max_turns: int = 40  # ceiling, not a target
    # If a spec is rejected on the explorer's last turn, grant this many extra turns once.
    explorer_last_turn_grace: int = 10
    # AppWorld only: append a remaining-turn reminder to each execution result, escalating
    # to "stop exploring and emit" near the ceiling. Used for the Qwen and DeepSeek banks,
    # whose explorers otherwise did not converge on a spec.
    explorer_turn_steering: bool = False
    judge_model: str = "gpt-5.4"
    judge_reasoning_effort: str = "high"
    num_sessions: int = 90  # sessions to run (a resume runs this many more)
    num_explorers: int = 5  # concurrent explorer processes
    # ── The Surveyor (coverage tags; core/generation/coverage.py) ──
    # One survey before the workers fan out fixes a closed set of tags with target shares;
    # every session sees the live per-tag tally of accepted tasks. False disables it.
    coverage_tags: bool = True
    # Reuse a saved survey (`coverage_tags.txt` of a previous run) instead of surveying.
    coverage_tags_path: str | None = None
    coverage_tags_max_turns: int = 100
    # Past this fraction of the run, the most under-covered tag (realized share below
    # target_share * coverage_escalate_ratio) becomes a directive rather than feedback.
    coverage_escalate_after: float = 0.5
    coverage_escalate_ratio: float = 0.5
    # ── The explorer guideline bank (core/generation/guidelines.py) ──
    # After a session that needed refinement, distil what made tasks too easy or too hard
    # into guidelines every later explorer prompt carries. False disables it.
    explorer_guidelines: bool = True
    max_refinements: int = 5  # Nr
    # Drop a task whose tool-path novelty against the accepted tasks is below this.
    # -1 disables the filter, which is not part of the method. The 90-session AppWorld
    # run used 0.05 (paper, Appendix E).
    min_novelty: float = -1.0
    # AppWorld: apps held out entirely (absent from the catalog, pruned from every listing,
    # refused on the channel — benchmarks/appworld/hidden_apps.py), e.g. ["gmail", "amazon"].
    excluded_apps: list[str] = field(default_factory=list)
    # τ²: rejected submit_task_spec calls a session tolerates before giving up.
    spec_max_rejections: int = 3
    # τ² retail: reject specs whose reference solution exchanges several items in one call
    # (the environment grades those by argument order; paper, Appendix E).
    reject_multi_item_exchange: bool = False
    # τ²: reject specs whose ticket leaks an identifier the solver could look up.
    withhold_discoverable_ids: bool = False


@dataclass
class ProviderRoutingConfig:
    """OpenRouter provider routing — which upstream serves an "openrouter/" model.

    OpenRouter fronts ONE model id with many upstreams at different quantizations. The
    AppWorld DeepSeek arm drew fp4, fp8 and bf16 endpoints inside a single run, so the
    weights actually executed varied call to call. That is a noise source no config
    recorded. These fields pin it.

    Sent only for a model string starting "openrouter/"; ignored for every other provider,
    which has no such concept.

    Prefer `quantizations: ["fp8"]` over pinning one provider: it removes the precision
    spread — the part that changes results — while leaving several upstreams available to
    absorb a 429. Pinning one tag with allow_fallbacks false turns any provider hiccup
    into a failed task.

    `order` and `only` take provider TAGS, which encode quantization, not display names.
    List them with the endpoints API, reading the `tag` field of
    https://openrouter.ai/api/v1/models/<author>/<slug>/endpoints —
    e.g. "baidu/fp8", "open-inference/fp8", "sail-research/fp4", "digitalocean".
    """

    # Upstreams to try, in order. Tags, e.g. ["baidu/fp8"].
    order: list[str] = field(default_factory=list)
    # Restrict routing to these upstreams (unordered).
    only: list[str] = field(default_factory=list)
    # Never route to these upstreams.
    ignore: list[str] = field(default_factory=list)
    # Allowed weight precisions, e.g. ["fp8"]. The lever that matters for reproducibility.
    quantizations: list[str] = field(default_factory=list)
    # "price" | "throughput" | "latency". "" leaves OpenRouter's default balance.
    sort: str = ""
    # False makes a request FAIL rather than route to an upstream outside the filter.
    allow_fallbacks: bool = True

    def to_body(self) -> dict[str, Any] | None:
        """The `provider` object for the request body, or None when nothing is set."""
        body: dict[str, Any] = {}
        for name in ("order", "only", "ignore", "quantizations"):
            value = getattr(self, name)
            if value:
                body[name] = list(value)
        if self.sort:
            body["sort"] = self.sort
        # Only meaningful alongside a filter, and sending it alone would pin nothing while
        # still disabling the fallbacks that keep a 168-task run alive.
        if not self.allow_fallbacks and body:
            body["allow_fallbacks"] = False
        return body or None


@dataclass
class RunConfig:
    """Execution settings (parallelism, task limits)."""

    max_tasks: int | None = None
    parallel: int = 1  # Number of parallel workers
    force: bool = False  # Re-run even if results already exist
    num_runs: int = 1  # Number of independent runs (for variance estimation)


@dataclass
class CostAccountingConfig:
    """LLM spend accounting for this experiment (see core/logging/usage*.py).

    Every experiment entry point requires `enabled`: with it off there is no usage ledger,
    so completed calls leave no record and the run cannot report what it spent.
    """

    enabled: bool = True
    # Where this process writes its usage ledger. "" = the experiment's own folder.
    # Set by a worker whose `experiment_name` was renamed for isolation (self-play
    # generation gives each explorer `<name>_w0`, …): without it every accounting object
    # built inside that process — the explorer's, the accumulation loop's and the solver
    # agent's — writes to `outputs/daedalus/<name>_w0/usage/`, outside the run folder
    # where the run's own UsageAggregator looks.
    ledger_dir: str = ""
    # Which dated price snapshot prices this run. "" = the current default
    # (cost.DEFAULT_PRICE_SNAPSHOT_ID). Pin it to make a historical run reproducible.
    price_snapshot: str = ""
    # Stop BEFORE any paid work when a configured model or feature cannot be priced
    # exactly. Off only for a run that deliberately accepts an incomplete estimate.
    require_complete_estimates: bool = True
    # This process launch's ledger directory, set by the entry point and carried into
    # spawned task workers through the serialized config. A resume gets a NEW id, so
    # earlier launches' ledgers stay immutable and both launches' spend is counted.
    launch_id: str = ""


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    appworld: AppWorldConfig = field(default_factory=AppWorldConfig)
    tau2: Tau2Config = field(default_factory=Tau2Config)
    automationbench: AutomationBenchConfig = field(default_factory=AutomationBenchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    accumulation: AccumulationConfig = field(default_factory=AccumulationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run: RunConfig = field(default_factory=RunConfig)
    cost_accounting: CostAccountingConfig = field(default_factory=CostAccountingConfig)
    provider_routing: ProviderRoutingConfig = field(
        default_factory=ProviderRoutingConfig
    )
    # Which benchmark this experiment runs on: appworld | tau2 | automationbench.
    # Selects the Benchmark implementation (see benchmarks/) that provides tasks, the
    # solver agent, run evaluation, and the GenerationBackend used by self-play.
    benchmark: str = "appworld"
    # Experiment name. Optional at the top level for backward compatibility:
    # older configs set appworld.experiment_name, which `name` falls back to.
    experiment_name: str | None = None
    # Which family of experiment this is; selects the output sub-tree (KIND_DIRNAMES):
    # inference, accumulation (→ daedalus-curated/), generation (→ daedalus/), testset
    # (→ evaluation-proxy/, a generated test set — see daedalus.scripts.tasks_as_test_set).
    # Set by the entry script of the same name (daedalus.scripts.<kind>); rarely needs
    # to be in the YAML.
    kind: str = "inference"
    # The section of the paper this run belongs to, e.g. "main-results", "memory-use",
    # "exploration-budget". Inference runs are filed under it —
    # outputs/inference/<benchmark>/<category>/<name>/ — so a folder holds exactly the runs one
    # table or figure reads, and the run browser shows it as a column. Omitted → "misc" (warned
    # about at load time). Only daedalus's own inference runs use it.
    category: str = ""
    # Which METHOD produced this run. "daedalus" (the default) is this repo's own method. Any
    # other value names a baseline implemented under references/<setup>/, whose runs are
    # namespaced as outputs/baselines/<setup>/{memory,inference}/<name>/ so the two trees never
    # mix. Set by that method's entry script, not by hand. The run browser shows it as the
    # first column of every run table, and the figure scripts label non-daedalus runs with it.
    setup: str = DEFAULT_SETUP

    @property
    def name(self) -> str:
        """Experiment name, falling back to the legacy appworld-nested field."""
        return self.experiment_name or self.appworld.experiment_name

    @property
    def benchmark_config(self) -> Any:
        """The config block of the active benchmark (e.g. `self.tau2`)."""
        return getattr(self, self.benchmark)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict (for JSON logging)."""
        return dataclasses.asdict(self)


class ConfigKeyError(ValueError):
    """A config block carries a key this schema does not define."""


def _build_dataclass(cls, data: dict[str, Any] | None, block: str = ""):
    """Build a dataclass from a dict, REFUSING unknown keys.

    Unknown keys used to be dropped silently, which turned every typo into a silent
    fallback to the default: `topk` for `top_k`, `num_run` for `num_runs`, and the run
    proceeded looking exactly like the one you meant to launch. A misspelled knob is
    always a mistake, so it stops here and names the near-misses.

    Only keys INSIDE a known block are checked. Unknown TOP-LEVEL keys stay legal, because
    that is how a competing method under references/ carries its own settings in the same
    file (`erl:`, `expel:`, … — see references/common/config.py).
    """
    if data is None:
        return cls()
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in field_names]
    if unknown:
        where = f"{block}." if block else ""
        hints = []
        for key in unknown:
            near = [f for f in sorted(field_names) if _looks_like(key, f)]
            hints.append(f"{where}{key}" + (f" (did you mean {where}{near[0]}?)" if near else ""))
        raise ConfigKeyError(
            f"unknown config key(s): {', '.join(hints)}. "
            f"Known keys in `{block or cls.__name__}`: {', '.join(sorted(field_names))}"
        )
    return cls(**data)


def _looks_like(a: str, b: str) -> bool:
    """Whether `a` is plausibly a typo of `b` (same letters ignoring _ and case)."""
    norm = lambda s: s.replace("_", "").lower()  # noqa: E731
    return norm(a) == norm(b) or (
        len(norm(a)) > 3 and (norm(a) in norm(b) or norm(b) in norm(a))
    )


def _build_memory_config(data: dict[str, Any] | None) -> MemoryConfig:
    """Build MemoryConfig with nested sub-dataclasses, refusing unknown keys."""
    data = dict(data or {})
    retriever = _build_dataclass(RetrieverConfig, data.pop("retriever", None), "memory.retriever")
    injection = _build_dataclass(InjectionConfig, data.pop("injection", None), "memory.injection")
    memory = _build_dataclass(MemoryConfig, data, "memory")
    memory.retriever, memory.injection = retriever, injection
    return memory


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build an ExperimentConfig from a plain dict (YAML data or to_dict() output).

    Single construction path shared by load_config and the parallel-worker
    config rebuild, so new top-level fields can't silently drop in one of them.
    """
    # PyYAML 1.1 parses all-digit+underscore values (e.g. "6104387_1") as integers,
    # stripping the underscore (61043871). str() cannot recover the original ID.
    # The real fix is to quote such IDs in YAML. This coercion only helps for
    # purely-numeric IDs that happen to be valid as strings.
    bench = {}
    for name in BENCHMARKS:
        block = dict(data.get(name) or {})
        if block.get("task_ids") is not None:
            block["task_ids"] = [str(t) for t in block["task_ids"]]
        bench[name] = block

    return ExperimentConfig(
        agent=_build_dataclass(AgentConfig, data.get("agent"), "agent"),
        appworld=_build_dataclass(AppWorldConfig, bench["appworld"], "appworld"),
        tau2=_build_dataclass(Tau2Config, bench["tau2"], "tau2"),
        automationbench=_build_dataclass(
            AutomationBenchConfig, bench["automationbench"], "automationbench"
        ),
        memory=_build_memory_config(data.get("memory")),
        logging=_build_dataclass(LoggingConfig, data.get("logging"), "logging"),
        accumulation=_build_dataclass(AccumulationConfig, data.get("accumulation"), "accumulation"),
        generation=_build_dataclass(GenerationConfig, data.get("generation"), "generation"),
        run=_build_dataclass(RunConfig, data.get("run"), "run"),
        cost_accounting=_build_dataclass(
            CostAccountingConfig, data.get("cost_accounting"), "cost_accounting"
        ),
        provider_routing=_build_dataclass(
            ProviderRoutingConfig, data.get("provider_routing"), "provider_routing"
        ),
        benchmark=data.get("benchmark", "appworld"),
        experiment_name=data.get("experiment_name"),
        category=(data.get("category") or "").strip(),
        kind=data.get("kind", "inference"),
        setup=data.get("setup") or DEFAULT_SETUP,
    )


def config_from_snapshot(data: dict[str, Any]) -> ExperimentConfig:
    """Build a config from a run's `config.yaml` snapshot, dropping keys this schema no
    longer has.

    Snapshots of the paper's runs predate the release schema; the knobs removed since held
    the value the code now hard-wires, so dropping them preserves what the run did.
    """
    known = {f.name: f for f in dataclasses.fields(ExperimentConfig)}
    nested = {"retriever": RetrieverConfig, "injection": InjectionConfig}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            continue
        if not isinstance(value, dict) or not dataclasses.is_dataclass(
            getattr(ExperimentConfig(), key)
        ):
            out[key] = value
            continue
        block_fields = {f.name for f in dataclasses.fields(getattr(ExperimentConfig(), key))}
        block = {k: v for k, v in value.items() if k in block_fields}
        for sub, cls in nested.items():
            if key == "memory" and isinstance(block.get(sub), dict):
                sub_fields = {f.name for f in dataclasses.fields(cls)}
                block[sub] = {k: v for k, v in block[sub].items() if k in sub_fields}
        out[key] = block
    return config_from_dict(out)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config file and build the ExperimentConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = config_from_dict(data)
    return cfg


# ── Output layout ──────────────────────────────────────────────────────────
# Every experiment owns one self-contained folder, named in the paper's terms:
#   daedalus/<name>/          generation: tasks.json, sessions/, run_summary.json, pool.json,
#                             consolidated_pool.json, traces/<sandbox>_attemptN.json
#   daedalus-curated/<name>/  accumulation: pool.json, pool.log.json, traces/<task>_attemptN.json
#   inference/<benchmark>/<category>/<name>/   run_0/<task>.json … , evaluation.json
#   evaluation-proxy/<generation run>/<model>/ testset: the generated tasks as a test set
# Paths derive from (setup, kind, experiment_name); cross-references (an inference run
# reading another experiment's pool) stay explicit via memory.pool_path.
#
# A baseline method (setup != "daedalus", implemented under references/<setup>/) writes its
# memory and its evaluation under outputs/baselines/<setup>/, so the two trees never mix and
# either can be wiped on its own; its configs name each run after the benchmark:
#   outputs/baselines/erl/memory/<name>/ , outputs/baselines/erl/inference/<name>/
OUTPUTS_ROOT = Path("outputs")


def set_outputs_root(path: str | Path) -> None:
    """Read runs from another folder than ./outputs (the run browser's `--outputs`)."""
    global OUTPUTS_ROOT
    OUTPUTS_ROOT = Path(path)

KIND_DIRNAMES = {
    "generation": "daedalus",
    "accumulation": "daedalus-curated",
    "testset": "evaluation-proxy",
}
BASELINE_KIND_DIRNAMES = {"accumulation": "memory"}


def kind_root(kind: str, setup: str = DEFAULT_SETUP) -> Path:
    """The folder holding every `kind` run of one setup (see ExperimentConfig.setup)."""
    if setup and setup != DEFAULT_SETUP:
        return OUTPUTS_ROOT / BASELINES_DIRNAME / setup / BASELINE_KIND_DIRNAMES.get(kind, kind)
    return OUTPUTS_ROOT / KIND_DIRNAMES.get(kind, kind)


def kind_roots(kind: str) -> list[tuple[str, Path]]:
    """Every (setup, folder) that holds `kind` runs on disk, daedalus first.

    The discovery seam for anything that reads across runs — the run browser and the figure
    scripts — so a new method under references/ shows up in both without further wiring.
    """
    roots = [(DEFAULT_SETUP, kind_root(kind))]
    baselines = OUTPUTS_ROOT / BASELINES_DIRNAME
    if baselines.is_dir():
        roots += [
            (d.name, kind_root(kind, d.name))
            for d in sorted(baselines.iterdir())
            if d.is_dir()
        ]
    return [(setup, path) for setup, path in roots if path.is_dir()]


MISC_CATEGORY = "misc"
# Kinds filed under <benchmark>/<category>/. Only daedalus's own inference: it is the kind that
# accumulates dozens of runs across many separate experiments, and the one every figure script
# reads. A memory-building run is referenced by path from the configs that consume its pool,
# and a baseline holds one evaluation per benchmark, so neither needs the extra levels.
CATEGORIZED_KINDS = ("inference",)


def is_categorized(kind: str, setup: str = DEFAULT_SETUP) -> bool:
    return kind in CATEGORIZED_KINDS and (not setup or setup == DEFAULT_SETUP)
# What marks a folder as a run rather than an organisational level. config.yaml alone is
# enough: it is written before the first task, so a crashed run is still found (and still
# listed as unscored) instead of being walked into.
_RUN_MARKERS = ("config.yaml", "run_meta.json", "evaluation.json")
_MAX_RUN_DEPTH = 3  # <benchmark>/<category>/<name>


def experiment_dir(cfg: ExperimentConfig) -> Path:
    """This experiment's own artifact folder.

    Inference: outputs/inference/<benchmark>/<category>/<name>/, so one folder holds exactly
    the runs one table or figure reads. Other kinds: <kind root>/<name>/ (see kind_root). A
    baseline (setup != daedalus) writes outputs/baselines/<setup>/{memory,inference}/<name>/.
    """
    root = kind_root(cfg.kind, cfg.setup)
    if is_categorized(cfg.kind, cfg.setup):
        return root / cfg.benchmark / (cfg.category or MISC_CATEGORY) / cfg.name
    return root / cfg.name


def walk_runs(root: Path, depth: int = _MAX_RUN_DEPTH) -> list[Path]:
    """Run folders at or below `root`, deepest-last, sorted.

    Depth-agnostic on purpose: it finds a run whether it sits directly under the kind root
    (the pre-category layout, and any tree not yet migrated) or under <benchmark>/<category>/.
    A folder holding a marker file IS the run, so the walk stops there and never descends into
    its run_<i>/ trace dirs.
    """
    out: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any((d / m).exists() for m in _RUN_MARKERS):
            out.append(d)
        elif depth > 1:
            out.extend(walk_runs(d, depth - 1))
    return out


def run_dirs(kind: str) -> list[tuple[str, Path]]:
    """Every (setup, run folder) of `kind` on disk, daedalus first.

    The discovery seam for anything that reads across runs — the run browser and the figure
    scripts. It replaced `kind_roots` + `root.iterdir()` at those call sites when inference
    runs gained the <benchmark>/<category>/ levels: iterdir() would have handed back the
    benchmark folders instead of the runs.
    """
    return [(setup, d) for setup, root in kind_roots(kind) for d in walk_runs(root)]


def run_slot(kind: str, setup: str, path: Path) -> tuple[str, str]:
    """(benchmark folder, category) read back off a run's path.

    ("", "") for a run sitting directly under the kind root — an unmigrated run, or a kind
    that is not categorized at all. The path is the source of truth here rather than the run's
    own config.yaml, so a run that was moved by hand reports where it actually is.
    """
    try:
        parts = path.relative_to(kind_root(kind, setup)).parts
    except ValueError:
        return "", ""
    return (parts[0], parts[1]) if len(parts) >= 3 else ("", "")


def run_id(kind: str, setup: str, path: Path) -> str:
    """A run's identity within its setup: its path under the kind root, e.g.
    "appworld/main-results/no-memory". Folder names repeat across benchmarks and sections
    (every benchmark has a `main-results/no-memory`), so the bare name is not enough."""
    try:
        return path.relative_to(kind_root(kind, setup)).as_posix()
    except ValueError:
        return path.name


def find_run_dir(kind: str, setup: str, name: str) -> Path | None:
    """The run folder `name` within one setup: a run_id, or a bare folder name when it is
    unique in that setup (the first match wins otherwise)."""
    for s, d in run_dirs(kind):
        if s == setup and run_id(kind, s, d) == name:
            return d
    for s, d in run_dirs(kind):
        if s == setup and d.name == name:
            return d
    return None


def resolve_trace_dir(cfg: ExperimentConfig, run_idx: int | None = None) -> Path:
    """Where per-task traces are written.

    Inference and testset runs use run_<idx>/ subfolders (one per repetition, scored for
    pass^k); generation and accumulation solver traces go in a single traces/ subfolder.
    """
    if cfg.logging.trace_dir:
        path = Path(cfg.logging.trace_dir)
        return path / f"run_{run_idx}" if run_idx is not None else path
    base = experiment_dir(cfg)
    if cfg.kind in ("inference", "testset"):
        return base / f"run_{run_idx}" if run_idx is not None else base
    return base / "traces"


def pool_output_path(cfg: ExperimentConfig) -> Path:
    """Where an accumulation/generation run writes its banked-heuristic pool."""
    return experiment_dir(cfg) / "pool.json"


def generation_paths(cfg: ExperimentConfig) -> dict[str, Path]:
    """The generation run's artifact paths (tasks bank, sessions, pool, summary)."""
    base = experiment_dir(cfg)
    return {
        "tasks": base / "tasks.json",
        "sessions": base / "sessions",
        "pool": base / "pool.json",
        "run_summary": base / "run_summary.json",
        "traces": base / "traces",
        "lock": base / "tasks.lock",
        "coverage": base
        / "coverage_tags.txt",  # survey artifact (reusable as coverage_tags_path)
    }


def _benchmark_domain(cfg: ExperimentConfig) -> str:
    """The benchmark's domain/split, for naming."""
    return {
        "tau2": cfg.tau2.domain,
        # A list, since a run may span several domains; "all" when it is every scored one.
        "automationbench": "-".join(cfg.automationbench.domains) or "all",
    }.get(cfg.benchmark, cfg.appworld.dataset)


def resolve_experiment_name(raw: str, cfg: ExperimentConfig) -> str:
    """Expand naming tokens in an experiment name.

    Supported tokens: {benchmark} {domain} {model} {date} (YYYYMMDD)
    {time} (HHMMSS) {datetime} (YYYYMMDD_HHMMSS). Use {datetime} for a fresh,
    timestamped folder every run, e.g. experiment_name: "{benchmark}_{domain}_{datetime}".
    """
    from datetime import datetime

    now = datetime.now()
    return raw.format(
        benchmark=cfg.benchmark,
        domain=_benchmark_domain(cfg),
        model=cfg.agent.model.replace("/", "-"),
        date=now.strftime("%Y%m%d"),
        time=now.strftime("%H%M%S"),
        datetime=now.strftime("%Y%m%d_%H%M%S"),
    )


def _guard_foreign_run(cfg: ExperimentConfig, base: Path) -> None:
    """Refuse to write into an experiment folder that already holds a DIFFERENT run.

    `experiment_name` is the identity of a run: it names the output folder and, for
    AppWorld, the on-disk world directory the official evaluator reads. Pointing a second
    config at the same name mixes two splits' traces in one folder and makes evaluation
    score this split's task ids against the other split's saved worlds. The stamped
    run_meta.json is the witness, so compare against it and stop before anything is
    overwritten. Same benchmark + same split = the normal re-run/resume path, allowed.
    """
    import json

    meta_path = base / "run_meta.json"
    if not meta_path.exists():
        return
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return  # unreadable stamp — nothing to compare against
    domain = _benchmark_domain(cfg)
    clashes = [
        (k, prev.get(k), new)
        for k, new in (("benchmark", cfg.benchmark), ("domain", domain))
        if prev.get(k) is not None and prev.get(k) != new
    ]
    if not clashes:
        return
    detail = "; ".join(
        f"{k}: existing {old!r} vs this run {new!r}" for k, old, new in clashes
    )
    raise FileExistsError(
        f"{base} already holds a different run ({detail}). Refusing to overwrite it — a "
        f"shared experiment_name mixes both runs' traces in one folder and makes scoring "
        f"read the wrong saved worlds. Give this run its own experiment_name (e.g. "
        f"{cfg.name}_{domain}), or remove/rename that folder to reuse the name."
    )


# Config blocks that decide HOW a task is executed. A change to any of them invalidates
# work already on disk, because a resume reuses that work verbatim (accumulation restores
# each task from `task_summaries/`, inference skips any run whose traces exist). `run` and
# `logging` are excluded: raising `num_runs` or `max_tasks` extends a run rather than
# contradicting it. `generation.num_sessions` is excluded for the same reason — a resume
# runs that many MORE sessions.
_FINGERPRINT_EXCLUDED_KEYS = {"generation": ("num_sessions",)}

# Bumped whenever the hash material changes (a block or field added or removed), so a
# folder fingerprinted by an older definition is passed through rather than refused.
# v5: the public-release schema.
_FINGERPRINT_VERSION = 5

_FINGERPRINT_BLOCKS = (
    "agent",
    "memory",
    "accumulation",
    "generation",
    "appworld",
    "tau2",
    "automationbench",
    "provider_routing",
)


def config_fingerprint(cfg: ExperimentConfig) -> str:
    """Hash of the settings that decide how a task is executed (see _FINGERPRINT_BLOCKS)."""
    import hashlib
    import json

    data = cfg.to_dict()
    material = {}
    for block in _FINGERPRINT_BLOCKS:
        value = data.get(block)
        dropped = _FINGERPRINT_EXCLUDED_KEYS.get(block)
        if dropped and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in dropped}
        material[block] = value
    canonical = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _guard_changed_config(cfg: ExperimentConfig, base: Path) -> None:
    """Refuse to reuse a run folder whose recorded config differs from this one.

    A resume REUSES completed work: accumulation restores every task from
    `task_summaries/`, and inference skips any run whose traces exist. So changing
    `agent.model`, `tau2.max_steps`, `memory.pool_path` or an `accumulation` knob and
    re-launching under the same `experiment_name` used to re-run NOTHING — it re-scored
    the old traces and reported them under the new config. Nothing compared the two:
    `_guard_foreign_run` only ever checked benchmark and domain.

    Runs made before the fingerprint existed carry no hash; those are passed through,
    since there is nothing to compare against.
    """
    import json

    meta_path = base / "run_meta.json"
    if cfg.run.force or not meta_path.exists():
        return
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    recorded = prev.get("config_fingerprint")
    if not recorded or recorded == config_fingerprint(cfg):
        return
    # A folder fingerprinted under an older definition of the hash material cannot be
    # compared against one computed here — the mismatch would say "different config" when
    # only the hash changed. Pass it through, as pre-fingerprint folders already are.
    if prev.get("config_fingerprint_version", 1) != _FINGERPRINT_VERSION:
        from daedalus.core.logging import console  # local: keeps this module stdlib-light

        console.info(
            f"  note: {base} was fingerprinted by an older definition "
            f"(v{prev.get('config_fingerprint_version', 1)} vs v{_FINGERPRINT_VERSION}); "
            f"the changed-config check is skipped for it. Verify the config yourself."
        )
        return
    raise FileExistsError(
        f"{base} was produced by a DIFFERENT config (fingerprint {recorded} vs "
        f"{config_fingerprint(cfg)}). Completed work in that folder would be reused as-is "
        f"rather than re-run, so the results would be reported under a config that did not "
        f"produce them.\n"
        f"Pick one: give this run its own `experiment_name`; pass --force to discard and "
        f"re-run; or revert the change to the "
        f"{', '.join(_FINGERPRINT_BLOCKS)} blocks. (`run` and `logging` are not "
        f"fingerprinted, so raising num_runs or max_tasks to extend a run is always fine.)"
    )


def save_run_config(cfg: ExperimentConfig) -> Path:
    """Snapshot the resolved config + run metadata into the experiment folder.

    Writes <experiment_dir>/config.yaml (the exact effective config) and run_meta.json
    (setup, kind, benchmark, category, name, model, config fingerprint, timestamp) so every
    experiment folder is self-describing and reproducible. Returns experiment_dir.

    Every previous snapshot is kept under `config_history/`. Overwriting config.yaml in
    place is how the record of a resumed run's real parameters was lost: a generation run
    executed as seven resumes of `num_sessions: 7` ended up with a config.yaml claiming 7
    sessions next to 50 session files.
    """
    import json
    from datetime import datetime, timezone

    base = experiment_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)
    _guard_foreign_run(cfg, base)
    _guard_changed_config(cfg, base)

    snapshot = yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
    current = base / "config.yaml"
    # Keep the outgoing snapshot when it differs, so a resume cannot erase what the
    # earlier launches actually ran with.
    if current.exists() and current.read_text(encoding="utf-8") != snapshot:
        history = base / "config_history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (history / f"config_{stamp}.yaml").write_text(
            current.read_text(encoding="utf-8"), encoding="utf-8"
        )
    current.write_text(snapshot, encoding="utf-8")

    (base / "run_meta.json").write_text(
        json.dumps(
            {
                "experiment_name": cfg.name,
                "setup": cfg.setup,
                "benchmark": cfg.benchmark,
                "category": (cfg.category or MISC_CATEGORY)
                if is_categorized(cfg.kind, cfg.setup) else cfg.category,
                "kind": cfg.kind,
                "domain": _benchmark_domain(cfg),
                "model": cfg.agent.model,
                "config_fingerprint": config_fingerprint(cfg),
                "config_fingerprint_version": _FINGERPRINT_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return base
