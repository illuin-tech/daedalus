# Reference methods

The memory baselines of the paper, re-implemented on daedalus's harness. Each one reuses
daedalus's benchmarks, solver agents, trace format, parallel runner and scoring, and replaces
only what a memory method owns: how memory is built (`accumulation`) and how it is retrieved
and injected at test time (`run`). Hyperparameters are those of the paper's Table 13
(Appendix E).

| method | folder | paper | configs (`<folder>/configs/`) |
| --- | --- | --- | --- |
| ExpeL | [`expel/`](expel) | Zhao et al., 2024 | `{appworld,tau2,automationbench}_{accumulation,inference}` |
| ERL | [`erl/`](erl) | Allard et al., 2026 | same six |
| AutoGuide | [`autoguide/`](autoguide) | Fu et al., 2024 | same six |
| ReasoningBank | [`reasoningbank/`](reasoningbank) | Ouyang et al., 2026 | same six |
| ACE | [`ace/`](ace) | Zhang et al., 2026 | same six |
| PREPING | [`preping/`](preping) | Choi et al., 2026 | run with the authors' code; its playbooks are evaluated through daedalus's `memory.pool_path` (see its README) |

The three benchmarks are AppWorld, τ²-bench retail and AutomationBench Operations. Each
config was regenerated from the snapshot of the run behind the paper's number (its header
names that run's output folder).

## Running a method

```bash
uv run python -m references.<method>.accumulation --config references/<method>/configs/<bench>_accumulation.yaml
uv run python -m references.<method>.run          --config references/<method>/configs/<bench>_inference.yaml
```

Both entry points take daedalus's usual flags (`--experiment-name --task-id --max-tasks
--parallel`; `run` adds `--num-runs --force` and one or two method overrides such as `--pool`
or `-k`), resume from what is already on disk, and refuse a config written for the other
entry point. Run them from the repo root. Artifacts go to
`outputs/baselines/<method>/{memory,inference}/<name>/`, where the paper's configs name each
run after its benchmark (`outputs/baselines/erl/memory/tau2/`), in daedalus's shapes
(`pool.json`, `task_summaries/`, `run_<i>/`, `evaluation.json`), and are tagged
`setup: <method>` in `serve` and the plots.

Each method's own block (`erl:`, `expel:`, …) sits at the top level of the same YAML and is
snapshotted to `<experiment_dir>/<method>.yaml`, which is how spawned workers read it back.
`memory.enabled` is forced off: every method does its own retrieval.

## Layout

- `common/` — shared plumbing: `config.py` (method config blocks), `run.py` (the inference
  driver every `run.py` hands its hooks to), `accumulation.py` (CLI, checkpoints/resume,
  worker pool), `experience.py` (ReAct+Reflexion experience gathering for ExpeL and
  AutoGuide), `solver.py`, `usage.py`.
- `<method>/` — `config.py`, `accumulation.py`, `run.py`, `agents/<benchmark>.py` (a small
  subclass of each benchmark's solver), `prompts/`, and the method's own modules.
