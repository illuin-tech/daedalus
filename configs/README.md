# Configs

One YAML decides everything about a run. Every config here was regenerated from the
`config.yaml` snapshot of the run that produced a number in the paper; its header names the
paper item and the output folder, and its `experiment_name` (plus `category` for inference) is
that folder's name in the released outputs, so re-running a config writes exactly where the
release holds the paper's run. A config lists only what differs from the defaults in
`daedalus/core/config.py`, whose dataclasses document every field.

```
configs/<benchmark>/generation/     DAEDALUS banks          (daedalus.scripts.generation)   → outputs/daedalus/
configs/<benchmark>/accumulation/   DAEDALUS-curated banks  (daedalus.scripts.accumulation) → outputs/daedalus-curated/
configs/<benchmark>/inference/      evaluation              (daedalus.scripts.inference)    → outputs/inference/
configs/appworld/inference/models/  other base agents (Table 2, Figures 1 and 8)
configs/smoke/                      one-task smoke tests   (tests/smoke_*.py)
```

The baselines keep their configs next to their code, in `references/<method>/configs/`; they
write to `outputs/baselines/<method>/memory/<benchmark>/` and
`outputs/baselines/<method>/inference/<benchmark>/`.
[`experiments-paths.md`](../experiments-paths.md) maps every table and figure to its configs
and outputs.

## Blocks

| block | read by | main fields |
| --- | --- | --- |
| top level | all | `benchmark` (appworld, tau2, automationbench), `experiment_name`, `category` (inference only: the paper section, i.e. the output sub-folder: `main-results`, `cross-family`, `pipeline-ablation`, …) |
| `agent` | all | the solver / base agent: `model`, `max_turns`, `temperature`, `reasoning_effort` |
| `appworld`, `tau2`, `automationbench` | all | task selection and environment (split, domain, step cap, τ² user simulator) |
| `memory` | inference | `enabled`, `pool_path`, `heuristics_at_start` (the method: whole bank at task start), or per-turn `retriever` + `injection.policy` + `retrieval_mode` |
| `accumulation` | accumulation, generation | the Solver loop: `max_failures` (Nf), `num_success_to_continue` (Ns), `extraction_model` |
| `generation` | generation | the Explorer, Judge, Surveyor (`coverage_tags`), guideline bank (`explorer_guidelines`), `max_refinements` (Nr), `num_sessions`, `stage` (Table 3 ablations B and C) |
| `run` | all | `num_runs`, `parallel`, `max_tasks` |
| `provider_routing` | all | OpenRouter upstream pinning (the Qwen and DeepSeek runs) |

Two safety rules:

- **Unknown keys inside a block are refused**, and the error names the near miss, so a typo
  cannot silently fall back to a default. Unknown top-level keys are allowed: that is how a
  baseline carries its own block (`expel:`, `erl:`, …).
- **A run folder is fingerprinted against its config.** Relaunching a changed config under an
  existing `experiment_name` is refused instead of silently reusing finished work; use a new
  name or `--force`. `run`, `logging` and `generation.num_sessions` are not fingerprinted, so
  raising `num_runs` or resuming a generation run with more sessions is always allowed.

Paths are relative to the repository root, so pools resolve once the released outputs
([`illuin/daedalus-traces`](https://huggingface.co/datasets/illuin/daedalus-traces)) are downloaded as `outputs/`
there. Put local, unshared configs in `configs/private/` (git-ignored).
