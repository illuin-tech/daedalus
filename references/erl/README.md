# ERL — Experiential Reflective Learning

Re-implementation of ERL (Allard et al., 2026) on daedalus's harness. Per training task, ERL
runs **one memory-free rollout**, reads the environment's binary reward, and reflects once
into a heuristic (`Analysis` + `Learned Guideline` with `Trigger:` / `Action:`); successes and
failures both bank. At test time, **one LLM ranker call per task** selects the top *k* = 20
heuristics from the whole pool on the task description, and they are injected into the
solver's system prompt for the whole task.

## Run

```bash
# 1. build the pool from the training split
uv run python -m references.erl.accumulation --config references/erl/configs/tau2_accumulation.yaml
# 2. evaluate on the test split (5 runs)
uv run python -m references.erl.run --config references/erl/configs/tau2_inference.yaml
```

Configs ship for AppWorld (`appworld_*`), τ²-bench retail (`tau2_accumulation`,
`tau2_inference`) and AutomationBench Operations (`automationbench_*`). `run` takes
`--pool` and `-k` overrides on top of the common flags. Outputs:
`outputs/baselines/erl/memory/<name>/` (`pool.json`, `pool.log.json`,
`task_summaries/`, `traces/`) and `outputs/baselines/erl/inference/<name>/` (`run_<i>/`,
`evaluation.json`, `retrieval/` with the ranker's selections and cost per task).

```yaml
erl:
  pool_path: "outputs/baselines/erl/memory/tau2/pool.json"  # inference
  k: 20                       # heuristics injected per task
  ranker_model: "gpt-5.4"     # ranks the whole pool, once per task
  ranker_reasoning_effort: "high"
```

The reflection LLM is daedalus's `accumulation.extraction_model` /
`extraction_reasoning_effort`.

## Paper → code

| paper | here |
| --- | --- |
| heuristic generation (§2, Fig. 8) | `reflection.py` + `prompts/heuristic_generation*.txt`, per task in `accumulation.py` |
| heuristic layout (Fig. 6) | `heuristics.py` (`Heuristic.block`, stored as a `MemoryItem`) |
| retrieval-augmented execution (§2, Fig. 9; LLM ranker, k = 20) | `retrieval.py` + `prompts/heuristic_retrieval.txt` |
| injection into the system prompt | `agents/<benchmark>.py` |

## Deviations from upstream

- The generation prompt's "operating in the ARE environment" reads "an interactive
  environment"; the rest of both prompts is the paper's text.
- The reward is the benchmark's own verdict (AppWorld's evaluator, τ²'s evaluator,
  AutomationBench's grader).
- The ranker is skipped when the pool holds at most *k* heuristics (the whole pool is
  injected), and falls back to the first *k* (logged) if its JSON cannot be parsed after one
  retry.
- On τ²-bench the ranker reads the task's `user_scenario`, which the solver itself never sees.
- Not implemented: the paper's ablations (embedding/random selection, failure-only or
  success-only pools, iterative ERL, the no-reward setting).
