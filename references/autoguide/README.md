# AutoGuide — context-aware guidelines

Re-implementation of AutoGuide (Fu et al., 2024) on daedalus's harness. Accumulation runs each
training task with ReAct+Reflexion (up to 3 retries) to collect (success, failure) pairs; for
each pair (at most 3 per task) it finds the step where the trajectories diverge, summarizes the
shared prefix into a **context**, and extracts one guideline for it (Algorithm 1). The memory
is a `context → guidelines` dictionary. At test time, **every turn**, the agent's context is
identified, matched against the bank's contexts, and up to *k* = 3 of the matched context's
guidelines are injected into that turn's prompt; when no context matches, nothing is injected
(Algorithm 2).

## Run

```bash
# 1. build the guideline bank from the training split
uv run python -m references.autoguide.accumulation --config references/autoguide/configs/tau2_accumulation.yaml
# 2. evaluate on the test split (5 runs)
uv run python -m references.autoguide.run --config references/autoguide/configs/tau2_inference.yaml
```

Configs ship for AppWorld (`appworld_*`), τ²-bench retail (`tau2_accumulation`,
`tau2_inference`) and AutomationBench Operations (`automationbench_*`). `run` takes
`--bank` and `-k` overrides on top of the common flags. Outputs:
`outputs/baselines/autoguide/memory/<name>/` (`pool.json` = the bank, `pool.log.json`
= one row per guideline, `experiences/`, `task_summaries/`, `traces/`) and
`outputs/baselines/autoguide/inference/<name>/` (`run_<i>/`, `evaluation.json`,
`retrieval/` = the per-turn context decisions and their cost).

```yaml
autoguide:
  bank_path: "outputs/baselines/autoguide/memory/tau2/pool.json"
  k: 3                          # guidelines injected per turn
  context_model: null           # context identification/matching/selection; null = solver model
  extraction_model: "gpt-5.4"   # guideline extraction (Eq. 2)
  extraction_reasoning_effort: "high"
  max_retries: 3                # Reflexion retries when gathering experience
  reflection_model: "gpt-5.4"
  max_pairs_per_task: 3
```

Inference cost: one context-identification call per turn, one matching call per context
string not yet seen by the worker, and one selection call when the matched context holds more
than *k* guidelines. Totals are printed at the end of a run and recorded in `retrieval/`.

## Paper → code

| paper | here |
| --- | --- |
| Algorithm 1 (bank construction), deviation step | `accumulation.py` (`build_bank`, `deviation_index`), over `references/common/experience.py` |
| context identification / matching / guideline extraction / selection | `modules.py` + `prompts/context_*.txt`, `prompts/guideline_*.txt` |
| Algorithm 2 (test time) | `retrieval.py::ContextAwareRetriever`, installed by `agents/<benchmark>.py` |
| the guideline dictionary | `guidelines.py::GuidelineBank`, stored as a daedalus pool |

## Deviations from upstream

- The **ALFWorld** prompt variants are used (the closest to a tool-calling environment), with
  "a text-based ALFRED task" reworded as "an interactive task".
- Context identification runs **zero-shot**: the ALFWorld examples teach a household
  vocabulary that would mis-key every context here. The extraction prompt keeps the paper's
  two format examples.
- Two inputs the figures abbreviate are filled explicitly: the identified context reaches the
  extraction prompt as `SUMMARIZATION: <context>`, and the matching prompt receives the new
  context under `Newly Generated Summarization:`. `{Task description}` is one line naming the
  benchmark and domain (`prompts/task_description.txt`).
- *k* is a parameter (the paper's figures hardcode 2 or 3); *k* = 3 is the best value in the
  paper's own ablation.
- Reflexion's reflection prompt is daedalus's post-attempt reflection prompt (AutoGuide ships
  none).
- On τ²-bench, context identification, matching and selection run on gpt-5.4, the auxiliary
  model of every method in the paper (`context_model: gpt-5.4`); AutoGuide runs them on the
  agent model, as the AppWorld and AutomationBench configs do.
- Not implemented: the WebArena extraction variant (against human demonstrations) and
  AutoGuide + Reflexion at test time.
