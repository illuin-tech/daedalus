# ExpeL — experiential learning from insights and past successes

Re-implementation of ExpeL (Zhao et al., 2024) on daedalus's harness. Accumulation runs each
training task with ReAct+Reflexion (up to 3 retries, Alg. 1), then walks every
(success, failure) pair of the same task and every batch of *L* = 3 successes, letting the
extraction LLM AGREE with, REMOVE, EDIT or ADD rules to **one shared insight list** capped at
20 (Alg. 2). At test time the whole insight list, plus the *k* = 2 successful trajectories
whose task is most similar (all-mpnet-base-v2 embeddings), are injected at task start, and
the task is solved in a single attempt (Alg. 3).

## Run

```bash
# 1. gather experience and extract insights from the training split
uv run python -m references.expel.accumulation --config references/expel/configs/tau2_accumulation.yaml
# 2. evaluate on the test split (5 runs)
uv run python -m references.expel.run --config references/expel/configs/tau2_inference.yaml
```

Configs ship for AppWorld (`appworld_*`), τ²-bench retail (`tau2_accumulation`,
`tau2_inference`) and AutomationBench Operations (`automationbench_*`). `run` takes
`--insights` and `-k` overrides on top of the common flags. Outputs:
`outputs/baselines/expel/memory/<name>/` (`pool.json` = the insight list with its
importance counts, `pool.log.json` = the list after every extraction call,
`demonstrations.json` = the successful trajectories recalled as few-shots, `experiences/` =
the raw Reflexion trials, `task_summaries/`, `traces/`) and
`outputs/baselines/expel/inference/<name>/` (`run_<i>/`, `evaluation.json`, `retrieval/`).

```yaml
expel:
  insights_path: "outputs/baselines/expel/memory/tau2/pool.json"
  fewshot_k: 2                   # recalled trajectories (demonstrations.json next to the list)
  embedder: "sentence-transformers/all-mpnet-base-v2"
  extraction_model: "gpt-5.4"    # insight extraction
  extraction_reasoning_effort: "high"
  max_num_rules: 20              # insight list cap
  success_batch_size: 3          # L
  max_retries: 3                 # Reflexion retries when gathering experience
  reflection_model: null         # null = the solver model, as in ExpeL
  max_pairs_per_task: 3
```

The first inference run downloads the all-mpnet-base-v2 weights into the Hugging Face cache;
recall is an embedding pass per task, not an LLM call.

## Paper → code

| paper | here |
| --- | --- |
| Alg. 1, experience gathering | `references/common/experience.py`, driven by `accumulation.py` |
| Alg. 2, insight extraction | `accumulation.py::extract_insights` |
| Fig. 2 critique prompts (compare / all-success) | `extraction.py` + `prompts/critique_*.txt` |
| the four operations and importance counts | `insights.py` |
| §4.3 inference (rules block, then few-shots) | `agents/base.py` |
| experience recall (kNN, all-mpnet-base-v2) | `retrieval.py::DemonstrationRetriever` |

## Deviations from upstream

- The prompts come from the authors' released implementation (the paper's figures elide
  text); the rule bookkeeping (ADD +2, AGREE/EDIT +1, REMOVE −1, −3 over budget, drop at 0)
  comes from the same place.
- The two domain-bound clauses ExpeL re-writes per domain (what the agent has access to, how
  a trial fails) are re-written per benchmark in `extraction.py::_DOMAIN_CLAUSES`.
- The acting prompt is daedalus's solver prompt, with the insight and few-shot blocks in its
  memory slot; ExpeL's few-shot wrapper is kept.
- *L* = 3 rather than the paper's 4–8, because these trajectories are much longer.
- Reflexion's reflection prompt is daedalus's post-attempt reflection prompt (ExpeL ships
  none of its own).
- On τ²-bench, Reflexion's reflector is gpt-5.4, the auxiliary model of every method in the
  paper (`reflection_model: gpt-5.4`); ExpeL reflects with the policy model, as the AppWorld
  and AutomationBench configs do.
- Not implemented: seeding with manual few-shots (`Fmanual`), the transfer-learning prompt
  (§4.4), and the repo's alternative few-shot / truncation / reranking strategies.
