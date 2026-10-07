# ReasoningBank — reasoning strategies distilled from self-judged experience

Re-implementation of ReasoningBank (Ouyang et al., 2026) on daedalus's harness. Accumulation
runs the paper's closed loop over the training split: for each task, **retrieve** the memory
items of the most similar past experience (*k* = 1, embedding search over past task queries),
solve the task once with them in the system prompt, label the trajectory with an
**LLM-as-a-judge** (no ground truth), **extract** up to 3 memory items
{title, description, content} with the success or the failure prompt (Fig. 9), and **add**
them to the bank without pruning. At test time the bank is frozen and each task retrieves the
items of its *k* = 1 most similar experience, injected into the system prompt at task start.
One rollout per training task: MaTTS test-time scaling is not used.

## Run

```bash
# 1. build the bank by streaming the training split (its labels are never read)
uv run python -m references.reasoningbank.accumulation --config references/reasoningbank/configs/tau2_accumulation.yaml
# 2. evaluate with the frozen bank on the test split (5 runs)
uv run python -m references.reasoningbank.run --config references/reasoningbank/configs/tau2_inference.yaml
```

Configs ship for AppWorld (`appworld_*`), τ²-bench retail (`tau2_accumulation`,
`tau2_inference`) and AutomationBench Operations (`automationbench_*`). `run` takes
`--bank` and `-k` overrides on top of the common flags. Outputs:
`outputs/baselines/reasoningbank/memory/<name>/` (`pool.json` = the bank, one item per
memory item; `pool.log.json` = per-task records with the judge's verdict; `task_summaries/`,
`traces/`, `retrieval/`) and `outputs/baselines/reasoningbank/inference/<name>/`
(`run_<i>/`, `evaluation.json`, `retrieval/`).

**`run.parallel` in an accumulation config is the number of tasks per wave**: everything
learned before a wave is available to every task in it, nothing learned inside it is. `1` is
the paper's exact stream; the shipped configs use 5 (AppWorld, τ²) and 8 (AutomationBench).

```yaml
reasoningbank:
  bank_path: "outputs/baselines/reasoningbank/memory/tau2/pool.json"
  k: 1                                 # EXPERIENCES retrieved per task (each brings ≤3 items)
  embedder: "text-embedding-3-large"   # stands in for the paper's gemini-embedding-001
  extraction_model: null               # null = agent.model, the paper's setting …
  extraction_temperature: 1.0          # … at temperature 1.0
  max_items: 3                         # per trajectory
  judge_model: null                    # null = agent.model, at temperature 0.0
```

In an accumulation config `bank_path` is ignored: the entry point points it at the run's own
`pool.json`, since the bank being built is the bank being retrieved from.

## Paper → code

| paper | here |
| --- | --- |
| memory item schema (§3.2) | `memory.py` (`Item`, `parse_items`, `to_pool`) |
| retrieval over task queries (§3.2, App. A.2) | `retrieval.py::ExperienceRetriever` |
| injection template (App. A.2) | `memory.py::injection_block` + `prompts/memory_injection.txt` |
| extraction, Fig. 9 | `extraction.py` + `prompts/extraction_{success,failure,user}.txt` |
| LLM-as-a-judge, Fig. 10 | `judge.py` + `prompts/judge_*.txt` |
| consolidation ("directly added") and the closed loop | `accumulation.py` (`_write_bank`, the wave loop) |

## Deviations from upstream

- **Source split instead of one continuous stream.** The paper grows the bank on the
  evaluation stream itself; here the closed loop runs over the training split and the bank is
  frozen for the test split, so within-stream transfer is not measured.
- Tasks run in parallel waves (see above) rather than strictly one at a time.
- The domain-bound clauses of the prompts ("an expert in web navigation", the judge's web-task
  taxonomy, "do not mention specific websites…") are re-written per benchmark in
  `domains.py`; the rest of each prompt is the paper's text.
- Two fields of the judge's input are dropped: the final webpage state (no analogue here) and
  the bot's response (already the last line of the trajectory).
- Retrieval uses `text-embedding-3-large` instead of `gemini-embedding-001`.
- The acting prompt is daedalus's solver prompt, with the items in its memory slot.
- On τ²-bench, the extractor and the judge are gpt-5.4, the auxiliary model of every method
  in the paper (`extraction_model`, `judge_model`); ReasoningBank runs both on the agent
  model, as the AppWorld and AutomationBench configs do.
- Not implemented: MaTTS (parallel and sequential test-time scaling) and Best-of-N selection.
