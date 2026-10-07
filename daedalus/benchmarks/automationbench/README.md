# AutomationBench connector

[AutomationBench](https://github.com/zapier/AutomationBench) is a business-workflow benchmark.
It has 47 simulated SaaS apps (CRM, inbox, sheets, calendar, helpdesk, ...) behind ~500 REST
endpoints, one in-process pydantic `WorldState` per task, and 800 public tasks: 100 in each of
six scored domains, plus 200 `simple` tasks that the benchmark leaves out of its score. Grading
is a deterministic sweep of final-state assertions. Assertions that already held before the
agent acted are excluded, and there is no LLM in the grader. The agent gets three tools:
`api_search` (BM25 over endpoints), `api_fetch` and `base64_encode`. Finding the right endpoint
is part of the task.

| file | role |
| --- | --- |
| `config.py` | `AutomationBenchConfig`: `domains` (default `["operations"]`), `split_file`/`split`, `task_ids`, `repo_root`, `max_steps`, `seed` |
| `__init__.py` | locates the checkout and puts it on `sys.path` (see Setup) |
| `task_loader.py` | loads tasks through the benchmark's own `get_domain_dataset`, which injects its noise records; ids are `info["task_name"]` |
| `env.py` | `AutomationBenchSession`: the world, the `api` tool schemas, tool execution and grading with the benchmark's rubric |
| `mirrored.py` | behaviour copies of two upstream functions (deviation 2) |
| `agent.py` | `AutomationBenchTaskAgent`, the solver: a native tool-calling loop |
| `runner.py` | `AutomationBenchBenchmark`: aggregates pass rate, mean `partial_credit`, `step_cap_rate` and a per-domain breakdown |
| `explorer.py` | the Explorer and the Surveyor, in a throwaway copy of a training-task world |
| `spec.py` | the generated-task spec (`{task, expected_path, success_conditions}`) and its conversion to a task |
| `backend.py` | `AutomationBenchGenerationBackend`: LLM-judge grading; the survey visits the train worlds that cover the services graded most often |
| `splits/` | `operations_30_70.json`, the paper's split, and `public_30_70.json`, the same recipe over all six domains |

## How the paper uses it

- **Domain:** Operations, 100 tasks, split 30 train / 70 test (`splits/operations_30_70.json`).
  The split is stratified by how many services a task involves and was built by
  `daedalus/scripts/make_automationbench_split.py` with seed 42. Evaluation runs 5 times on `test`.
- **Models:** solver `gpt-5.6-luna` at medium effort; explorer, judge and extractor `gpt-5.6-terra`
  at high effort. Step cap 50 (`automationbench.max_steps`, not `agent.max_turns`).
- **Generation** runs 30 sessions and uses only the worlds of the 30 training tasks, with their
  assertions and tool grants stripped. Each world is subscribed to exactly the services it seeds.
- Configs are in `configs/automationbench/`.

## Setup

Fetch the `benchmarks/AutomationBench` submodule (package 1.0.6), or
set `automationbench.repo_root`, and run `uv sync --extra automationbench`. **Keep
`OPENAI_API_KEY` set** whichever provider the solver uses. The simulated ChatGPT app makes a real
OpenAI call, and falls back to a stub without a key. This affects 15 of the 600 scored tasks,
which are therefore not deterministic.

## Prompts (`prompts/`)

| file | used by |
| --- | --- |
| `solver_env.txt`, `solver_protocol.txt` | the solver. The protocol describes the three tools and ends with **the benchmark's own system message, verbatim** |
| `explorer_env.txt`, `explorer_protocol.txt`, `explorer_task_guidelines.txt` | the Explorer (Prompts 2 and 3) |
| `explorer_refine_protocol.txt` | refining an off-target task (Prompt 4) |
| `coverage_survey_protocol.txt` | the Surveyor (Prompt 1) |
| `accumulation_memory_extraction.txt` | **override** of the shared Extractor: the business-operations version, paper Prompt 12 |

## Deviations from upstream

1. **The package is read from the checkout, not installed.** `automation-bench` depends on
   `verifiers>=0.2.0`, which pulls in a large dependency tree for a runner we replace. The modules
   we import (world schema, domain loaders, `api` tools, rubric, task contract) need only
   `pydantic` and `datasets`. Any upstream module that imports `verifiers` is therefore out of
   reach, so the tool schemas are derived from the function signatures in `env.py`.
2. **Two functions are mirrored instead of imported.** `compute_allowed_services` and
   `strip_none_values` live in upstream `runner.py`, which imports `verifiers`. `mirrored.py` holds
   behaviour copies of both. `tests/test_automationbench_adapter.py` extracts the originals from the
   checkout with `ast` and compares their outputs on all 800 tasks. The first function matters
   most: `api_fetch` refuses any service outside `world.meta.allowed_services`.
3. **The agent loop is ours.** Upstream runs a `verifiers` `StatefulToolEnv`. Here the loop is a
   native tool-calling loop through the harness `LLMClient`, with the same tasks, tool surface and
   rubric. Their transports (the Anthropic streaming client, the OpenAI Responses path, the
   Batch-API mode) are not used. Scores are comparable in kind to the published table, not a
   reproduction of it.
4. **The prompt is composed, not replaced.** Upstream's system prompt is the task's own system
   message and nothing else. Ours is the shared solver template, then `solver_env.txt`, then the
   tool protocol. The protocol carries their system message verbatim, because it states rules that
   affect the score: the ~50-turn budget, and not listing skipped items, which the
   `*_not_contains` assertions penalise.
5. **Only the `api` toolset is supported,** which is the one the published table uses. The
   `zapier` and `limited_zapier` toolsets (named actions) are not implemented.
6. **Hitting the step cap is not a failure.** As in upstream, the run stops at 50 model steps and
   the world is scored as it stands. The trace records `extra.hit_step_cap`, and `evaluate_run`
   reports `step_cap_rate`.
7. **`_end_state` is dropped.** Upstream's rubric stores the whole final world for its exporter.
   We keep only the per-assertion verdicts (`extra.assertions`), together with `partial_credit` and
   the benchmark's `task_contract_sha256`. The contract hash makes a task that changed under a run
   detectable.
8. **The split is ours.** Upstream ships no train/test split, so we provide
   `splits/operations_30_70.json`.
9. **Generated tasks are graded by the LLM judge.** The benchmark checks its own tasks with 571
   assertion types, and an explorer cannot realistically be asked to write assertions in that
   vocabulary. Generated tasks are therefore judged against natural-language success conditions.
   The explorer's `submit_task` is refused unless at least one write succeeded, every write the
   spec declares was actually executed, and no attempted write was left failing.
