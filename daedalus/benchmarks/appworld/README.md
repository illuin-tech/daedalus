# AppWorld connector

[AppWorld](https://github.com/StonyBrookNLP/appworld) is a code-execution benchmark: nine
simulated apps behind ~450 APIs, driven by arbitrary Python that the agent writes each turn
against a per-task database, and graded by each task's own programmatic checker over the
start-vs-end database pair. This package implements both seams: `Benchmark` (inference and
accumulation) and `GenerationBackend` (self-play generation).

| file | role |
| --- | --- |
| `config.py` | `AppWorldConfig`: `dataset` (train / dev / test_normal / test_challenge) and an optional `task_ids` filter |
| `__init__.py` | resolves the AppWorld data root (`appworld_root`) before `appworld` is imported (see Setup) |
| `runner.py` | `AppWorldBenchmark`: task listing, agent dispatch, scoring with AppWorld's own evaluator |
| `agent.py` | `ReActAgent`, the solver: `Thought:` + one ```` ```python ```` block per turn, executed through `appworld.AppWorld` |
| `memory_agent.py` | `MemoryReActAgent`: the same loop with per-turn retrieval (`pre_generation` or `reason_then_retrieve`) |
| `explorer.py` | the Explorer and the Surveyor: design a task inside a sandbox world, solve it in code, emit `{task, expected_path, success_conditions, tags}` |
| `backend.py` | `AppWorldGenerationBackend`: sandbox worlds, LLM-judge grading, held-out-app screening of mined heuristics |
| `hidden_apps.py` | hides `generation.excluded_apps` from the explorer, on the code channel and in every output |
| `direct_explore.py` | explorer-only sessions for the two exploration-only ablations (Table 3) |
| `paths.py` | `app.tool` call paths from code and from a spec, used for grounding and for the novelty filter |

## How the paper uses it

- **Evaluation:** `test_normal`, 168 tasks, 5 runs. **Training side:** the 90 `train` tasks.
  Both are AppWorld's own splits, and task ids are AppWorld's.
- **Models:** solver `gpt-5.4-mini` (provider-default reasoning), step ceiling 30
  (`agent.max_turns`). Every auxiliary role (explorer, judge, extractor) is `gpt-5.4` at high effort.
- **Generation** runs 90 sessions. Each session plays in the initial world of one **train** task
  (`generation.sandbox_dataset: train`); that task's instruction and verifier are never read, so
  generation never touches a test-split world. **Gmail and Amazon are held out**
  (`generation.excluded_apps: [gmail, amazon]`) because they appear only in `test_challenge`. This
  run alone also used an early novelty filter (`min_novelty: 0.05`), which is not part of the
  general method (Appendix E).
- Configs are in `configs/appworld/{inference,accumulation,generation}/`.

## Setup

Fetch the `benchmarks/appworld` submodule, `uv sync --extra appworld`, then
`uv run appworld install && uv run appworld download data`. AppWorld's `apply_db_changes()`
refuses any database path that contains the substring `memory`. Importing this package therefore
points `APPWORLD_ROOT` at a sibling `appworld/` checkout that has a `data/` directory, unless the
variable already names a usable root. The agent and the evaluator both read the root through
`appworld_root()`, so they always use the same on-disk experiment directory.

## Prompts (`prompts/`)

Each one is composed with a shared template in `daedalus/core/prompts/` (see that README).

| file | used by |
| --- | --- |
| `solver_env.txt`, `solver_protocol.txt` | the solver: the simplified AppWorld solver of paper Prompt 10 (four auth/inspection facts, then the code protocol and the app catalogue) |
| `explorer_env.txt`, `explorer_protocol.txt`, `explorer_task_guidelines.txt` | the Explorer (Prompt 2; the guidelines import the shared doctrine of Prompt 3) |
| `explorer_refine_protocol.txt` | refining an off-target task (Prompt 4) |
| `coverage_survey_protocol.txt`, `coverage_survey_deliverable_partition.txt` | the Surveyor (Prompt 1) |
| `naive_explorer_env.txt`, `naive_explorer_protocol.txt` | the exploration-only ablations |

AppWorld has no Extractor override, so it uses the shared `accumulation/memory_extraction.txt`
(Prompt 7) unchanged. The extractor is also shown `solver_env.txt` and `solver_protocol.txt`, so
that a mined heuristic does not repeat what the solver is already told.

## Deviations from upstream

1. **The agent loop is ours.** AppWorld's `experiments/code/` agents and CLI are not used. The
   ReAct loop parses `Thought:` and one Python block, executes the block through `AppWorld`, and
   sends a format nudge on any turn that contains no code. Completions go through the harness
   `LLMClient`, which provides cost accounting, retries, reasoning effort and memory injection.
   The environment, `load_task_ids` and the evaluator are AppWorld's, unmodified.
2. **The solver prompt is simplified.** AppWorld's official prompt is a multi-turn few-shot
   transcript of one complete worked task, and it is full of human-written heuristics for the
   environment. Every method here uses a short written protocol instead (paper Prompt 10), so that
   all of them get the same, smaller amount of hand-engineered guidance.
3. **The solver prompt contains a short environment block.** `solver_env.txt` states four concrete
   facts about authentication and record inspection. The block is identical in every arm,
   including the no-memory baseline, so "no memory" is not literally free of environment
   knowledge.
4. **Scores come from AppWorld's own outputs.** `evaluate_run` calls `evaluate_tasks` on the
   run-scoped experiment directory that the agent wrote. A task passes only with zero failures, and
   a task whose result has no `failures` key counts as failed.
5. **Held-out apps are hidden at the channel as well as in the prompt.** The channel is arbitrary
   Python, so an explorer could otherwise find a held-out app through
   `show_app_descriptions()`, `dir(apis)`, doc search or an error message. `hidden_apps.py`
   refuses any code block that names a hidden app and filters every execution result. Only the
   app's *identity* is hidden: email addresses such as `…@gmail.com` are left intact, because the
   supervisor's own address is one of them. One leak remains open: code that prints a count
   derived from the live catalogue shows the true total. The solver of a generated task still sees
   the full catalogue, so `backend.py::heuristic_admissible` also refuses to bank any heuristic that
   names a held-out app.
6. **Generated tasks are graded by the LLM judge.** A generated task has no ground-truth checker, so
   the judge grades it against the explorer's success conditions. It reads the trajectory through
   the shared renderer, which cuts each execution output at 500 characters. The explorer's spec is accepted only if its `expected_path` was actually executed.
7. **Code runs on the host, not in a sandbox.** `AppWorld(...)` executes the agent's code inside
   this process. AppWorld's safety guard blocks `open`, writes and `subprocess`, but not
   `os.listdir`, `os.scandir`, `Path.iterdir`/`rglob`/`read_text` or `os.environ`. The macOS disk is
   also case-insensitive, so AppWorld-style paths such as `~/documents/...` open the real
   `~/Documents`. Some agents used this to list the machine's folders and read `.env` files. Run
   AppWorld in Docker (`remote_docker`), or as a user with no access to anyone's home folder.

## Anonymized traces

The shipped `outputs/` were anonymized after the runs; scores and every non-text value are
unchanged.

- **Redacted outputs.** 2,840 turn outputs that showed the host machine are replaced by
  `[REDACTED: output from the host machine, outside the benchmark sandbox]`. A turn is redacted
  when any of these holds:
  - its output shows a home path outside the repo and tooling, or another host marker (hostname,
    key, environment dump, temp folder, repo listing);
  - its code touched the host filesystem or environment and printed more than a refusal;
  - it repeats a file or folder name leaked earlier in the same trace.

  The same rule applies inside composite texts (`last_trace_text`, ExpeL and AutoGuide
  trajectories). The code of a redacted turn is kept.
- **Placeholders.** Other strings use `<REPO>`, `<APPWORLD_ROOT>`, `<HOME>/<REDACTED_PATH>`,
  `<PERSON>`, `<USER>`, `<EMAIL>`, `<REDACTED_KEY>` and `<HOSTNAME>`. Host file names that agents
  copied into code, thoughts or queries became `<REDACTED_NAME>`.
- **Where this shows.** 797 files changed. One ExpeL demonstration
  (`references/expel/accumulation/expel_appworld_train/demonstrations.json`) is among them, so an
  ExpeL rerun reads that demonstration redacted.
