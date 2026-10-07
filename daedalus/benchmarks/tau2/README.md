# τ²-bench connector

[τ²-bench](https://github.com/sierra-research/tau2-bench) is a dual-control customer-service
benchmark. An agent and a *simulated customer* act on one shared environment through tools,
under a written domain policy. The agent has to discover the request through conversation,
and a correct refusal counts as a success. Grading compares the database end state with the
gold state and runs communication checks. This is the connector that reuses the most of its
upstream: the `Orchestrator`, the environment, the user simulator and the evaluator are all
τ²'s. We replace the **agent inside the loop**.

| file | role |
| --- | --- |
| `config.py` | `Tau2Config`: `domain`, `task_split`, `task_ids`, `solver_mode`, `user_llm`, `max_steps`, `max_errors`, `seed` |
| `__init__.py` | `prompt_dirs(domain)`: the prompt search path `[prompts/<domain>/, prompts/]` |
| `runner.py` | `Tau2TaskRunner` (one τ² simulation per task, mapped onto the shared trace format) and `Tau2Benchmark` |
| `agent.py` | `HarnessTau2Agent` (the solver, a τ² `HalfDuplexAgent`) and `TicketUser` (a scripted customer) |
| `explorer.py` | the Explorer and the Surveyor: the domain tools plus `reset_environment`, `read_db` (explorer only) and `submit_task_spec` |
| `spec.py` | the generated-task schema, grounding by replay, conversion to a native τ² `Task`, and the two spec gates below |
| `backend.py` | `Tau2GenerationBackend`: LLM-judge grading, plus a triage call that keeps spec defects out of the too-hard guidelines |

## How the paper uses it

- **Domain:** `retail` only. It uses τ²'s own splits: 74 `train` tasks and 40 `test` tasks
  (`task_split`). Evaluation runs 5 times on `test`.
- **Models:** solver `gpt-5.4-mini` (provider-default reasoning); auxiliary roles `gpt-5.4` at
  high effort; user simulator `gpt-4.1-2025-04-14` at temperature 0, as in the original benchmark.
  **Step cap: 200** (`tau2.max_steps`; see deviation 5).
- **Generation** runs 74 sessions, one per training task, with `reject_multi_item_exchange` and
  `withhold_discoverable_ids` on. All retail tasks share a single database, which methods that
  use training tasks also see.
- Every paper config sets `solver_mode: conversational`: learning phases talk to the LLM
  customer too. Configs are in `configs/tau2/`.

## Setup

Fetch the `benchmarks/tau2-bench` submodule and run `uv sync --extra tau2`.
τ² is installed as an **editable** package (`[tool.uv.sources]`), because it reads `data/` from
its source tree. Nothing in the checkout is modified.

## Prompts (`prompts/`)

The search path is `prompts/retail/`, then `prompts/`: a file under the domain directory
shadows the shared copy, and a domain without a directory of its own falls through to it.

| file | used by |
| --- | --- |
| `solver_protocol.txt` | the solver's protocol block (policy, one message *or* one tool call per turn, ticket-mode wording); it includes `retail/solver_identification.txt`, the retail lookup advice |
| `explorer_env.txt`, `explorer_protocol.txt`, `retail/explorer_task_guidelines.txt` | the Explorer (Prompts 2 and 3) |
| `explorer_refine_protocol.txt` | refining an off-target task (Prompt 4) |
| `coverage_survey_protocol.txt`, `coverage_survey_deliverable_partition.txt` | the Surveyor (Prompt 1) |
| `accumulation_memory_extraction.txt` | **override** of the shared Extractor: the customer-service version, paper Prompt 11 |
| `accumulation_self_reflection.txt`, `accumulation_self_reflection_system.txt` | overrides of the shared self-reflection prompt, used by the reference methods |
| `triage.txt` | `backend.py`'s triage of failed generated tasks (capability vs. spec defect) |

## Deviations from upstream

1. **The agent is ours, inside their loop.** `HarnessTau2Agent` is a `HalfDuplexAgent`, so the
   Orchestrator drives it the way it drives `LLMAgent`. Its completions go through the harness
   `LLMClient` instead of τ²'s litellm call. The tasks, tools, user simulator and scorer are the
   same, but the agent scaffold is different, so scores are comparable *in kind* to the published
   table and are not a reproduction of it. τ²'s `cli.py` and batch runner are not used.
2. **A message that has both text and tool calls keeps the calls.** τ² requires an assistant
   message to be text XOR tool calls. When a model emits both, the connector keeps the tool calls
   and stores the text on the trace (`extra.xor_fixups` counts these cases). Upstream would instead
   fail validation. The Orchestrator runs with `validate_communication=False`.
3. **`ticket` mode swaps the simulated customer for a script.** `TicketUser` delivers the whole
   task in its first message, sends one blanket confirmation, and then stops. This mode is the
   default for non-inference runs when `solver_mode` is unset. No paper number uses it.
4. **Memory injection happens on the agent side.** The Orchestrator owns the message history, so
   the agent keeps its own accumulator of injected heuristics and adds it to every turn's payload.
   `memory.retrieval_mode` supports only `pre_generation`.
5. **`max_steps` defaults to 100, not τ²'s 200.** The paper's configs set 200 explicitly. Results at
   different caps are different conditions and should not be pooled.
6. **Generated tasks are ours.** τ² has no task generator. A spec is grounded by replaying its
   reference actions on a fresh environment: every call must succeed, every argument literal must
   be one the customer can state or one an earlier call returned, and the database must change
   unless the spec carries NL assertions. The spec is then converted to a native τ² `Task`
   (written to `native_tasks.json`). The solver attempts it against the simulated customer, and the
   **LLM judge** grades the attempt against the spec's success conditions.
7. **Multi-item exchanges are rejected (`reject_multi_item_exchange`).** In retail, when
   `modify_pending_order_items` is called with several items, every modified item gets the price
   and options of the *last* pair. Two equally correct solutions that list the items in a different
   order therefore reach different graded states. We leave the environment and the benchmark tasks
   unmodified. Instead, the explorer rejects any spec whose reference solution exchanges two or
   more items in a single call (Appendix E).
8. **Generated customers may not volunteer identifiers the agent could look up
   (`withhold_discoverable_ids`).** A leaked order id deletes the lookup step the task is meant to
   test. `spec.py::_ID_CLASSES` declares which identifier classes are discoverable. For retail that
   is every id class, since `find_user_id_by_name_zip` leads to all of them. The explorer rejects
   any scenario that names an instance of one of those classes. `tests/test_tau2_identifier_gate.py`
   checks, offline, that every identifier class in the database is classified and pins retail's
   gated set.
