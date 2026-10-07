# Prompts

`daedalus/core/prompts/` holds the prompt text shared by every benchmark, and
`daedalus/benchmarks/<name>/prompts/` holds what is specific to each one. Prompts are Jinja
templates. There are two kinds.

## 1. Composed agent prompts: general + env + protocol

The solver, the Explorer, the refinement step and the Surveyor all have prompts built by
`daedalus/core/resources.py::render_prompt(kind, prompt_dirs, **vars)`, from these blocks:

| block | source | slot in the general template |
| --- | --- | --- |
| general | `core/prompts/<sub>/<file>` (table below); always present | — |
| env | the benchmark's `<kind>_env.txt`; left empty when the benchmark has none | `{{ env_knowledge }}` |
| protocol | the benchmark's `<kind>_protocol.txt` (or the file named by `protocol=`); required | `{{ action_protocol }}` |
| task guidelines | the benchmark's `<kind>_task_guidelines.txt` (the refiner reuses the explorer's) | `{{ task_guidelines }}` |
| deliverable partition | the benchmark's `<kind>_deliverable_partition.txt`, if any | `{{ deliverable_partition }}` |

| kind | general template |
| --- | --- |
| `solver` | `agent/solver_system.txt` (also renders the `heuristics` list, or a pre-formatted `playbook` block) |
| `explorer` | `generation/explorer_system.txt` |
| `explorer_refine` | `generation/explorer_refine.txt` |
| `coverage_survey` | `generation/coverage_survey_system.txt` |
| `naive_explorer` | `generation/naive_explorer_system.txt` (AppWorld exploration-only ablations) |

The benchmark blocks are rendered first with the caller's variables (app catalogue, domain
policy, ...), and the general template is rendered around them. Benchmark files are resolved
along a **search path**: a single directory, or several listed most-specific first. τ² passes
`[benchmarks/tau2/prompts/<domain>, benchmarks/tau2/prompts]` (`Benchmark.prompt_dirs`), so a
file in `prompts/retail/` shadows the shared τ² copy. The Jinja loader searches the benchmark
directories and then `core/prompts/`. This is how every benchmark's
`explorer_task_guidelines.txt` can `{% from "generation/task_doctrine.txt" import ... %}` the
shared doctrine, and how τ²'s `solver_protocol.txt` can include `solver_identification.txt`.
`render_prompt` still accepts `add_env_knowledge=False`, which would substitute a core
`<kind>_general_env.txt`. No such file ships, and every caller leaves the flag on.

## 2. Standalone prompts and benchmark overrides

All other prompts are loaded directly by the code that uses them. The accumulation prompts can
be overridden per benchmark by naming convention
(`daedalus/core/memory/extraction.py::_benchmark_override`): `core/prompts/<subdir>/<name>` is
replaced by `benchmarks/<bench>/prompts/<subdir>_<name>` whenever that file exists. A
`<subdir>_<stem>_system.txt` file replaces the inline system message in the same way. The
overrides that exist today:

- `tau2/prompts/accumulation_memory_extraction.txt`
- `tau2/prompts/accumulation_self_reflection.txt`, `tau2/prompts/accumulation_self_reflection_system.txt`
- `automationbench/prompts/accumulation_memory_extraction.txt`

AppWorld has no override and uses the shared files. Check this list before you conclude that a
prompt is shared.

## Files

| file | purpose |
| --- | --- |
| `agent/solver_system.txt` | general solver template |
| `agent/reason_first.txt`, `agent/generate_with_memory.txt` | AppWorld `reason_then_retrieve` turns: reason first, then write code with the retrieved memory |
| `generation/coverage_survey_system.txt` | general Surveyor template |
| `generation/explorer_system.txt` | general Explorer template |
| `generation/task_doctrine.txt` | Jinja macros for the shared task-design doctrine, imported by each benchmark's `explorer_task_guidelines.txt` |
| `generation/explorer_refine.txt` | general template for refining an off-target task |
| `generation/explorer_memory.txt` | explorer guideline update (`core/generation/guidelines.py`) |
| `generation/judge.txt` | LLM judge (`core/generation/judge.py`; the system message is inline there) |
| `generation/judge_conditions.txt` | the same judge with one verdict per condition, used to score generated tasks as a test set (`core/testset/judge.py`) |
| `generation/naive_explorer_system.txt` | general template for the exploration-only ablations |
| `accumulation/memory_extraction.txt` | the Extractor (`generate_memory_item`; the system message is inline in `extraction.py`) |
| `accumulation/self_reflection.txt` | post-attempt reflection, used by the reference methods (`references/common/experience.py`) |
| `consolidate.txt` | the Consolidator (`daedalus.scripts.consolidate`, default `single` strategy) |
| `consolidate_dedup_single.txt` | the `dedup` consolidation strategy: duplicate removal only |

Benchmark directories (see each benchmark's README):

- `appworld/prompts/`: `solver_{env,protocol}`, `explorer_{env,protocol,task_guidelines}`,
  `explorer_refine_protocol`, `coverage_survey_{protocol,deliverable_partition}`,
  `naive_explorer_{env,protocol}`
- `tau2/prompts/`: `solver_protocol`, `explorer_{env,protocol}`, `explorer_refine_protocol`,
  `coverage_survey_{protocol,deliverable_partition}`, the accumulation overrides above, `triage`;
  `retail/`: `explorer_task_guidelines`, `solver_identification`
- `automationbench/prompts/`: `solver_{env,protocol}`, `explorer_{env,protocol,task_guidelines}`,
  `explorer_refine_protocol`, `coverage_survey_protocol`, `accumulation_memory_extraction`

## Paper prompts → files

| paper | file(s) |
| --- | --- |
| Prompt 1, Surveyor | `generation/coverage_survey_system.txt` + `<bench>/prompts/coverage_survey_protocol.txt` (+ `coverage_survey_deliverable_partition.txt`) |
| Prompt 2, Explorer | `generation/explorer_system.txt` + `<bench>/prompts/explorer_{env,protocol,task_guidelines}.txt` |
| Prompt 3, task-design doctrine | `generation/task_doctrine.txt`, via each `explorer_task_guidelines.txt` |
| Prompt 4, refinement | `generation/explorer_refine.txt` + `<bench>/prompts/explorer_refine_protocol.txt` |
| Prompt 5, guideline update | `generation/explorer_memory.txt` |
| Prompt 6, LLM judge | `generation/judge.txt` (system: `core/generation/judge.py`) |
| Prompt 7, Extractor | `accumulation/memory_extraction.txt` (system: `core/memory/extraction.py`) |
| Prompt 8, Consolidation | `consolidate.txt` |
| Prompt 9, Solver | `agent/solver_system.txt` |
| Prompt 10, AppWorld solver | `benchmarks/appworld/prompts/solver_env.txt` + `solver_protocol.txt` |
| Prompt 11, τ²-bench Extractor | `benchmarks/tau2/prompts/accumulation_memory_extraction.txt` |
| Prompt 12, AutomationBench Extractor | `benchmarks/automationbench/prompts/accumulation_memory_extraction.txt` |
