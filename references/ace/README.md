# ACE — evolving a playbook from an unlabelled training split

Re-implementation of ACE (Zhang et al., 2026) on daedalus's harness, following the agentic
half of the authors' release (the `ace-appworld` code). We use the **offline variant without
ground truth**: each training task is solved once, sequentially, with the playbook as it
stands (the Generator is the benchmark's own solver); a **Reflector** diagnoses the finished
trajectory and a **Curator** turns that diagnosis into `ADD` operations on the playbook. There
is no success signal anywhere in the loop (the environment's verdict is recorded, never read).
The playbook starts empty (8 sections) and the Reflector and Curator run on `gpt-5.4` at high
effort. At test time the **whole playbook** is injected into the system prompt at task start
and the task is solved in a single attempt.

## Run

```bash
# 1. grow a playbook over the training split
uv run python -m references.ace.accumulation --config references/ace/configs/appworld_accumulation.yaml
# 2. evaluate with that playbook on the test split (5 runs)
uv run python -m references.ace.run --config references/ace/configs/appworld_inference.yaml
```

Configs ship for AppWorld (`appworld_*`), τ²-bench retail (`tau2_accumulation`,
`tau2_inference`) and AutomationBench Operations (`automationbench_*`). `run` takes a
`--playbook` override on top of the common flags.

```
outputs/baselines/ace/memory/<name>/
    playbook.txt              the artifact inference injects
    pool.json                 one MemoryItem per bullet (serve and the plots read this)
    pool.log.json, task_summaries/, traces/, curator_operations.jsonl
    playbook_snapshots/       the playbook every `snapshot_every` tasks
outputs/baselines/ace/inference/<name>/
    run_<i>/, evaluation.json, retrieval/   (retrieval/ = size of what was injected)
```

```yaml
ace:
  reflector_model: "gpt-5.4"
  reflector_reasoning_effort: "high"
  curator_model: "gpt-5.4"
  curator_reasoning_effort: "high"
  snapshot_every: 30          # playbook snapshots, as upstream
  playbook_path: null         # accumulation pins it to its own playbook.txt; inference
                              # points it at a finished one
```

**`run.parallel` in an accumulation config is the wave size.** `1` is upstream's sequential
loop and is what every shipped config uses; a wider wave switches on upstream's ComBEE
reducer (`prompts/curator_reduce.txt`), which is a different algorithm and was not used in
the paper.

## Paper / upstream → code

| upstream | here |
| --- | --- |
| Generator | the benchmark's solver, with the playbook in its system prompt (`agents/`) |
| Reflector, Curator (`reflector_call`, `curator_call`) | `curation.py` + `prompts/reflector.txt`, `prompts/curator.txt` |
| playbook format, `ADD` operation | `playbook.py` |
| offline adaptation loop | `accumulation.py` |
| evaluation (inject whole playbook, solve) | `run.py` |

## Deviations from upstream

- **The solver prompt is daedalus's**, not ACE's AppWorld prompt (which adds a second worked
  demonstration); the playbook goes into its memory slot, so the ACE row measures the
  playbook, not a prompt change.
- **The seed playbook is empty** rather than ACE's 8 AppWorld bullets that restate the
  benchmark's own instructions.
- **The Reflector reads the trajectory untruncated**, as upstream does; daedalus's shared
  AppWorld renderer would cut tool output at 500 characters.
- **Workers propose, the parent applies**: the playbook has a single writer and is rebuilt
  deterministically from the per-task checkpoints; for one-task waves the applied sequence
  and ids match upstream's.
- **Reflector and Curator run on `gpt-5.4`** rather than ACE's single model for all three
  roles, matching the writer model of the other methods.
- The eight AppWorld-flavoured section names are kept on every benchmark; only the clauses
  ACE itself writes per domain are rewritten (`domains.py`).
- Upstream behaviour kept as is: `ADD` is the only operation (UPDATE/MERGE/DELETE are
  unimplemented upstream), so the playbook is append-only, and an `ADD` to
  `problem_solving_heuristics_and_workflows` is routed to `OTHERS` (pinned by a test).
