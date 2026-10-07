# PREPING playbooks

PREPING (Choi et al., 2026) was run with the authors' released code, not re-implemented here.
Its playbooks ship with the paper's outputs, in `outputs/baselines/preping/memory/<benchmark>/`,
already converted to daedalus memory pools, so the PREPING rows are evaluated on exactly the
same inference path as every other pool: the whole playbook injected at the start of each test
task. This folder holds the converter.

| file (under `outputs/baselines/preping/memory/`) | what it is |
| --- | --- |
| `appworld/playbook.json` | the raw AppWorld playbook, as PREPING writes it (`strategies`, `code_snippets`, `pitfalls`, `apis`) |
| `appworld/pool.json` | `appworld/playbook.json` converted to a daedalus pool (Table 1) |
| `tau2/pool.json` | the τ²-bench retail playbook built with gpt-5.4 auxiliaries, converted (Table 1) |
| `automationbench/pool.json` | the AutomationBench Operations playbook, converted (Table 1) |

The inference runs that read them are `outputs/inference/<benchmark>/main-results/preping/`.

## Converting a playbook

```bash
uv run python references/preping/playbook_to_pool.py \
    --playbook outputs/baselines/preping/memory/appworld/playbook.json \
    --out outputs/baselines/preping/memory/appworld/pool.json
```

With no arguments it performs exactly that conversion (the output is byte-identical to the
shipped file). Each bullet becomes one `MemoryItem`: its `content` is the item text, its
source task becomes `source_task_id` (`preping:<n>`), and PREPING's own fields (id, section,
Curator tag, version, usage counters) are kept in `tags`. Bullets keep PREPING's section
order, exact duplicates are merged, and the result is reloaded as a `MemoryPool` before the
script exits. `--notes` prints the two mapping choices (every item has
`source_trajectory_success: true`, and ids are daedalus content hashes unless `--keep-ids`).
Optional filters (`--sections`, `--drop-harmful`, `--exclude-apps` / `--drop-excluded`) are
off by default; the AppWorld file is the unfiltered conversion.

To evaluate a converted playbook, point an inference config's `memory.pool_path` at it and
inject it whole at task start (e.g. `configs/appworld/inference/preping.yaml`):

```yaml
memory:
  enabled: true
  pool_path: outputs/baselines/preping/memory/appworld/pool.json
  heuristics_at_start: true
```
