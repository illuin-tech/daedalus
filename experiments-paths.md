# Experiments → configs → outputs

Every table and figure of the paper, with the configs that produce it and the output
folders that hold its artifacts. Outputs are distributed as the Hugging Face dataset
[`illuin/daedalus-traces`](https://huggingface.co/datasets/illuin/daedalus-traces); download it as `outputs/` at the repository root and unpack its
`appworld.zip` (see the README). It holds only the runs listed here. Paths are relative to the root. Values
are the paper's (MSR / pass^5 unless stated).

Folders use the paper's terms: `daedalus/` (DAEDALUS banks), `daedalus-curated/`
(DAEDALUS-curated banks), `baselines/<method>/{memory,inference}/<benchmark>`,
`environment-survey/` (the Surveyor), `evaluation-proxy/` (generated tasks as a test set,
§5.3), and `inference/<benchmark>/<section>/<run>` with one section per paper table or
figure. Each config's `experiment_name` (and `category`) is its run's folder, so a re-run
writes exactly there. Inside a bank folder the bank the paper evaluated is always
`consolidated_pool.json` (the raw, unconsolidated heuristics are `pool.json`).

Configs were regenerated from each run's `config.yaml` snapshot. The τ² configs of the
gpt-5.4-auxiliary reruns and of the DAEDALUS bank were rebuilt from their launch configs, then
checked against the run snapshots (see the notes below).

## Table 1 — AppWorld (test_normal)

MSR / pass^5 over five inference runs. Memory is built once (first output), then evaluated (second).

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| No-memory baseline | 44.3 / 14.9 | [`appworld/inference/baseline.yaml`](configs/appworld/inference/baseline.yaml) | `outputs/inference/appworld/main-results/no-memory` |
| AutoGuide | 44.0 / 17.9 | [`references/autoguide/configs/appworld_accumulation.yaml`](references/autoguide/configs/appworld_accumulation.yaml)<br>[`references/autoguide/configs/appworld_inference.yaml`](references/autoguide/configs/appworld_inference.yaml) | `outputs/baselines/autoguide/memory/appworld`<br>`outputs/baselines/autoguide/inference/appworld` |
| ReasoningBank | 48.0 / 19.6 | [`references/reasoningbank/configs/appworld_accumulation.yaml`](references/reasoningbank/configs/appworld_accumulation.yaml)<br>[`references/reasoningbank/configs/appworld_inference.yaml`](references/reasoningbank/configs/appworld_inference.yaml) | `outputs/baselines/reasoningbank/memory/appworld`<br>`outputs/baselines/reasoningbank/inference/appworld` |
| ERL | 58.6 / 29.2 | [`references/erl/configs/appworld_accumulation.yaml`](references/erl/configs/appworld_accumulation.yaml)<br>[`references/erl/configs/appworld_inference.yaml`](references/erl/configs/appworld_inference.yaml) | `outputs/baselines/erl/memory/appworld`<br>`outputs/baselines/erl/inference/appworld` |
| ExpeL | 59.0 / 33.3 | [`references/expel/configs/appworld_accumulation.yaml`](references/expel/configs/appworld_accumulation.yaml)<br>[`references/expel/configs/appworld_inference.yaml`](references/expel/configs/appworld_inference.yaml) | `outputs/baselines/expel/memory/appworld`<br>`outputs/baselines/expel/inference/appworld` |
| ACE | 60.5 / 41.1 | [`references/ace/configs/appworld_accumulation.yaml`](references/ace/configs/appworld_accumulation.yaml)<br>[`references/ace/configs/appworld_inference.yaml`](references/ace/configs/appworld_inference.yaml) | `outputs/baselines/ace/memory/appworld`<br>`outputs/baselines/ace/inference/appworld` |
| DAEDALUS-curated | 60.8 / 36.3 | [`appworld/accumulation/curated.yaml`](configs/appworld/accumulation/curated.yaml)<br>[`appworld/inference/curated.yaml`](configs/appworld/inference/curated.yaml) | `outputs/daedalus-curated/appworld`<br>`outputs/inference/appworld/main-results/daedalus-curated` |
| PREPING | 56.0 / 25.6 | [`appworld/inference/preping.yaml`](configs/appworld/inference/preping.yaml) | playbook `outputs/baselines/preping/memory/appworld/`<br>`outputs/inference/appworld/main-results/preping` |
| DAEDALUS | 60.2 / 32.1 | [`appworld/generation/daedalus.yaml`](configs/appworld/generation/daedalus.yaml)<br>[`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | `outputs/daedalus/appworld`<br>`outputs/inference/appworld/main-results/daedalus` |

## Table 1 — τ²-bench (retail)

MSR / pass^5 over five inference runs. Memory is built once (first output), then evaluated (second).

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| No-memory baseline | 57.5 / 22.5 | [`tau2/inference/baseline.yaml`](configs/tau2/inference/baseline.yaml) | `outputs/inference/tau2/main-results/no-memory` |
| AutoGuide | 61.5 / 27.5 | [`references/autoguide/configs/tau2_accumulation.yaml`](references/autoguide/configs/tau2_accumulation.yaml)<br>[`references/autoguide/configs/tau2_inference.yaml`](references/autoguide/configs/tau2_inference.yaml) | `outputs/baselines/autoguide/memory/tau2`<br>`outputs/baselines/autoguide/inference/tau2` |
| ReasoningBank | 64.0 / 32.5 | [`references/reasoningbank/configs/tau2_accumulation.yaml`](references/reasoningbank/configs/tau2_accumulation.yaml)<br>[`references/reasoningbank/configs/tau2_inference.yaml`](references/reasoningbank/configs/tau2_inference.yaml) | `outputs/baselines/reasoningbank/memory/tau2`<br>`outputs/baselines/reasoningbank/inference/tau2` |
| ERL | 64.0 / 30.0 | [`references/erl/configs/tau2_accumulation.yaml`](references/erl/configs/tau2_accumulation.yaml)<br>[`references/erl/configs/tau2_inference.yaml`](references/erl/configs/tau2_inference.yaml) | `outputs/baselines/erl/memory/tau2`<br>`outputs/baselines/erl/inference/tau2` |
| ExpeL | 67.0 / 37.5 | [`references/expel/configs/tau2_accumulation.yaml`](references/expel/configs/tau2_accumulation.yaml)<br>[`references/expel/configs/tau2_inference.yaml`](references/expel/configs/tau2_inference.yaml) | `outputs/baselines/expel/memory/tau2`<br>`outputs/baselines/expel/inference/tau2` |
| ACE | 70.5 / 37.5 | [`references/ace/configs/tau2_accumulation.yaml`](references/ace/configs/tau2_accumulation.yaml)<br>[`references/ace/configs/tau2_inference.yaml`](references/ace/configs/tau2_inference.yaml) | `outputs/baselines/ace/memory/tau2`<br>`outputs/baselines/ace/inference/tau2` |
| DAEDALUS-curated | 70.0 / 35.0 | [`tau2/accumulation/curated.yaml`](configs/tau2/accumulation/curated.yaml)<br>[`tau2/inference/curated.yaml`](configs/tau2/inference/curated.yaml) | `outputs/daedalus-curated/tau2`<br>`outputs/inference/tau2/main-results/daedalus-curated` |
| PREPING | 60.0 / 25.0 | [`tau2/inference/preping.yaml`](configs/tau2/inference/preping.yaml) | playbook `outputs/baselines/preping/memory/tau2/`<br>`outputs/inference/tau2/main-results/preping` |
| DAEDALUS | 67.5 / 37.5 | [`tau2/generation/daedalus.yaml`](configs/tau2/generation/daedalus.yaml)<br>[`tau2/inference/daedalus.yaml`](configs/tau2/inference/daedalus.yaml) | `outputs/daedalus/tau2`<br>`outputs/inference/tau2/main-results/daedalus` |

## Table 1 — AutomationBench (Operations)

MSR / pass^5 over five inference runs. Memory is built once (first output), then evaluated (second).

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| No-memory baseline | 31.7 / 12.9 | [`automationbench/inference/baseline.yaml`](configs/automationbench/inference/baseline.yaml) | `outputs/inference/automationbench/main-results/no-memory` |
| AutoGuide | 35.4 / 18.6 | [`references/autoguide/configs/automationbench_accumulation.yaml`](references/autoguide/configs/automationbench_accumulation.yaml)<br>[`references/autoguide/configs/automationbench_inference.yaml`](references/autoguide/configs/automationbench_inference.yaml) | `outputs/baselines/autoguide/memory/automationbench`<br>`outputs/baselines/autoguide/inference/automationbench` |
| ReasoningBank | 19.1 / 2.9 | [`references/reasoningbank/configs/automationbench_accumulation.yaml`](references/reasoningbank/configs/automationbench_accumulation.yaml)<br>[`references/reasoningbank/configs/automationbench_inference.yaml`](references/reasoningbank/configs/automationbench_inference.yaml) | `outputs/baselines/reasoningbank/memory/automationbench`<br>`outputs/baselines/reasoningbank/inference/automationbench` |
| ERL | 33.4 / 15.7 | [`references/erl/configs/automationbench_accumulation.yaml`](references/erl/configs/automationbench_accumulation.yaml)<br>[`references/erl/configs/automationbench_inference.yaml`](references/erl/configs/automationbench_inference.yaml) | `outputs/baselines/erl/memory/automationbench`<br>`outputs/baselines/erl/inference/automationbench` |
| ExpeL | 42.6 / 21.4 | [`references/expel/configs/automationbench_accumulation.yaml`](references/expel/configs/automationbench_accumulation.yaml)<br>[`references/expel/configs/automationbench_inference.yaml`](references/expel/configs/automationbench_inference.yaml) | `outputs/baselines/expel/memory/automationbench`<br>`outputs/baselines/expel/inference/automationbench` |
| ACE | 26.3 / 10.0 | [`references/ace/configs/automationbench_accumulation.yaml`](references/ace/configs/automationbench_accumulation.yaml)<br>[`references/ace/configs/automationbench_inference.yaml`](references/ace/configs/automationbench_inference.yaml) | `outputs/baselines/ace/memory/automationbench`<br>`outputs/baselines/ace/inference/automationbench` (ran with `ace.playbook_preamble: true`, not yet in this code) |
| DAEDALUS-curated | 40.0 / 24.3 | [`automationbench/accumulation/curated.yaml`](configs/automationbench/accumulation/curated.yaml)<br>[`automationbench/inference/curated.yaml`](configs/automationbench/inference/curated.yaml) | `outputs/daedalus-curated/automationbench`<br>`outputs/inference/automationbench/main-results/daedalus-curated` |
| PREPING | 34.9 / 15.7 | [`automationbench/inference/preping.yaml`](configs/automationbench/inference/preping.yaml) | playbook `outputs/baselines/preping/memory/automationbench/`<br>`outputs/inference/automationbench/main-results/preping` |
| DAEDALUS | 36.0 / 21.4 | [`automationbench/generation/daedalus.yaml`](configs/automationbench/generation/daedalus.yaml)<br>[`automationbench/inference/daedalus.yaml`](configs/automationbench/inference/daedalus.yaml) | `outputs/daedalus/automationbench`<br>`outputs/inference/automationbench/main-results/daedalus` |

## Table 2 — cross-family transfer (AppWorld)

MSR gain of each base agent (column) with each bank (row). Banks: GPT-5.4 = the Table 1 bank; the Qwen and DeepSeek banks are generated with `configs/appworld/generation/{qwen,deepseek}.yaml`.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| Qwen bank (generation) | — | [`appworld/generation/qwen.yaml`](configs/appworld/generation/qwen.yaml) | `outputs/daedalus/appworld_qwen` |
| DeepSeek bank (generation) | — | [`appworld/generation/deepseek.yaml`](configs/appworld/generation/deepseek.yaml) | `outputs/daedalus/appworld_deepseek` |
| GPT-5.4-mini, no memory | 44.3 | [`appworld/inference/baseline.yaml`](configs/appworld/inference/baseline.yaml) | `outputs/inference/appworld/main-results/no-memory` |
| GPT-5.4-mini + GPT-5.4 bank | +15.9 | [`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | `outputs/inference/appworld/main-results/daedalus` |
| GPT-5.4-mini + Qwen bank | +16.8 | [`appworld/inference/models/qwenbank_on_gpt54mini.yaml`](configs/appworld/inference/models/qwenbank_on_gpt54mini.yaml) | `outputs/inference/appworld/cross-family/qwen-bank_on_gpt-5.4-mini` |
| GPT-5.4-mini + DeepSeek bank | +11.4 | [`appworld/inference/models/dsbank_on_gpt54mini.yaml`](configs/appworld/inference/models/dsbank_on_gpt54mini.yaml) | `outputs/inference/appworld/cross-family/deepseek-bank_on_gpt-5.4-mini` |
| Qwen3.6-35B-A3B, no memory | 44.8 | [`appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml`](configs/appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml) | `outputs/inference/appworld/cross-family/no-memory_qwen3.6-35b-a3b` |
| Qwen3.6-35B-A3B + GPT-5.4 bank | +16.1 | [`appworld/inference/models/gpt54bank_on_qwen36.yaml`](configs/appworld/inference/models/gpt54bank_on_qwen36.yaml) | `outputs/inference/appworld/cross-family/gpt-bank_on_qwen3.6-35b-a3b` |
| Qwen3.6-35B-A3B + Qwen bank | +29.3 | [`appworld/inference/models/qwenbank_on_qwen36.yaml`](configs/appworld/inference/models/qwenbank_on_qwen36.yaml) | `outputs/inference/appworld/cross-family/qwen-bank_on_qwen3.6-35b-a3b` |
| Qwen3.6-35B-A3B + DeepSeek bank | +14.5 | [`appworld/inference/models/dsbank_on_qwen36.yaml`](configs/appworld/inference/models/dsbank_on_qwen36.yaml) | `outputs/inference/appworld/cross-family/deepseek-bank_on_qwen3.6-35b-a3b` |
| DeepSeek-V4-Flash, no memory | 80.6 | [`appworld/inference/models/baseline_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/baseline_deepseek_v4_flash_0731_t30.yaml) | `outputs/inference/appworld/cross-family/no-memory_deepseek-v4-flash-0731` |
| DeepSeek-V4-Flash + GPT-5.4 bank | +8.3 | [`appworld/inference/models/gpt54bank_on_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/gpt54bank_on_deepseek_v4_flash_0731_t30.yaml) | `outputs/inference/appworld/cross-family/gpt-bank_on_deepseek-v4-flash-0731` |
| DeepSeek-V4-Flash + Qwen bank | +4.9 | [`appworld/inference/models/qwenbank_on_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/qwenbank_on_deepseek_v4_flash_0731_t30.yaml) | `outputs/inference/appworld/cross-family/qwen-bank_on_deepseek-v4-flash-0731` |
| DeepSeek-V4-Flash + DeepSeek bank | +3.0 | [`appworld/inference/models/dsbank_on_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/dsbank_on_deepseek_v4_flash_0731_t30.yaml) | `outputs/inference/appworld/cross-family/deepseek-bank_on_deepseek-v4-flash-0731` |

## Figure 1 — MSR vs. mean turns, seven base agents

`plots/scripts/transferability.py`; each point pair = a base agent without memory and with the GPT-5.4 bank.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash |  | [`appworld/inference/models/baseline_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/baseline_deepseek_v4_flash_0731_t30.yaml)<br>[`appworld/inference/models/gpt54bank_on_deepseek_v4_flash_0731_t30.yaml`](configs/appworld/inference/models/gpt54bank_on_deepseek_v4_flash_0731_t30.yaml) | `outputs/inference/appworld/cross-family/no-memory_deepseek-v4-flash-0731`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_deepseek-v4-flash-0731` |
| GLM-5.3-Flash |  | [`appworld/inference/models/baseline_glm_53_flash.yaml`](configs/appworld/inference/models/baseline_glm_53_flash.yaml)<br>[`appworld/inference/models/gpt54bank_on_glm_53_flash.yaml`](configs/appworld/inference/models/gpt54bank_on_glm_53_flash.yaml) | `outputs/inference/appworld/cross-family/no-memory_glm-5.3-flash`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_glm-5.3-flash` |
| Qwen3.6-35B-A3B |  | [`appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml`](configs/appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml)<br>[`appworld/inference/models/gpt54bank_on_qwen36.yaml`](configs/appworld/inference/models/gpt54bank_on_qwen36.yaml) | `outputs/inference/appworld/cross-family/no-memory_qwen3.6-35b-a3b`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_qwen3.6-35b-a3b` |
| MiniMax-M2.7 |  | [`appworld/inference/models/baseline_minimax_m27.yaml`](configs/appworld/inference/models/baseline_minimax_m27.yaml)<br>[`appworld/inference/models/gpt54bank_on_minimax_m27.yaml`](configs/appworld/inference/models/gpt54bank_on_minimax_m27.yaml) | `outputs/inference/appworld/cross-family/no-memory_minimax-m2.7`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_minimax-m2.7` |
| GPT-5.4-nano |  | [`appworld/inference/models/baseline_gpt_5.4_nano.yaml`](configs/appworld/inference/models/baseline_gpt_5.4_nano.yaml)<br>[`appworld/inference/models/gpt54bank_on_gpt_5.4_nano.yaml`](configs/appworld/inference/models/gpt54bank_on_gpt_5.4_nano.yaml) | `outputs/inference/appworld/cross-family/no-memory_gpt-5.4-nano`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_gpt-5.4-nano` |
| Nemotron-3.5-Lightning |  | [`appworld/inference/models/baseline_nemotron35_lightning.yaml`](configs/appworld/inference/models/baseline_nemotron35_lightning.yaml)<br>[`appworld/inference/models/gpt54bank_on_nemotron35_lightning.yaml`](configs/appworld/inference/models/gpt54bank_on_nemotron35_lightning.yaml) | `outputs/inference/appworld/cross-family/no-memory_nemotron-3.5-lightning`<br>`outputs/inference/appworld/cross-family/gpt-bank_on_nemotron-3.5-lightning` |
| GPT-5.4-mini |  | [`appworld/inference/baseline.yaml`](configs/appworld/inference/baseline.yaml)<br>[`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | same as Table 1 |

## Table 3 — cumulative pipeline ablation (AppWorld)

MSR / pass^5 / generation $. Each row adds a component to the previous one.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| (A) Single explorer, 100 turns | 35.7 / 10.1 / 5.2 | `daedalus.scripts.naive_explorer --max-turns 100 --exclude-apps gmail,amazon --name appworld_ablation-A`<br>[`appworld/inference/ablation_single_explorer.yaml`](configs/appworld/inference/ablation_single_explorer.yaml) | `outputs/daedalus/appworld_ablation-A`<br>`outputs/inference/appworld/pipeline-ablation/A_single-explorer` |
| (B) Multi-session explorer | 38.7 / 11.3 / 92.0 | [`appworld/generation/ablation_explorer_with_bank.yaml`](configs/appworld/generation/ablation_explorer_with_bank.yaml)<br>[`appworld/inference/ablation_explorer_with_bank.yaml`](configs/appworld/inference/ablation_explorer_with_bank.yaml) | `outputs/daedalus/appworld_ablation-B`<br>`outputs/inference/appworld/pipeline-ablation/B_multi-session-explorer` |
| (C) Explorer tasks and solver traces | 54.5 / 25.0 / 46.6 | [`appworld/generation/ablation_solver_traces.yaml`](configs/appworld/generation/ablation_solver_traces.yaml)<br>[`appworld/inference/ablation_solver_traces.yaml`](configs/appworld/inference/ablation_solver_traces.yaml) | `outputs/daedalus/appworld_ablation-C`<br>`outputs/inference/appworld/pipeline-ablation/C_solver-traces` |
| (D) + Solver loop | 59.0 / 30.4 / 210.4 | [`appworld/generation/ablation_no_survey_no_guidelines.yaml`](configs/appworld/generation/ablation_no_survey_no_guidelines.yaml)<br>[`appworld/inference/ablation_no_survey_no_guidelines.yaml`](configs/appworld/inference/ablation_no_survey_no_guidelines.yaml) | `outputs/daedalus/appworld_ablation-D`<br>`outputs/inference/appworld/pipeline-ablation/D_solver-loop` |
| (E) + Explorer guideline bank | 57.9 / 31.5 / 185.3 | [`appworld/generation/ablation_no_survey.yaml`](configs/appworld/generation/ablation_no_survey.yaml)<br>[`appworld/inference/ablation_no_survey.yaml`](configs/appworld/inference/ablation_no_survey.yaml) | `outputs/daedalus/appworld_ablation-E`<br>`outputs/inference/appworld/pipeline-ablation/E_guideline-bank` |
| (F) + Environment survey (DAEDALUS) | 60.2 / 32.1 / 109.7 | [`appworld/generation/daedalus.yaml`](configs/appworld/generation/daedalus.yaml)<br>[`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | `outputs/daedalus/appworld`<br>`outputs/inference/appworld/main-results/daedalus` |
| Figure 5 — generation cost by role |  | `plots/scripts/generation_compare.py` | the (D), (E), (F) generation folders |

## Table 4 — LLM judge vs. official verifiers

Precision / recall / κbench / κinter. The judge re-reads run 0 of each benchmark's no-memory baseline five times, against success conditions the Explorer wrote by inspecting the environment and restating the benchmark's oracle requirements (`judge_conditions/`). `judge_summary.json` holds the metrics (κinter is `repeat_analysis.pairwise_kappa`), `judge_records.json` every verdict.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| AppWorld (N = 168) | .943 / .846 / .807 / .848 | — | `outputs/inference/appworld/judge-validation/gpt-5.4-judge`<br>judged: `outputs/inference/appworld/main-results/no-memory` (run 0) |
| τ²-bench (N = 40) | .889 / .842 / .749 / 1.000 | — | `outputs/inference/tau2/judge-validation/gpt-5.4-judge`<br>judged: `outputs/inference/tau2/judge-validation/no-memory_judged` (run 0; an earlier no-memory run, MSR 43.0, not the Table 1 baseline) |
| AutomationBench (N = 70) | .863 / .807 / .729 / .902 | — | `outputs/inference/automationbench/judge-validation/gpt-5.6-terra-judge`<br>judged: `outputs/inference/automationbench/main-results/no-memory` (run 0) |

## Figure 6 and Table 7 — scaling with the number of sessions (AppWorld)

`plots/scripts/num_sessions.py`. Checkpoint banks are `consolidated_pool_first<N>.json` in `outputs/daedalus/appworld`, each the consolidation of `pool_first<N>.json`. N counts the first N banked heuristics, one per successful session: 81 of the 90 sessions banked one, so N heuristics cost about N sessions (the 50th arrived at session 52), and the figure places each checkpoint at x = N.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| 1 session | 43.3 | [`appworld/inference/scaling_first1.yaml`](configs/appworld/inference/scaling_first1.yaml) | `outputs/inference/appworld/exploration-budget/1-sessions` |
| 5 sessions | 52.7 | [`appworld/inference/scaling_first5.yaml`](configs/appworld/inference/scaling_first5.yaml) | `outputs/inference/appworld/exploration-budget/5-sessions` |
| 10 sessions | 52.6 | [`appworld/inference/scaling_first10.yaml`](configs/appworld/inference/scaling_first10.yaml) | `outputs/inference/appworld/exploration-budget/10-sessions` |
| 50 sessions | 54.6 | [`appworld/inference/scaling_first50.yaml`](configs/appworld/inference/scaling_first50.yaml) | `outputs/inference/appworld/exploration-budget/50-sessions` |
| 90 sessions | 60.2 | [`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | same as Table 1 |
| 140 sessions, single-call consolidation | 53.7 | [`appworld/generation/scaling_140.yaml`](configs/appworld/generation/scaling_140.yaml)<br>[`appworld/inference/scaling_140.yaml`](configs/appworld/inference/scaling_140.yaml) | `outputs/daedalus/appworld_140-sessions`<br>`outputs/inference/appworld/exploration-budget/140-sessions` |
| 140 sessions, hierarchical consolidation | 54.6 | [`appworld/inference/scaling_140_hierarchical.yaml`](configs/appworld/inference/scaling_140_hierarchical.yaml)<br>`consolidate --strategy hierarchical` | `outputs/inference/appworld/exploration-budget/140-sessions_hierarchical` |

## Table 5 — auxiliary agents on GPT-5.4-mini (AppWorld)

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| GPT-5.4-mini auxiliary agents | 51.2 / 20.2 / 67.2 | [`appworld/generation/aux_gpt54mini.yaml`](configs/appworld/generation/aux_gpt54mini.yaml)<br>[`appworld/inference/aux_gpt54mini.yaml`](configs/appworld/inference/aux_gpt54mini.yaml) | `outputs/daedalus/appworld_aux-gpt-5.4-mini`<br>`outputs/inference/appworld/aux-model/aux-gpt-5.4-mini` |

## Table 6 — consolidation and injection (AppWorld)

MSR / pass^5.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| Top-5 per turn, random | 51.7 / 19.0 | [`appworld/inference/retrieval_random_k5.yaml`](configs/appworld/inference/retrieval_random_k5.yaml) | `outputs/inference/appworld/memory-use/random_k5` |
| Top-5 per turn, BM25 | 52.0 / 22.0 | [`appworld/inference/bm25_k5.yaml`](configs/appworld/inference/bm25_k5.yaml) | `outputs/inference/appworld/memory-use/bm25_k5` |
| Top-5 per turn, Qwen3-Embedding-4B | 54.3 / 24.4 | [`appworld/inference/retrieval_dense_k5.yaml`](configs/appworld/inference/retrieval_dense_k5.yaml) | `outputs/inference/appworld/memory-use/qwen3-emb_k5` |
| Whole bank per turn | 51.5 / 28.0 | [`appworld/inference/all_at_turn.yaml`](configs/appworld/inference/all_at_turn.yaml) | `outputs/inference/appworld/memory-use/all-at-turn` |
| Whole bank at start, raw | 49.5 / 26.2 | [`appworld/inference/bank_raw.yaml`](configs/appworld/inference/bank_raw.yaml) | `outputs/inference/appworld/memory-use/all-at-start_raw` |
| Whole bank at start, dedup only | 49.8 / 27.4 | [`appworld/inference/bank_dedup.yaml`](configs/appworld/inference/bank_dedup.yaml)<br>`consolidate --strategy dedup` | `outputs/inference/appworld/memory-use/all-at-start_dedup` |
| Whole bank at start, consolidated (DAEDALUS) | 60.2 / 32.1 | [`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | same as Table 1 |

## Figure 7 — BM25 top-k per turn (AppWorld)

`plots/scripts/retriever_top_k_analysis.py`.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| k = 1 | 46.8 | [`appworld/inference/bm25_k1.yaml`](configs/appworld/inference/bm25_k1.yaml) | `outputs/inference/appworld/memory-use/bm25_k1` |
| k = 2 | 48.7 | [`appworld/inference/bm25_k2.yaml`](configs/appworld/inference/bm25_k2.yaml) | `outputs/inference/appworld/memory-use/bm25_k2` |
| k = 3 | 50.4 | [`appworld/inference/bm25_k3.yaml`](configs/appworld/inference/bm25_k3.yaml) | `outputs/inference/appworld/memory-use/bm25_k3` |
| k = 5 | 52.0 | [`appworld/inference/bm25_k5.yaml`](configs/appworld/inference/bm25_k5.yaml) | `outputs/inference/appworld/memory-use/bm25_k5` |
| k = 10 | 53.1 | [`appworld/inference/bm25_k10.yaml`](configs/appworld/inference/bm25_k10.yaml) | `outputs/inference/appworld/memory-use/bm25_k10` |
| k = 20 | 53.0 | [`appworld/inference/bm25_k20.yaml`](configs/appworld/inference/bm25_k20.yaml) | `outputs/inference/appworld/memory-use/bm25_k20` |
| k = 30 | 58.2 | [`appworld/inference/bm25_k30.yaml`](configs/appworld/inference/bm25_k30.yaml) | `outputs/inference/appworld/memory-use/bm25_k30` |

## Table 9 — retrieval query (AppWorld, BM25, k = 5)

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| PRE-GEN | 52.0 / 22.0 | [`appworld/inference/bm25_k5.yaml`](configs/appworld/inference/bm25_k5.yaml) | same as Table 6 |
| R2R | 44.2 / 11.9 | [`appworld/inference/retrieval_bm25_r2r_k5.yaml`](configs/appworld/inference/retrieval_bm25_r2r_k5.yaml) | `outputs/inference/appworld/memory-use/bm25_k5_r2r` |

## Table 10 — injection modes (AppWorld, Qwen3-Embedding-4B, k = 5)

The Distinct column is `daedalus.scripts.retrieval_diversity`.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| Transient | 48.5 / 13.1 | [`appworld/inference/injection_transient.yaml`](configs/appworld/inference/injection_transient.yaml) | `outputs/inference/appworld/memory-use/qwen3-emb_k5_transient` |
| Cumulative, with repetition | 50.0 / 17.3 | [`appworld/inference/injection_with_repetition.yaml`](configs/appworld/inference/injection_with_repetition.yaml) | `outputs/inference/appworld/memory-use/qwen3-emb_k5_with-repetition` |
| Cumulative, deduplicated | 54.3 / 24.4 | [`appworld/inference/retrieval_dense_k5.yaml`](configs/appworld/inference/retrieval_dense_k5.yaml) | same as Table 6 |
| Cumulative, no replacement | 56.0 / 25.6 | [`appworld/inference/injection_no_replacement.yaml`](configs/appworld/inference/injection_no_replacement.yaml) | `outputs/inference/appworld/memory-use/qwen3-emb_k5_no-replacement` |
| ALL@TURN / ALL@START | 51.5 / 60.2 | [`appworld/inference/all_at_turn.yaml`](configs/appworld/inference/all_at_turn.yaml)<br>[`appworld/inference/daedalus.yaml`](configs/appworld/inference/daedalus.yaml) | same as Table 6 |

## Figure 8 — generated tasks as a test set (AppWorld)

`plots/scripts/testset_correlation.py`. Generated side: `daedalus.scripts.tasks_as_test_set --generation outputs/daedalus/appworld --model <model> --num-runs 3` (the 81 accepted tasks). Real side: the no-memory test_normal runs. The `gpt-5.4-mini_own_heuristic` folder next to the models is Appendix C's counterfactual replay, not a Figure 8 point.

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| gpt-5.4-mini |  | [`appworld/inference/baseline.yaml`](configs/appworld/inference/baseline.yaml) | `outputs/evaluation-proxy/appworld/gpt-5.4-mini`<br>`outputs/inference/appworld/main-results/no-memory` |
| qwen3.6-35b-a3b |  | [`appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml`](configs/appworld/inference/models/baseline_qwen36_35b_a3b_parasail.yaml) | `outputs/evaluation-proxy/appworld/qwen3.6-35b-a3b`<br>`outputs/inference/appworld/cross-family/no-memory_qwen3.6-35b-a3b` |
| deepseek-v4-flash |  | [`appworld/inference/models/baseline_deepseek_v4_flash.yaml`](configs/appworld/inference/models/baseline_deepseek_v4_flash.yaml) | `outputs/evaluation-proxy/appworld/deepseek-v4-flash`<br>`outputs/inference/appworld/evaluation-proxy/no-memory_deepseek-v4-flash` |
| nemotron-3.5-lightning |  | [`appworld/inference/models/baseline_nemotron35_lightning.yaml`](configs/appworld/inference/models/baseline_nemotron35_lightning.yaml) | `outputs/evaluation-proxy/appworld/nemotron-3.5-lightning`<br>`outputs/inference/appworld/cross-family/no-memory_nemotron-3.5-lightning` |
| minimax-m2.7 |  | [`appworld/inference/models/baseline_minimax_m27.yaml`](configs/appworld/inference/models/baseline_minimax_m27.yaml) | `outputs/evaluation-proxy/appworld/minimax-m2.7`<br>`outputs/inference/appworld/cross-family/no-memory_minimax-m2.7` |
| gpt-oss-120b |  | [`appworld/inference/models/baseline_gpt_oss_120b.yaml`](configs/appworld/inference/models/baseline_gpt_oss_120b.yaml) | `outputs/evaluation-proxy/appworld/gpt-oss-120b`<br>`outputs/inference/appworld/evaluation-proxy/no-memory_gpt-oss-120b` |
| gemma-4-31b-it |  | [`appworld/inference/models/baseline_gemma4_31b.yaml`](configs/appworld/inference/models/baseline_gemma4_31b.yaml) | `outputs/evaluation-proxy/appworld/gemma-4-31b-it`<br>`outputs/inference/appworld/evaluation-proxy/no-memory_gemma-4-31b-it` |
| mimo-v2.5 |  | [`appworld/inference/models/baseline_mimo_v25.yaml`](configs/appworld/inference/models/baseline_mimo_v25.yaml) | `outputs/evaluation-proxy/appworld/mimo-v2.5`<br>`outputs/inference/appworld/evaluation-proxy/no-memory_mimo-v2.5` |
| gpt-5.6-luna |  | [`appworld/inference/models/baseline_gpt56_luna.yaml`](configs/appworld/inference/models/baseline_gpt56_luna.yaml) | `outputs/evaluation-proxy/appworld/gpt-5.6-luna`<br>`outputs/inference/appworld/evaluation-proxy/no-memory_gpt-5.6-luna` |

## Appendix C

| item | paper | configs | outputs |
| --- | --- | --- | --- |
| Figure 9 — failures and refinements per accepted heuristic |  | `plots/scripts/generation.py` | `outputs/daedalus/appworld` |
| Table 8 — extractor GPT-5.4-mini, no reasoning | 53.8 / 28.6 | [`appworld/accumulation/curated_extractor_mini_no_reasoning.yaml`](configs/appworld/accumulation/curated_extractor_mini_no_reasoning.yaml)<br>[`appworld/inference/curated_extractor_mini_no_reasoning.yaml`](configs/appworld/inference/curated_extractor_mini_no_reasoning.yaml) | `outputs/daedalus-curated/appworld_extractor-gpt-5.4-mini-no-reasoning`<br>`outputs/inference/appworld/extractor-choice/extractor-gpt-5.4-mini-no-reasoning` |
| Table 8 — extractor GPT-5.4-mini, high | 57.1 / 28.6 | [`appworld/accumulation/curated_extractor_mini.yaml`](configs/appworld/accumulation/curated_extractor_mini.yaml)<br>[`appworld/inference/curated_extractor_mini.yaml`](configs/appworld/inference/curated_extractor_mini.yaml) | `outputs/daedalus-curated/appworld_extractor-gpt-5.4-mini`<br>`outputs/inference/appworld/extractor-choice/extractor-gpt-5.4-mini` |
| Table 8 — extractor GPT-5.4, high | 60.8 / 36.3 | [`appworld/accumulation/curated.yaml`](configs/appworld/accumulation/curated.yaml)<br>[`appworld/inference/curated.yaml`](configs/appworld/inference/curated.yaml) | same as Table 1 |
| Counterfactual replay, without heuristic | 66.7 / 40.7 (pass^3) | `tasks_as_test_set ... --model gpt-5.4-mini --num-runs 3` | `outputs/evaluation-proxy/appworld/gpt-5.4-mini` |
| Counterfactual replay, with heuristic | 77.0 / 55.6 (pass^3) | `tasks_as_test_set ... --model gpt-5.4-mini --num-runs 3 --own-heuristic` | `outputs/evaluation-proxy/appworld/gpt-5.4-mini_own_heuristic` |
| Table 14 — solver rollouts |  | — | count the traces of the Table 1 memory-construction folders |

## Notes on reproducing

- **Near-duplicate filter.** The 90-session AppWorld run dropped seven accepted tasks whose
  tool paths near-duplicated banked ones (`generation.min_novelty: 0.05`). The filter is not
  part of the method and is off by default (`min_novelty: -1`); the paper's bank is the 81
  heuristics of that run.
- **Generation-time extractor effort.** The generation pipeline sends no reasoning effort to
  the extractor for the DAEDALUS stage (provider default), as in the paper's runs; the
  `accumulation.extraction_reasoning_effort` of a generation config is read by the Table 3
  row C stage and by accumulation on labeled tasks.
- **τ² baselines with gpt-5.4 auxiliaries.** AutoGuide, ExpeL, ReasoningBank and PREPING were
  rerun on τ² with gpt-5.4 in every auxiliary role, as Section 4 states; their configs were
  rebuilt from the launch configs and match the run snapshots (`config.yaml` and the
  method's own `<method>.yaml`) on every setting the runs read. The snapshots differ only in
  blocks these runs never read (`generation`, `automationbench`, and
  `accumulation.extraction_model`, which AutoGuide and ExpeL replace with their own
  `extraction_model`), set to the defaults of the time. The earlier runs, in which some
  auxiliary roles used the solver model, are not the paper's and are not shipped.
- **Concurrency.** The 90-session AppWorld run used 3 concurrent explorers (`num_explorers`);
  row B of Table 3 used one.
- **τ² bank.** `outputs/daedalus/tau2` ran as 50 sessions and a resume of 24; its
  `consolidated_pool.json` was reconsolidated over all 74 sessions (the consolidation of the
  first 50 is not shipped).
- **AutomationBench bank.** The heuristics in `outputs/daedalus/automationbench/pool.json` (and
  its `consolidated_pool.json`, which the inference run reads) were re-extracted from the run's
  traces with the benchmark's own extraction prompt
  (`daedalus/benchmarks/automationbench/prompts/accumulation_memory_extraction.txt`), which a
  fresh run uses directly. The `sessions/` files keep the heuristics extracted during the run.
- **Figure 8.** The gpt-5.4-mini point now uses the standard no-memory baseline (44.3), like
  every other model; with it, Kendall's τ is 0.89 (34/36 pairs) for both MSR and pass^3.
