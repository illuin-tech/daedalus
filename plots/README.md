# Paper figures

`bash plots/all.sh` regenerates the data figures of the paper into `plots/pdf/`. Run it from the
repo root with the run artifacts under `outputs/` (the Hugging Face dataset [`illuin/daedalus-traces`](https://huggingface.co/datasets/illuin/daedalus-traces), downloaded as `outputs/` with `appworld.zip` unpacked; see the top-level README). Figures
2, 3, 4, 10 and 11 are hand-drawn diagrams and are not produced here.

Inference runs are found by scanning `outputs/inference/` and matching each run's own
`config.yaml` and `run_meta.json` (benchmark, split, model, memory block), not its folder name.
The run folders listed below are the ones the scripts select today.

| Fig. | Script | Input runs | Output PDF (`plots/pdf/…`) |
|---|---|---|---|
| 1 | `transferability.py --pool outputs/daedalus/appworld/consolidated_pool.json` | For each of 7 models, a memory-off baseline and an at-start run on that pool. gpt-5.4-mini: `inference/appworld/main-results/no-memory` → `main-results/daedalus`. Other models: `cross-family/no-memory_<model>` → `cross-family/gpt-bank_on_<model>` | `cross-family/appworld_test_normal_appworld_transferability_success.pdf` |
| 5 | `generation_compare.py <3 generation runs>` | `outputs/daedalus/appworld_ablation-D` (D), `appworld_ablation-E` (E), `appworld` (F) | `generation-dynamics/generation_cost_by_role.pdf` |
| 6 | `num_sessions.py --generation-folder outputs/daedalus/appworld --generation-folder outputs/daedalus/appworld_140-sessions` | `outputs/inference/appworld/main-results/no-memory`, `exploration-budget/{1,5,10,50}-sessions`, `main-results/daedalus`, `exploration-budget/140-sessions`, plus the two generation folders' `sessions/` | `exploration-budget/appworld_test_normal_gpt-5.4-mini_appworld_num_sessions_success_pass5_poolheuristics.pdf` |
| 7 | `retriever_top_k_analysis.py --retriever bm25 --runs 5` | The BM25 sweep `outputs/inference/appworld/memory-use/bm25_k{1,2,3,5,10,20,30}` (5 repeats each), plus `main-results/no-memory` (k=0) and `main-results/daedalus` (reference line). | `memory-use/appworld_test_normal_gpt-5.4-mini_appworld_topk_bm25_{success,pass5}.pdf` |
| 8 | `testset_correlation.py --generation-folder appworld` | Generated side: `outputs/evaluation-proxy/appworld/<model>/` (81 tasks, 3 repeats; the `_own_heuristic` replay of Appendix C is skipped). Real side: the memory-off test_normal baseline of each model on daedalus's solver prompt (5 repeats): `main-results/no-memory`, `cross-family/no-memory_*` and `evaluation-proxy/no-memory_*` under `outputs/inference/appworld/` | `evaluation-proxy/appworld_test_normal_appworld_testset_combined_v2.pdf` |
| 9 | `generation.py outputs/daedalus/appworld` | the banked sessions in `outputs/daedalus/appworld/sessions/` | `generation-dynamics/appworld_{failures_before_banking,refinements}_hist.pdf` |

Shared helpers: `_style.py` (palette, rcParams, save), `_family.py` (marks for Fig. 1, `finish_plain`),
`_runs.py` (reading runs, pass^k), `_agent_cost.py` (per-agent cost of the pre-ledger run in Fig. 5).
