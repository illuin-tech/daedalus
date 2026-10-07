#!/usr/bin/env bash
# Regenerates the paper's data figures into plots/pdf/. Run from the repo root, with the run
# artifacts under outputs/ (the illuin/daedalus-traces dataset, see README.md). See plots/README.md for the
# runs each figure reads.
set -euo pipefail

# The 90-session AppWorld generation run every pool-based figure hangs off, and its
# continuation to 140 sessions (sessions 0-89 are identical in both folders).
GEN=outputs/daedalus/appworld
GEN_SCALING=outputs/daedalus/appworld_140-sessions
POOL=$GEN/consolidated_pool.json
# The test-set tree is keyed by its own folder NAME under outputs/evaluation-proxy/.
TESTSET=appworld

# Figure 1: MSR vs mean turns, memory-off baseline -> whole pool at start, 7 models.
uv run python plots/scripts/transferability.py --benchmark appworld --split test_normal \
    --pool $POOL

# Figure 5: generation cost by agent, ablations (D) / (E) / (F).
uv run python plots/scripts/generation_compare.py \
    $GEN \
    outputs/daedalus/appworld_ablation-E \
    outputs/daedalus/appworld_ablation-D

# Figure 6: MSR and pass^5 against the sessions the pool cost, pool size in heuristics.
uv run python plots/scripts/num_sessions.py --benchmark appworld --split test_normal \
    --model gpt-5.4-mini \
    --generation-folder $GEN \
    --generation-folder $GEN_SCALING

# Figure 7: MSR and pass^5 against BM25 top-k, whole bank at start as a reference line
# (the sweep of configs/appworld/inference/bm25_k*.yaml).
uv run python plots/scripts/retriever_top_k_analysis.py --benchmark appworld --split test_normal \
    --model gpt-5.4-mini --retriever bm25 --pool $POOL --runs 5

# Figure 8: MSR and pass^3 on the generated tasks against the AppWorld test split, 9 models.
uv run python plots/scripts/testset_correlation.py --generation-folder $TESTSET --split test_normal

# Figure 9: failed solver attempts before banking, and refinements per banked session.
uv run python plots/scripts/generation.py $GEN
