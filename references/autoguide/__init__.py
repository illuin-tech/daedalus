"""AutoGuide — Automated Generation and Selection of Context-Aware Guidelines (Fu et al., NeurIPS 2024).

A baseline for DAEDALUS, implemented on the same harness: it reuses daedalus's benchmarks,
solver agents, traces and scoring, and replaces only the two things a memory method owns —
how guidelines are learned (`accumulation`) and how they are retrieved (`run`).

See references/autoguide/README.md for the paper → code map.
"""
