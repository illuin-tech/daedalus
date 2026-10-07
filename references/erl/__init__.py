"""ERL — Experiential Reflective Learning (Allard et al., 2026).

A baseline for DAEDALUS, implemented on the same harness: it reuses daedalus's benchmarks,
solver agents, traces and scoring, and replaces only the two things a memory method owns —
how heuristics are learned (`accumulation`) and how they are retrieved (`run`).

See references/erl/README.md for the paper → code map.
"""
