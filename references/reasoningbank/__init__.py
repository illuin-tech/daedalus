"""ReasoningBank — Scaling Agent Self-Evolving with Reasoning Memory (Ouyang et al., 2026).

A baseline for DAEDALUS, implemented on the same harness: it reuses daedalus's benchmarks,
solver agents, traces and scoring, and replaces only the two things a memory method owns —
how memory items are learned (`accumulation`) and how they are retrieved (`run`).

See references/reasoningbank/README.md for the paper → code map.
"""
