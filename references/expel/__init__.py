"""ExpeL — LLM Agents Are Experiential Learners (Zhao et al., AAAI 2024).

A baseline for DAEDALUS, implemented on the same harness: it reuses daedalus's benchmarks,
solver agents, traces and scoring, and replaces only the two things a memory method owns —
how insights are learned (`accumulation`) and how they and past successes are recalled
(`run`).

See references/expel/README.md for the paper → code map.
"""
