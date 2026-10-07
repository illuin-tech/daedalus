"""daedalus — self-play memory generation + memory-augmented inference across
benchmarks (AppWorld, tau2-bench, AutomationBench).

Shared, benchmark-neutral machinery lives in `daedalus.core`; each benchmark is a
self-contained connector under `daedalus.benchmarks.<name>` that plugs into the two
seams `daedalus.core.benchmark.Benchmark` (inference/accumulation) and
`daedalus.core.generation.backend.GenerationBackend` (self-play generation).
"""

import warnings as _warnings

# AppWorld imports `fastapi.testclient`, which emits a StarletteDeprecationWarning about
# httpx on every single process start — including every plot script and every worker. It is
# a third-party dependency's own upgrade notice, not actionable here, and it buried real
# output. Matched on the message rather than the class so importing starlette is not needed
# just to silence it; installed here because `daedalus` is imported before appworld in every
# entry point, and a filter must be in place before the warning is raised.
_warnings.filterwarnings(
    "ignore", message=r"Using `httpx` with `starlette\.testclient` is deprecated.*"
)
