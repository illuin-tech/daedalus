"""DAEDALUS self-play memory generation (paper Algorithms 1-2).

    uv run python -m daedalus.scripts.generation --config configs/appworld/generation/daedalus.yaml
    uv run python -m daedalus.scripts.generation --config <cfg> --resume   # num_sessions more

Writes `outputs/daedalus/<experiment_name>/`: `tasks.json` (the task bank and explorer
guidelines), `pool.json` (the accepted heuristics), `sessions/` (one record per session,
with the explorer transcript and the Solver-loop summary), `traces/` (every solver attempt)
and `usage/` (the per-call cost ledger). Consolidate the pool before inference with
`daedalus.scripts.consolidate`. A fresh run refuses to reuse an existing folder.
"""

from __future__ import annotations

import argparse

from daedalus.core.config import load_config
from daedalus.core.env import load_dotenv
from daedalus.core.generation.pipeline import generate


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Self-play memory generation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--num-sessions", type=int, default=None, help="Override num_sessions")
    parser.add_argument("--num-explorers", type=int, default=None, help="Override num_explorers")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a prior run with num_sessions more sessions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_sessions is not None:
        cfg.generation.num_sessions = args.num_sessions
    if args.num_explorers is not None:
        cfg.generation.num_explorers = args.num_explorers
    generate(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
