"""ACE's own config block.

Everything shared with DAEDALUS stays in daedalus's blocks (`agent`, the benchmark block,
`run`); only what ACE itself adds lives here. See `references/common/config.py` for how
the block is loaded and snapshotted into the run folder.

ACE names three roles — Generator, Reflector, Curator. The Generator IS the benchmark's
solver (`agent.model`), so it has no field here; the other two do.
"""

from __future__ import annotations

from dataclasses import dataclass

from references.common.config import MethodConfig, retrieval_dir  # noqa: F401 (re-export)

# This method's setup name: it namespaces every artifact under
# outputs/baselines/ace/{memory,inference}/<name>/, keeps ACE runs out of daedalus's own tree, and is
# what the run browser and the figure scripts label these runs with. Both entry points set
# it on the config, so it cannot be lost by editing a YAML.
SETUP = "ace"


@dataclass
class ACEConfig(MethodConfig):
    """ACE's knobs: the playbook, and the two models that grow it."""

    SETUP = SETUP

    # ── inference ────────────────────────────────────────────────────────────
    # The trained playbook to inject, verbatim, at task start: the `playbook.txt` written
    # by `references.ace.accumulation`. During accumulation this is pinned by the entry
    # point to the run's own playbook, so a worker can never read another run's.
    playbook_path: str | None = None

    # ── accumulation ─────────────────────────────────────────────────────────
    # Reflector (ace-appworld `adaptation_react.py::reflector_call`): one call over the
    # finished trajectory, producing a diagnosis. Defaulted to daedalus's own
    # `accumulation.extraction_model` setting so the ACE row is paired with ours on
    # writer strength rather than on ACE's single-model published setup.
    reflector_model: str = "gpt-5.4"
    reflector_reasoning_effort: str | None = "high"
    # Curator (`curator_call`): one call over (playbook, reflection, trajectory),
    # emitting ADD operations. ADD is the only operation ACE implements.
    curator_model: str = "gpt-5.4"
    curator_reasoning_effort: str | None = "high"

    # Snapshot the playbook every N finished tasks (ACE: 30), so its growth curve is
    # recoverable after the fact.
    snapshot_every: int = 30
