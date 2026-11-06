"""Interface contract shared by the modules developed in parallel.

Dependency direction: editors.py, env.py and policies.py import only from here
and never from each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

OBJECTIVES = ("gsk3b", "jnk3", "qed", "sa")
N_OBJ = len(OBJECTIVES)

# Expert names; this is also the action space
EXPERTS = ("gsk3b", "jnk3", "qed", "sa")
N_ACTIONS = len(EXPERTS)


@dataclass(frozen=True)
class Candidate:
    """Result of one proposal. Invalid, duplicate and scaffold-breaking
    candidates are returned too, so they can be counted."""

    smiles: str | None          # canonical SMILES, None if invalid
    parent: str                 # molecule it was derived from
    expert: str                 # expert that proposed it
    valid: bool
    scaffold_ok: bool
    scores: tuple[float, ...] | None = None   # in OBJECTIVES order, higher is better

    @property
    def eligible(self) -> bool:
        return self.valid and self.scaffold_ok


@dataclass
class StepLog:
    """One record per step; all failure analysis and diagnostics read from this."""

    t: int
    expert: str
    current: str
    n_proposed: int
    n_valid: int
    n_scaffold_ok: int
    n_novel: int                # unseen within this episode
    hv_before: float
    hv_after: float
    reward: float
    archive_size: int
    entered_archive: bool       # for diagnosing sparse reward: did anything enter the archive
    best_single_gain: float     # largest single-objective gain, visible even when HV is flat


class Editor(Protocol):
    """Interface shared by all experts. See editors.py for implementations.

    Must be deterministic: the same (smiles, seed) must give the same output,
    otherwise the proposal cache breaks and policies are no longer compared
    on matched trajectories.
    """

    name: str

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        """Return at most k candidate SMILES; env validates them."""
        ...


class Policy(Protocol):
    """Interface shared by all routers. See policies.py for implementations."""

    name: str

    def act(self, obs: Sequence[float], t: int) -> int:
        """Return an action index into EXPERTS."""
        ...

    def update(self, *args, **kwargs) -> None:
        """No-op for baselines that do not learn."""
        ...


@dataclass
class EpisodeConfig:
    """Shared by every policy; otherwise the budget is not matched."""

    max_steps: int = 20
    k_per_call: int = 4
    enforce_scaffold: bool = True
    ref_point: tuple[float, ...] = field(default=(0.0,) * N_OBJ)
    lam: float = 0.0            # call penalty coefficient
    cost_unit: str = "generation"  # "generation" | "oracle"; see README
