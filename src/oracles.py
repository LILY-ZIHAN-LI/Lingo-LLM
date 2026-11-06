"""Four-objective oracle wrapper: one direction (higher is better), one cache,
one call counter.

All policies share a single instance so that c_t (the number of new scalar
scorings) is measured identically across baselines, which is what makes the
-lambda * c_t term in the reward comparable.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

# Shim for `from rdkit.six import iteritems`, still used by PyTDC 1.1.x
if "rdkit.six" not in sys.modules:
    _six = types.ModuleType("rdkit.six")
    _six.iteritems = lambda d: iter(d.items())
    sys.modules["rdkit.six"] = _six

from rdkit import Chem, RDLogger  # noqa: E402
from tdc import Oracle  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OBJECTIVES = ("gsk3b", "jnk3", "qed", "sa")

# Raw SA is 1-10 and lower is better; map to [0,1] where higher is better
_SA_LO, _SA_HI = 1.0, 10.0


def canonical(smiles: str) -> str | None:
    """Canonical SMILES, or None for an invalid molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@dataclass
class OracleBank:
    """Scorer for the four objectives. score() returns a dict where higher is better."""

    _oracles: dict = field(default_factory=dict, init=False)
    _cache: dict = field(default_factory=dict, init=False)
    n_calls: int = field(default=0, init=False)  # (molecule, objective) pairs actually computed
    n_hits: int = field(default=0, init=False)   # cache hits

    def __post_init__(self) -> None:
        self._oracles = {
            "gsk3b": Oracle(name="GSK3B"),
            "jnk3": Oracle(name="JNK3"),
            "qed": Oracle(name="QED"),
            "sa": Oracle(name="SA"),
        }

    def score(self, smiles: str) -> dict[str, float] | None:
        """Returns None for invalid SMILES; the caller should log a failed proposal."""
        key = canonical(smiles)
        if key is None:
            return None
        if key in self._cache:
            self._hits_add(len(OBJECTIVES))
            return self._cache[key]

        raw = {name: float(orc(key)) for name, orc in self._oracles.items()}
        self.n_calls += len(OBJECTIVES)
        vec = {
            "gsk3b": raw["gsk3b"],
            "jnk3": raw["jnk3"],
            "qed": raw["qed"],
            "sa": (_SA_HI - raw["sa"]) / (_SA_HI - _SA_LO),
        }
        vec = {k: min(max(v, 0.0), 1.0) for k, v in vec.items()}
        self._cache[key] = vec
        return vec

    def _hits_add(self, n: int) -> None:
        self.n_hits += n

    def vector(self, smiles: str) -> list[float] | None:
        v = self.score(smiles)
        return None if v is None else [v[k] for k in OBJECTIVES]

    def stats(self) -> dict[str, int]:
        return {"logical_calls": self.n_calls, "cache_hits": self.n_hits,
                "unique_molecules": len(self._cache)}
