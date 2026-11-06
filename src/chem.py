"""Molecule validity and scaffold constraint checks."""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=200_000)
def canonical(smiles: str) -> str | None:
    """Canonical SMILES, or None if invalid. Cached; all callers go through this."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=200_000)
def murcko_scaffold(smiles: str) -> str | None:
    """Canonical SMILES of the Bemis-Murcko scaffold.

    Acyclic molecules have an empty scaffold "". The grouping rule is that the
    empty scaffold forms its own group and never mixes with cyclic molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        # Drop stereochemistry, otherwise the SA editor's strip_stereo looks like
        # it broke the scaffold. The constraint is on connectivity, not configuration.
        Chem.RemoveStereochemistry(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def scaffold_preserved(parent: str, child: str) -> bool:
    """Whether child keeps the parent's Murcko scaffold."""
    ps, cs = murcko_scaffold(parent), murcko_scaffold(child)
    if ps is None or cs is None:
        return False
    return ps == cs


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return 0 if mol is None else mol.GetNumHeavyAtoms()


def is_reasonable(smiles: str, max_heavy: int = 70) -> bool:
    """Reject the degenerate molecules editors occasionally emit.

    Not a chemistry plausibility judgement, just a guard against feeding junk to
    the oracles. Rejected proposals still count as generation attempts.
    """
    n = heavy_atom_count(smiles)
    return 3 <= n <= max_heavy
