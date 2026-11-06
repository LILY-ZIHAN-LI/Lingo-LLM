"""Four objective-conditioned editors backed by deterministic RDKit reaction pools.

Design notes:
1. QED is a bell-shaped function of MW/logP/TPSA, so a fixed reaction pool cannot
   serve it: large molecules need atoms removed, small ones need atoms added. The
   qed and sa editors therefore switch sub-pools on RDKit descriptors. Descriptors
   are free structural features, not objectives, and do not count against the
   oracle budget.
2. The SA editor needs generic deletion reactions that fire on any molecule,
   otherwise it proposes nothing on most seeds.
3. The gsk3b and jnk3 pools are deliberately chemically disjoint. The two RF
   oracles are themselves highly correlated (both kinases), so whether the editors
   actually diverge is an empirical question; see check_editor_specialization.py.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

from .chem import canonical, is_reasonable

RDLogger.DisableLog("rdApp.*")

# --- GSK3B: polar groups, hydrogen bond donors and acceptors --------------
_GSK3B = [
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]C(N)=O",
    "[cH:1]>>[c:1]NC(C)=O",
    "[cH:1]>>[c:1]C#N",
    "[cH:1]>>[c:1]S(N)(=O)=O",
    "[cH:1]>>[c:1]N1CCNCC1",
    "[cH:1]>>[c:1]NC(N)=O",
    "[cH:1]>>[c:1]C(=O)O",
]

# --- JNK3: halogens, hydrophobic groups, sulfur ---------------------------
_JNK3 = [
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]Br",
    "[cH:1]>>[c:1]C(F)(F)F",
    "[cH:1]>>[c:1]OC(F)(F)F",
    "[cH:1]>>[c:1]SC",
    "[cH:1]>>[c:1]C(C)C",
    "[cH:1]>>[c:1]C(C)(C)C",
    "[cH:1]>>[c:1]OCC",
]

# --- Generic deletions, fire on any cyclic molecule (shared by SA and QED-shrink)
_SHRINK = [
    "[R:1][!R;D1]>>[R:1]",                 # drop any single-atom ring substituent
    "[c:1][C;!R;H3]>>[c:1]",               # remove aromatic methyl
    "[c:1][F,Cl,Br,I]>>[c:1]",             # dehalogenate
    "[c:1]C(F)(F)F>>[c:1]C",               # CF3 -> Me
    "[c:1]S(N)(=O)=O>>[c:1]",              # remove sulfonamide
    "[c:1][N+](=O)[O-]>>[c:1]",            # remove nitro
    "[c:1]C#N>>[c:1]C",
    "[c:1]OC(F)(F)F>>[c:1]OC",
    "[C;!R;H2:1][C;!R;H3]>>[C;!R;H3:1]",   # shorten alkyl chain
    "[c:1][O;!R][CH3]>>[c:1]O",            # demethylate
]

# --- Generic growth: add small polar groups (used by QED-grow) ------------
_GROW = [
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]OC",
    "[cH:1]>>[c:1]C",
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]C(N)=O",
]

# --- QED: like-for-like swaps rather than deletions; shares no reaction with _SHRINK
_QED_SWAP = [
    "[c:1]C(C)(C)C>>[c:1]C",
    "[c:1]Br>>[c:1]F",
    "[c:1]I>>[c:1]Cl",
    "[c:1]C(F)(F)F>>[c:1]C",
    "[c:1][N+](=O)[O-]>>[c:1]N",
    "[c:1]OC(F)(F)F>>[c:1]OC",
    "[c:1]C(=O)OC>>[c:1]C(=O)O",
    "[c:1]S(C)(=O)=O>>[c:1]C",
]

def _compile(smarts_list):
    out = []
    for s in smarts_list:
        rxn = AllChem.ReactionFromSmarts(s)
        if rxn is not None:
            out.append(rxn)
    return out


# Union of all reactions, used by _mix to blend pools together.
_ALL_SMARTS = sorted(set(_GSK3B + _JNK3 + _SHRINK + _GROW + _QED_SWAP))


def _mix(own: list[str], alpha: float) -> list[str]:
    """Deterministically blend `alpha` of the union into `own`.

    alpha=0 leaves the pool unchanged, alpha=1 gives the full union. The sort key
    is a blake2b digest rather than source order, so prefixes form a nested chain
    of pools: raising alpha grows every pool monotonically and makes the four
    editors converge. At alpha=1 all four share the union and are identical by
    construction, so the effective action dimension should collapse to 1.
    """
    if alpha <= 0:
        return list(own)
    have = set(own)
    extra = [x for x in _ALL_SMARTS if x not in have]
    extra.sort(key=lambda x: hashlib.blake2b(x.encode(),
                                             digest_size=8).digest())
    return list(own) + extra[:int(round(alpha * len(extra)))]


def _too_heavy(mol) -> bool:
    """The QED bell is centred near MW 300-380 and logP 1-3: shrink above, grow below."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    return mw > 360 or logp > 3.0


@dataclass
class RDKitEditor:
    """Deterministic: the same (smiles, seed) gives the same output, which is what
    the proposal cache relies on."""

    name: str
    strip_stereo: bool = False
    # Blend ratio, see _mix. 0 keeps the pools disjoint as designed.
    overlap: float = 0.0
    # Use _QED_SWAP instead of _SHRINK for heavy molecules. With the default
    # False, qed and sa share the exact same pool on heavy molecules, which is
    # why they proposed identical candidates on 70 of 100 seeds. Turning this on
    # raises the effective action dimension.
    qed_disjoint: bool = False
    _heavy: list = field(default=None, init=False, repr=False)
    _light: list = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name == "gsk3b":
            heavy = light = _GSK3B
        elif self.name == "jnk3":
            heavy = light = _JNK3
        elif self.name == "sa":
            heavy = light = _SHRINK
        else:   # qed: conditioned on descriptors (shrink if heavy, grow if light)
            heavy = _QED_SWAP if self.qed_disjoint else _SHRINK
            light = _GROW
        # Compile once per instance. propose sorts products before sampling, so
        # pool order does not affect the output.
        self._heavy = _compile(_mix(heavy, self.overlap))
        self._light = _compile(_mix(light, self.overlap))

    def _pools(self, mol) -> list:
        if self.name != "qed":
            return self._heavy
        return self._heavy if _too_heavy(mol) else self._light

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        products: list[str] = []
        seen: set[str] = {canonical(smiles) or smiles}

        for rxn in self._pools(mol):
            try:
                outcomes = rxn.RunReactants((mol,))
            except Exception:
                continue
            for tup in outcomes:
                for p in tup:
                    try:
                        Chem.SanitizeMol(p)
                        smi = Chem.MolToSmiles(p)
                    except Exception:
                        continue
                    if smi in seen or not is_reasonable(smi):
                        continue
                    seen.add(smi)
                    products.append(smi)

        if self.strip_stereo:
            flat = Chem.MolFromSmiles(smiles)
            if flat is not None and Chem.FindMolChiralCenters(
                flat, includeUnassigned=False
            ):
                Chem.RemoveStereochemistry(flat)
                smi = Chem.MolToSmiles(flat)
                if smi not in seen:
                    seen.add(smi)
                    products.append(smi)

        if not products:
            return []

        products.sort()
        rng = random.Random(seed)
        if len(products) <= k:
            return products
        return rng.sample(products, k)


def make_editors(overlap: float = 0.0,
                 qed_disjoint: bool = False) -> dict[str, RDKitEditor]:
    """The default arguments reproduce the original pools."""
    kw = {"overlap": overlap, "qed_disjoint": qed_disjoint}
    return {
        "gsk3b": RDKitEditor("gsk3b", **kw),
        "jnk3": RDKitEditor("jnk3", **kw),
        "qed": RDKitEditor("qed", **kw),
        "sa": RDKitEditor("sa", strip_stereo=True, **kw),
    }
