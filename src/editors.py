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
    # Optional pool produced by scripts/design_pool.py.
    designed: list = None
    # Allocate proposal slots by template, with the selected product tie-break.
    sample_by_template: bool = False
    product_pick: str = "random"
    _heavy: list = field(default=None, init=False, repr=False)
    _light: list = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.designed is not None:
            self._heavy = self._light = _compile(self.designed)
            return
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

    def applicability(self, smiles: str) -> dict[str, float]:
        """这个 expert 在该分子上「能动多少」—— 纯决策前信息。

        只看分子本身：跑一遍自己的模板池，数有多少条模板真的 fire、
        产出多少个通过 is_reasonable 的产物。不打分、不看未来、不消耗
        oracle，所以可以安全地当 selector 的输入特征，也可以用来搭一个
        无泄漏的 Applicability Rule 基线。

        注意这和 fixed_val.json 里的 `eligible` 不是一回事：那个是整条
        episode **跑完之后**的合格候选累计数，含未来信息，只能当诊断。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"n_template_fire": 0.0, "n_product": 0.0,
                    "frac_template_fire": 0.0, "heavy_pool": 0.0}
        pool = self._pools(mol)
        seen: set[str] = {canonical(smiles) or smiles}
        n_fire = 0
        for rxn in pool:
            fired = False
            try:
                outcomes = rxn.RunReactants((mol,))
            except Exception:
                outcomes = ()
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
                    fired = True
            if fired:
                n_fire += 1
        n_prod = len(seen) - 1
        return {
            "n_template_fire": float(n_fire),
            "n_product": float(n_prod),
            "frac_template_fire": n_fire / len(pool) if pool else 0.0,
            # qed 会按描述符在 heavy/light 池之间切换，记下走的是哪一支
            "heavy_pool": float(self._heavy is self._pools(mol)),
        }

    def template_sites(self, smiles: str) -> dict:
        """逐条模板在该分子上的匹配**位点数**，heavy / light 两支分开给。

        比 applicability() 细一档：那个只数"有没有 fire"和"出了几个产物"，
        这个按模板逐条给位点数，所以 gsk3b / jnk3 这种"模板总是全部 fire"
        （frac_template_fire 恒为 1.0，标准化后是死列）的 expert 也能有变化。

        两支必须分开返回、不能拼成一个定长向量：qed 的 heavy 是 _QED_SWAP
        （8 条）、light 是 _GROW（6 条），是**完全不同的模板集**，同一个下标
        在两支里语义不同，补零拼接会让线性模型学到错的东西。
        调用方应按 active_heavy 掩码。

        只做子结构匹配，不跑反应、不打分 —— 比 applicability() 便宜。
        """
        mol = Chem.MolFromSmiles(smiles)
        shared = self._heavy is self._light

        def count(pool):
            if mol is None:
                return [0] * len(pool)
            out = []
            for rxn in pool:
                try:
                    patt = rxn.GetReactantTemplate(0)
                    out.append(len(mol.GetSubstructMatches(patt, uniquify=True)))
                except Exception:
                    out.append(0)
            return out

        return {
            "heavy": count(self._heavy),
            "light": None if shared else count(self._light),
            "active_heavy": True if mol is None else (self._heavy is self._pools(mol)),
            "shared": shared,
        }

    def _pools(self, mol) -> list:
        if self.name != "qed":
            return self._heavy
        return self._heavy if _too_heavy(mol) else self._light

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        products: list[str] = []
        by_rxn: list[list[str]] = []
        seen: set[str] = {canonical(smiles) or smiles}

        for rxn in self._pools(mol):
            mine: list[str] = []
            try:
                outcomes = rxn.RunReactants((mol,))
            except Exception:
                outcomes = ()
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
                    mine.append(smi)
                    products.append(smi)
            if mine:
                by_rxn.append(sorted(mine))

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

        rng = random.Random(seed)
        if self.sample_by_template and by_rxn:
            # k 个名额分给 k 条不同的模板，每条随机出一个产物。
            # 每条取排序后的首个也能保证确定性，但那样同一个 (分子, 模板)
            # 永远给同一个产物 —— §2 的吸收态正是"重选同一个 editor 又拿到
            # 同样的东西"，不能再加一条让它更容易发生的规则。
            # 这里的随机性仍然由 (episode, t) 导出的 seed 决定，
            # 所以 paired 比较照样成立。
            picks = (by_rxn if len(by_rxn) <= k
                     else rng.sample(by_rxn, k))
            if self.product_pick == "first":
                out = [m[0] for m in picks]
            elif self.product_pick == "last":
                out = [m[-1] for m in picks]
            else:
                out = [rng.choice(m) for m in picks]
            if len(out) < k:
                rest = sorted(set(products) - set(out))
                if rest:
                    out += rng.sample(rest, min(k - len(out), len(rest)))
            return out

        products.sort()
        if len(products) <= k:
            return products
        return rng.sample(products, k)


def make_editors(overlap: float = 0.0,
                 qed_disjoint: bool = False,
                 pool_design: str | None = None,
                 sample_by_template: bool = False,
                 product_pick: str = "random") -> dict[str, RDKitEditor]:
    """The default arguments reproduce the original pools.

    A pool_design from scripts/design_pool.py replaces the hand-written pool
    and its overlap and qed_disjoint settings.
    """
    if pool_design is not None:
        import json
        from pathlib import Path
        blob = json.loads(Path(pool_design).read_text())
        pools = blob["pools"]
        missing = [n for n in ("gsk3b", "jnk3", "qed", "sa") if n not in pools]
        if missing:
            raise ValueError(f"{pool_design} 缺少 expert: {missing}")
        sbt = bool(blob.get("sample_by_template", True))
        return {n: RDKitEditor(n, designed=pools[n], sample_by_template=sbt,
                               strip_stereo=(n == "sa"))
                for n in ("gsk3b", "jnk3", "qed", "sa")}
    kw = {"overlap": overlap, "qed_disjoint": qed_disjoint,
          "sample_by_template": sample_by_template,
          "product_pick": product_pick}
    return {
        "gsk3b": RDKitEditor("gsk3b", **kw),
        "jnk3": RDKitEditor("jnk3", **kw),
        "qed": RDKitEditor("qed", **kw),
        "sa": RDKitEditor("sa", strip_stereo=True, **kw),
    }
