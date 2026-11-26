"""Selector / Router 的分子级特征。

全部是**决策前**可得的量：只看当前分子和 expert 池本身，不看任何
已经跑出来的候选、不看未来、不消耗 oracle 调用（F1 例外，见下）。

特征组按 EXP 1.4 定义：

  F1  四个初始属性分数（gsk3b/jnk3/qed/sa）—— 要 4 次 oracle 打分，
      但种子分子在正式跑之前本来就会被打一次分，所以不算新增成本。
  F2  Morgan fingerprint，折叠到 64 位（和 env.py 的 _FP_BITS 一致，
      刻意不用 1024 位：50 个分子撑不起那个维度）。
  F3  MW / LogP / TPSA / HBD / HBA / RotatableBonds。
  F4  四个 expert 的模板适用性（fire 条数、产物数、fire 比例、走哪支池）。
  F5  F1 + F3 + F4
  F6  全部

F4 调 RDKitEditor.applicability()，和 fixed_val.json 里那个含未来信息的
`eligible` 不是一回事 —— 后者是整条 episode 跑完的累计数，只能当诊断。
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .interfaces import EXPERTS

FP_BITS = 64   # 与 env.py 的 _FP_BITS 对齐
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

FEATURE_SETS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")


def _descriptors(mol) -> tuple[list[float], list[str]]:
    vals = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
    ]
    return vals, ["mw", "logp", "tpsa", "hbd", "hba", "rotb"]


def _fingerprint(mol) -> tuple[list[float], list[str]]:
    fp = _fpgen.GetFingerprintAsNumPy(mol).astype(np.float64)
    return list(fp), [f"fp{i}" for i in range(FP_BITS)]


def _applicability(smiles: str, editors: dict) -> tuple[list[float], list[str]]:
    vals, names = [], []
    for e in EXPERTS:
        a = editors[e].applicability(smiles)
        for k in ("n_template_fire", "n_product", "frac_template_fire", "heavy_pool"):
            vals.append(a[k])
            names.append(f"app_{e}_{k}")
    return vals, names


def _structural(mol) -> tuple[list[float], list[str]]:
    """EXP 1.8：针对 GSK3β/JNK3 的结构特征。

    这两个 expert 在 F1+F3+F4 下的 Ridge 系数全是 0 量级 —— 模型对它们
    没学到任何结构，而它们在 30/50 个分子上是最优的。激酶活性和芳环数、
    卤素、杂原子分布相关，所以这里补的是这几类计数。
    """
    ri = mol.GetRingInfo()
    arom_rings = sum(1 for r in ri.AtomRings()
                     if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    vals = [
        float(ri.NumRings()),
        float(arom_rings),
        float(sum(1 for s in sym if s in ("F", "Cl", "Br", "I"))),
        float(sym.count("N")),
        float(sym.count("O")),
        float(sym.count("S")),
        float(sum(1 for s in sym if s not in ("C", "H"))),
        float(mol.GetNumHeavyAtoms()),
        rdMolDescriptors.CalcFractionCSP3(mol),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        float(rdMolDescriptors.CalcNumHeterocycles(mol)),
    ]
    names = ["n_ring", "n_arom_ring", "n_halogen", "n_N", "n_O", "n_S",
             "n_hetero", "n_heavy", "frac_csp3", "n_arom_ring2",
             "n_aliph_ring", "n_heterocycle"]
    return vals, names


def _template_sites(smiles: str, editors: dict) -> tuple[list[float], list[str]]:
    """每个 expert 逐条模板的匹配位点数（见 RDKitEditor.template_sites）。

    heavy / light 两支各占一块定长列，非激活的那支整块置 0 —— 这样列的语义
    固定（某一支的第 i 条模板），而"走哪支"的信息由 F4 的 heavy_pool 携带。
    `shared=True` 的 expert（gsk3b / jnk3 / sa）两支同池，只出一块。
    """
    vals, names = [], []
    for e in EXPERTS:
        s = editors[e].template_sites(smiles)
        act = s["active_heavy"]
        branches = ([("", s["heavy"], True)] if s["shared"]
                    else [("h", s["heavy"], act), ("l", s["light"], not act)])
        for tag, sites, on in branches:
            for i, n in enumerate(sites):
                vals.append(float(n) if on else 0.0)
                names.append(f"site_{e}_{tag}t{i}")
            vals += [float(sum(sites)) if on else 0.0,
                     float(max(sites)) if (on and sites) else 0.0]
            names += [f"site_{e}_{tag}total", f"site_{e}_{tag}max"]
    return vals, names


def blocks(smiles: str, editors: dict, bank) -> dict[str, tuple[list[float], list[str]]]:
    """返回 {块名: (值, 列名)}。块名 P/FP/D/A 对应 F1/F2/F3/F4。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"非法 SMILES: {smiles}")
    props = bank.vector(smiles)
    if props is None:
        raise ValueError(f"无法打分: {smiles}")
    d_v, d_n = _descriptors(mol)
    f_v, f_n = _fingerprint(mol)
    a_v, a_n = _applicability(smiles, editors)
    s_v, s_n = _structural(mol)
    t_v, t_n = _template_sites(smiles, editors)
    return {
        "P": (list(props), [f"prop_{o}" for o in EXPERTS]),
        "FP": (f_v, f_n),
        "D": (d_v, d_n),
        "A": (a_v, a_n),
        "S": (s_v, s_n),
        "T": (t_v, t_n),
    }


_SETS = {
    "F1": ("P",),
    "F2": ("FP",),
    "F3": ("D",),
    "F4": ("A",),
    "F5": ("P", "D", "A"),
    "F6": ("P", "FP", "D", "A"),
    # EXP 1.8 新增：F5 是四个池上一致的赢家，所以新特征都以它为底叠加，
    # 这样 F7/F8/F9 与 F5 的差就是新信息的净贡献。
    "F7": ("P", "D", "A", "S"),            # + 结构计数
    "F8": ("P", "D", "A", "T"),            # + 逐模板位点数
    "F9": ("P", "D", "A", "S", "T"),       # 两者都加
}


def build_matrix(smiles_list: list[str], editors: dict, bank,
                 feature_set: str) -> tuple[np.ndarray, list[str]]:
    """(n_mol, n_feat) 特征矩阵 + 列名。行序与 smiles_list 一致。"""
    if feature_set not in _SETS:
        raise ValueError(f"未知特征组 {feature_set}，可选 {list(_SETS)}")
    keys = _SETS[feature_set]
    rows, names = [], None
    for s in smiles_list:
        b = blocks(s, editors, bank)
        vals, nm = [], []
        for k in keys:
            v, n = b[k]
            vals += v
            nm += n
        rows.append(vals)
        names = nm
    return np.asarray(rows, dtype=np.float64), names
