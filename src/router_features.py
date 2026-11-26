"""Dynamic Router 的状态特征 —— 离线训练与在线部署共用同一份定义。

**同一份实现服务两条路径**，这是刻意的：
  离线  从 cf_*.json 的状态记录里取（反事实标签已经采好）
  在线  从活着的 MolEnv 里取（推理时只能看决策前信息）
两条路径走同一个 `_feature_blocks()`，所以特征定义、顺序、预处理不可能
漂移。Step 7 要求"训练和部署使用完全相同的特征定义和特征顺序"，
把它交给类型系统比交给纪律可靠。

决策前信息的边界（禁止泄漏）：
  - 当前分子的四项性质分数：env 在转移时已经打过分并缓存，读缓存免费
  - expert 历史：只用本 episode **已经观察到**的调用次数和累计收益
  - template_sites：只做子结构匹配，0.20 ms/expert
  - applicability：跑反应产生产物，8.94 ms/expert ≈ 一次 propose
                   -> Router 每步付 4 次 = OneStepOracle 的探查成本，
                      所以它只进 R6 消融组，不进默认部署特征
  - **不含**任何未选中 expert 的真实收益、未来状态、反事实标签

特征块：
  ST    step, budget_left                                      2
  SC    当前四项性质分数                                        4
  MOL   描述符 6 + Morgan fingerprint 64                       70
  HIST  calls/gain/stale 各 4 + archive_size + n_seen          14
  SITE  每个 expert 的 template_sites: total/max/nfire         12
  APP   每个 expert 的 applicability: fire/prod/frac/heavy     16
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import (Crippen, Descriptors, rdFingerprintGenerator,
                        rdMolDescriptors)

from .interfaces import EXPERTS

FP_BITS = 64                       # 与 env.py 的 _FP_BITS 对齐
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

BLOCKS = ("ST", "SC", "MOL", "HIST", "SITE", "XI", "APP")

FEATURE_GROUPS = {
    "R1": ("ST",),
    "R2": ("ST", "SC"),
    "R3": ("ST", "SC", "MOL"),
    "R4": ("ST", "SC", "MOL", "HIST"),
    "R5": ("ST", "SC", "MOL", "HIST", "SITE"),     # v1 部署
    "R6": ("ST", "SC", "MOL", "HIST", "SITE", "APP"),  # 高成本消融
    # XI = 分子 x expert 交互（见文件末尾的说明）。R7 是 v2 的候选部署组。
    "R7": ("ST", "SC", "MOL", "HIST", "SITE", "XI"),
    "R8": ("ST", "SC", "MOL", "SITE", "XI"),       # 去掉消融证明无用的 HIST
    "R9": ("ST", "SC", "MOL", "HIST", "SITE", "XI", "APP"),
}

_desc_names = ["mw", "logp", "tpsa", "n_heavy", "n_arom", "rotb"]


def block_names(block: str) -> list[str]:
    if block == "ST":
        return ["step", "budget_left"]
    if block == "SC":
        return [f"score_{e}" for e in EXPERTS]
    if block == "MOL":
        return list(_desc_names) + [f"fp{i}" for i in range(FP_BITS)]
    if block == "HIST":
        return ([f"calls_{e}" for e in EXPERTS]
                + [f"gain_{e}" for e in EXPERTS]
                + [f"stale_{e}" for e in EXPERTS]
                + ["archive_size", "n_seen"])
    if block == "SITE":
        return [f"site_{e}_{k}" for e in EXPERTS
                for k in ("total", "max", "nfire")]
    if block == "XI":
        return [f"xi_{e}_{k}" for e in EXPERTS
                for k in ("logp_after", "tpsa_after", "hbd_after",
                          "logp_toward_window", "mw_after", "size_pressure")]
    if block == "APP":
        return [f"app_{e}_{k}" for e in EXPERTS
                for k in ("fire", "prod", "frac", "heavy")]
    raise ValueError(block)


def feature_names(group: str) -> list[str]:
    out = []
    for b in FEATURE_GROUPS[group]:
        out += block_names(b)
    return out


# --------------------------------------------------------------------------
def _site_vec(editors, smiles):
    v = []
    for e in EXPERTS:
        ts = editors[e].template_sites(smiles)
        act = ts["heavy"] if (ts["shared"] or ts["active_heavy"]) else ts["light"]
        v += [float(sum(act)), float(max(act) if act else 0),
              float(sum(1 for x in act if x > 0))]
    return v


def _app_vec(editors, smiles):
    v = []
    for e in EXPERTS:
        a = editors[e].applicability(smiles)
        v += [a["n_template_fire"], a["n_product"],
              a["frac_template_fire"], a["heavy_pool"]]
    return v


def _feature_blocks(ctx: dict, editors, want) -> dict[str, list[float]]:
    """ctx 的字段两条路径都必须提供，缺字段直接 KeyError —— 宁可炸也不要
    悄悄补 0 造成训练/部署不一致。"""
    out = {}
    if "ST" in want:
        out["ST"] = [float(ctx["step"]), float(ctx["budget_left"])]
    if "SC" in want:
        out["SC"] = [float(x) for x in ctx["scores"]]
    if "MOL" in want:
        mol = Chem.MolFromSmiles(ctx["current_smiles"])
        if mol is None:
            out["MOL"] = [0.0] * (len(_desc_names) + FP_BITS)
        else:
            d = [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                 rdMolDescriptors.CalcTPSA(mol), float(mol.GetNumHeavyAtoms()),
                 float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                 float(rdMolDescriptors.CalcNumRotatableBonds(mol))]
            out["MOL"] = d + _fpgen.GetFingerprintAsNumPy(mol).astype(
                np.float64).tolist()
    if "HIST" in want:
        calls = [float(x) for x in ctx["calls_so_far"]]
        gain = [float(x) for x in ctx["gain_so_far"]]
        # stale：调用过但至今零收益。两条路径都能从 calls/gain 推出，
        # 不需要额外记录"上次是否失败"（cf 数据里没有那一项）。
        stale = [1.0 if (c > 0 and g <= 0) else 0.0 for c, g in zip(calls, gain)]
        out["HIST"] = calls + gain + stale + [float(ctx["archive_size"]),
                                              float(ctx["n_seen"])]
    if "SITE" in want:
        out["SITE"] = _site_vec(editors, ctx["current_smiles"])
    if "XI" in want:
        out["XI"] = _interaction(Chem.MolFromSmiles(ctx["current_smiles"]),
                                 editors)[0]
    if "APP" in want:
        out["APP"] = _app_vec(editors, ctx["current_smiles"])
    return out


# --------------------------------------------------------------------------
# 离线：从反事实 json 建训练矩阵
# --------------------------------------------------------------------------
def build_states(cf_path: Path, pool: str):
    """-> (S, blocks, D, step, seed_smiles, cost_ms)

    blocks[b] 是 (N, d_b) 的矩阵；cost_ms[b] 是该块的提取成本（ms/状态）。
    """
    from .editors import make_editors
    kw = {"results_base0": {"qed_disjoint": False}}.get(
        f"results_{pool}", {"qed_disjoint": True})
    editors = make_editors(**kw)

    S = json.loads(Path(cf_path).read_text())
    D = np.array([[s["actions"][e]["d_hv"] for e in EXPERTS] for s in S])
    step = np.array([s["step"] for s in S])
    seed_smi = [s["seed_smiles"] for s in S]

    acc: dict[str, list] = {b: [] for b in BLOCKS}
    cost: dict[str, float] = {b: 0.0 for b in BLOCKS}
    for s in S:
        ctx = {
            "step": s["step"], "budget_left": s["budget_left"],
            "scores": s["obs"][:len(EXPERTS)],
            "current_smiles": s["current_smiles"],
            "calls_so_far": s["calls_so_far"],
            "gain_so_far": s["gain_so_far"],
            "archive_size": s["archive_size"], "n_seen": s["n_seen"],
        }
        for b in BLOCKS:
            t0 = time.perf_counter()
            v = _feature_blocks(ctx, editors, (b,))[b]
            cost[b] += time.perf_counter() - t0
            acc[b].append(v)
    blocks = {b: np.asarray(acc[b], float) for b in BLOCKS}
    cost = {b: 1000 * cost[b] / len(S) for b in BLOCKS}
    return S, blocks, D, step, seed_smi, cost


# --------------------------------------------------------------------------
# 在线：从活着的 env 取特征（推理路径）
# --------------------------------------------------------------------------
def features_from_env(env, group: str) -> np.ndarray:
    """(1, d) 特征行。只读 env 的当前状态，不执行任何 expert、不克隆环境。"""
    want = FEATURE_GROUPS[group]
    sc = env.bank.vector(env.current) or [0.0] * len(EXPERTS)
    ctx = {
        "step": env.t,
        "budget_left": 1.0 - env.t / env.cfg.max_steps,
        "scores": sc,
        "current_smiles": env.current,
        "calls_so_far": list(env.calls_so_far),
        "gain_so_far": list(env.gain_so_far),
        "archive_size": len(env.archive),
        "n_seen": len(env.seen),
    }
    blk = _feature_blocks(ctx, env.editors, want)
    return np.asarray([sum((blk[b] for b in want), [])], dtype=float)


# --------------------------------------------------------------------------
# 块 XI：分子 x expert 交互 —— 攻 jnk3 盲点
# --------------------------------------------------------------------------
# 错误分析显示：jnk3 在 344 个状态上是最优，模型只召回 14.8%，最常被误选成
# gsk3b（149 次）。根因不是学力，是特征里没有能分开它们的信息 ——
# 两个池的反应物模式**逐字相同**（都是 [cH:1]，任意芳香 CH），所以
# template_sites 在它们上恒等。它们只在"接什么"上不同：
#     gsk3b 接极性基团   ΔlogP -0.50  ΔTPSA +34.5  ΔHBD +1.00
#     jnk3  接亲脂基团   ΔlogP +0.78  ΔTPSA  +2.1  ΔHBD  0.00
# 所以判别信号是"接上去之后分子会变成什么样"，而这不需要跑反应：
# 每条模板片段对描述符的贡献是固定的，用探针分子预先算一次即可。
#
# 成本：描述符在 MOL 块里已经算过，这里只是加减法，约 0 ms。

_PROBES = [
    "c1ccccc1",                    # 苯：增长型模板
    "Cc1ccccc1",                   # 甲苯：带可删单原子取代
    "Nc1ccc(C)cc1O",               # 多官能团芳香环
    "CC(C)c1ccc(OC)cc1C(N)=O",     # 兼顾缩小/替换型
]
_FRAG_DESC = ("mw", "logp", "tpsa", "hbd", "hba")
_frag_cache: dict | None = None


def _expert_fragment_delta(editors) -> dict[str, list[float]]:
    """每个 expert 的模板片段对 (MW, logP, TPSA, HBD, HBA) 的平均效应。

    对一组探针分子跑一遍该 expert 的池，取产物与探针的描述符差再平均。
    只在模块首次使用时算一次（约 40 次 RunReactants），之后走缓存。
    """
    global _frag_cache
    if _frag_cache is not None:
        return _frag_cache

    def desc(m):
        return [Descriptors.MolWt(m), Crippen.MolLogP(m),
                rdMolDescriptors.CalcTPSA(m),
                float(rdMolDescriptors.CalcNumHBD(m)),
                float(rdMolDescriptors.CalcNumHBA(m))]

    out = {}
    for e in EXPERTS:
        acc = []
        for smi in _PROBES:
            probe = Chem.MolFromSmiles(smi)
            if probe is None:
                continue
            base = desc(probe)
            for rxn in editors[e]._pools(probe):
                try:
                    res = rxn.RunReactants((probe,))
                except Exception:
                    continue
                for tup in res[:1]:
                    for p in tup[:1]:
                        try:
                            Chem.SanitizeMol(p)
                        except Exception:
                            continue
                        acc.append([a - b for a, b in zip(desc(p), base)])
        out[e] = (np.mean(acc, axis=0).tolist() if acc
                  else [0.0] * len(_FRAG_DESC))
    _frag_cache = out
    return out


def _interaction(mol, editors) -> tuple[list[float], list[str]]:
    """每个 expert 的"编辑后会变成什么样"，以及它与当前位置的关系。"""
    frag = _expert_fragment_delta(editors)
    if mol is None:
        return [0.0] * (len(EXPERTS) * 6), [
            f"xi_{e}_{k}" for e in EXPERTS
            for k in ("logp_after", "tpsa_after", "hbd_after",
                      "logp_toward_window", "mw_after", "size_pressure")]
    now = {"mw": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
           "tpsa": rdMolDescriptors.CalcTPSA(mol),
           "hbd": float(rdMolDescriptors.CalcNumHBD(mol))}
    # QED 的类药窗口中心，取自 src/editors.py:_too_heavy 的同一组阈值
    LOGP_LO, LOGP_HI, MW_HI = 1.0, 3.0, 360.0

    def dist_window(x):
        return max(LOGP_LO - x, 0.0) + max(x - LOGP_HI, 0.0)

    vals, names = [], []
    d_now = dist_window(now["logp"])
    for e in EXPERTS:
        dmw, dlogp, dtpsa, dhbd, _ = frag[e]
        lp = now["logp"] + dlogp
        vals += [
            lp,                                   # 编辑后的 logP
            now["tpsa"] + dtpsa,                  # 编辑后的 TPSA
            now["hbd"] + dhbd,                    # 编辑后的 HBD
            d_now - dist_window(lp),              # 是否把 logP 推回类药窗口（正=推回）
            now["mw"] + dmw,                      # 编辑后的 MW
            (now["mw"] + dmw) - MW_HI,            # 相对 MW 上限的压力
        ]
        names += [f"xi_{e}_{k}" for k in
                  ("logp_after", "tpsa_after", "hbd_after",
                   "logp_toward_window", "mw_after", "size_pressure")]
    return vals, names
