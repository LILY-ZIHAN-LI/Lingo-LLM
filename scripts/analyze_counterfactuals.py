"""EXP 3 / 阶段一 Step 1：3000 个反事实状态的完整统计，输出 counterfactual_analysis.md。

目的是在训任何 Dynamic Router 之前，先定量回答「这批状态里有多少可被
机器学习利用的决策信息」。

两个 pilot 阶段踩过的坑，这里一开始就避开：
  1. `D.argmax(1)` 在平局和全零状态上一律落到下标 0（= gsk3b），会伪造出
     "gsk3b 在后期接管" 的假迁移。所以所有"最佳 expert"统计只在**干净
     状态**（有正收益、且最优与次优之差 > 平局阈值）上算。
  2. "全零" 必须拆开：真吸收态（连合法分子都生不出）、无合格候选、
     有候选但无即时 HV 收益（= 饱和）。pilot 里前两者只占 1-2%，
     被我错报成"什么都动不了"。

状态分四类（按阶段一的定义）：
  A 存在明显最佳 expert   最优 − 次优 > tie_tol
  B 选谁差别不大          有正收益但最优 ≈ 次优
  C 只有一个 expert 有效   恰好一个 expert 正收益
  D 全部无即时收益
ML 只可能在 A 和 C 上产生价值。

用法:
  python scripts/analyze_counterfactuals.py --cf results_v2/counterfactuals/cf_sbt_first_200mol.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import RDLogger                  # noqa: E402

from src.chem import murcko_scaffold        # noqa: E402
from src.interfaces import EXPERTS          # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)


def load(path: Path):
    S = json.loads(path.read_text())
    D = np.array([[s["actions"][e]["d_hv"] for e in EXPERTS] for s in S])
    EL = np.array([[s["actions"][e]["n_eligible"] for e in EXPERTS] for s in S])
    NV = np.array([[s["actions"][e]["n_valid"] for e in EXPERTS] for s in S])
    step = np.array([s["step"] for s in S])
    beh = np.array([s["behavior"] for s in S])
    seed = np.array([s["seed_smiles"] for s in S])
    return S, D, EL, NV, step, beh, seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="results_v2/counterfactuals/cf_sbt_first_200mol.json")
    ap.add_argument("--tie-tol", type=float, default=2e-4,
                    help="平局阈值；默认 2e-4 = pilot 测到的生成噪声 p90 的一半")
    ap.add_argument("--out", default="docs/counterfactual_analysis.md")
    args = ap.parse_args()

    S, D, EL, NV, step, beh, seed = load(ROOT / args.cf)
    N = len(S)
    TOL = args.tie_tol
    L = []                                   # markdown 行
    def p(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        L.append(line)

    meta = json.loads((ROOT / args.cf).with_name(
        Path(args.cf).stem + "_meta.json").read_text())
    p(f"# 反事实状态分析（{Path(args.cf).name}）\n")
    p(f"- pool `{meta['pool']}`，editor `{meta['env_kw']}`")
    p(f"- {meta['n_mol']} 个起始分子 x {len(meta['behaviors'])} 行为策略 "
      f"x {len(meta['states'])} 个状态 = **{N} 个状态**，"
      f"{meta['n_counterfactual_probe']} 次反事实探查")
    p(f"- 逻辑 oracle 打分 {meta['oracle_logical_calls']}，"
      f"缓存命中 {meta['oracle_cache_hits']}，墙钟 {meta['wall_seconds'] / 60:.1f} 分钟")
    p(f"- 平局阈值 `tie_tol = {TOL:g}`（pilot 实测生成噪声 p90 的一半）\n")

    # ---------- 状态分类 ----------
    pos = (D > 0).sum(1)
    srt = np.sort(D, 1)
    gap = srt[:, -1] - srt[:, -2]
    absorb = (NV.sum(1) == 0)
    no_cand = (EL.sum(1) == 0)
    cls = np.full(N, "B", dtype="<U1")
    cls[pos == 0] = "D"
    cls[(pos == 1)] = "C"
    cls[(pos >= 2) & (gap > TOL)] = "A"
    p("## 1. 状态分类\n")
    p("| 类型 | 定义 | 数量 | 占比 |")
    p("|---|---|---|---|")
    for c, desc in [("A", "有正收益且最优−次优 > tie_tol（**ML 有价值**）"),
                    ("B", "有多个正收益但最优≈次优（选谁都差不多）"),
                    ("C", "恰好一个 expert 有正收益（**ML 有价值**）"),
                    ("D", "全部无即时 HV 收益")]:
        m = cls == c
        p(f"| {c} | {desc} | {m.sum()} | {m.mean():.1%} |")
    useful = np.isin(cls, ["A", "C"])
    p(f"\n**A+C = {useful.sum()} 个状态（{useful.mean():.1%}）是 ML 可能产生价值的。**\n")

    p("### D 类必须再拆开（pilot 阶段我把这三类混报成『全零』）\n")
    p("| step | 真吸收态(无合法分子) | 无合格候选 | 有候选但无 HV 收益(饱和) | 有收益 | n |")
    p("|---|---|---|---|---|---|")
    for t in sorted(set(step.tolist())):
        m = step == t
        sat = m & (pos == 0) & ~no_cand
        p(f"| {t} | {(m & absorb).sum() / m.sum():.1%} | "
          f"{(m & no_cand & ~absorb).sum() / m.sum():.1%} | "
          f"{sat.sum() / m.sum():.1%} | {(m & (pos > 0)).sum() / m.sum():.1%} | {m.sum()} |")
    sat_all = (pos == 0) & ~no_cand
    p(f"| **全体** | {absorb.mean():.1%} | {(no_cand & ~absorb).mean():.1%} | "
      f"{sat_all.mean():.1%} | {(pos > 0).mean():.1%} | {N} |\n")

    # ---------- 最佳 expert 分布与阶段迁移 ----------
    p("## 2. 最佳 expert 分布与阶段迁移（只用干净状态 A+C）\n")
    p("| step | " + " | ".join(EXPERTS) + " | n |")
    p("|---|" + "---|" * (NE + 1))
    for t in sorted(set(step.tolist())):
        m = useful & (step == t)
        c = np.bincount(D[m].argmax(1), minlength=NE)
        tot = max(m.sum(), 1)
        p(f"| {t} | " + " | ".join(f"{v} ({v / tot:.0%})" for v in c) +
          f" | {m.sum()} |")
    c = np.bincount(D[useful].argmax(1), minlength=NE)
    p(f"| **全体** | " + " | ".join(f"{v} ({v / useful.sum():.0%})" for v in c) +
      f" | {useful.sum()} |\n")

    p("对照：若按 pilot 那样用全部状态的 `argmax`（平局/全零落到下标 0 = "
      f"{EXPERTS[0]}）\n")
    p("| step | " + " | ".join(EXPERTS) + " |")
    p("|---|" + "---|" * NE)
    for t in sorted(set(step.tolist())):
        m = step == t
        c2 = np.bincount(D[m].argmax(1), minlength=NE)
        p(f"| {t} | " + " | ".join(str(v) for v in c2) + " |")
    p("")

    # ---------- 切换率 ----------
    p("## 3. 最佳 expert 切换率\n")
    g = defaultdict(list)
    for i in range(N):
        g[(seed[i], beh[i])].append(i)
    sw, nst = [], []
    for k, ix in g.items():
        b = [int(D[i].argmax()) for i in sorted(ix, key=lambda i: step[i])
             if useful[i]]
        if len(b) >= 2:
            sw.append(len(set(b)) > 1)
            nst.append(len(set(b)))
    p(f"- 轨迹内（只算干净状态，至少 2 个）最佳 expert 发生切换: "
      f"**{np.mean(sw):.0%}** ({sum(sw)}/{len(sw)} 条轨迹)")
    p(f"- 一条轨迹内出现过的不同最佳 expert 个数: 均值 {np.mean(nst):.2f}\n")

    # ---------- 收益差距 ----------
    p("## 4. 选错 expert 损失多少 HV\n")
    p("| 状态子集 | n | 最优 | 次优 | 平均 | 最差 | 最优−平均 | 最优−次优 |")
    p("|---|---|---|---|---|---|---|---|")
    for nm, m in [("A+C（干净）", useful), ("A", cls == "A"), ("C", cls == "C"),
                  ("B", cls == "B"), ("全部有收益", pos > 0)]:
        if m.sum() == 0:
            continue
        s = np.sort(D[m], 1)
        p(f"| {nm} | {m.sum()} | {s[:, -1].mean():.6f} | {s[:, -2].mean():.6f} | "
          f"{D[m].mean():.6f} | {s[:, 0].mean():.6f} | "
          f"{(s[:, -1] - D[m].mean(1)).mean():.6f} | "
          f"{(s[:, -1] - s[:, -2]).mean():.6f} |")
    p("")
    p("随机选 vs 最优选的即时 regret（对全部 3000 个状态，含 D 类）:")
    p(f"- 随机: {(D.max(1) - D.mean(1)).mean():.6f}")
    for j, e in enumerate(EXPERTS):
        p(f"- 固定 {e}: {(D.max(1) - D[:, j]).mean():.6f}")
    p("")

    # ---------- expert 可用性 ----------
    p("## 5. expert 可用性（产出合格候选的比例）\n")
    p("| step | " + " | ".join(EXPERTS) + " |")
    p("|---|" + "---|" * NE)
    for t in sorted(set(step.tolist())):
        m = step == t
        p(f"| {t} | " + " | ".join(f"{(EL[m, j] > 0).mean():.0%}" for j in range(NE)) + " |")
    p(f"| **全体** | " + " | ".join(f"{(EL[:, j] > 0).mean():.0%}" for j in range(NE)) + " |")
    p(f"\n正收益比例: " + "，".join(f"{e} {(D[:, j] > 0).mean():.0%}"
                                 for j, e in enumerate(EXPERTS)) + "\n")

    # ---------- 行为策略影响 ----------
    p("## 6. 行为策略之间是否一致\n")
    p("| 行为策略 | n | A+C 占比 | D 占比 | 最佳 expert 分布 | 切换率 |")
    p("|---|---|---|---|---|---|")
    for b in sorted(set(beh.tolist())):
        m = beh == b
        c = np.bincount(D[m & useful].argmax(1), minlength=NE)
        sb = [len({int(D[i].argmax()) for i in ix if useful[i]}) > 1
              for k, ix in g.items() if k[1] == b
              and sum(useful[i] for i in ix) >= 2]
        p(f"| {b} | {m.sum()} | {useful[m].mean():.0%} | {(cls[m] == 'D').mean():.0%} | "
          + "/".join(str(v) for v in c) + f" | {np.mean(sb):.0%} |")
    p("")

    # ---------- 分子结构相关性 / 分组 ----------
    p("## 7. 决策信息是分子级还是状态级\n")
    scaf = {s: murcko_scaffold(s) for s in set(seed.tolist())}
    nscaf = len({v for v in scaf.values() if v})
    p(f"- {len(set(seed.tolist()))} 个起始分子 / {nscaf} 个 Murcko scaffold")
    p(f"- **3000 个状态只有 {len(set(seed.tolist()))} 个独立分子**，"
      f"交叉验证必须按分子/scaffold 分组，不能随机拆状态\n")
    # 同一分子内的最佳 expert 是否一致 -> 分子级 vs 状态级
    same = []
    for m_ in set(seed.tolist()):
        ix = [i for i in range(N) if seed[i] == m_ and useful[i]]
        if len(ix) >= 2:
            same.append(len({int(D[i].argmax()) for i in ix}) == 1)
    p(f"- 同一起始分子的全部干净状态都指向同一个最佳 expert 的比例: "
      f"**{np.mean(same):.0%}**（{sum(same)}/{len(same)} 个分子）")
    p(f"  - 越低说明决策信息越偏**状态级**（Static Selector 吃不到）、"
      f"越高说明越偏**分子级**（静态选择就够）\n")

    # ---------- 可学习空间的定量判断 ----------
    p("## 8. 可学习空间的定量判断\n")
    rnd = (D.max(1) - D.mean(1)).mean()
    best_fixed = min((D.max(1) - D[:, j]).mean() for j in range(NE))
    # 只看 step 的上界：每个 step 用该 step 的最优 expert
    stp = 0.0
    for t in sorted(set(step.tolist())):
        m = step == t
        j = int(np.argmax(D[m].sum(0)))
        stp += (D[m].max(1) - D[m, j]).sum()
    stp /= N
    p("| 决策规则 | 平均即时 regret | 相对随机消除的 regret |")
    p("|---|---|---|")
    for nm, v in [("随机", rnd), ("最好的固定 expert", best_fixed),
                  ("只看 step 的查表", stp), ("oracle（上界）", 0.0)]:
        p(f"| {nm} | {v:.6f} | {(rnd - v) / rnd:.0%} |")
    p(f"\n- 一个只看 step 的查表策略就能消除 {(rnd - stp) / rnd:.0%} 的 regret；"
      f"**ML 必须超过这条线才算用到了分子/状态信息**")
    p(f"- 剩余可争取的空间 = {stp:.6f}/步（相对 oracle）"
      f"，20 步累计上界约 {20 * stp:.5f} HV\n")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text("\n".join(L) + "\n")
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
