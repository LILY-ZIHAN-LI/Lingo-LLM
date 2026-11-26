"""Step 2：Dynamic Router 在 3000 个反事实状态上到底选错了什么。

模型已经消除约 50% 的随机决策损失，剩下那一半集中在哪里？把 Router 的
选择和 oracle 的选择逐状态对比，按错误类型归类，并按**损失大小**加权 ——
一个损失 0.00001 的"错"和一个损失 0.01 的错不是一回事。

错误分类（互斥，按优先级判定）：
  no_gain_possible   四个 expert 的真实收益都 <= 0，选谁都一样 -> 不算错
  tie                选中的收益与最优之差 <= tie_tol -> 实质无损
  picked_dead        选中的 expert 产出 0 个合格候选（缺适用性信息？）
  picked_zero        有合格候选但真实收益 <= 0（缺"修改质量"信息？）
  picked_low         有正收益但不是最优（缺结构/专家匹配信息？）

预测侧再分两类，用来区分"没预测到高收益"和"高估了低收益"：
  missed_high   oracle 选中的那个 expert 被模型显著低估
  overrated     模型选中的那个 expert 被显著高估

评估用**与训练相同的分组 CV 的留出预测**，不是在训练数据上自评。

用法:
  python scripts/router_error_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                              # noqa: E402

from src.cv import grouped_folds, scaffold_of           # noqa: E402
from src.interfaces import EXPERTS                      # noqa: E402
from src.router_features import FEATURE_GROUPS, build_states  # noqa: E402
from train_dynamic_router import fit_fold, refs_per_fold       # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="results_v2/counterfactuals/cf_sbt_first_200mol.json")
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--group", default="R5")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tie-tol", type=float, default=2e-4)
    ap.add_argument("--out", default="results_v2/dynamic_router/router_error_analysis.csv")
    args = ap.parse_args()

    print("提取特征 ...", flush=True)
    S, blocks, D, step, seed_smi, costs = build_states(ROOT / args.cf, args.pool)
    X = np.concatenate([blocks[b] for b in FEATURE_GROUPS[args.group]], axis=1)
    N = len(D)
    EL = np.array([[s["actions"][e]["n_eligible"] for e in EXPERTS] for s in S])
    beh = np.array([s["behavior"] for s in S])
    SCAF = scaffold_of(seed_smi)
    FOLD = grouped_folds(SCAF, args.folds, seed=0)

    print(f"留出预测（{args.group}/{args.model}，{args.folds}-fold 按 scaffold 分组）...",
          flush=True)
    P = np.zeros((N, NE))
    for f in range(args.folds):
        tr, te = FOLD != f, FOLD == f
        P[te], _ = fit_fold(args.model, X[tr], D[tr], X[te], SCAF[tr], 0)
    pick = P.argmax(1)
    idx = np.arange(N)
    best = D.max(1)
    got = D[idx, pick]
    reg = best - got
    total = float(reg.sum())

    # ---- 错误分类 ----
    cls = np.full(N, "", dtype="<U18")
    cls[best <= 0] = "no_gain_possible"
    rest = cls == ""
    cls[rest & (reg <= args.tie_tol)] = "tie"
    rest = cls == ""
    cls[rest & (EL[idx, pick] == 0)] = "picked_dead"
    rest = cls == ""
    cls[rest & (got <= 0)] = "picked_zero"
    cls[cls == ""] = "picked_low"

    print(f"\n{N} 个状态，总 regret {total:.4f}（均值 {reg.mean():.6f}）\n")
    print(f"{'错误类型':<20}{'状态数':>7}{'占比':>7}{'总 regret':>11}"
          f"{'占总损失':>9}{'均 regret':>11}")
    print("-" * 68)
    rows = []
    order = ["no_gain_possible", "tie", "picked_dead", "picked_zero", "picked_low"]
    for c in order:
        m = cls == c
        if not m.any():
            continue
        s = float(reg[m].sum())
        print(f"{c:<20}{m.sum():>7}{m.mean():>7.1%}{s:>11.5f}"
              f"{s / total if total else 0:>9.1%}{reg[m].mean():>11.6f}")
        rows.append({"kind": "error_class", "key": c, "n": int(m.sum()),
                     "share_states": float(m.mean()), "regret_sum": s,
                     "share_regret": float(s / total) if total else 0.0,
                     "regret_mean": float(reg[m].mean())})

    # ---- 损失集中在少数状态吗 ----
    o = np.argsort(-reg)
    print()
    for k in (0.01, 0.05, 0.10, 0.25):
        n = max(int(k * N), 1)
        s = float(reg[o[:n]].sum())
        print(f"  损失最大的 {k:>4.0%} 状态（n={n:>4}）贡献了总损失的 {s / total:>5.1%}")
        rows.append({"kind": "concentration", "key": f"top_{int(100 * k)}pct",
                     "n": n, "share_regret": float(s / total)})

    # ---- 预测侧：低估 vs 高估 ----
    bidx = D.argmax(1)
    miss = (cls == "picked_low") | (cls == "picked_zero") | (cls == "picked_dead")
    under = P[idx, bidx] - D[idx, bidx]        # oracle 选的那个被低估多少（负=低估）
    over = P[idx, pick] - D[idx, pick]         # 模型选的那个被高估多少（正=高估）
    print(f"\n在 {miss.sum()} 个真错状态上：")
    print(f"  oracle 选中的 expert 被低估 {np.mean(under[miss] < 0):.0%}，"
          f"平均偏差 {under[miss].mean():+.6f}")
    print(f"  模型选中的 expert 被高估 {np.mean(over[miss] > 0):.0%}，"
          f"平均偏差 {over[miss].mean():+.6f}")
    rows += [
        {"kind": "prediction", "key": "missed_high_rate",
         "share_states": float(np.mean(under[miss] < 0)),
         "regret_mean": float(under[miss].mean())},
        {"kind": "prediction", "key": "overrated_rate",
         "share_states": float(np.mean(over[miss] > 0)),
         "regret_mean": float(over[miss].mean())},
    ]

    # ---- 按阶段 / 按行为策略 ----
    print(f"\n{'step':<8}{'状态':>6}{'均 regret':>11}{'Top-1':>8}"
          f"{'无可得收益':>10}{'选到死 expert':>13}")
    print("-" * 58)
    srt = np.sort(D, 1)
    strict = (best > 0) & ((srt[:, -1] - srt[:, -2]) > args.tie_tol)
    for t in sorted(set(step.tolist())):
        m = step == t
        ms = m & strict
        print(f"{t:<8}{m.sum():>6}{reg[m].mean():>11.6f}"
              f"{np.mean(pick[ms] == bidx[ms]) if ms.any() else float('nan'):>8.1%}"
              f"{np.mean(best[m] <= 0):>10.0%}"
              f"{np.mean(EL[m][np.arange(m.sum()), pick[m]] == 0):>13.0%}")
        rows.append({"kind": "by_step", "key": str(t), "n": int(m.sum()),
                     "regret_mean": float(reg[m].mean()),
                     "share_states": float(np.mean(pick[ms] == bidx[ms]))
                     if ms.any() else float("nan")})

    print(f"\n{'行为策略':<14}{'状态':>6}{'均 regret':>11}{'Top-1(严格)':>12}")
    print("-" * 44)
    for b in sorted(set(beh.tolist())):
        m = beh == b
        ms = m & strict
        print(f"{b:<14}{m.sum():>6}{reg[m].mean():>11.6f}"
              f"{np.mean(pick[ms] == bidx[ms]):>12.1%}")
        rows.append({"kind": "by_behavior", "key": b, "n": int(m.sum()),
                     "regret_mean": float(reg[m].mean()),
                     "share_states": float(np.mean(pick[ms] == bidx[ms]))})

    # ---- 选中/应选 的 expert 混淆 ----
    print(f"\n混淆（行=oracle 最优，列=模型选中），只用严格非平局状态")
    print(f"{'':<8}" + "".join(f"{e:>9}" for e in EXPERTS))
    for i, e in enumerate(EXPERTS):
        m = strict & (bidx == i)
        c = np.bincount(pick[m], minlength=NE)
        print(f"{e:<8}" + "".join(f"{v:>9}" for v in c))
        for j, v in enumerate(c):
            rows.append({"kind": "confusion", "key": f"{e}->{EXPERTS[j]}",
                         "n": int(v)})

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    keys = ["kind", "key", "n", "share_states", "regret_sum", "share_regret",
            "regret_mean"]
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
