"""Step 5+6：Dynamic Router 的模型比较与特征消融（离线，不跑完整轨迹）。

数据：results_v2/counterfactuals/cf_sbt_first_200mol.json 的 3000 个状态，
每个状态含四个 expert 从**逐字相同**的 (current, archive, seen, t, 历史)
出发各走一步的真实 ΔHV。

硬约束（来自评估审计）：
  - 按 scaffold 分组切 fold（src/cv.py），所有模型、所有特征组共用同一套
  - 标准化、超参选择只用训练 fold；Ridge 的 alpha 在训练 fold 内再做
    **分组** CV（不能用 RidgeCV 的默认行级 CV）
  - 「最好的固定 expert」「只看 step 的查表」两条参照线同样 fold 内确定
  - Top-1 只在严格非平局状态上算（平局没有唯一正确答案）
  - 完整 applicability 只作为高成本消融项（实测 8.94 ms/expert ≈ 一次
    propose；Router 每步付 4 次 = OneStepOracle 的探查成本，没有部署意义）

特征组（Step 6）：
  R1 step + 预算
  R2 R1 + 当前四项性质
  R3 R2 + 分子结构描述符 + Morgan fingerprint
  R4 R3 + expert 历史（调用次数/累计收益/上次是否失败）+ Archive 摘要
  R5 R4 + template_sites            <- 默认部署特征
  R6 R5 + 完整 applicability         <- 高成本消融

用法:
  python scripts/train_dynamic_router.py --models ridge,rf,gb,twostage
  python scripts/train_dynamic_router.py --ablation-only
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                                   # noqa: E402
from sklearn.ensemble import (GradientBoostingRegressor,     # noqa: E402
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge   # noqa: E402
from sklearn.model_selection import GroupKFold               # noqa: E402
from sklearn.neural_network import MLPRegressor              # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from src.cv import grouped_folds, scaffold_of                # noqa: E402
from src.interfaces import EXPERTS                           # noqa: E402
from src.router_features import FEATURE_GROUPS, build_states  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
ALPHAS = np.logspace(-3, 3, 13)


# --------------------------------------------------------------------------
def pick_alpha(A, y, groups):
    ug = np.unique(groups)
    k = min(5, len(ug))
    if k < 2:
        return 1.0
    gkf = GroupKFold(n_splits=k)
    err = np.zeros(len(ALPHAS))
    for tr, va in gkf.split(A, y, groups):
        for i, al in enumerate(ALPHAS):
            pr = Ridge(alpha=al, solver="svd").fit(A[tr], y[tr]).predict(A[va])
            err[i] += float(np.sum((pr - y[va]) ** 2))
    return float(ALPHAS[int(np.argmin(err))])


def make_model(kind, seed):
    """超参固定、规模匹配 200 个独立分子 —— 不做无边界网格搜索。"""
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=8,
                                     min_samples_leaf=5, random_state=seed,
                                     n_jobs=1)
    if kind == "gb":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05,
                                         min_samples_leaf=5, random_state=seed)
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                            max_iter=600, early_stopping=True,
                            random_state=seed)
    raise ValueError(kind)


def fit_rank(Xtr, Dtr, Xte, seed, weighted=True):
    """收益加权 pairwise ranking。

    任务不是把每个数值预测准，而是选对 expert —— 而且选错的代价差别
    很大：错误分析显示损失最大的 1% 状态贡献了 31.9% 的总损失，模型在
    真错状态上把 oracle 选中的那个低估 81%（均值 -0.00245）。普通回归
    把"损失 0.00001 的错"和"损失 0.01 的错"同等看待，正是向均值回归的
    来源。

    这里对每个 expert 对 (a,b) 训一个二分类器判"a 是否优于 b"，
    **样本权重 = |ΔHV_a - ΔHV_b|** —— 收益差距越大，选错惩罚越重。
    打分用 Borda（各对胜率之和）。
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    score = np.zeros((len(B), NE))
    for a in range(NE):
        for b in range(a + 1, NE):
            diff = Dtr[:, a] - Dtr[:, b]
            lab = (diff > 0).astype(int)
            if lab.min() == lab.max():
                score[:, a if lab[0] == 1 else b] += 1.0
                continue
            w = np.abs(diff) if weighted else None
            if w is not None:
                w = w / (w.mean() + 1e-12)
                # 完全平局的样本权重为 0 会让部分分类器拿不到信号，给个地板
                w = np.maximum(w, 1e-3)
            mdl = LogisticRegression(C=1.0, max_iter=3000)
            mdl.fit(A, lab, sample_weight=w)
            p = mdl.predict_proba(B)[:, 1]
            score[:, a] += p
            score[:, b] += 1.0 - p
    return score


def fit_fold(kind, Xtr, Dtr, Xte, groups, seed):
    """返回 (m_te, 4) 的预测打分，以及拟合用掉的秒数。"""
    t0 = time.perf_counter()
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = np.zeros((len(B), NE))
    for j in range(NE):
        y = Dtr[:, j]
        if kind == "ridge":
            out[:, j] = Ridge(alpha=pick_alpha(A, y, groups), solver="svd").fit(A, y).predict(B)
        elif kind in ("rf_w", "gb_w"):
            # 把"按损失大小加权"装回回归里：样本权重 = 该状态的赌注
            # (max - mean 的真实收益)。ranking 验证了加权有效（加权 42%
            # vs 不加权 35%），但 Borda 丢掉量级信息又输给回归（50%）。
            # 这个变体两者都保留。
            stake = Dtr.max(1) - Dtr.mean(1)
            w = np.maximum(stake / (stake.mean() + 1e-12), 1e-3)
            out[:, j] = make_model(kind[:-2], seed).fit(
                A, y, sample_weight=w).predict(B)
        elif kind in ("rank", "rank_unw"):
            if j == 0:      # ranking 一次给出全部 4 列
                out[:] = fit_rank(Xtr, Dtr, Xte, seed,
                                  weighted=(kind == "rank"))
            continue
        elif kind == "twostage":
            lab = (y > 0).astype(int)
            if lab.min() == lab.max():
                p = np.full(len(B), float(lab[0]))
            else:
                p = RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=5,
                    random_state=seed, n_jobs=1).fit(A, lab).predict_proba(B)[:, 1]
            m = lab == 1
            if m.sum() >= 20:
                q = make_model("rf", seed).fit(A[m], y[m]).predict(B)
            else:
                q = np.full(len(B), float(y[m].mean()) if m.sum() else 0.0)
            out[:, j] = p * np.maximum(q, 0.0)
        else:
            out[:, j] = make_model(kind, seed).fit(A, y).predict(B)
    return out, time.perf_counter() - t0


def evaluate(D, pick, strict, oracle_ref, rand_ref):
    reg = D.max(1) - D[np.arange(len(D)), pick]
    return {
        "mean_regret": float(reg.mean()),
        "regret_reduction": float((rand_ref - reg.mean()) / rand_ref),
        "top1_strict": float(np.mean(pick[strict] == D[strict].argmax(1))),
        "positive_gain_rate": float(np.mean(D[np.arange(len(D)), pick] > 0)),
        "oracle_gap_recovery": float(1 - reg.mean() / rand_ref)
        if rand_ref > 0 else float("nan"),
        "mae": float(np.nan),     # 由调用方填
    }


def refs_per_fold(D, step, fold, n_folds):
    """fold 内确定的两条参照线。"""
    N = len(D)
    fix, stp = np.zeros(N), np.zeros(N)
    picks_fix, picks_step = np.zeros(N, int), np.zeros(N, int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        bj = int(np.argmin([(D[tr].max(1) - D[tr][:, j]).mean() for j in range(NE)]))
        picks_fix[te] = bj
        fix[te] = D[te].max(1) - D[te][:, bj]
        tbl = {}
        for t in np.unique(step):
            m = tr & (step == t)
            tbl[t] = int(np.argmax(D[m].sum(0))) if m.sum() else bj
        pk = np.array([tbl[t] for t in step[te]])
        picks_step[te] = pk
        stp[te] = D[te].max(1) - D[te][np.arange(te.sum()), pk]
    return fix.mean(), stp.mean(), picks_fix, picks_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="results_v2/counterfactuals/cf_sbt_first_200mol.json")
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="ridge,rf,gb,twostage")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tie-tol", type=float, default=2e-4)
    ap.add_argument("--groups", default="R1,R2,R3,R4,R5,R6")
    ap.add_argument("--main-group", default="R5")
    ap.add_argument("--outdir", default="results_v2/dynamic_router")
    args = ap.parse_args()

    od = ROOT / args.outdir
    (od / "models").mkdir(parents=True, exist_ok=True)
    (od / "logs").mkdir(parents=True, exist_ok=True)

    print("提取特征 ...", flush=True)
    S, blocks, D, step, seed_smi, costs = build_states(ROOT / args.cf, args.pool)
    N = len(D)
    SCAF = scaffold_of(seed_smi)
    FOLD = grouped_folds(SCAF, args.folds, seed=0)
    srt = np.sort(D, 1)
    strict = ((D > 0).sum(1) >= 1) & ((srt[:, -1] - srt[:, -2]) > args.tie_tol)
    rand_ref = float((D.max(1) - D.mean(1)).mean())
    fix_ref, stp_ref, _, pk_step = refs_per_fold(D, step, FOLD, args.folds)

    print(f"{N} 个状态 / {len(np.unique(SCAF))} 个 scaffold / {args.folds} folds")
    print(f"严格非平局状态 {strict.sum()}")
    print(f"参照线: 随机 {rand_ref:.6f} | 最好固定 expert {fix_ref:.6f} "
          f"({(rand_ref - fix_ref) / rand_ref:.0%}) | "
          f"只看 step {stp_ref:.6f} ({(rand_ref - stp_ref) / rand_ref:.0%})\n")
    print("特征组维度与提取成本:")
    for g in args.groups.split(","):
        X = np.concatenate([blocks[b] for b in FEATURE_GROUPS[g]], axis=1)
        c = sum(costs[b] for b in FEATURE_GROUPS[g])
        print(f"  {g}: {X.shape[1]:>4} 维   {c:>7.2f} ms/状态   "
              f"({'+'.join(FEATURE_GROUPS[g])})")
    print()

    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for g in args.groups.split(","):
        X = np.concatenate([blocks[b] for b in FEATURE_GROUPS[g]], axis=1)
        cost = sum(costs[b] for b in FEATURE_GROUPS[g])
        for mk in args.models.split(","):
            use = seeds if mk in ("rf", "gb", "twostage", "mlp") else [0]
            accs = []
            fit_s, inf_s = 0.0, 0.0
            for sd in use:
                pick = np.zeros(N, int)
                mae = 0.0
                for f in range(args.folds):
                    tr, te = FOLD != f, FOLD == f
                    P, ft = fit_fold(mk, X[tr], D[tr], X[te], SCAF[tr], sd)
                    fit_s += ft
                    t1 = time.perf_counter()
                    pick[te] = P.argmax(1)
                    inf_s += time.perf_counter() - t1
                    mae += float(np.abs(P - D[te]).mean()) * te.sum()
                m = evaluate(D, pick, strict, 0.0, rand_ref)
                m["mae"] = mae / N
                accs.append(m)
            agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
            sd_ = {k + "_sd": float(np.std([a[k] for a in accs]))
                   for k in accs[0]} if len(accs) > 1 else {}
            rec = {"feature_group": g, "n_feat": int(X.shape[1]),
                   "feat_cost_ms": cost, "model": mk, "n_seed": len(use),
                   **agg, **sd_,
                   "fit_seconds": fit_s / len(use),
                   "infer_ms_per_state": 1000 * inf_s / len(use) / N,
                   "beats_step_only": int(agg["mean_regret"] < stp_ref)}
            rows.append(rec)
            print(f"  {g}/{mk:<9} regret {agg['mean_regret']:.6f} "
                  f"消除 {agg['regret_reduction']:>4.0%}  "
                  f"Top-1 {agg['top1_strict']:>5.1%}  "
                  f"正收益 {agg['positive_gain_rate']:>4.0%}  "
                  f"MAE {agg['mae']:.6f}  "
                  f"{'跨过 StepOnly' if rec['beats_step_only'] else '未跨过'}",
                  flush=True)

    # 参照行也写进 CSV，便于直接读
    for nm, v, pk in [("[ref] random", rand_ref, None),
                      ("[ref] best_fixed_expert", fix_ref, None),
                      ("[ref] step_only", stp_ref, pk_step)]:
        r = {"feature_group": "-", "n_feat": 0, "feat_cost_ms": 0.0,
             "model": nm, "n_seed": 0, "mean_regret": v,
             "regret_reduction": (rand_ref - v) / rand_ref,
             "top1_strict": float(np.mean(pk[strict] == D[strict].argmax(1)))
             if pk is not None else float("nan"),
             "positive_gain_rate": float(np.mean(
                 D[np.arange(N), pk] > 0)) if pk is not None else float("nan"),
             "oracle_gap_recovery": 1 - v / rand_ref, "mae": float("nan"),
             "fit_seconds": 0.0, "infer_ms_per_state": 0.0,
             "beats_step_only": 0}
        rows.append(r)

    keys = ["feature_group", "n_feat", "feat_cost_ms", "model", "n_seed",
            "mean_regret", "mean_regret_sd", "regret_reduction",
            "top1_strict", "top1_strict_sd", "positive_gain_rate",
            "oracle_gap_recovery", "mae", "fit_seconds",
            "infer_ms_per_state", "beats_step_only"]
    cmp_csv = od / "model_comparison.csv"
    with cmp_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n-> {cmp_csv}")

    # ---- 消融：主特征组逐块移除 ----
    main_g = args.main_group
    best_mk = min((r for r in rows if r["feature_group"] == main_g
                   and not r["model"].startswith("[ref]")),
                  key=lambda r: r["mean_regret"])["model"]
    print(f"\n消融（主组 {main_g}，模型 {best_mk}，逐块移除）")
    abl = []
    full_blocks = FEATURE_GROUPS[main_g]
    for drop in [None] + list(full_blocks):
        use_b = [b for b in full_blocks if b != drop]
        if not use_b:
            continue
        X = np.concatenate([blocks[b] for b in use_b], axis=1)
        pick = np.zeros(N, int)
        for f in range(args.folds):
            tr, te = FOLD != f, FOLD == f
            P, _ = fit_fold(best_mk, X[tr], D[tr], X[te], SCAF[tr], 0)
            pick[te] = P.argmax(1)
        m = evaluate(D, pick, strict, 0.0, rand_ref)
        abl.append({"removed": drop or "(none)", "n_feat": int(X.shape[1]),
                    "feat_cost_ms": sum(costs[b] for b in use_b), **m})
        print(f"  移除 {str(drop or '(无)'):<14} {X.shape[1]:>4} 维  "
              f"regret {m['mean_regret']:.6f}  消除 {m['regret_reduction']:>4.0%}  "
              f"Top-1 {m['top1_strict']:>5.1%}", flush=True)
    abl_csv = od / "ablation_results.csv"
    with abl_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abl[0]))
        w.writeheader()
        w.writerows(abl)
    print(f"-> {abl_csv}")

    # ---- 保存最终模型（全数据重训，供 Step 7/8 部署）----
    X = np.concatenate([blocks[b] for b in FEATURE_GROUPS[main_g]], axis=1)
    sc = StandardScaler().fit(X)
    A = sc.transform(X)
    mdls = []
    if best_mk in ("rank", "rank_unw"):
        # ranking 不是 per-expert 模型，而是 6 个 expert 对的分类器；
        # 单独存成一个条目，DynamicRouter 按 kind 分派。
        from sklearn.linear_model import LogisticRegression as _LR
        pair = {}
        for a in range(NE):
            for b in range(a + 1, NE):
                diff = D[:, a] - D[:, b]
                lab = (diff > 0).astype(int)
                if lab.min() == lab.max():
                    pair[(a, b)] = ("const", float(lab[0]))
                    continue
                w = np.abs(diff)
                w = np.maximum(w / (w.mean() + 1e-12), 1e-3) \
                    if best_mk == "rank" else None
                m = _LR(C=1.0, max_iter=3000)
                m.fit(A, lab, sample_weight=w)
                pair[(a, b)] = ("model", m)
        mdls = [("rank", pair)]
    for j in (range(NE) if best_mk not in ("rank", "rank_unw") else []):
        y = D[:, j]
        if best_mk == "ridge":
            mdls.append(Ridge(alpha=pick_alpha(A, y, SCAF), solver="svd").fit(A, y))
        elif best_mk == "twostage":
            lab = (y > 0).astype(int)
            clf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                         min_samples_leaf=5, random_state=0,
                                         n_jobs=1).fit(A, lab) \
                if lab.min() != lab.max() else None
            m = lab == 1
            reg = make_model("rf", 0).fit(A[m], y[m]) if m.sum() >= 20 else None
            mdls.append(("twostage", clf, reg,
                         float(y[m].mean()) if m.sum() else 0.0,
                         float(lab[0]) if lab.min() == lab.max() else None))
        else:
            mdls.append(make_model(best_mk, 0).fit(A, y))
    bundle = {"kind": best_mk, "feature_group": main_g,
              "blocks": list(FEATURE_GROUPS[main_g]),
              "scaler": sc, "models": mdls, "experts": list(EXPERTS),
              "pool": args.pool, "cf": args.cf, "folds": args.folds,
              "tie_tol": args.tie_tol, "trained_on_states": int(N),
              "feat_cost_ms": sum(costs[b] for b in FEATURE_GROUPS[main_g])}
    mp = od / "models" / f"router_{main_g}_{best_mk}.pkl"
    mp.write_bytes(pickle.dumps(bundle))
    (od / "models" / "router_meta.json").write_text(json.dumps({
        "kind": best_mk, "feature_group": main_g,
        "blocks": list(FEATURE_GROUPS[main_g]),
        "n_feat": int(X.shape[1]),
        "feat_cost_ms": bundle["feat_cost_ms"],
        "refs": {"random": rand_ref, "best_fixed": fix_ref, "step_only": stp_ref},
        "path": str(mp.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    print(f"-> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
