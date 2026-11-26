"""EXP 1 正式版：Nested LOO + 强规则基线 + Two-head selector。

相对 run_loo_selector.py（探索版）的三处改动，对应任务 1 / 4 / 7：

**任务 1 —— 选择全部进内层。** 探索版在 50 个 fold 外面比较 9x5 个
(特征组, 模型) 组合，再把最好的那个当头条 —— 那是在留出数据上选模型。
这里改成：外层 LOO 留一个分子，内层对剩下 49 个做 K-fold，在内层上选
(特征组, 模型, 超参)，用选中的组合在留出分子上预测一次。置换检验包住
整条链（每次置换都重跑完整的内层选择），所以零分布含选择带来的乐观偏差。

**任务 4 —— 强规则。** 探索版的 ApplicabilityRule 是无参数的
argmax(产物数)，它在四个池上的方向会翻（-0.00127 ~ +0.00175），不是稳定
基线。这里给一族规则，参数（阈值、权重、expert 先验）只在训练 fold 内
拟合，规则族内部也在训练 fold 内选。

**任务 7 —— Two-head。** 实测"选中 expert 的预算达成率 78-88% vs SBS
50-62%"说明收益主要来自"挑一个真的能动的 expert"。所以把
P(攒够候选) 和 E[HV | 攒够] 分开预测再相乘，和直接回归 HV 对比。

指标按任务 2 的诊断和任务 10 的要求：ΔHV 的均值会被重尾主导
（2.20 上剔掉 2 个分子就从 +0.00353* 掉到 +0.00100），所以均值、
10% winsorized、中位数、赢面四个并列，固定调用预算与固定候选预算并列，
外加覆盖率和计算成本。

用法:
  python scripts/run_nested_selector.py --run-dir results_sbt_first --nperm 0
  python scripts/run_nested_selector.py --run-dir results_sbt_first --nperm 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold                     # noqa: E402
from scipy.stats import trim_mean                                   # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV        # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

from run_loo_selector import (EDITOR_KWARGS, boot_ci, fit_predict,  # noqa: E402
                              load_labels)
from src.editors import make_editors                                # noqa: E402
from src.features import build_matrix                               # noqa: E402
from src.interfaces import EXPERTS                                  # noqa: E402
from src.oracles import OracleBank                                  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)


# --------------------------------------------------------------------------
# 任务 4：强规则族。参数只能看 train fold。
# --------------------------------------------------------------------------
class RuleFamily:
    """一族非学习型规则，族内选择和参数拟合都只用训练 fold。

    每条规则给四个 expert 打分，argmax 即决策。全部只用决策前信息：
    模板适用性（产物数 / fire 条数 / 位点数）、分子描述符、heavy/light 分支，
    以及训练 fold 上各 expert 的先验强度。
    """

    # 规则名 -> 需要的列前缀
    RULES = ("prod", "fire", "sites", "prior", "prior_prod", "mw_gate")

    @staticmethod
    def supports(names: list[str]) -> bool:
        """这组特征里有没有规则需要的适用性列。

        F1（只有属性分数）和 F3（只有描述符）没有 `app_*` 列，构造
        RuleFamily 会在 names.index 上抛 ValueError。inner_select 里有
        try/except 兜住，但 outer_loo 最后那次拟合没有 —— 内层一旦选中
        (F1, rule)，整个 run 就静默死掉。所以调用方要先问这个。
        """
        return all(f"app_{e}_n_product" in names for e in EXPERTS)

    def __init__(self, names: list[str]):
        if not self.supports(names):
            raise ValueError(
                "这组特征没有 app_*_n_product 列，规则族不适用；"
                "调用前先用 RuleFamily.supports(names) 判断")
        self.n = names
        self.c_prod = [names.index(f"app_{e}_n_product") for e in EXPERTS]
        self.c_fire = [names.index(f"app_{e}_n_template_fire") for e in EXPERTS]
        self.c_site = [[i for i, nm in enumerate(names)
                        if nm.startswith(f"site_{e}_") and nm.endswith("total")]
                       for e in EXPERTS]
        self.i_mw = names.index("mw") if "mw" in names else None

    def _score(self, rule: str, X, prior, par):
        """X: (m,d) -> (m,4) 每个 expert 的规则分。"""
        prod = X[:, self.c_prod]
        if rule == "prod":
            return prod
        if rule == "fire":
            return X[:, self.c_fire]
        if rule == "sites":
            if not all(self.c_site):
                return prod
            return np.stack([X[:, c].sum(1) if c else np.zeros(len(X))
                             for c in self.c_site], 1)
        if rule == "prior":
            return np.tile(prior, (len(X), 1))
        if rule == "prior_prod":
            # expert 先验强度 + b * 适用性。b 在训练 fold 上网格选。
            return np.tile(prior, (len(X), 1)) + par * prod
        if rule == "mw_gate":
            # 重于阈值 theta 的分子交给"能缩"的 expert（产物数最多那个），
            # 否则按先验。theta 在训练 fold 上网格选。
            if self.i_mw is None:
                return np.tile(prior, (len(X), 1))
            heavy = (X[:, self.i_mw] > par)[:, None]
            return np.where(heavy, prod / (prod.max() + 1e-12) * prior.max(),
                            np.tile(prior, (len(X), 1)))
        raise ValueError(rule)

    def fit(self, Xtr, Ytr):
        """在训练 fold 上选规则和参数。"""
        prior = Ytr.mean(0)
        best, bs = None, -np.inf
        for rule in self.RULES:
            if rule == "prior_prod":
                grid = [0.0] + list(np.logspace(-5, -2, 8))
            elif rule == "mw_gate":
                grid = list(np.percentile(Xtr[:, self.i_mw], [25, 50, 75])) \
                    if self.i_mw is not None else [0.0]
            else:
                grid = [0.0]
            for par in grid:
                pick = self._score(rule, Xtr, prior, par).argmax(1)
                s = Ytr[np.arange(len(Ytr)), pick].mean()
                if s > bs:
                    best, bs = (rule, par, prior), s
        self.best = best
        return self

    def predict(self, Xte):
        rule, par, prior = self.best
        return self._score(rule, Xte, prior, par).argmax(1)


# --------------------------------------------------------------------------
# 任务 7：Two-head
# --------------------------------------------------------------------------
def two_head(Xtr, Ytr, Atr, Xte):
    """score_e = P(攒够候选 | 分子, e) * E[HV | 攒够]。

    Atr: (m,4) 训练分子上该 expert 的预算达成率（0 / 0.5 / 1）。
    两个 head 都只看决策前特征。达成率在训练 fold 内无变化时退回常数。
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = np.empty((len(Xte), NE))
    for j in range(NE):
        lab = (Atr[:, j] > 0.5).astype(int)
        if lab.min() == lab.max():
            p = np.full(len(Xte), float(lab[0]))
        else:
            p = LogisticRegression(C=1.0, max_iter=2000).fit(A, lab).predict_proba(B)[:, 1]
        m = lab == 1
        if m.sum() >= 5:
            h = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(A[m], Ytr[m, j]).predict(B)
        else:
            h = np.full(len(Xte), Ytr[:, j].mean())
        out[:, j] = p * np.maximum(h, 0.0)
    return out


# --------------------------------------------------------------------------
def inner_select(X_by_fs, names, Y, A, tr, models, folds, rng):
    """在 tr 这批分子上做 K-fold，返回内层最好的 (特征组, 模型)。"""
    m = len(tr)
    order = rng.permutation(m)
    cuts = np.array_split(order, folds)
    best, bs = None, -np.inf
    for fs, Xall in X_by_fs.items():
        X = Xall[tr]
        for mk in models:
            if mk == "rule" and not RuleFamily.supports(names[fs]):
                continue   # 这组特征没有适用性列，规则族不适用
            tot = 0.0
            for cut in cuts:
                msk = np.ones(m, bool)
                msk[cut] = False
                if msk.sum() < 8:
                    continue
                Xa, Ya, Aa = X[msk], Y[tr][msk], A[tr][msk]
                Xb = X[~msk]
                try:
                    if mk == "twohead":
                        s = two_head(Xa, Ya, Aa, Xb)
                    elif mk == "rule":
                        pk = RuleFamily(names[fs]).fit(Xa, Ya).predict(Xb)
                        s = np.eye(NE)[pk]
                    else:
                        s = fit_predict(mk, Xa, Ya, Xb, rng)
                except Exception:
                    s = np.zeros((len(Xb), NE))
                pk = s.argmax(1)
                tot += Y[tr][~msk][np.arange(len(pk)), pk].sum()
            if tot > bs:
                best, bs = (fs, mk), tot
    return best


def outer_loo(X_by_fs, names, Y, A, models, folds, rng):
    """嵌套 LOO。返回 (pick, sbs, 被选中的组合列表)。"""
    M = len(Y)
    pick = np.zeros(M, int)
    sbs = np.zeros(M, int)
    chosen = []
    for i in range(M):
        tr = np.flatnonzero(np.arange(M) != i)
        sbs[i] = int(np.argmax(Y[tr].mean(0)))
        fs, mk = inner_select(X_by_fs, names, Y, A, tr, models, folds, rng)
        if mk == "rule" and not RuleFamily.supports(names[fs]):
            mk = "A"      # 不该发生；真发生了就退回绝对回归而不是崩掉
        chosen.append((fs, mk))
        X = X_by_fs[fs]
        if mk == "twohead":
            s = two_head(X[tr], Y[tr], A[tr], X[i:i + 1])[0]
        elif mk == "rule":
            pk = RuleFamily(names[fs]).fit(X[tr], Y[tr]).predict(X[i:i + 1])
            s = np.eye(NE)[pk[0]]
        else:
            s = fit_predict(mk, X[tr], Y[tr], X[i:i + 1], rng)
        pick[i] = int(np.argmax(s))
    return pick, sbs, chosen


def robust(d):
    return {"mean": float(d.mean()), "winsor10": float(trim_mean(d, 0.1)),
            "median": float(np.median(d)), "winrate": float(np.mean(d > 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results_sbt_first")
    ap.add_argument("--features", default="F1,F3,F4,F5,F7,F8")
    ap.add_argument("--models", default="A,B,C,rf,twohead,rule")
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--nperm", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    ekw = EDITOR_KWARGS[args.run_dir]
    mols, Y, Yr, H5, AT = load_labels(args.run_dir)
    M = len(mols)
    ed, bank = make_editors(**ekw), OracleBank()
    sets = args.features.split(",")
    models = args.models.split(",")

    X_by_fs, names = {}, {}
    for fs in sets:
        X_by_fs[fs], names[fs] = build_matrix(mols, ed, bank, fs)

    nscaf = len({MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
                 for s in mols})
    print(f"run={args.run_dir}  editor={ekw}")
    print(f"{M} 分子 / {nscaf} 个互不相同 scaffold -> "
          f"{'LOO == leave-one-scaffold-out' if nscaf == M else '需按 scaffold 分组'}")
    print(f"内层 {args.inner_folds}-fold，网格 = {len(sets)} 特征组 x {len(models)} 模型 "
          f"= {len(sets) * len(models)} 个组合\n")

    pick, sbs, chosen = outer_loo(X_by_fs, names, Y, AT, models, args.inner_folds, rng)
    idx = np.arange(M)
    d = Y[idx, pick] - Y[idx, sbs]
    lo, hi = boot_ci(d, rng)
    vbs = Y.max(1).mean()

    # 固定 fold 外的强规则（任务 4），作为唯一要打败的基线
    rpick = np.zeros(M, int)
    for i in range(M):
        tr = np.flatnonzero(idx != i)
        rpick[i] = RuleFamily(names["F5"]).fit(X_by_fs["F5"][tr], Y[tr]).predict(
            X_by_fs["F5"][i:i + 1])[0]
    dr = Y[idx, pick] - Y[idx, rpick]
    rlo, rhi = boot_ci(dr, rng)
    drs = Y[idx, rpick] - Y[idx, sbs]
    slo, shi = boot_ci(drs, rng)

    print("=== 固定调用预算 (HV @ 20 calls) ===")
    print(f"{'策略':<20}{'HV':>9}{'Δ vs SBS':>11}{'95% CI':>24}{'Top-1':>8}"
          f"{'winsor10':>10}{'中位数':>10}{'赢面':>7}")
    print("-" * 100)
    print(f"{'SBS (per-fold)':<20}{Y[idx, sbs].mean():>9.5f}{0.0:>+11.5f}"
          f"{'—':>24}{np.mean(sbs == Y.argmax(1)):>8.1%}{'—':>10}{'—':>10}{'—':>7}")
    rb = robust(drs)
    print(f"{'StrongRule (任务4)':<20}{Y[idx, rpick].mean():>9.5f}{drs.mean():>+11.5f}"
          f"{'*' if slo > 0 else ' '}[{slo:+.5f}, {shi:+.5f}]"
          f"{np.mean(rpick == Y.argmax(1)):>8.1%}{rb['winsor10']:>+10.5f}"
          f"{rb['median']:>+10.5f}{rb['winrate']:>7.0%}")
    nb = robust(d)
    print(f"{'Nested Selector':<20}{Y[idx, pick].mean():>9.5f}{d.mean():>+11.5f}"
          f"{'*' if lo > 0 else ' '}[{lo:+.5f}, {hi:+.5f}]"
          f"{np.mean(pick == Y.argmax(1)):>8.1%}{nb['winsor10']:>+10.5f}"
          f"{nb['median']:>+10.5f}{nb['winrate']:>7.0%}")
    print(f"{'VBS (诊断)':<20}{vbs:>9.5f}{vbs - Y[idx, sbs].mean():>+11.5f}"
          f"{'—':>24}{1.0:>8.1%}{'—':>10}{'—':>10}{'—':>7}")

    print(f"\nNested Selector − StrongRule（配对，这是唯一未决的核心命题）")
    print(f"  {dr.mean():+.5f}{'*' if rlo > 0 else ' '} [{rlo:+.5f}, {rhi:+.5f}]"
          f"   Top-1 差 {np.mean(pick == Y.argmax(1)) - np.mean(rpick == Y.argmax(1)):+.1%}")

    # 内层选了什么
    from collections import Counter
    cc = Counter(chosen)
    print(f"\n内层在 50 个 fold 里选中的组合（前 6）:")
    for (fs, mk), c in cc.most_common(6):
        print(f"  {fs} / {mk:<10}{c:>3}/{M} 次")

    # 固定候选预算（任务 10）
    print(f"\n=== 固定候选预算 (HV @ 5 candidates) ===")
    rows = []
    for nm, pk in [("StrongRule", rpick), ("Nested Selector", pick)]:
        ok = ~np.isnan(H5[idx, pk]) & ~np.isnan(H5[idx, sbs])
        d5 = H5[idx, pk][ok] - H5[idx, sbs][ok]
        l5, h5 = boot_ci(d5, rng) if ok.sum() > 5 else (np.nan, np.nan)
        print(f"  {nm:<18}支撑 {ok.sum():>2}/{M}  Δ vs SBS {d5.mean():>+.5f}"
              f"{'*' if l5 > 0 else ' '}[{l5:+.5f}, {h5:+.5f}]"
              f"  覆盖率 {np.mean(AT[idx, pk]):.0%} (SBS {np.mean(AT[idx, sbs]):.0%})")
        rows.append({"policy": nm, "support": int(ok.sum()), "d_hv5": float(d5.mean()),
                     "ci": [l5, h5], "coverage": float(np.mean(AT[idx, pk]))})

    # 置换（包住整条选择链）
    perm = None
    if args.nperm:
        print(f"\n=== 置换负对照（每次重跑完整内层选择，{args.nperm} 次）===")
        nd, nt = np.empty(args.nperm), np.empty(args.nperm)
        for t in range(args.nperm):
            p = rng.permutation(M)
            Yp, Ap = Y[p], AT[p]
            pk, sb, _ = outer_loo(X_by_fs, names, Yp, Ap, models, args.inner_folds, rng)
            nd[t] = (Yp[idx, pk] - Yp[idx, sb]).mean()
            nt[t] = np.mean(pk == Yp.argmax(1))
            print(f"  [{t + 1}/{args.nperm}] ΔHV {nd[t]:+.5f}  Top-1 {nt[t]:.1%}",
                  flush=True)
        p_d = float(np.mean(nd >= d.mean()))
        p_t = float(np.mean(nt >= np.mean(pick == Y.argmax(1))))
        print(f"  实测 ΔHV {d.mean():+.5f} vs 置换均值 {nd.mean():+.5f} "
              f"(p95 {np.percentile(nd, 95):+.5f})  p = {p_d:.3f}")
        print(f"  实测 Top-1 {np.mean(pick == Y.argmax(1)):.1%} vs 置换均值 "
              f"{nt.mean():.1%} (p95 {np.percentile(nt, 95):.1%})  p = {p_t:.3f}")
        perm = {"nperm": args.nperm, "p_d_hv": p_d, "p_top1": p_t,
                "null_d_hv_mean": float(nd.mean()), "null_top1_mean": float(nt.mean())}

    el = time.perf_counter() - t0
    print(f"\n计算成本: 墙钟 {el:.0f}s；新增 oracle 调用 {bank.stats()['logical_calls']}"
          f"（标签全部复用 {args.run_dir}/fixed_val.json）")

    out = {
        "run_dir": args.run_dir, "editor_kwargs": ekw, "n_mol": M, "n_scaffold": nscaf,
        "grid": {"features": sets, "models": models, "inner_folds": args.inner_folds},
        "vbs": float(vbs), "sbs_hv": float(Y[idx, sbs].mean()),
        "strong_rule": {"hv": float(Y[idx, rpick].mean()), **robust(drs),
                        "ci_vs_sbs": [slo, shi],
                        "top1": float(np.mean(rpick == Y.argmax(1)))},
        "nested": {"hv": float(Y[idx, pick].mean()), **robust(d), "ci_vs_sbs": [lo, hi],
                   "top1": float(np.mean(pick == Y.argmax(1)))},
        "nested_vs_rule": {"d_hv": float(dr.mean()), "ci": [rlo, rhi], **robust(dr)},
        "inner_choices": {f"{a}/{b}": c for (a, b), c in cc.items()},
        "matched_candidate": rows, "permutation": perm, "wall_seconds": el,
        "new_oracle_calls": bank.stats()["logical_calls"],
    }
    op = ROOT / (args.out or
                 f"results_v2/selector/nested_{args.run_dir.replace('results_', '')}.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"-> {op}")


if __name__ == "__main__":
    main()
