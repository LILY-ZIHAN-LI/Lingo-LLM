"""EXP 1：LOO Expert Selector —— 分子级 expert 差异到底能不能从结构预测出来。

零新增 oracle 调用：标签全部来自 results_base0/fixed_val.json 里已有的
50 分子 x 4 expert x 2 重复 = 400 条轨迹。

三个训练目标（EXP 1.5）：
  A  绝对回归      y = [HV_gsk3b, ..., HV_sa]
  B  分子内中心化  y' = y - mean(y)       <- 方差分解说目标方差全在交互项上，
  C  pairwise 排序 6 个 expert 对的胜负      所以 B/C 比 A 更对症

对照：
  SBS                 每个 fold 内重新选的全局最优固定 expert（主基线）
  ApplicabilityRule   无泄漏的化学规则：按决策前的模板产物数取 argmax
  VBS                 事后 oracle，只作诊断，不是可达性能

判据（已和你商定）：n=50 时 ΔHV vs SBS 的最小可检出效应是 oracle 上限的
47%，所以**不以 ΔHV 的显著性为门槛**，主看排序准确率、Gap Closed、
Learning Curve 斜率，以及标签置换负对照。

用法:
  python scripts/run_loo_selector.py
  python scripts/run_loo_selector.py --features F5 --nperm 200
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold                     # noqa: E402
from sklearn.ensemble import RandomForestRegressor                  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV        # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

from src.analysis import hv_at_eligible                             # noqa: E402
from src.editors import make_editors                                # noqa: E402
from src.features import FEATURE_SETS, build_matrix                 # noqa: E402
from src.interfaces import EXPERTS                                  # noqa: E402
from src.oracles import OracleBank                                  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

ALPHAS = np.logspace(-3, 3, 13)

# 各 run 当初是用什么 editor 配置跑出来的 —— 从对应的 *_log.txt 头部读出来。
# 必须按 run 取：qed_disjoint 决定 qed 的 heavy 池是 _SHRINK（= sa 的池，
# 导致 38/50 个分子上 qed 与 sa 退化成同一个 editor）还是 _QED_SWAP。
# 用错了，app_qed_* / site_qed_* / heavy_pool 这几列描述的就是另一个池。
# sample_by_template 和 product_pick 只影响抽样，不改池子，所以不进特征。
EDITOR_KWARGS = {
    "results_base0":      {"qed_disjoint": False},
    "results_sbt_random": {"qed_disjoint": True},
    "results_sbt_last":   {"qed_disjoint": True},
    "results_sbt_first":  {"qed_disjoint": True},
}
CS = [0.01, 0.1, 1.0, 10.0]
PAIRS = [(a, b) for a in range(len(EXPERTS)) for b in range(a + 1, len(EXPERTS))]


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_labels(run_dir: str, metric: str = "final"):
    """-> (mols, Y, Y_reps, HV5, ATT)

    Y      (M,4) 两次重复的均值
    Y_reps (M,4,R) 每次重复，给噪声诊断用
    HV5    (M,4) 攒够 5 个合格候选时的 HV（没攒够 = nan）
    ATT    (M,4) 该 (分子,expert) 有多少比例的重复攒够了 5 个候选
    """
    recs = [r for r in json.loads((ROOT / run_dir / "fixed_val.json").read_text())
            if r["policy"].startswith("only_")]
    g: dict[tuple[str, str], list] = defaultdict(list)
    for r in recs:
        g[(r["smiles"], r["policy"])].append(r)
    mols = sorted({m for m, _ in g})
    pols = [f"only_{e}" for e in EXPERTS]
    R = max(len(v) for v in g.values())

    Y = np.full((len(mols), len(EXPERTS)), np.nan)
    Yr = np.full((len(mols), len(EXPERTS), R), np.nan)
    H5 = np.full((len(mols), len(EXPERTS)), np.nan)
    AT = np.zeros((len(mols), len(EXPERTS)))
    for i, m in enumerate(mols):
        for j, p in enumerate(pols):
            rs = sorted(g[(m, p)], key=lambda r: r["episode_seed"])
            vals = [float(r[metric]) for r in rs]
            Y[i, j] = np.mean(vals)
            Yr[i, j, :len(vals)] = vals
            h = [hv_at_eligible(r, 5) for r in rs]
            ok = [v for v in h if v is not None]
            if ok:
                H5[i, j] = float(np.mean(ok))
            AT[i, j] = len(ok) / len(rs)
    return mols, Y, Yr, H5, AT


# --------------------------------------------------------------------------
# 三个训练目标 + 对照
# --------------------------------------------------------------------------
def fit_predict(kind: str, Xtr, Ytr, Xte, rng):
    """在 train fold 上拟合一次，返回对 Xte(m,d) 全部行的打分 (m,4)。

    注意是**拟合一次、预测 m 行**。早先的版本只取 predict(B)[0]，调用方
    因此对每个测试样本重新拟合一遍 —— 在 50 个分子的 LOO 里每次只预测
    1 行，看不出来；200 个分子做 5 折内层选择时就是几千次重复拟合。
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(np.atleast_2d(Xte))
    m = len(B)

    if kind in ("A", "B", "rf"):
        T = Ytr - Ytr.mean(axis=1, keepdims=True) if kind == "B" else Ytr
        out = np.empty((m, len(EXPERTS)))
        for j in range(len(EXPERTS)):
            if kind == "rf":
                mdl = RandomForestRegressor(
                    n_estimators=300, max_depth=4, min_samples_leaf=3,
                    random_state=0, n_jobs=1).fit(A, T[:, j])
            else:
                mdl = RidgeCV(alphas=ALPHAS).fit(A, T[:, j])
            out[:, j] = mdl.predict(B)
        return out

    if kind == "C":
        # 6 个 expert 对各训一个二分类器，Borda 计分
        score = np.zeros((m, len(EXPERTS)))
        for a, b in PAIRS:
            lab = (Ytr[:, a] > Ytr[:, b]).astype(int)
            if lab.min() == lab.max():          # train fold 内该对无胜负变化
                score[:, a if lab[0] == 1 else b] += 1.0
                continue
            best, bs = None, -np.inf
            for C in CS:                        # 小网格，只用 train fold
                mdl = LogisticRegression(C=C, max_iter=2000).fit(A, lab)
                sc_ = mdl.score(A, lab)
                if sc_ > bs:
                    best, bs = mdl, sc_
            p = best.predict_proba(B)[:, 1]
            score[:, a] += p
            score[:, b] += 1.0 - p
        return score

    raise ValueError(kind)


def metrics(pick, Y, sbs_pick, name):
    """pick/sbs_pick: 长度 M 的 expert 下标。"""
    M = len(Y)
    idx = np.arange(M)
    hv = Y[idx, pick]
    hv_sbs = Y[idx, sbs_pick]
    best = Y.max(axis=1)
    top1 = float(np.mean(pick == Y.argmax(axis=1)))
    return {
        "policy": name, "hv": float(hv.mean()),
        "d_hv": float((hv - hv_sbs).mean()),
        "d_hv_per_mol": (hv - hv_sbs),
        "regret": float((best - hv).mean()),
        "top1_acc": top1,
    }


def boot_ci(d, rng, n=4000):
    bs = np.array([d[i].mean() for i in rng.integers(0, len(d), (n, len(d)))])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_loo(X, Y, kinds, rng):
    """返回 {kind: pick 数组}，外加 sbs_pick / app 的 pick 由调用方给。"""
    M = len(Y)
    picks = {k: np.zeros(M, dtype=int) for k in kinds}
    sbs = np.zeros(M, dtype=int)
    for i in range(M):
        tr = np.arange(M) != i
        sbs[i] = int(np.argmax(Y[tr].mean(axis=0)))   # SBS 每个 fold 重选
        for k in kinds:
            picks[k][i] = int(np.argmax(
                fit_predict(k, X[tr], Y[tr], X[i:i + 1], rng)))
    return picks, sbs


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results_base0")
    ap.add_argument("--metric", default="final")
    ap.add_argument("--features", default="all")
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--lc-reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_v2/selector/loo_selector.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    mols, Y, Yr, H5, AT = load_labels(args.run_dir, args.metric)
    M = len(mols)
    print(f"数据: {M} 分子 x {len(EXPERTS)} expert x {Yr.shape[2]} 重复 "
          f"(来自 {args.run_dir}/fixed_val.json，新增 oracle 调用 = 0)\n")

    # --- scaffold 检查：LOO 是否等价于 leave-one-scaffold-out ---
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in mols]
    nuniq = len(set(scaf))
    print(f"Murcko scaffold: {nuniq} 个互不相同 / {M} 个分子 "
          f"-> {'LOO 等价于 leave-one-scaffold-out' if nuniq == M else '存在共享 scaffold，需按 scaffold 分组留出'}")

    # --- 标签噪声 ---
    sd = np.nanstd(Yr, axis=2, ddof=1)
    print(f"标签重复噪声: 单次 SD 中位数 {np.nanmedian(sd):.5f}，"
          f"两次均值的 SE {np.nanmedian(sd) / np.sqrt(2):.5f}")

    vbs = Y.max(axis=1).mean()
    print(f"VBS (事后 oracle，仅诊断) = {vbs:.5f}\n")

    if args.run_dir not in EDITOR_KWARGS:
        raise SystemExit(
            f"{args.run_dir} 不在 EDITOR_KWARGS 里。先去 {args.run_dir}_log.txt "
            f"头部确认 qed_disjoint 等配置再加进去 —— 用错 editor，"
            f"qed 的适用性特征描述的会是另一个池。")
    ekw = EDITOR_KWARGS[args.run_dir]
    print(f"editor 配置: {ekw}  (来自 {args.run_dir}_log.txt)")
    ed, bank = make_editors(**ekw), OracleBank()
    sets = FEATURE_SETS if args.features == "all" else tuple(args.features.split(","))

    out = {"n_mol": M, "n_scaffold": nuniq, "vbs": float(vbs),
           "label_noise_sd": float(np.nanmedian(sd)), "by_feature_set": {}}
    best = None

    for fs in sets:
        X, names = build_matrix(mols, ed, bank, fs)
        picks, sbs = run_loo(X, Y, ("A", "B", "C", "rf"), rng)

        # 无泄漏 Applicability Rule：决策前的模板产物数取 argmax
        Xa, na = build_matrix(mols, ed, bank, "F4")
        col = [na.index(f"app_{e}_n_product") for e in EXPERTS]
        app = np.argmax(Xa[:, col], axis=1)

        rows = [metrics(sbs, Y, sbs, "SBS (per-fold)"),
                metrics(app, Y, sbs, "ApplicabilityRule"),
                metrics(picks["A"], Y, sbs, "A 绝对回归"),
                metrics(picks["B"], Y, sbs, "B 中心化回归"),
                metrics(picks["C"], Y, sbs, "C pairwise 排序"),
                metrics(picks["rf"], Y, sbs, "RandomForest (绝对)")]
        gapden = vbs - Y[np.arange(M), sbs].mean()

        print(f"=== 特征组 {fs}  ({X.shape[1]} 维) " + "=" * (46 - len(fs)))
        print(f"{'策略':<22}{'HV':>9}{'ΔHV vs SBS':>13}{'95% CI':>24}"
              f"{'Regret':>10}{'Top-1':>8}{'GapClosed':>11}")
        print("-" * 98)
        for r in rows:
            lo, hi = boot_ci(r["d_hv_per_mol"], rng) if r["policy"] != "SBS (per-fold)" else (0.0, 0.0)
            st = "*" if lo > 0 else " "
            gc = r["d_hv"] / gapden if gapden > 0 else float("nan")
            print(f"{r['policy']:<22}{r['hv']:>9.5f}{r['d_hv']:>+13.5f}{st}"
                  f" [{lo:+.5f}, {hi:+.5f}]{r['regret']:>10.5f}"
                  f"{r['top1_acc']:>8.1%}{gc:>11.1%}")
            r["ci"] = [lo, hi]
            r["gap_closed"] = gc
            del r["d_hv_per_mol"]
        print(f"{'VBS (诊断)':<22}{vbs:>9.5f}{vbs - Y[np.arange(M), sbs].mean():>+13.5f} "
              f"{'—':>25}{0.0:>10.5f}{1.0:>8.1%}{1.0:>11.1%}")

        # ApplicabilityRule 才是真正要打败的强基线：SBS 只是零参数固定 expert，
        # 而 F4 里的模板适用性是一行化学规则就能用的信息。模型只有在配对意义上
        # 超过这条规则，才能说"机器学习"本身带来了增量。
        idx = np.arange(M)
        hv_app = Y[idx, app]
        print(f"\n  vs ApplicabilityRule（配对，这才是强基线）")
        print(f"  {'策略':<22}{'ΔHV vs 规则':>14}{'95% CI':>24}{'Top-1 差':>10}")
        print("  " + "-" * 68)
        vs_rule = {}
        for r, pk in [(rows[2], picks["A"]), (rows[3], picks["B"]),
                      (rows[4], picks["C"]), (rows[5], picks["rf"])]:
            d = Y[idx, pk] - hv_app
            lo, hi = boot_ci(d, rng)
            st = "*" if lo > 0 else " "
            print(f"  {r['policy']:<22}{d.mean():>+14.5f}{st}[{lo:+.5f}, {hi:+.5f}]"
                  f"{r['top1_acc'] - rows[1]['top1_acc']:>+10.1%}")
            vs_rule[r["policy"]] = {"d_hv": float(d.mean()), "ci": [lo, hi]}
            r["d_hv_vs_rule"] = float(d.mean())
            r["ci_vs_rule"] = [lo, hi]
        print()

        out["by_feature_set"][fs] = {"n_feat": int(X.shape[1]), "rows": rows,
                                     "vs_rule": vs_rule}
        for r in rows[2:]:
            if best is None or r["d_hv"] > best[2]:
                best = (fs, r["policy"], r["d_hv"])

    # ---------------- 负对照：标签置换 ----------------
    fs, pol, _ = best
    kind = {"A 绝对回归": "A", "B 中心化回归": "B",
            "C pairwise 排序": "C", "RandomForest (绝对)": "rf"}[pol]
    X, _ = build_matrix(mols, ed, bank, fs)
    print(f"=== 负对照：标签置换 ({fs} / {pol}，{args.nperm} 次) ===")
    obs = out["by_feature_set"][fs]["rows"][
        [r["policy"] for r in out["by_feature_set"][fs]["rows"]].index(pol)]
    nulls = np.empty(args.nperm)
    ntop = np.empty(args.nperm)
    for t in range(args.nperm):
        perm = rng.permutation(M)                 # 打乱 分子->标签 的对应
        Yp = Y[perm]
        pk, sb = run_loo(X, Yp, (kind,), rng)
        idx = np.arange(M)
        nulls[t] = (Yp[idx, pk[kind]] - Yp[idx, sb]).mean()
        ntop[t] = np.mean(pk[kind] == Yp.argmax(axis=1))
    p_hv = float(np.mean(nulls >= obs["d_hv"]))
    p_t1 = float(np.mean(ntop >= obs["top1_acc"]))
    print(f"实测 ΔHV {obs['d_hv']:+.5f} vs 置换均值 {nulls.mean():+.5f} "
          f"(p95 {np.percentile(nulls, 95):+.5f})  p = {p_hv:.3f}")
    print(f"实测 Top-1 {obs['top1_acc']:.1%} vs 置换均值 {ntop.mean():.1%} "
          f"(p95 {np.percentile(ntop, 95):.1%})  p = {p_t1:.3f}")
    out["permutation"] = {"feature_set": fs, "policy": pol, "nperm": args.nperm,
                          "obs_d_hv": obs["d_hv"], "null_d_hv_mean": float(nulls.mean()),
                          "p_d_hv": p_hv, "obs_top1": obs["top1_acc"],
                          "null_top1_mean": float(ntop.mean()), "p_top1": p_t1}

    # ---------------- Learning curve ----------------
    print(f"\n=== Learning Curve ({fs} / {pol}) ===")
    print(f"{'训练分子数':<12}{'Top-1':>9}{'ΔHV vs SBS':>13}{'Regret':>10}")
    print("-" * 46)
    lc = {}
    for n_tr in (10, 20, 30, 40, M - 1):
        t1, dh, rg = [], [], []
        for _ in range(args.lc_reps):
            for i in range(M):
                pool = np.array([j for j in range(M) if j != i])
                tr = rng.choice(pool, size=min(n_tr, len(pool)), replace=False)
                s = int(np.argmax(Y[tr].mean(axis=0)))
                p = int(np.argmax(fit_predict(kind, X[tr], Y[tr], X[i:i + 1], rng)))
                t1.append(p == int(Y[i].argmax()))
                dh.append(Y[i, p] - Y[i, s])
                rg.append(Y[i].max() - Y[i, p])
        print(f"{n_tr:<12}{np.mean(t1):>9.1%}{np.mean(dh):>+13.5f}{np.mean(rg):>10.5f}")
        lc[n_tr] = {"top1": float(np.mean(t1)), "d_hv": float(np.mean(dh)),
                    "regret": float(np.mean(rg))}
    out["learning_curve"] = lc

    # ---------------- 匹配候选预算（次要口径）----------------
    print(f"\n=== 匹配候选预算 HV@5（次要口径，按 pair 各自支撑）===")
    picks, sbs = run_loo(X, Y, (kind,), rng)
    pk = picks[kind]
    idx = np.arange(M)
    ok = ~np.isnan(H5[idx, pk]) & ~np.isnan(H5[idx, sbs])
    d5 = H5[idx, pk][ok] - H5[idx, sbs][ok]
    lo, hi = boot_ci(d5, rng) if ok.sum() > 5 else (float("nan"),) * 2
    print(f"支撑 {ok.sum()}/{M} 分子（双方都攒够 5 个候选）")
    print(f"ΔHV@5 = {d5.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    print(f"预算达成率: 选中 expert {np.mean(AT[idx, pk]):.1%}，SBS {np.mean(AT[idx, sbs]):.1%}")
    out["matched_candidate"] = {"support": int(ok.sum()), "d_hv5": float(d5.mean()),
                               "ci": [lo, hi],
                               "attain_pick": float(np.mean(AT[idx, pk])),
                               "attain_sbs": float(np.mean(AT[idx, sbs]))}

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
