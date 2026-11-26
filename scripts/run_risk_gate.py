"""分子级 Risk Gate：在 episode 开始时，二选一地决定整条轨迹用 Router 还是 TwoPhase。

动机（数据来自 test_v2，仅作方向参考，不作开发目标）：
    Router − TwoPhase 的均值 +0.00044，中位数 +0.00056，winsor10 +0.00078
    最差 1 个分子 -0.0866，独自吃掉 99% 的总收益
    亏 >0.002 的 38 个，赚 >0.002 的 67 个
灾难是**整条轨迹级别**的 —— 一个分子亏掉它自身 HV 的好几倍，不像"某一步
选错"，更像"这个分子整体不适合 Router 的风格"。所以在 episode 开始时
一次性决定用哪条策略，比逐步微调更对症，也避开了 Hybrid 那个状态不匹配
问题（已实测：PhaseSplit 对照显示换规则本身就亏 -0.00025）。

**交叉拟合是必须的，不是可选的。** Router 的 3000 个训练状态全部来自
train 的 200 个分子（已核实，零例外），所以在 train 上直接跑 Router 产生
标签是彻底样本内。这里按 scaffold 分 K 折：
    第 k 折：用**不含该折分子**的反事实状态重训 Router → 在该折分子上
             跑 Router 和 TwoPhase → 得到诚实的配对标签 y(m)
    y(m) = HV_Router(m) - HV_TwoPhase(m)
Gate 再用留一折的方式在这些标签上训练和评估。

判据（三个一起报，缺一不可）：
    避免的损失   回退掉的分子里，本来会亏多少
    错失的收益   回退掉的分子里，本来会赚多少
    回退率       全部回退到 TwoPhase 也能消灭全部灾难，所以单看"灾难
                 减少了"没有意义
最终看的是 Gate 策略的整体 HV 是否高于纯 Router 和纯 TwoPhase。

用法:
  python scripts/run_risk_gate.py --folds 5 --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                                  # noqa: E402
from sklearn.ensemble import RandomForestRegressor          # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

from run_loo_selector import boot_ci                        # noqa: E402
from run_oracle_router import pool_env_kw, rollout          # noqa: E402
from src.cv import grouped_folds, scaffold_of               # noqa: E402
from src.dynamic_router import DynamicRouter                # noqa: E402
from src.env import MolEnv                                  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig           # noqa: E402
from src.oracles import OracleBank                          # noqa: E402
from src.router_features import FEATURE_GROUPS, build_states, features_from_env  # noqa: E402
from train_dynamic_router import fit_fold, pick_alpha       # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 970000


class FoldRouter:
    """用部分状态训出来的 Router，接口与 DynamicRouter 一致。"""

    def __init__(self, X, D, scaler, models, group):
        self.scaler, self.models, self.group = scaler, models, group

    def act_env(self, env, t=None):
        x = self.scaler.transform(features_from_env(env, self.group))
        s = np.array([m.predict(x)[0] for m in self.models])
        calls = np.asarray(env.calls_so_far, float)
        gain = np.asarray(env.gain_so_far, float)
        bad = (calls > 0) & (gain <= 0)
        if not bad.all():
            s[bad] = -np.inf
        return int(np.argmax(s))

    def chooser(self):
        return lambda e, t: (self.act_env(e, t), 0)


def train_fold_router(X, D, group, seed=0):
    sc = StandardScaler().fit(X)
    A = sc.transform(X)
    models = [RandomForestRegressor(n_estimators=300, max_depth=8,
                                    min_samples_leaf=5, random_state=seed,
                                    n_jobs=1).fit(A, D[:, j])
              for j in range(NE)]
    return FoldRouter(X, D, sc, models, group)


def mol_features(mols, pool):
    """Gate 的输入：只看起始分子，决策前可得。复用 R5 的块定义。"""
    from src.editors import make_editors
    from src.features import build_matrix
    ed = make_editors(qed_disjoint=(pool != "base0"))
    bank = OracleBank()
    X, names = build_matrix(list(mols), ed, bank, "F5")
    return X, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--cf", default="results_v2/counterfactuals/cf_sbt_first_200mol.json")
    ap.add_argument("--group", default="R5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", default="results_v2/risk/risk_gate.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    od = (ROOT / args.out).parent
    od.mkdir(parents=True, exist_ok=True)
    ck = od / "ckpt"
    ck.mkdir(exist_ok=True)

    env_kw = pool_env_kw(args.pool)
    two = tuple(json.loads(
        (ROOT / "results_v2/baselines/baseline_meta.json").read_text())["two_phase"])
    e1, e2, k = two

    print("提取反事实特征 ...", flush=True)
    S, blocks, D, step, seed_smi, costs = build_states(ROOT / args.cf, args.pool)
    Xs = np.concatenate([blocks[b] for b in FEATURE_GROUPS[args.group]], axis=1)
    smi_arr = np.asarray(seed_smi)
    mols = sorted(set(seed_smi))
    MS = scaffold_of(mols)
    MFOLD = grouped_folds(MS, args.folds, seed=0)
    print(f"{len(mols)} 个 train 分子 / {len(np.unique(MS))} scaffold / "
          f"{args.folds} 折；Router 的 {len(D)} 个状态全部来自这些分子，"
          f"所以必须按折重训\n")

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    two_fn = lambda e, t: ((e1 if t < k else e2), 0)   # noqa: E731

    # ---------- 交叉拟合：每折重训 Router，在留出分子上取诚实标签 ----------
    hv_r = np.full((len(mols), args.repeats), np.nan)
    hv_t = np.full((len(mols), args.repeats), np.nan)
    t0 = time.perf_counter()
    for f in range(args.folds):
        cp = ck / f"fold{f}.json"
        if cp.exists():
            blob = json.loads(cp.read_text())
            for i, m in enumerate(mols):
                if m in blob["hv_r"]:
                    hv_r[i] = blob["hv_r"][m]
                    hv_t[i] = blob["hv_t"][m]
            print(f"  fold {f}: [复用 checkpoint]", flush=True)
            continue
        te_mol = [m for i, m in enumerate(mols) if MFOLD[i] == f]
        tr_state = ~np.isin(smi_arr, te_mol)
        r = train_fold_router(Xs[tr_state], D[tr_state], args.group)
        fn = r.chooser()
        hr, ht = {}, {}
        for m in te_mol:
            i = mols.index(m)
            for rep in range(args.repeats):
                s = SEED_BASE + rep * 10000 + i
                hv_r[i, rep] = rollout(env, fn, m, s, args.max_steps)["final"]
                hv_t[i, rep] = rollout(env, two_fn, m, s, args.max_steps)["final"]
            hr[m], ht[m] = hv_r[i].tolist(), hv_t[i].tolist()
        cp.write_text(json.dumps({"hv_r": hr, "hv_t": ht}, default=float))
        print(f"  fold {f}: {len(te_mol)} 个留出分子，"
              f"用 {tr_state.sum()} 个状态重训 "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    R, T = hv_r.mean(1), hv_t.mean(1)
    y = R - T
    print(f"\n交叉拟合标签（train {len(mols)} 分子，Router 为样本外）")
    print(f"  Router {R.mean():.5f}   TwoPhase {T.mean():.5f}   "
          f"Δ {y.mean():+.5f}")
    print(f"  中位数 {np.median(y):+.5f}   赢面 {np.mean(y > 0):.0%}   "
          f"亏 >0.002 的 {int(np.sum(y < -0.002))} 个   "
          f"赚 >0.002 的 {int(np.sum(y > 0.002))} 个")
    o = np.argsort(y)
    for kk in (1, 3, 5, 10):
        print(f"  最差 {kk:>2} 个合计 {y[o[:kk]].sum():+.5f}，"
              f"剔掉后均值 {y[o[kk:]].mean():+.5f}  <- 完美预知的上界")

    # ---------- 训练 Gate（留一折），评估 ----------
    Xm, mnames = mol_features(mols, args.pool)
    gate_pred = np.zeros(len(mols))
    for f in range(args.folds):
        tr, te = MFOLD != f, MFOLD == f
        sc = StandardScaler().fit(Xm[tr])
        mdl = RandomForestRegressor(n_estimators=400, max_depth=6,
                                    min_samples_leaf=5, random_state=0,
                                    n_jobs=1).fit(sc.transform(Xm[tr]), y[tr])
        gate_pred[te] = mdl.predict(sc.transform(Xm[te]))

    print(f"\nGate 的预测能力（留一折）")
    from scipy.stats import spearmanr
    print(f"  预测 vs 真实 Δ 的 Spearman {spearmanr(gate_pred, y).statistic:+.3f}")
    print(f"  预测为负的分子里，真实也为负的比例 "
          f"{np.mean(y[gate_pred < 0] < 0) if np.any(gate_pred < 0) else float('nan'):.0%}")

    print(f"\n{'阈值 τ':<10}{'回退率':>8}{'Gate HV':>10}{'Δ vs Router':>13}"
          f"{'95% CI':>22}{'避免的损失':>12}{'错失的收益':>12}")
    print("-" * 90)
    rows = []
    for tau in (-np.inf, -0.004, -0.002, -0.001, 0.0, 0.001, np.inf):
        use_two = gate_pred < tau
        hv = np.where(use_two, T, R)
        d = hv - R
        lo, hi = boot_ci(d, rng) if use_two.any() and not use_two.all() else (0.0, 0.0)
        avoided = float(-y[use_two & (y < 0)].sum())      # 回退掉的亏损
        missed = float(y[use_two & (y > 0)].sum())        # 回退掉的盈利
        name = ("纯 Router" if tau == -np.inf else
                "纯 TwoPhase" if tau == np.inf else f"τ={tau:+.3f}")
        print(f"{name:<10}{use_two.mean():>8.0%}{hv.mean():>10.5f}"
              f"{d.mean():>+13.5f}{'' if tau in (-np.inf, np.inf) else ('*' if lo > 0 else ' ')}"
              f"[{lo:+.5f},{hi:+.5f}]{avoided:>12.5f}{missed:>12.5f}")
        rows.append({"threshold": float(tau), "fallback_rate": float(use_two.mean()),
                     "gate_hv": float(hv.mean()), "d_vs_router": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi, "loss_avoided": avoided,
                     "gain_missed": missed,
                     "net": avoided - missed,
                     "router_hv": float(R.mean()), "twophase_hv": float(T.mean())})

    op = ROOT / args.out
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (od / "risk_gate_labels.json").write_text(json.dumps(
        {"mols": mols, "hv_router": R.tolist(), "hv_twophase": T.tolist(),
         "y": y.tolist(), "gate_pred": gate_pred.tolist(),
         "folds": MFOLD.tolist(), "repeats": args.repeats,
         "wall_seconds": time.perf_counter() - t0}, ensure_ascii=False))
    print(f"\n墙钟 {(time.perf_counter() - t0) / 60:.1f} 分钟")
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
