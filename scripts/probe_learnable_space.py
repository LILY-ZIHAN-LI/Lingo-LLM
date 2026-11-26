"""阶段一 Step 6：按分子/scaffold 分组的即时决策损失对比。

问题：3000 个反事实状态里，有多少调度空间是**可预测的**？

做法不是训正式 Router，而是用最简单的模型（Ridge / RF / 分类×回归两段）
在**分组交叉验证**下预测四个 expert 的即时 ΔHV，取 argmax，和一组不学习
的规则比即时 regret。只回答"有没有可学的信号"，不回答"最终 HV 能涨多少"。

三条必须跨过的线（来自 counterfactual_analysis.md 的定量判断）：
  随机              regret 0.001362   消除 0%
  最好的固定 expert   regret 0.000955   消除 30%
  只看 step 的查表    regret 0.000852   消除 37%   <- ML 必须超过这条
  oracle            regret 0           消除 100%

分组是硬要求：3000 个状态只有 200 个独立分子，随机拆状态会把同一个分子的
不同 step 放进训练和测试两边 —— 那不是泛化，是插值。

特征成本（实测）：
  template_sites  0.20 ms   便宜，Router 每步付得起
  applicability   8.94 ms   和 propose 同价，Router 每步付 4 次 = OneStepOracle
                            的探查成本，那就没有存在理由
所以这里把两组分开评，并把成本一起报。

用法:
  python scripts/probe_learnable_space.py
  python scripts/probe_learnable_space.py --folds 5 --models ridge,rf,twostage
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

from rdkit import Chem, RDLogger                       # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV                # noqa: E402
from sklearn.model_selection import GroupKFold                 # noqa: E402
from sklearn.preprocessing import StandardScaler       # noqa: E402

from run_loo_selector import EDITOR_KWARGS             # noqa: E402
from src.cv import grouped_folds, scaffold_of          # noqa: E402
from src.editors import make_editors                   # noqa: E402
from src.interfaces import EXPERTS                     # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
ALPHAS = np.logspace(-3, 3, 13)


def build(cf_path: Path, pool: str, cheap_only: bool):
    """-> (X, names, D, step, seed, cost_ms_per_state)"""
    S = json.loads(cf_path.read_text())
    ed = make_editors(**EDITOR_KWARGS[f"results_{pool}"])
    D = np.array([[s["actions"][e]["d_hv"] for e in EXPERTS] for s in S])
    step = np.array([s["step"] for s in S])
    seed = np.array([s["seed_smiles"] for s in S])

    rows, names, t0 = [], None, time.perf_counter()
    for s in S:
        cur = s["current_smiles"]
        mol = Chem.MolFromSmiles(cur)
        v, n = [], []
        # 步数 / 预算（免费）
        v += [s["step"], s["budget_left"], s["archive_size"], s["n_seen"]]
        n += ["step", "budget_left", "archive_size", "n_seen"]
        # 当前四项分数：obs 的前 4 维（env 已经算过并缓存，免费）
        v += list(s["obs"][:len(EXPERTS)])
        n += [f"score_{e}" for e in EXPERTS]
        # 历史（免费）
        v += list(s["calls_so_far"]) + list(s["gain_so_far"])
        n += [f"calls_{e}" for e in EXPERTS] + [f"gain_{e}" for e in EXPERTS]
        # 结构描述符（便宜）
        if mol is not None:
            v += [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                  rdMolDescriptors.CalcTPSA(mol),
                  float(mol.GetNumHeavyAtoms()),
                  rdMolDescriptors.CalcNumAromaticRings(mol),
                  rdMolDescriptors.CalcNumRotatableBonds(mol)]
        else:
            v += [0.0] * 6
        n += ["mw", "logp", "tpsa", "n_heavy", "n_arom", "rotb"]
        # 模板位点数（0.20 ms/expert，便宜）
        for e in EXPERTS:
            ts = ed[e].template_sites(cur)
            act = ts["heavy"] if (ts["shared"] or ts["active_heavy"]) else ts["light"]
            v += [float(sum(act)), float(max(act) if act else 0),
                  float(sum(1 for x in act if x > 0))]
            n += [f"site_{e}_total", f"site_{e}_max", f"site_{e}_nfire"]
        if not cheap_only:
            # 完整适用性（8.94 ms/expert = 1 次 propose）
            for e in EXPERTS:
                a = ed[e].applicability(cur)
                v += [a["n_template_fire"], a["n_product"],
                      a["frac_template_fire"], a["heavy_pool"]]
                n += [f"app_{e}_fire", f"app_{e}_prod",
                      f"app_{e}_frac", f"app_{e}_heavy"]
        rows.append(v)
        names = n
    cost = 1000 * (time.perf_counter() - t0) / len(S)
    return np.asarray(rows, float), names, D, step, seed, cost


def _pick_alpha(A, y, groups):
    """在训练 fold 内部按 group 再切一次来选 alpha。

    不能用 RidgeCV 的默认内层 CV：它对行做 LOO/KFold，会把同一个起始分子
    的不同 step 分到内层的训练和验证两边 —— 那是插值，选出来的 alpha 偏小。
    """
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


def fit_predict(kind, Xtr, Dtr, Xte, groups=None):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = np.zeros((len(B), NE))
    for j in range(NE):
        y = Dtr[:, j]
        if kind == "ridge":
            al = _pick_alpha(A, y, groups) if groups is not None else 1.0
            out[:, j] = Ridge(alpha=al, solver="svd").fit(A, y).predict(B)
        elif kind == "rf":
            out[:, j] = RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                random_state=0, n_jobs=1).fit(A, y).predict(B)
        elif kind == "twostage":
            # 阶段一：正收益的概率；阶段二：正收益条件下的 ΔHV
            lab = (y > 0).astype(int)
            if lab.min() == lab.max():
                p = np.full(len(B), float(lab[0]))
            else:
                p = RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=5,
                    random_state=0, n_jobs=1).fit(A, lab).predict_proba(B)[:, 1]
            m = lab == 1
            if m.sum() >= 20:
                q = RandomForestRegressor(
                    n_estimators=200, max_depth=8, min_samples_leaf=5,
                    random_state=0, n_jobs=1).fit(A[m], y[m]).predict(B)
            else:
                q = np.full(len(B), y[m].mean() if m.sum() else 0.0)
            out[:, j] = p * np.maximum(q, 0.0)
        else:
            raise ValueError(kind)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", default="results_v2/counterfactuals/cf_sbt_first_200mol.json")
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="ridge,rf,twostage")
    ap.add_argument("--tie-tol", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_v2/baselines/learnable_space.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = {"cf": args.cf, "pool": args.pool, "folds": args.folds, "sets": {}}
    # fold 只算一次，两组特征、所有模型共用 —— 否则连 StepOnly 这条参照线
    # 都会因为划分不同而变化，跨组比较不成立。
    _, _, _D0, _, _seed0, _ = build(ROOT / args.cf, args.pool, True)
    SCAF = scaffold_of(_seed0)
    FOLD = grouped_folds(SCAF, args.folds, seed=args.seed)
    print(f"分组划分：{len(np.unique(SCAF))} 个 scaffold -> {args.folds} folds "
          f"(seed={args.seed})，两组特征共用")

    for tag, cheap in [("便宜特征 (step+分数+历史+描述符+位点)", True),
                       ("+完整 applicability", False)]:
        X, names, D, step, seed, cost = build(ROOT / args.cf, args.pool, cheap)
        N = len(D)
        scaf, fold = SCAF, FOLD
        groups = np.unique(scaf)

        print(f"\n{'=' * 78}")
        print(f"{tag}   {X.shape[1]} 维，特征提取 {cost:.2f} ms/状态")
        print(f"{N} 个状态 / {len(groups)} 个 scaffold，"
              f"{args.folds}-fold 按 scaffold 分组")
        print(f"{'=' * 78}")

        # 不学习的参照线（同样按 fold 评，参数只用训练 fold）
        ref = {}
        reg_rand = (D.max(1) - D.mean(1)).mean()
        ref["随机"] = reg_rand
        # 「最好的固定 expert」和「只看 step 的查表」都必须在**训练 fold 内**
        # 确定，再到留出 fold 上计分。用全部数据选 expert 等于偷看测试标签。
        fix_r, stp = np.zeros(N), np.zeros(N)
        fix_pick = []
        for f in range(args.folds):
            tr, te = fold != f, fold == f
            bj = int(np.argmin([(D[tr].max(1) - D[tr][:, j]).mean()
                                for j in range(NE)]))
            fix_pick.append(EXPERTS[bj])
            fix_r[te] = D[te].max(1) - D[te][:, bj]
            tbl = {}
            for t in np.unique(step):
                m = tr & (step == t)
                tbl[t] = int(np.argmax(D[m].sum(0))) if m.sum() else bj
            pick = np.array([tbl[t] for t in step[te]])
            stp[te] = D[te].max(1) - D[te][np.arange(te.sum()), pick]
        ref[f"最好的固定 expert (fold 内选: {'/'.join(sorted(set(fix_pick)))})"] = \
            fix_r.mean()
        ref["只看 step 的查表 (fold 内定)"] = stp.mean()

        print(f"{'决策规则':<34}{'即时 regret':>13}{'消除比例':>10}"
              f"{'Top-1':>8}{'正收益率':>9}")
        print("-" * 76)
        for nm, v in ref.items():
            print(f"{nm:<34}{v:>13.6f}{(reg_rand - v) / reg_rand:>10.0%}"
                  f"{'—':>8}{'—':>9}")

        clean = (D > 0).sum(1) >= 1
        srt = np.sort(D, 1)
        strict = clean & ((srt[:, -1] - srt[:, -2]) > args.tie_tol)

        rows = {}
        for mk in args.models.split(","):
            reg = np.zeros(N)
            pick = np.zeros(N, int)
            t0 = time.perf_counter()
            for f in range(args.folds):
                tr, te = fold != f, fold == f
                P = fit_predict(mk, X[tr], D[tr], X[te], groups=scaf[tr])
                pk = P.argmax(1)
                pick[te] = pk
                reg[te] = D[te].max(1) - D[te][np.arange(te.sum()), pk]
            # Top-1 只在严格非平局状态上算（平局没有唯一正确答案）
            top1 = float(np.mean(pick[strict] == D[strict].argmax(1)))
            pgr = float(np.mean(D[np.arange(N), pick] > 0))
            print(f"{('ML: ' + mk):<34}{reg.mean():>13.6f}"
                  f"{(reg_rand - reg.mean()) / reg_rand:>10.0%}{top1:>8.1%}"
                  f"{pgr:>9.0%}   ({time.perf_counter() - t0:.0f}s)")
            rows[mk] = {"regret": float(reg.mean()),
                        "removed": float((reg_rand - reg.mean()) / reg_rand),
                        "top1_strict": top1, "positive_gain_rate": pgr}
        print(f"{'oracle（上界）':<34}{0.0:>13.6f}{1.0:>10.0%}{1.0:>8.1%}"
              f"{float(np.mean(D.max(1) > 0)):>9.0%}")

        line = ref["只看 step 的查表 (fold 内定)"]
        beat = [m for m, r in rows.items() if r["regret"] < line]
        print(f"\n跨过『只看 step』这条线的模型: {beat or '无'}  (线 = {line:.6f})")
        out["sets"][tag] = {"n_feat": int(X.shape[1]), "cost_ms_per_state": cost,
                            "n_scaffold": len(groups), "ref": ref,
                            "models": rows, "beat_step_only": beat,
                            "n_strict": int(strict.sum())}

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
