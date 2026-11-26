"""Step 7：TwoStep Router —— 监督学习版的"两步累计收益"路由。

两步前瞻（300 个状态）给了依据，也给了限度：
    改第一步决策的平均两步累计增益  +0.000392 [+0.000253, +0.000551]  显著
    生成噪声                        0.000176
    一步决策本身的 regret            +0.001479      <- 大 3.8 倍
所以长期结构是真的，但只占一步决策价值的 26%；而且几乎全在开局
（step 0 增益 +0.000875，step 14 只剩 +0.000070）。

据此**不训 DQN**：两步增益的量级（0.0004）比 DQN 实测的 learning noise
（0.00355）小一个数量级，训了也测不出来。这里只做监督版：

  目标 y_a = ΔHV(a) + max_b ΔHV(b | a)     <- 两步累计，标签来自 lookahead_raw
  对照 y_a = ΔHV(a)                        <- 一步，同一批状态、同一套特征

两者用**同一批状态、同一套 fold、同一套特征**训练和评估，差异只在标签，
这样"学两步目标值不值"是干净的单变量比较。

评估分两个口径：
  一步 regret   按 ΔHV(a) 算 —— 两步模型在这个口径上可能更差，正常
  两步 regret   按两步累计算 —— 这才是两步模型该赢的地方

注意样本量：lookahead 只有 300 个状态（vs 主训练的 3000），所以这里的
结论比主实验弱一档，CI 会明显更宽。不要拿它和 3000 状态的数字直接比。

用法:
  python scripts/train_twostep_router.py
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

from rdkit import Chem, RDLogger                        # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402

from run_loo_selector import boot_ci                    # noqa: E402
from src.cv import grouped_folds, scaffold_of           # noqa: E402
from src.interfaces import EXPERTS                      # noqa: E402
from train_dynamic_router import fit_fold               # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)


def build(raw):
    """lookahead_raw.json -> (X, ONE, TWO, step, seed_smiles)

    lookahead 记录里没有存 obs/历史，所以特征只能用从 SMILES 和 step 推得的
    那部分（描述符 + 位点 + 片段交互 + step）。这是个真实限制，不是遗漏：
    换特征集合会让它和主实验不可比，所以这里**两个目标用完全相同的特征**，
    只比标签的差异。
    """
    from src.editors import make_editors
    from src.router_features import _expert_fragment_delta
    ed = make_editors(qed_disjoint=True)
    frag = _expert_fragment_delta(ed)

    X, names = [], None
    for r in raw:
        smi = r["current_smiles"]   # 必须是当前分子，不是种子
        mol = Chem.MolFromSmiles(smi)
        v, n = [float(r["step"])], ["step"]
        if mol is None:
            v += [0.0] * 6
        else:
            v += [Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                  rdMolDescriptors.CalcTPSA(mol),
                  float(rdMolDescriptors.CalcNumHBD(mol)),
                  float(rdMolDescriptors.CalcNumHBA(mol)),
                  float(mol.GetNumHeavyAtoms())]
        n += ["mw", "logp", "tpsa", "hbd", "hba", "n_heavy"]
        for j, e in enumerate(EXPERTS):
            ts = ed[e].template_sites(smi)
            act = ts["heavy"] if (ts["shared"] or ts["active_heavy"]) else ts["light"]
            v += [float(sum(act)), float(r["eligible"][j])]
            n += [f"site_{e}", f"elig_{e}"]
            dmw, dlogp, dtpsa, dhbd, _ = frag[e]
            lp = (Crippen.MolLogP(mol) if mol is not None else 0.0) + dlogp
            v += [lp, dtpsa, dhbd]
            n += [f"xi_{e}_logp_after", f"xi_{e}_dtpsa", f"xi_{e}_dhbd"]
        X.append(v)
        names = n
    return (np.asarray(X, float), names,
            np.array([r["one"] for r in raw]),
            np.array([r["two_best_per_a"] for r in raw]),
            np.array([r["step"] for r in raw]),
            [r["smiles"] for r in raw])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results_v2/dynamic_router/lookahead_raw.json")
    ap.add_argument("--models", default="rf,twostage,rf_w")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results_v2/dynamic_router/twostep_comparison.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    raw = json.loads((ROOT / args.raw).read_text())
    X, names, ONE, TWO, step, seed_smi = build(raw)
    N = len(ONE)
    SCAF = scaffold_of(seed_smi)
    FOLD = grouped_folds(SCAF, args.folds, seed=0)
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"{N} 个状态 / {len(np.unique(SCAF))} 个 scaffold / {X.shape[1]} 维特征")
    print(f"（主实验是 3000 个状态，这里只有 {N} 个 —— 结论弱一档，CI 更宽）\n")
    print(f"参照：随机的一步 regret {np.mean(ONE.max(1) - ONE.mean(1)):.6f}，"
          f"两步 regret {np.mean(TWO.max(1) - TWO.mean(1)):.6f}\n")

    rows = []
    print(f"{'标签':<12}{'模型':<10}{'一步 regret':>13}{'两步 regret':>13}"
          f"{'一步 Top-1':>12}{'两步 Top-1':>12}")
    print("-" * 74)
    for tag, Y in [("一步 ΔHV", ONE), ("两步累计", TWO)]:
        for mk in args.models.split(","):
            use = seeds if mk in ("rf", "twostage", "rf_w", "gb_w") else [0]
            r1, r2, t1, t2 = [], [], [], []
            for sd in use:
                pick = np.zeros(N, int)
                for f in range(args.folds):
                    tr, te = FOLD != f, FOLD == f
                    P, _ = fit_fold(mk, X[tr], Y[tr], X[te], SCAF[tr], sd)
                    pick[te] = P.argmax(1)
                i = np.arange(N)
                r1.append(float(np.mean(ONE.max(1) - ONE[i, pick])))
                r2.append(float(np.mean(TWO.max(1) - TWO[i, pick])))
                t1.append(float(np.mean(pick == ONE.argmax(1))))
                t2.append(float(np.mean(pick == TWO.argmax(1))))
            print(f"{tag:<12}{mk:<10}{np.mean(r1):>13.6f}{np.mean(r2):>13.6f}"
                  f"{np.mean(t1):>12.1%}{np.mean(t2):>12.1%}")
            rows.append({"label": tag, "model": mk, "n_seed": len(use),
                         "one_step_regret": float(np.mean(r1)),
                         "two_step_regret": float(np.mean(r2)),
                         "one_step_top1": float(np.mean(t1)),
                         "two_step_top1": float(np.mean(t2)),
                         "n_states": N})

    # 两步标签到底有没有换来两步 regret 的下降
    print()
    best1 = min((r for r in rows if r["label"] == "一步 ΔHV"),
                key=lambda r: r["two_step_regret"])
    best2 = min((r for r in rows if r["label"] == "两步累计"),
                key=lambda r: r["two_step_regret"])
    d = best1["two_step_regret"] - best2["two_step_regret"]
    print(f"按**两步 regret** 比较（这才是两步模型该赢的口径）：")
    print(f"  一步标签最好 {best1['model']:<10}{best1['two_step_regret']:.6f}")
    print(f"  两步标签最好 {best2['model']:<10}{best2['two_step_regret']:.6f}")
    print(f"  差 {d:+.6f}  -> "
          f"{'学两步目标有收益' if d > 0 else '学两步目标没有收益'}")
    print(f"\n注意：{N} 个状态的样本量下，这个差要和两步前瞻测到的"
          f"噪声 0.000176 比；小于它就不能当真。")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
