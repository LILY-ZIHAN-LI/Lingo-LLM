"""EXP 2 / Step 3：在 200 个 train 分子上训练 Static Selector，在 50 个 val 分子上评。

这是第一次真正的 train -> held-out 评估。之前 EXP 1 全部是 50 个分子上的
LOO，受两个限制：检出下限是 oracle 缺口的 47%（n=200 后降到 24%），
而且那 50 个分子已经被反复用于模型/特征/规则选择，任何在它们上面
"最好" 的数字都含选择偏差。

本脚本的分工：
  train (200 分子，results_v2/labels/*.json)  -> 拟合模型、选超参、选 SBS、拟合 StrongRule
  dev   ( 50 分子，results_*/fixed_val.json)  -> 评估，不参与任何选择

**纯 ML 与 Hybrid 分开报。** 纯 ML 组只含 A/B/C/rf/twohead；Hybrid 组允许
在 train 上从 {纯 ML 模型, StrongRule} 里挑一个。两者混在一起报会让
"机器学习的增量" 变得不可读 —— 如果 Hybrid 赢而纯 ML 不赢，赢的是规则。

dev 上的 50 个分子已经被此前的分析反复使用，所以这里的数字是 **开发集
结果**，不是最终结论；最终数字只能来自冻结的 test 集（100 分子）。

用法:
  python scripts/train_expert_selector.py --pool sbt_first
  python scripts/train_expert_selector.py --pool base0 --features F1,F3,F4,F5,F7,F8
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem, RDLogger                      # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold       # noqa: E402
from scipy.stats import trim_mean                     # noqa: E402

from run_loo_selector import (EDITOR_KWARGS, boot_ci,  # noqa: E402
                              fit_predict, load_labels)
from run_nested_selector import RuleFamily, two_head   # noqa: E402
from src.analysis import hv_at_eligible                # noqa: E402
from src.editors import make_editors                   # noqa: E402
from src.features import build_matrix                  # noqa: E402
from src.interfaces import EXPERTS                     # noqa: E402
from src.oracles import OracleBank                     # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
ML_MODELS = ("A", "B", "C", "rf", "twohead")


def load_train(path: Path):
    """train 标签 -> (mols, Y, AT, n_rep_used)。只取每个 (分子,expert) 的第一次重复
    作为主标签；额外重复只用于噪声诊断，不进训练，否则 train 里 25 个分子
    的标签精度高于其余 175 个，等于偷偷加权。"""
    rows = json.loads(path.read_text())
    g = defaultdict(list)
    for r in rows:
        g[(r["smiles"], r["policy"])].append(r)
    mols = sorted({m for m, _ in g})
    pols = [f"only_{e}" for e in EXPERTS]
    Y = np.zeros((len(mols), NE))
    AT = np.zeros((len(mols), NE))
    for i, m in enumerate(mols):
        for j, p in enumerate(pols):
            rs = sorted(g[(m, p)], key=lambda r: r["episode_seed"])
            Y[i, j] = float(rs[0]["final"])
            AT[i, j] = float(hv_at_eligible(rs[0], 5) is not None)
    return mols, Y, AT


def robust(d):
    return {"mean": float(d.mean()), "winsor10": float(trim_mean(d, 0.1)),
            "median": float(np.median(d)), "winrate": float(np.mean(d > 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--features", default="F1,F3,F4,F5,F7,F8,F9")
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    run_dir = f"results_{args.pool}"
    ekw = EDITOR_KWARGS[run_dir]
    tr_mols, Ytr, ATtr = load_train(ROOT / f"results_v2/labels/train_{args.pool}.json")
    dv_mols, Ydv, _, H5dv, ATdv = load_labels(run_dir)

    def scafs(ms):
        return {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
                for s in ms}
    overlap = scafs(tr_mols) & scafs(dv_mols)
    print(f"pool {args.pool}  editor {ekw}")
    print(f"train {len(tr_mols)} 分子 / dev {len(dv_mols)} 分子；"
          f"scaffold 交集 {len(overlap)} 个 "
          f"-> {'互不重叠 ✓' if not overlap else '有重叠，结果不可用 ✗'}")
    if overlap:
        return 1

    ed, bank = make_editors(**ekw), OracleBank()
    sets = args.features.split(",")
    Xtr, Xdv, names = {}, {}, {}
    for fs in sets:
        Xtr[fs], names[fs] = build_matrix(tr_mols, ed, bank, fs)
        Xdv[fs], _ = build_matrix(dv_mols, ed, bank, fs)

    # --- 基线：SBS 和 StrongRule，参数全部在 train 上定 ---
    sbs = int(np.argmax(Ytr.mean(0)))
    rule = RuleFamily(names["F5"]).fit(Xtr["F5"], Ytr)
    rpick = rule.predict(Xdv["F5"])
    print(f"\nSBS（在 train 上选）= {EXPERTS[sbs]}")
    print(f"StrongRule（在 train 上选）= 规则 {rule.best[0]}，参数 {rule.best[1]:.4g}")

    idx = np.arange(len(dv_mols))
    hv_sbs = Ydv[:, sbs]
    hv_rule = Ydv[idx, rpick]

    # --- 内层 CV 在 train 上选 (特征组, 模型) ---
    def inner_score(cands):
        m = len(tr_mols)
        cuts = np.array_split(rng.permutation(m), args.inner_folds)
        best, bs = None, -np.inf
        for fs in sets:
            for mk in cands:
                if mk == "rule" and not RuleFamily.supports(names[fs]):
                    continue
                tot = 0.0
                for cut in cuts:
                    msk = np.ones(m, bool); msk[cut] = False
                    A, Ya, Aa = Xtr[fs][msk], Ytr[msk], ATtr[msk]
                    B = Xtr[fs][~msk]
                    try:
                        if mk == "twohead":
                            s = two_head(A, Ya, Aa, B)
                        elif mk == "rule":
                            pk = RuleFamily(names[fs]).fit(A, Ya).predict(B)
                            s = np.eye(NE)[pk]
                        else:
                            s = fit_predict(mk, A, Ya, B, rng)
                    except Exception:
                        s = np.zeros((len(B), NE))
                    pk = s.argmax(1)
                    tot += Ytr[~msk][np.arange(len(pk)), pk].sum()
                if tot > bs:
                    best, bs = (fs, mk), tot
        return best

    def fit_full(fs, mk):
        if mk == "twohead":
            return two_head(Xtr[fs], Ytr, ATtr, Xdv[fs]).argmax(1)
        if mk == "rule":
            return RuleFamily(names[fs]).fit(Xtr[fs], Ytr).predict(Xdv[fs])
        return fit_predict(mk, Xtr[fs], Ytr, Xdv[fs], rng).argmax(1)

    print("\n在 train 上做 5-fold 内层选择 ...")
    sel_ml = inner_score(ML_MODELS)
    sel_hy = inner_score(tuple(ML_MODELS) + ("rule",))
    print(f"  纯 ML 组选中   : {sel_ml[0]} / {sel_ml[1]}")
    print(f"  Hybrid 组选中  : {sel_hy[0]} / {sel_hy[1]}")
    pick_ml = fit_full(*sel_ml)
    pick_hy = fit_full(*sel_hy)

    vbs = Ydv.max(1).mean()
    print(f"\n=== dev（50 分子）固定调用预算 HV@20 ===")
    print(f"{'策略':<26}{'HV':>9}{'Δ vs SBS':>11}{'95% CI':>24}{'Top-1':>8}"
          f"{'winsor10':>10}{'中位数':>10}{'赢面':>7}")
    print("-" * 106)
    print(f"{'SBS (train 选 ' + EXPERTS[sbs] + ')':<26}{hv_sbs.mean():>9.5f}"
          f"{0.0:>+11.5f}{'—':>24}{np.mean(sbs == Ydv.argmax(1)):>8.1%}"
          f"{'—':>10}{'—':>10}{'—':>7}")
    out_rows = {}
    for nm, pk in [("StrongRule", rpick),
                   (f"纯 ML ({sel_ml[0]}/{sel_ml[1]})", pick_ml),
                   (f"Hybrid ({sel_hy[0]}/{sel_hy[1]})", pick_hy)]:
        d = Ydv[idx, pk] - hv_sbs
        lo, hi = boot_ci(d, rng)
        rb = robust(d)
        print(f"{nm:<26}{Ydv[idx, pk].mean():>9.5f}{d.mean():>+11.5f}"
              f"{'*' if lo > 0 else ' '}[{lo:+.5f}, {hi:+.5f}]"
              f"{np.mean(pk == Ydv.argmax(1)):>8.1%}{rb['winsor10']:>+10.5f}"
              f"{rb['median']:>+10.5f}{rb['winrate']:>7.0%}")
        out_rows[nm] = {"hv": float(Ydv[idx, pk].mean()), "ci_vs_sbs": [lo, hi],
                        "top1": float(np.mean(pk == Ydv.argmax(1))), **rb}
    print(f"{'VBS (诊断)':<26}{vbs:>9.5f}{vbs - hv_sbs.mean():>+11.5f}"
          f"{'—':>24}{1.0:>8.1%}{'—':>10}{'—':>10}{'—':>7}")

    print(f"\n配对比较（核心命题）")
    for nm, pk in [("纯 ML − StrongRule", pick_ml), ("Hybrid − StrongRule", pick_hy)]:
        d = Ydv[idx, pk] - hv_rule
        lo, hi = boot_ci(d, rng)
        print(f"  {nm:<22}{d.mean():>+11.5f}{'*' if lo > 0 else ' '}"
              f"[{lo:+.5f}, {hi:+.5f}]   中位数 {np.median(d):+.5f}  赢面 {np.mean(d > 0):.0%}")
        out_rows[nm] = {"d_hv": float(d.mean()), "ci": [lo, hi], **robust(d)}

    print(f"\n=== dev 固定候选预算 HV@5（按 pair 支撑 + 覆盖率）===")
    mc = {}
    for nm, pk in [("StrongRule", rpick), ("纯 ML", pick_ml), ("Hybrid", pick_hy)]:
        ok = ~np.isnan(H5dv[idx, pk]) & ~np.isnan(H5dv[:, sbs])
        d5 = H5dv[idx, pk][ok] - H5dv[:, sbs][ok]
        l5, h5 = boot_ci(d5, rng) if ok.sum() > 5 else (np.nan, np.nan)
        print(f"  {nm:<14}支撑 {ok.sum():>2}/50  Δ vs SBS {d5.mean():>+.5f}"
              f"{'*' if l5 > 0 else ' '}[{l5:+.5f}, {h5:+.5f}]"
              f"  覆盖率 {np.mean(ATdv[idx, pk]):.0%} (SBS {np.mean(ATdv[:, sbs]):.0%})")
        mc[nm] = {"support": int(ok.sum()), "d_hv5": float(d5.mean()),
                  "ci": [l5, h5], "coverage": float(np.mean(ATdv[idx, pk]))}

    op = ROOT / (args.out or f"results_v2/selector/static_{args.pool}.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({
        "pool": args.pool, "editor_kwargs": ekw,
        "n_train": len(tr_mols), "n_dev": len(dv_mols),
        "scaffold_overlap": len(overlap),
        "sbs_expert": EXPERTS[sbs], "strong_rule": list(map(str, rule.best[:2])),
        "selected_ml": list(sel_ml), "selected_hybrid": list(sel_hy),
        "vbs": float(vbs), "sbs_hv": float(hv_sbs.mean()),
        "rows": out_rows, "matched_candidate": mc,
        "new_oracle_calls": bank.stats()["logical_calls"],
    }, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
