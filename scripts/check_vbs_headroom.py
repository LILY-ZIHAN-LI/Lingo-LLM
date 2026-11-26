"""检验 SBS→VBS 的"潜在收益"里有多少是真实的分子级异质性，多少是选择噪声。

背景：有人用 fixed_val.json 算了 SBS（全局最优固定 expert）和 VBS（每个分子
事后挑最好的 expert），得到 16%~65% 的"潜在提升"，并据此主张「不同分子需要
不同 expert，值得训一个 selector」。

问题在于 VBS 是 max over 4 experts of 一个用 2 个重复估出来的均值。max 是
有偏的：即使四个 expert 对每个分子的真实期望完全相同，VBS 也会系统性高于
SBS，高出来的量就是 4 个噪声样本的最大值的期望。这是 winner's curse，不是信号。

本脚本给同一批数据算四个量：

  SBS       全局最优固定 expert 的均值（和原分析一致）
  VBS_naive 每分子事后取 max（和原分析一致，乐观有偏）
  VBS_cv    交叉拟合：用 seed A 挑 expert，在 seed B 上计分，两个方向平均。
            这是"只凭一次带噪观测做分子级选择"能真实兑现的收益，无偏但被
            选择噪声衰减，是下界。
  VBS_null  参数化零假设：假设不存在分子×expert 交互（每个分子的 expert
            效应 = 全局 expert 均值 + 分子主效应），用实测的重复间方差注入
            噪声，重采样 2000 次重算 VBS_naive。这给出"纯噪声能伪造多少
            潜在收益"的分布。

外加一个两因素含重复的方差分解：把 final 的方差拆成分子主效应 / expert
主效应 / 分子×expert 交互 / 重复残差。selector 唯一能吃到的就是交互项，
所以 sigma2_interaction 是不是显著大于 0 才是这条路线的生死线。

用法:
  python scripts/check_vbs_headroom.py
  python scripts/check_vbs_headroom.py --metric final --nboot 2000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 标签 -> (fixed_val.json 所在目录, 有效维度)
MANIFEST = {
    "RDKit 原始 (2.20)": ("results_base0", 2.20),
    "RDKit sbt-random (2.42)": ("results_sbt_random", 2.42),
    "RDKit sbt-last (2.68)": ("results_sbt_last", 2.68),
    "RDKit sbt-first (3.01)": ("results_sbt_first", 3.01),
    "Transformer 生成式": ("results_gen_router", None),
}


def load_cube(path: Path, metric: str):
    """返回 (cube, mols, experts)，cube[m, e, r] = 第 r 次重复的指标值。"""
    recs = json.loads(path.read_text())
    by: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in recs:
        pol = r["policy"]
        if not pol.startswith("only_"):
            continue
        by[(r["smiles"], pol)].append((r["episode_seed"], float(r[metric])))

    mols = sorted({m for m, _ in by})
    experts = sorted({e for _, e in by})
    n_rep = max(len(v) for v in by.values())

    cube = np.full((len(mols), len(experts), n_rep), np.nan)
    for (m, e), vals in by.items():
        vals.sort()  # 按 seed 排序，保证所有 (分子,expert) 的重复下标对齐同一个 seed
        for r, (_, v) in enumerate(vals):
            cube[mols.index(m), experts.index(e), r] = v
    return cube, mols, experts


def sbs_vbs(cube):
    """SBS / VBS_naive / VBS_cv。cube[m,e,r]。"""
    per_me = np.nanmean(cube, axis=2)           # (M, E) 每分子每 expert 的均值
    sbs_idx = int(np.argmax(np.nanmean(per_me, axis=0)))
    sbs = float(np.nanmean(per_me[:, sbs_idx]))
    vbs_naive = float(np.nanmean(np.nanmax(per_me, axis=1)))

    # 交叉拟合：seed r 上挑 expert，在另一个 seed 上计分
    n_rep = cube.shape[2]
    scores = []
    for a in range(n_rep):
        for b in range(n_rep):
            if a == b:
                continue
            pick = np.nanargmax(cube[:, :, a], axis=1)
            scores.append(cube[np.arange(cube.shape[0]), pick, b])
    vbs_cv = float(np.nanmean(np.concatenate(scores)))
    return sbs, vbs_naive, vbs_cv, sbs_idx


def null_vbs(cube, nboot, rng):
    """零假设下 VBS_naive − SBS 的分布：无分子×expert 交互，只有主效应+噪声。"""
    M, E, R = cube.shape
    per_me = np.nanmean(cube, axis=2)
    mol_eff = np.nanmean(per_me, axis=1)                  # 分子主效应
    exp_eff = np.nanmean(per_me, axis=0) - np.nanmean(per_me)  # expert 主效应（中心化）
    # 重复间噪声标准差（按 (分子,expert) 单元的组内方差池化）
    resid = cube - per_me[:, :, None]
    dof = np.sum(~np.isnan(cube)) - M * E
    sigma = float(np.sqrt(np.nansum(resid ** 2) / max(dof, 1)))

    truth = mol_eff[:, None] + exp_eff[None, :]           # 无交互的真实均值
    gaps = np.empty(nboot)
    for i in range(nboot):
        sim = truth[:, :, None] + rng.normal(0, sigma, size=(M, E, R))
        pm = sim.mean(axis=2)
        gaps[i] = pm.max(axis=1).mean() - pm[:, int(np.argmax(pm.mean(axis=0)))].mean()
    return gaps, sigma


def anova(cube):
    """两因素含重复的方差分解，返回各方差成分（负值截到 0）。"""
    M, E, R = cube.shape
    grand = np.nanmean(cube)
    per_me = np.nanmean(cube, axis=2)
    mol_m = np.nanmean(per_me, axis=1)
    exp_m = np.nanmean(per_me, axis=0)

    ss_mol = E * R * np.sum((mol_m - grand) ** 2)
    ss_exp = M * R * np.sum((exp_m - grand) ** 2)
    inter = per_me - mol_m[:, None] - exp_m[None, :] + grand
    ss_int = R * np.nansum(inter ** 2)
    ss_res = np.nansum((cube - per_me[:, :, None]) ** 2)

    ms_res = ss_res / max(M * E * (R - 1), 1)
    ms_int = ss_int / max((M - 1) * (E - 1), 1)
    ms_mol = ss_mol / max(M - 1, 1)
    ms_exp = ss_exp / max(E - 1, 1)

    return {
        "sigma2_resid": ms_res,
        "sigma2_inter": max((ms_int - ms_res) / R, 0.0),
        "sigma2_mol": max((ms_mol - ms_int) / (E * R), 0.0),
        "sigma2_exp": max((ms_exp - ms_int) / (M * R), 0.0),
        "F_inter": ms_int / ms_res if ms_res > 0 else float("nan"),
        "df_inter": ((M - 1) * (E - 1), M * E * (R - 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="final", choices=["final", "auc"])
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/vbs_headroom.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = {}
    print(f"指标 = {args.metric}\n")
    hdr = (f"{'专家池':<26}{'SBS':>9}{'VBS_naive':>11}{'VBS_cv':>10}"
           f"{'naive gap':>11}{'cv gap':>10}{'null gap p95':>13}{'p':>8}")
    print(hdr)
    print("-" * len(hdr.replace("专家池", "x" * 6)))

    for label, (d, dim) in MANIFEST.items():
        p = ROOT / d / "fixed_val.json"
        if not p.exists():
            print(f"{label:<26}  (缺 {p})")
            continue
        cube, mols, experts = load_cube(p, args.metric)
        if cube.shape[2] < 2:
            print(f"{label:<26}  (只有 {cube.shape[2]} 次重复，无法交叉拟合)")
            continue

        sbs, vbs_n, vbs_cv, sbs_idx = sbs_vbs(cube)
        gaps, sigma = null_vbs(cube, args.nboot, rng)
        obs = vbs_n - sbs
        pval = float(np.mean(gaps >= obs))
        av = anova(cube)

        print(f"{label:<26}{sbs:>9.5f}{vbs_n:>11.5f}{vbs_cv:>10.5f}"
              f"{obs:>+11.5f}{vbs_cv - sbs:>+10.5f}"
              f"{np.percentile(gaps, 95):>+13.5f}{pval:>8.3f}")

        out[label] = {
            "effdim": dim, "n_mol": len(mols), "n_expert": len(experts),
            "n_rep": int(cube.shape[2]), "sbs_expert": experts[sbs_idx],
            "sbs": sbs, "vbs_naive": vbs_n, "vbs_cv": vbs_cv,
            "gap_naive": obs, "gap_cv": vbs_cv - sbs,
            "gap_naive_pct": 100 * obs / sbs if sbs else None,
            "gap_cv_pct": 100 * (vbs_cv - sbs) / sbs if sbs else None,
            "null_gap_mean": float(gaps.mean()),
            "null_gap_p95": float(np.percentile(gaps, 95)),
            "p_value": pval, "rep_noise_sigma": sigma,
            "anova": {k: (v if not isinstance(v, tuple) else list(v))
                      for k, v in av.items()},
        }

    print("\n方差分解（selector 只能吃到 sigma2_inter 这一项）\n")
    hdr2 = (f"{'专家池':<26}{'sigma2_mol':>12}{'sigma2_exp':>12}"
            f"{'sigma2_inter':>14}{'sigma2_resid':>14}{'F_inter':>9}")
    print(hdr2)
    print("-" * len(hdr2.replace("专家池", "x" * 6)))
    for label, r in out.items():
        a = r["anova"]
        print(f"{label:<26}{a['sigma2_mol']:>12.3e}{a['sigma2_exp']:>12.3e}"
              f"{a['sigma2_inter']:>14.3e}{a['sigma2_resid']:>14.3e}"
              f"{a['F_inter']:>9.2f}")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
