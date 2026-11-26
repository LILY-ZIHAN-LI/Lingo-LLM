"""Dynamic Router 的三张图。

形式选择（先定形式，颜色最后）：
  1. regret vs 特征成本   -> 散点。两个不同单位的量在两个轴上，不是双 y 轴。
     成本跨 0.00~30 ms，用 log x。颜色按模型（分类，固定槽位），marker
     作第二编码（报告可能灰度打印）。三条参照线用中性灰横线，不占序列色。
  2. 特征消融             -> 横向条形，单序列（量级比较），按 regret 排序，
     选择性直标数值。单序列不需要图例，标题已经说明是什么。
  3. 分阶段收益           -> 分组条形。阶段是有序的 (1-5 / 6-10 / 11-20)，
     策略是身份 -> 策略上色、阶段分组。

配色取 src/figstyle.py 的 PALETTE（固定槽位，不循环；report 里 fig7-fig12
用的同一套）。本机没有 node，无法重跑色板校验器，所以不新造颜色，
只复用这套已验证的槽位。

用法: python scripts/plot_router.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import GRID, INK, INK_SECONDARY, PALETTE, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_v2/dynamic_router/figures"

# 分类槽位固定顺序，不循环。模型身份 -> 槽位。
SLOT = ["round_robin", "gap", "greedy", "bandit", "dqn"]
MODEL_STYLE = {
    "ridge":    (PALETTE["round_robin"], "o"),
    "rf":       (PALETTE["gap"], "s"),
    "gb":       (PALETTE["greedy"], "^"),
    "twostage": (PALETTE["bandit"], "D"),
    "mlp":      (PALETTE["dqn"], "v"),
}
MODEL_LABEL = {"ridge": "Ridge", "rf": "Random forest",
               "gb": "Gradient boosting", "twostage": "Two-stage",
               "mlp": "MLP"}


def rd(p):
    p = ROOT / p
    if not p.exists():
        return None
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fig_regret_cost(mc):
    """特征组是**有序**的（R1..R6 按成本递增），所以用类别 x 轴 + 每个模型
    一条折线，而不是散点：散点在 log 轴上把 R1/R2（成本都约 0 ms）压到同一个
    x 上，点和标签全叠在一起。成本放进刻度标签，信息不丢。"""
    refs = {r["model"]: float(r["mean_regret"]) for r in mc
            if r["model"].startswith("[ref]")}
    rows = [r for r in mc if not r["model"].startswith("[ref]")]
    groups = sorted({r["feature_group"] for r in rows},
                    key=lambda g: float(next(x["feat_cost_ms"] for x in rows
                                             if x["feature_group"] == g)))
    cost = {g: float(next(x["feat_cost_ms"] for x in rows
                          if x["feature_group"] == g)) for g in groups}
    models = [m for m in ("ridge", "rf", "gb", "twostage", "mlp")
              if any(r["model"] == m for r in rows)]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(groups))
    # best_fixed 和 step_only 只差 0.00002，标签必须一上一下，否则重叠
    for key, lab, dy, va in [
            ("[ref] random", "Random", 3, "bottom"),
            ("[ref] best_fixed_expert", "Best fixed expert", 3, "bottom"),
            ("[ref] step_only", "Step-only lookup", -4, "top")]:
        if key not in refs:
            continue
        ax.axhline(refs[key], color=INK_SECONDARY, lw=0.9, ls=(0, (4, 2)),
                   zorder=1)
        ax.annotate(lab, xy=(0.012, refs[key]),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, dy), textcoords="offset points",
                    va=va, ha="left", fontsize=8, color=INK_SECONDARY)

    for mk in models:
        c, m = MODEL_STYLE[mk]
        y = [float(next((r["mean_regret"] for r in rows
                         if r["model"] == mk and r["feature_group"] == g),
                        np.nan)) for g in groups]
        ax.plot(x, y, marker=m, ms=7.5, color=c, lw=1.8, mec="white", mew=1.2,
                zorder=3, label=MODEL_LABEL[mk])

    ax.set_xticks(x, [f"{g}\n{cost[g]:.2f} ms" for g in groups])
    ax.set_xlabel("Feature group (ordered by extraction cost per decision)")
    ax.set_ylabel("Mean immediate regret (HV)")
    ax.set_title("Routing regret against the compute it costs to decide",
                 loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=4,
              fontsize=8.5)
    fig.text(0.0, -0.30,
             "Lower is better. R1 = step+budget only; R5 adds template_sites; "
             "R6 adds full applicability.\nR6 costs one propose() per expert "
             "per step — the same compute as probing all four outright — and "
             "buys 0-2 points.",
             fontsize=7.6, color=INK_SECONDARY, va="top", linespacing=1.45)
    fig.savefig(OUT / "regret_vs_cost.png")
    plt.close(fig)


def fig_ablation(ab):
    base = next((r for r in ab if r["removed"] == "(none)"), None)
    rows = [r for r in ab if r["removed"] != "(none)"]
    rows.sort(key=lambda r: float(r["mean_regret"]))
    fig, ax = plt.subplots(figsize=(6.2, 0.52 * len(rows) + 1.9))
    y = np.arange(len(rows))
    vals = [float(r["mean_regret"]) for r in rows]
    # 单序列 -> 一个颜色，标题说明是什么，不需要图例
    ax.barh(y, vals, height=0.62, color=PALETTE["round_robin"],
            edgecolor="white", lw=1.2, zorder=3)
    ax.set_yticks(y, [r["removed"] for r in rows])
    ax.invert_yaxis()
    if base:
        b = float(base["mean_regret"])
        ax.axvline(b, color=INK_SECONDARY, lw=1.0, ls=(0, (4, 2)), zorder=4)
        ax.annotate(f"full model {b:.5f}", xy=(b, len(rows) - 0.4),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY, va="center")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.5f}", xy=(v, yi), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.set_xlabel("Mean immediate regret after removing that feature block (HV)")
    ax.set_title("Which feature blocks the router actually needs",
                 loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.16)
    fig.text(0.0, -0.06 / (0.52 * len(rows) + 1.9) * 4,
             "Bars further right hurt more when removed, so they matter more. "
             "Dashed line is the full model.",
             fontsize=7.6, color=INK_SECONDARY, va="top")
    fig.savefig(OUT / "ablation.png")
    plt.close(fig)


def fig_stages(ro):
    """固定挑 5 条策略，正好等于固定槽位数 —— 色相绝不循环。
    必须包含 DynamicRouter（主角）和当时最强的那条普通基线。"""
    rows = [r for r in ro if r["mode"] == "raw"]
    if not rows:
        return
    by = {r["policy"]: r for r in rows}
    strongest = rows[0].get("strongest", "")
    tp = next((p for p in by if p.startswith("TwoPhase(")), None)
    want = ["DynamicRouter", tp, "RoundRobin",
            next((p for p in by if p.startswith("SBS")), None),
            "OneStepOracle"]
    keep = [by[p] for p in want if p and p in by]
    if len(keep) > len(SLOT):
        keep = keep[:len(SLOT)]

    stages = [("gain_1_5", "Steps 1–5"), ("gain_6_10", "Steps 6–10"),
              ("gain_11_20", "Steps 11–20")]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    n = len(keep)
    w = 0.78 / n
    x = np.arange(len(stages))
    for i, r in enumerate(keep):
        lab = r["policy"]
        if lab == strongest:
            lab += "  (strongest baseline)"
        ax.bar(x + i * w - 0.39 + w / 2,
               [float(r[k]) for k, _ in stages], width=w * 0.86,
               color=PALETTE[SLOT[i]], edgecolor="white", lw=1.2, zorder=3,
               label=lab)
    ax.set_xticks(x, [lab for _, lab in stages])
    ax.set_ylabel("Hypervolume gained in that stage")
    ax.set_title("Where in the episode each policy earns its hypervolume",
                 loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3,
              fontsize=8.5)
    fig.text(0.0, -0.26,
             "31% of mid and late states are saturated — candidates are "
             "produced but none enters the Pareto front — so the routing "
             "headroom sits in the first stage.\nOneStepOracle probes all "
             "four experts each step (80 extra editor calls per episode) and "
             "is a diagnostic ceiling, not a fair-budget policy.",
             fontsize=7.6, color=INK_SECONDARY, va="top", linespacing=1.45)
    fig.savefig(OUT / "stage_gain.png")
    plt.close(fig)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    mc = rd("results_v2/dynamic_router/model_comparison.csv")
    ab = rd("results_v2/dynamic_router/ablation_results.csv")
    ro = rd("results_v2/dynamic_router/rollout_comparison.csv")
    made = []
    if mc:
        fig_regret_cost(mc)
        made.append("regret_vs_cost.png")
    if ab:
        fig_ablation(ab)
        made.append("ablation.png")
    if ro:
        fig_stages(ro)
        made.append("stage_gain.png")
    for m in made:
        p = OUT / m
        print(f"-> {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")
    if not made:
        print("没有可画的 csv")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
