"""fig12：§16c 的预注册预测打分（findings §18）。

两栏讲同一件事的两半，必须放在一起看，单看任何一栏都会得到相反的结论：

  (a) 相对 random 的收益 —— §12d 的拟合线（实线段是拟合覆盖的
      1.02–2.50，点线是外推）预测在 3.01 维处 bandit 应得 +0.0034，
      实测 +0.0035。预测 (1) 兑现。
  (b) 相对**零参数固定 expert** 的收益 —— 同样的四个点上全部压在 0 附近。
      预测 (2) 不成立：(a) 里那 +0.0035 并不是"路由"挣来的，
      而是"别用弱 expert"挣来的，后者不需要学习。

两栏共用 y 轴范围会让 (b) 退化成一条零线，所以刻意分开标尺，并在 (b) 上
画出 ±§7 的基线噪声带作为参照，避免把 (b) 的小尺度读成"几乎赢了"。

  python scripts/plot_sbt.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import (  # noqa: E402
    DISPLAY, GRID, INK, INK_SECONDARY, MARKER, PALETTE, apply_style,
)

SHOW = ["greedy", "bandit", "dqn"]
# 三个 policy 在同一个 x 上，不错开就会叠成一条贯穿全图的竖线，
# 读起来像分隔线而不是三组数据（fig11 的生成式点踩过同一个坑）。
JITTER = {"greedy": -0.045, "bandit": 0.0, "dqn": 0.045}


def load_curve(path: str) -> dict:
    """§12d 的七个同族点，只取 bandit 以外也要用的三个 policy。"""
    tab = json.loads(Path(path).read_text())["table"]
    out = {}
    for k, v in tab.items():
        if k == "generative" or not v.get("eff_dim"):
            continue
        out[v["eff_dim"]] = {
            n: float(np.mean([s[n][0] for s in v["seeds"]]))
            for n in SHOW if n in v["seeds"][0]
        }
    return out


def load_sbt(path: str) -> tuple[dict, dict]:
    """(相对 random, 相对共用固定 expert 横杆)，键是有效维度。

    2.20 那个点（§14 的 base run）只进 (b) 栏。(a) 栏那里已经有 §12d 扫出来
    的三种子点了，而两者的公共支撑不同（把 only_sa 并进来会筛掉它够不着的
    episode），数值因此不同。同一个横坐标上画两个值会被读成矛盾，
    实际上是两个口径 —— 所以 (a) 栏只画三个新池。
    """
    blob = json.loads(Path(path).read_text())
    vs_rand, vs_bar = {}, {}
    for p in blob["points"]:
        if p.get("eff_dim") is None:
            continue
        d = p["eff_dim"]
        vs_bar[d] = {r["a"]: (r["mean"], r["lo"], r["hi"])
                     for r in p["vs_fixed_bar"]}
        if p.get("product_pick", "").startswith("base"):
            continue
        vs_rand[d] = {r["a"]: (r["mean"], r["lo"], r["hi"])
                      for r in p["vs_random"]}
    return vs_rand, vs_bar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results_sweep/dimension_curve.json")
    ap.add_argument("--sbt", default="results/sbt_score.json")
    ap.add_argument("--out", default="figures/fig12_fixed_expert_bar.png")
    ap.add_argument("--no-caption", action="store_true",
                    help="不画图内说明。放进论文时用：LaTeX 的 \\caption 已经"
                         "讲了同样的话，两份重复还会把图撑高一页。")
    args = ap.parse_args()

    fam = load_curve(args.curve)
    vs_rand, vs_bar = load_sbt(args.sbt)

    apply_style()
    # 上下两栏共享 x 轴，不并排：两栏画的是同一个横坐标上的两种纵量，
    # 竖着对读才是重点（每个维度上"赢 random 多少"对"赢固定 expert 多少"）。
    # 并排还会让 (b) 栏左边三分之二空着 —— 那里没有固定 expert 的数据。
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(7.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.16})

    # ---------------- (a) 相对 random ----------------
    axa.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axa.set_axisbelow(True)
    axa.axhline(0, color=PALETTE["random"], linestyle=(0, (4, 2)),
                linewidth=1.2, zorder=1)

    fx = np.array(sorted(fam))
    for name in SHOW:
        c = PALETTE[name]
        fy = np.array([fam[d][name] for d in fx])
        b, a = np.polyfit(fx, fy, 1)
        gx = np.linspace(fx.min() - 0.04, fx.max() + 0.04, 40)
        axa.plot(gx, a + b * gx, color=c, linewidth=1.0, alpha=0.45, zorder=2)
        # 拟合只覆盖到 2.50；再往右是外推，必须换成点线，否则会被读成拟合
        ex = np.linspace(fx.max() + 0.04, max(vs_rand) + 0.06, 24)
        axa.plot(ex, a + b * ex, color=c, linewidth=1.0, alpha=0.35,
                 linestyle=(0, (1, 2)), zorder=2)
        axa.plot(fx, fy, marker=MARKER[name], markersize=5.5, color=c,
                 linewidth=0, alpha=0.5, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
        # §16c 的三个点：1 个训练种子，所以画的是该种子自己的 bootstrap CI
        sx = np.array(sorted(vs_rand)) + JITTER[name]
        sm = np.array([vs_rand[d][name][0] for d in sorted(vs_rand)])
        slo = np.array([vs_rand[d][name][1] for d in sorted(vs_rand)])
        shi = np.array([vs_rand[d][name][2] for d in sorted(vs_rand)])
        axa.vlines(sx, slo, shi, color=c, linewidth=1.5, alpha=0.7, zorder=4)
        axa.plot(sx, sm, marker=MARKER[name], markersize=8.0, color=c,
                 linewidth=0, markeredgecolor=INK, markeredgewidth=1.0,
                 label=DISPLAY[name], zorder=5)

    axa.annotate("pale: §12d sweep (3 seeds, mean)\n"
                 "dark: new pools (1 seed, bars are that seed's CI)\n"
                 "dotted: fit extrapolated past 2.50",
                 xy=(1.02, 0.00695), fontsize=8.0, color=INK_SECONDARY,
                 ha="left", va="top")
    axa.set_title("(a) vs random — the prediction held", fontsize=10.5,
                  loc="left", pad=6)
    axa.set_ylabel("HV gain vs random")
    # 白底：图例落在零参考线上方，无底色时虚线会从文字中间穿过
    axa.legend(loc="lower right", handletextpad=0.6, fontsize=8.5,
               ncol=3, columnspacing=1.2, frameon=True, framealpha=1.0,
               edgecolor="none", borderpad=0.5)

    # ---------------- (b) 相对零参数固定 expert ----------------
    axb.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axb.set_axisbelow(True)
    axb.axhline(0, color=INK, linestyle=(0, (4, 2)), linewidth=1.2, zorder=1)

    base = np.array(sorted(vs_bar))
    # 四个共有维度上下各画一条极淡的竖线，帮助在两栏之间对齐视线
    for ax in (axa, axb):
        for d in base:
            ax.axvline(d, color=GRID, linewidth=0.7, zorder=0)
    for name in SHOW:
        c = PALETTE[name]
        bx = base + JITTER[name]
        m = np.array([vs_bar[d][name][0] for d in base])
        lo = np.array([vs_bar[d][name][1] for d in base])
        hi = np.array([vs_bar[d][name][2] for d in base])
        axb.vlines(bx, lo, hi, color=c, linewidth=1.5, alpha=0.7, zorder=3)
        axb.plot(bx, m, marker=MARKER[name], markersize=8.0, color=c,
                 linewidth=0, markeredgecolor=INK, markeredgewidth=1.0,
                 zorder=4)

    # (a) 栏右端 bandit 的量级，搬到 (b) 的标尺上作对照 —— 没有这条线，
    # (b) 栏的点会被读成"差一点就赢了"，而实际上它们小了一个数量级。
    ref = vs_rand[max(vs_rand)]["bandit"][0]
    axb.annotate(f"the fixed-expert baseline was only run on these four pools;\n"
                 f"prediction (2) said all four should be > 0 by 3.01.\n"
                 f"Note the scale: (a)'s bandit gain at 3.01 is {ref:+.4f},\n"
                 f"{ref / 0.0008:.0f}× the largest value this panel can show.",
                 xy=(1.02, 0.00074), fontsize=8.0, color=INK_SECONDARY,
                 ha="left", va="top", linespacing=1.5)
    axb.set_title("(b) vs the zero-parameter fixed expert — it did not",
                  fontsize=10.5, loc="left", pad=6)
    axb.set_ylim(-0.00082, 0.00082)
    axb.set_ylabel("HV gain vs\nfixed expert")
    axb.set_xlabel("Effective action dimension  "
                   "(participation ratio, per-objective normalised)")
    axa.set_xlim(0.95, 3.12)

    # 必须手动折行：savefig 的 bbox="tight" 会把画布撑到最宽元素的宽度，
    # 一行到底的说明比绘图区宽 60%，于是图右边空出一大片。
    if not args.no_caption:
        _caption(fig)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"写入 {args.out}")
    return 0


def _caption(fig) -> None:
    cap = ("All values are paired differences on identical held-out episodes at the same candidate\n"
           "budget (x = 5). In (b) the comparison is against only_sa, selected on validation in §14\n"
           "before these pools existed; both sides are deterministic given the pool, so neither\n"
           "noise source in §7 applies to (b).")
    fig.text(0.0, -0.012, cap, fontsize=7.5, color=INK_SECONDARY, ha="left",
             va="top", linespacing=1.45)


if __name__ == "__main__":
    raise SystemExit(main())
