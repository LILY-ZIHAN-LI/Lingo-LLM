"""给 findings §16c 的预注册预测打分。

§16c 在跑之前写下了三条预测，针对 per-template 采样的三个 pool
（2.42 / 2.68 / 3.01 有效维度，四个 pool 的模板集合逐字节相同，
只有"一个模板选哪个产物"这个无化学含义的 tie-break 不同）：

  (1) bandit − random 在匹配候选预算下应随维度上升，在 3.01 处 ≈ +0.0034
      （从 §12d 拟合的 +0.0017/维，以 1.02 空对照为起点）。
  (2) bandit − 最强单 expert > 0 在 3.01 处成立。
  (3) 若 (2) 在 3.01（可达上限 3.45 的 87%）处仍不成立，则"routing 收益
      随动作维度上升"只是相关性、不是可用的杠杆。

这个脚本不挑预算：三个点统一在 x=5 上算，和 §14、§16c 引用的基准一致。
各 run 自己的 pick_candidate_budget 会自适应选不同的 x，那样三个点不可比
（§12c 踩过同一个坑）。

公共支撑取「六个 router + 被选中的单 expert」都攒够 x 个合格候选的 episode，
所有 pair 用同一批，避免按 pair 丢弃低产出轨迹反而污染要控制的变量。

用法:
  python scripts/score_sbt.py
  python scripts/score_sbt.py --budget 5 --out results/sbt_score.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, hv_at_eligible, summarize_pair,
)

# 标签 -> (结果目录, 有效维度 json)
# §14 的 2.20 维那个点放在最前面：它和下面三个必须用同一个预算、同一套
# 公共支撑规则重算，否则四行来自不同口径，放在一张表里就是误导。
MANIFEST = {
    "base (§14)": ("results_base0", "results_sweep/action_space_a000.json"),
    "random": ("results_sbt_random", "results/action_space_hand_sbt.json"),
    "last": ("results_sbt_last", "results/action_space_hand_sbt_last.json"),
    "first": ("results_sbt_first", "results/action_space_hand_sbt_first.json"),
}
ROUTERS = ["round_robin", "gap", "greedy", "bandit", "dqn"]
# §14 在原始 pool 上就是用 only_sa 当横杆的，早于这三个 run，所以把它固定下来
# 当作三个点共用的横杆是预先指定、不是事后在 test 上挑的。
FIXED_BAR = "only_sa"

# §12d 拟合：bandit 每单位有效维度 +0.00170，α=1 的空对照在 1.02 维处为 0
SLOPE = 0.00170
NULL_DIM = 1.02


def eff_dim(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("effective_action_dim_scaled")


def load_rows(run: str, keep: list[str]) -> tuple[list[dict], str, dict]:
    """router 的 raw.json 和单 expert 的 fixed_raw.json 合并成一份行表。

    返回 (rows, 在 val 上选中的 expert, val 分数)。选择必须在 val 上做完——
    在 test 上挑最好的那个等于做 4 次比较只报 1 次。`keep` 指定要并进来的
    固定 expert：只放一个时公共支撑最大，放四个时四根横杆等高、可横向比。
    """
    d = Path(run)
    rows = json.loads((d / "raw.json").read_text())
    fixed = json.loads((d / "fixed_raw.json").read_text())
    summary = json.loads((d / "fixed_summary.json").read_text())
    rows += [r for r in fixed if r["policy"] in keep]
    return rows, summary["best_on_val"], summary["val_score"]


def score_one(run: str, x: float) -> dict:
    # 先用「val 选中的那一个」算主结果：公共支撑最大，且符合 §14 的协议。
    rows, best, val_score = load_rows(run, [FIXED_BAR])
    if best != FIXED_BAR:
        rows, _, _ = load_rows(run, [best, FIXED_BAR])
    keys = common_support(rows, x)
    cov = coverage_at(rows, x)

    def val(r):
        return hv_at_eligible(r, x)

    vs_random = [summarize_pair(rows, p, "random", val, restrict=keys)
                 for p in ROUTERS]
    vs_sel = [summarize_pair(rows, p, best, val, restrict=keys)
              for p in ROUTERS + ["random"]]
    vs_bar = ([summarize_pair(rows, p, FIXED_BAR, val, restrict=keys)
               for p in ROUTERS + ["random"]] if best != FIXED_BAR else vs_sel)

    # 绝对水平，供 §15 的回本计算用
    level = {}
    for p in set(ROUTERS + ["random", best, FIXED_BAR]):
        vs = [val(r) for r in rows if r["policy"] == p]
        vs = [v for v in vs if v is not None]
        if vs:
            level[p] = sum(vs) / len(vs)

    return {
        "run": run,
        "best_on_val": best,
        "val_score": val_score,
        "fixed_bar": FIXED_BAR,
        "budget": x,
        "n_common": len(keys),
        "coverage_common": cov["_common"],
        "vs_random": vs_random,
        "vs_selected_expert": vs_sel,
        "vs_fixed_bar": vs_bar,
        "mean_level": level,
    }


def fmt(r: dict) -> str:
    if r["n"] == 0:
        return f"  {r['a']:12} (无可配对 episode)"
    star = "*" if r["sig"] else " "
    return (f"  {r['a']:12} {r['mean']:+.5f}{star} "
            f"[{r['lo']:+.5f}, {r['hi']:+.5f}]  n={r['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=5.0,
                    help="固定候选预算；三个点必须用同一个值才可比")
    ap.add_argument("--out", default="results/sbt_score.json")
    args = ap.parse_args()

    out = {"budget": args.budget, "slope_ref": SLOPE,
           "null_dim_ref": NULL_DIM, "points": []}

    for pick, (run, dim_path) in MANIFEST.items():
        if not Path(run).exists():
            print(f"[跳过] {run} 不存在")
            continue
        dim = eff_dim(dim_path)
        res = score_one(run, args.budget)
        res["product_pick"] = pick
        res["eff_dim"] = dim
        res["predicted_bandit_vs_random"] = (
            None if dim is None else SLOPE * (dim - NULL_DIM))
        out["points"].append(res)

        dim_s = f"{dim:.2f}" if dim is not None else "?"
        print(f"\n{'=' * 66}")
        print(f"product-pick={pick}  有效维度={dim_s}  {run}")
        print(f"公共支撑 {res['n_common']} episode "
              f"(覆盖率 {res['coverage_common']:.0%})，x={args.budget:g} 候选")
        print(f"\n相对 random:")
        for r in res["vs_random"]:
            print(fmt(r))
        if res["predicted_bandit_vs_random"] is not None:
            b = next(r for r in res["vs_random"] if r["a"] == "bandit")
            pred = res["predicted_bandit_vs_random"]
            print(f"  → §12d 拟合线预测 bandit {pred:+.5f}，"
                  f"实测 {b['mean']:+.5f}，差 {b['mean'] - pred:+.5f}")
        print(f"\n相对 {res['best_on_val']}（val 选出的零参数基线，§14 协议）:")
        for r in res["vs_selected_expert"]:
            print(fmt(r))
        if res["best_on_val"] != FIXED_BAR:
            print(f"  ⚠ val 选中的是 {res['best_on_val']}，不是三点共用横杆 "
                  f"{FIXED_BAR}；val 分数 "
                  f"{res['val_score'][res['best_on_val']]:+.5f} vs "
                  f"{res['val_score'][FIXED_BAR]:+.5f}")
            print(f"\n相对 {FIXED_BAR}（预先指定的共用横杆）:")
            for r in res["vs_fixed_bar"]:
                print(fmt(r))

    # 预测 (2) 的判定：3.01 那个点上 bandit 是否显著 > 单 expert
    pt = next((p for p in out["points"] if p["product_pick"] == "first"), None)
    if pt:
        b = next(r for r in pt["vs_selected_expert"] if r["a"] == "bandit")
        held = b["sig"] and b["mean"] > 0
        out["prediction_2_held"] = bool(held)
        print(f"\n{'=' * 66}")
        print(f"预测 (2) —— bandit > 最强单 expert @ 3.01 维："
              f"{'成立' if held else '不成立'}")
        print(f"  bandit − {pt['best_on_val']} = {b['mean']:+.5f} "
              f"[{b['lo']:+.5f}, {b['hi']:+.5f}]")
        if not held:
            print("  → 落入 §16c 的结局 (3)：维度相关性成立但不是可用杠杆。")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
