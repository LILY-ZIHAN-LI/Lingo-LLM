"""训练 router 要跑多少次新 campaign 才回本。

proposal 审稿意见 #3：
  "Extend the cost analysis to show when router training pays off relative to
   the strongest heuristic. Include generation and scoring during training and
   estimate the number of new campaigns needed to recover that cost."

记账方式
--------
训练一个 router 要先跑 n_train 条 rollout。那些 rollout 和一次真实 campaign
用的是同一个 env、同样的 max_steps、同样的 k_per_call —— 所以一条训练 rollout
的生成/打分开销**就等于**一次 campaign 的开销。这让换算与单位无关：
用 generation attempt、oracle 调用还是墙上时间算，得到的 campaign 数一样。

给定总预算 B（以 campaign 为单位）：
  全部交给最强免训练启发式：   B * H
  先训练再部署 router：        (B - n_train) * (H + delta)
  router 更划算需要
      B * delta > n_train * (H + delta)
  即
      N* = n_train * (H + delta) / delta

H     = 最强免训练启发式在同一候选预算下的 HV 增益（每次 campaign）
delta = router 相对它的配对增益（同一候选预算，同一批 episode）

delta 有置信区间，所以 N* 也给区间端点。delta 的 CI 跨 0 时 N* 无上界 ——
这不是数值问题，是"没有证据表明它会回本"。

哪些 policy 算有训练成本
------------------------
random / round_robin / only_*  : 0。没有任何参数要拟合。
gap / greedy                   : 共用同一批 random rollout，各记全额
                                 （单独要其中一个时就得跑那么多）。
bandit / dqn                   : 各自在线跑满 n_train 条。
DQN 的梯度更新另算：那是纯 CPU 秒，不消耗 oracle，单独报。

用法:
  python scripts/cost_analysis.py --results results_base0 \
      --train-seeds 200 --train-repeats 3 --log results_base0_log.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, episode_key, hv_at_eligible, pick_candidate_budget,
    summarize_pair,
)

# 没有任何参数要拟合的 policy 前缀
FREE = ("random", "round_robin", "only_")


def is_free(name: str) -> bool:
    return any(name == f or name.startswith(f) for f in FREE)


def parse_log(path: Path) -> dict:
    """从 run_experiment 的日志里抠出三段训练墙上时间。"""
    if not path or not path.exists():
        return {}
    txt = path.read_text()
    secs = [int(s) for s in re.findall(r"^   (\d+)s", txt, flags=re.M)]
    out = {}
    for key, i in (("collect_random", 0), ("bandit", 1), ("dqn", 2)):
        if len(secs) > i:
            out[key] = secs[i]
    m = re.search(r"^   ([\d.]+) (?:分钟|minutes)", txt, flags=re.M)
    if m:
        out["eval_minutes"] = float(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--log", default=None, help="对应的 *_log.txt，用来取墙上时间")
    ap.add_argument("--train-seeds", type=int, default=200)
    ap.add_argument("--train-repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--k-per-call", type=int, default=4)
    args = ap.parse_args()

    res = Path(args.results)
    rows = json.loads((res / "raw.json").read_text())
    fx = res / "fixed_raw.json"
    if fx.exists():
        rows += json.loads(fx.read_text())
        print(f"并入 {fx}（单 expert 基线也是免训练的，要一起参与"
              f"『最强启发式』的评选）\n")
    names = sorted({r["policy"] for r in rows})

    n_train = args.train_seeds * args.train_repeats

    # ---------- 1. 训练开销 ----------
    gen_per_ep = args.max_steps * args.k_per_call
    scored = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        scored[n] = float(np.mean([r["eligible"] for r in sub]))
    # 训练 rollout 的行为策略：greedy/gap 用 random，bandit/dqn 用自己
    behav = {"gap": "random", "greedy": "random", "bandit": "bandit",
             "dqn": "dqn"}
    log = parse_log(Path(args.log)) if args.log else {}
    wall = {"gap": log.get("collect_random"), "greedy": log.get("collect_random"),
            "bandit": log.get("bandit"), "dqn": log.get("dqn")}

    print(f"训练开销（n_train = {args.train_seeds} 种子 x {args.train_repeats} "
          f"repeat = {n_train} 条 rollout）\n")
    print(f"{'policy':13}{'rollout':>9}{'生成尝试':>11}{'打分候选':>11}{'墙上时间':>11}")
    for n in names:
        if is_free(n):
            print(f"{n:13}{0:>9}{0:>11}{0:>11}{'0':>11}")
            continue
        b = behav.get(n, n)
        gen = n_train * gen_per_ep
        sc = n_train * scored.get(b, scored[n])
        w = wall.get(n)
        ws = f"{w/3600:.1f}h" if w else "-"
        print(f"{n:13}{n_train:>9}{gen:>11,}{sc:>11,.0f}{ws:>11}")
    print("\n  『打分候选』= 该 policy 每条 episode 的合格候选均值 x rollout 数，"
          "\n  是 oracle 调用的上界（OracleBank 会缓存重复分子）。"
          "\n  DQN 的梯度更新（3 epoch x 300 次）不消耗 oracle，含在墙上时间里。")

    # ---------- 2. 同一候选预算下的收益 ----------
    x = pick_candidate_budget(rows)
    supp = common_support(rows, x)

    def at_x(r):
        return hv_at_eligible(r, x)

    abs_hv = {}
    for n in names:
        v = [at_x(r) for r in rows
             if r["policy"] == n and episode_key(r) in supp]
        v = [z for z in v if z is not None]
        abs_hv[n] = float(np.mean(v)) if v else float("nan")

    free_names = [n for n in names if is_free(n)]
    base = max(free_names, key=lambda n: abs_hv[n])
    H = abs_hv[base]
    print(f"\n候选预算 x={x}，公共支撑 {len(supp)} episode")
    print(f"最强免训练启发式：{base}，H = {H:+.5f} HV / campaign")
    print(f"  （候选：{', '.join(f'{n} {abs_hv[n]:+.5f}' for n in free_names)}）")

    # ---------- 3. 回本点 ----------
    print(f"\n回本点 N* = {n_train} * (H + delta) / delta\n")
    print(f"{'policy':13}{'delta':>10}{'95% CI':>22}{'N*':>12}{'N* 的 CI 端点':>24}")
    out_rows = []
    for n in names:
        if is_free(n) or n == base:
            continue
        s = summarize_pair(rows, n, base, at_x, restrict=supp)
        if s["n"] == 0:
            continue
        d = s["mean"]

        def nstar(dd):
            if dd <= 0:
                return None
            return n_train * (H + dd) / dd

        ns, lo, hi = nstar(d), nstar(s["lo"]), nstar(s["hi"])
        f = lambda v: "永不回本" if v is None else f"{v:,.0f}"  # noqa: E731
        ci = f"[{s['lo']:+.5f}, {s['hi']:+.5f}]" + ("*" if s["sig"] else " ")
        print(f"{n:13}{d:+10.5f}{ci:>22}{f(ns):>12}"
              f"{f(hi) + ' .. ' + f(lo):>24}")
        out_rows.append({"policy": n, "delta": d, "lo": s["lo"], "hi": s["hi"],
                         "n": s["n"], "sig": s["sig"], "nstar": ns,
                         "nstar_lo": hi, "nstar_hi": lo})

    print("\n  CI 端点一列：左边是乐观端（delta 取 CI 上界），右边是悲观端。"
          "\n  delta <= 0 记『永不回本』—— 再多 campaign 也摊不平，因为每次部署都在亏。")

    # ---------- 4. 墙上时间口径 ----------
    if log.get("collect_random"):
        # 用训练 rollout 的单进程耗时，不用评估的并行摊薄值 ——
        # 一次真实 campaign 是串行跑完的，那才是可比的单价。
        per = log["collect_random"] / n_train
        if per:
            print(f"\n墙上时间参考：一条 rollout 约 {per:.1f} s（单进程，"
                  f"= 一次 campaign 的单价）。"
                  f"\n  N* 与单位无关，但换算过去便于说人话：")
            for r in out_rows:
                if r["nstar"]:
                    print(f"    {r['policy']:9} 回本要再跑 {r['nstar']:,.0f} 次 "
                          f"campaign ≈ {r['nstar'] * per / 3600:,.0f} 核心小时")

    (res / "cost_summary.json").write_text(json.dumps({
        "n_train": n_train, "budget_x": x, "common_support": len(supp),
        "baseline": base, "H": H, "abs_hv": abs_hv,
        "gen_attempts_per_router": n_train * gen_per_ep,
        "wall_seconds": {k: v for k, v in wall.items() if v},
        "breakeven": out_rows,
    }, indent=2))
    print(f"\n写入 {res}/cost_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
