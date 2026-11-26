"""Step 3：两步前瞻诊断 —— 这个环境里有没有值得学的长期决策。

问题：Dynamic Router 预测的是一步收益，但一步收益最高的 expert 未必是
两步累计最高的。TwoPhase 在 6-10 步反超 Router（0.00488 vs 0.00255）
是个迹象，但还不足以证明。

做法：从 train 分子的状态出发，把环境 clone 成树：
    s0 --a--> s1(a) --b--> s2(a,b)        16 条两步路径
一步最优   = argmax_a  ΔHV(a)
两步最优   = argmax_a  max_b [ΔHV(a) + ΔHV(b | a)]
两者选的 a 不同，且两步累计差距显著，才说明长期决策有价值。

每个分支必须从**逐字相同**的快照出发（src/env.py:clone 共享 bank 和
editors，深拷贝 current/archive/seen/t/历史计数）。连续执行构造的序列
是错的 —— 那是一条轨迹，不是反事实树。

成本：每个状态 4 次一步探查 + 16 次两步探查 = 20 次，正好是你给的
「每个状态最多 20 次专家探查」上限。

关键区分（避免把噪声读成长期价值）：
  - 只看"选择不同的比例"不够，必须看**两步累计收益的差**有多大
  - 用同一状态换 proposal seed 重跑来量生成噪声，差距要显著大于它
  - 即时收益为 0 的动作能不能带来后续收益，单独统计

用法:
  python scripts/run_lookahead.py --n-state 300
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                           # noqa: E402

from run_loo_selector import EDITOR_KWARGS, boot_ci  # noqa: E402
from run_oracle_router import pool_env_kw            # noqa: E402
from src.env import MolEnv                           # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig    # noqa: E402
from src.oracles import OracleBank                   # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 900000


def step_from(env, a, seed_shift=0):
    """从 env 的当前状态克隆一份走一步，返回 (新 env, ΔHV)。env 不变。"""
    c = env.clone()
    if seed_shift:
        c.episode_seed += seed_shift
    h0 = c.hv
    _, _, _, info = c.step(a)
    return c, float(info["hv"] - h0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("--n-state", type=int, default=300)
    ap.add_argument("--states", default="0,4,9,14",
                    help="在哪些 step 上做两步前瞻（第 19 步没有第二步）")
    ap.add_argument("--behaviors", default="random,round_robin,bandit")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--noise-reps", type=int, default=1,
                    help="同状态同动作换 seed 重跑几次，用来量生成噪声")
    ap.add_argument("--tie-tol", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_v2/dynamic_router/lookahead.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kw = pool_env_kw(args.pool)
    steps = [int(x) for x in args.states.split(",")]
    behs = args.behaviors.split(",")
    pol_all = json.loads("{}")
    import pickle
    pol_all = pickle.loads((ROOT / f"results_{args.pool}/policies.pkl").read_bytes())
    pols = {b: pol_all[b] for b in behs}

    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()]
    # 每个状态要 4 + 16 = 20 次探查；按 n_state 反推需要多少分子
    per_mol = len(behs) * len(steps)
    n_mol = max(1, int(np.ceil(args.n_state / per_mol)))
    mols = mols[:n_mol]

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    print(f"pool {args.pool}  env {env_kw}")
    print(f"{len(mols)} 分子 x {len(behs)} 行为策略 x {len(steps)} 个 step "
          f"= {len(mols) * per_mol} 个状态")
    print(f"每状态 {NE} 次一步探查 + {NE * NE} 次两步探查 = {NE + NE * NE} 次\n")

    recs = []
    t0 = time.perf_counter()
    oc0 = bank.n_calls
    for bi, b in enumerate(behs):
        pol = pols[b]
        for mi, m in enumerate(mols):
            ep = SEED_BASE + bi * 100000 + mi
            obs = env.reset(m, ep)
            if hasattr(pol, "reset"):
                pol.reset(ep)
            for t in range(args.max_steps):
                if t in steps:
                    one = np.zeros(NE)
                    two = np.full((NE, NE), -np.inf)
                    elig = np.zeros(NE, dtype=int)
                    noise = []
                    for a in range(NE):
                        c1, g1 = step_from(env, a)
                        one[a] = g1
                        elig[a] = c1.logs[-1].n_scaffold_ok
                        for k in range(args.noise_reps):
                            _, gk = step_from(env, a, seed_shift=10_000 * (k + 1))
                            noise.append(abs(gk - g1))
                        if c1.t < args.max_steps:
                            for bb in range(NE):
                                _, g2 = step_from(c1, bb)
                                two[a, bb] = g1 + g2
                        else:
                            two[a, :] = g1
                    recs.append({
                        "smiles": m, "behavior": b, "step": t,
                        # 必须存**当前**分子：step>0 时它和种子不同，
                        # 下游 TwoStep Router 的特征要从它算
                        "current_smiles": env.current,
                        "archive_size": len(env.archive),
                        "n_seen": len(env.seen),
                        "one": one.tolist(), "two_best_per_a": two.max(1).tolist(),
                        "eligible": elig.tolist(),
                        "noise": float(np.mean(noise)) if noise else 0.0,
                    })
                a = pol.act(obs, t)
                obs, _, done, _ = env.step(a)
                if done:
                    break
        print(f"  {b}: 累计 {len(recs)} 个状态 ({time.perf_counter() - t0:.0f}s)",
              flush=True)

    el = time.perf_counter() - t0
    N = len(recs)
    ONE = np.array([r["one"] for r in recs])
    TWO = np.array([r["two_best_per_a"] for r in recs])
    EL = np.array([r["eligible"] for r in recs])
    STEP = np.array([r["step"] for r in recs])
    NOISE = np.array([r["noise"] for r in recs])

    a1 = ONE.argmax(1)
    a2 = TWO.argmax(1)
    idx = np.arange(N)
    # 核心量：改第一步决策能多拿多少两步累计 HV
    gain = TWO[idx, a2] - TWO[idx, a1]
    useful = ONE.max(1) > 0

    print(f"\n墙钟 {el / 60:.1f} 分钟；逻辑 oracle 打分 {bank.n_calls - oc0}"
          f"（缓存命中 {bank.n_hits}）")
    print(f"生成噪声（同状态同动作换 seed）：均值 {NOISE.mean():.6f}，"
          f"p90 {np.percentile(NOISE, 90):.6f}\n")

    print(f"=== {N} 个状态 ===")
    print(f"一步最优 与 两步最优 选的第一个动作不同: "
          f"{np.mean(a1 != a2):.1%}")
    strict = gain > args.tie_tol
    print(f"其中两步累计多拿的量 > 噪声阈值 {args.tie_tol:g} 的: "
          f"{strict.mean():.1%}（{strict.sum()} 个状态）")
    lo, hi = boot_ci(gain, rng)
    print(f"改第一步决策的平均两步累计增益: {gain.mean():+.6f} "
          f"[{lo:+.6f}, {hi:+.6f}]  中位数 {np.median(gain):+.6f}")
    print(f"  对比：一步决策本身的 regret（随机 vs 最优）"
          f"{np.mean(ONE.max(1) - ONE.mean(1)):+.6f}")

    print(f"\n即时收益为 0 的动作能不能带来后续收益")
    z = ONE <= 0
    z_two = np.where(z, TWO, np.nan)
    has = np.nansum(z_two > args.tie_tol, axis=1) > 0
    print(f"  有「即时 0 但两步 > {args.tie_tol:g}」动作的状态: {has.mean():.1%}")
    print(f"  这些动作里，两步收益均值 {np.nanmean(z_two[z_two > 0]):.6f}"
          if np.any(z_two > 0) else "  （无）")

    print(f"\n{'step':<7}{'状态':>6}{'选择不同':>9}{'显著不同':>9}"
          f"{'平均两步增益':>13}{'一步 regret':>12}")
    print("-" * 60)
    rows = []
    for t in sorted(set(STEP.tolist())):
        m = STEP == t
        lo_t, hi_t = boot_ci(gain[m], rng)
        print(f"{t:<7}{m.sum():>6}{np.mean(a1[m] != a2[m]):>9.1%}"
              f"{np.mean(gain[m] > args.tie_tol):>9.1%}"
              f"{gain[m].mean():>+13.6f}"
              f"{np.mean(ONE[m].max(1) - ONE[m].mean(1)):>+12.6f}")
        rows.append({"scope": f"step_{t}", "n": int(m.sum()),
                     "differ_rate": float(np.mean(a1[m] != a2[m])),
                     "strict_differ_rate": float(np.mean(gain[m] > args.tie_tol)),
                     "mean_two_step_gain": float(gain[m].mean()),
                     "ci_lo": lo_t, "ci_hi": hi_t,
                     "one_step_regret": float(np.mean(ONE[m].max(1) - ONE[m].mean(1)))})
    rows.insert(0, {"scope": "all", "n": N,
                    "differ_rate": float(np.mean(a1 != a2)),
                    "strict_differ_rate": float(strict.mean()),
                    "mean_two_step_gain": float(gain.mean()),
                    "ci_lo": lo, "ci_hi": hi,
                    "one_step_regret": float(np.mean(ONE.max(1) - ONE.mean(1))),
                    "noise_mean": float(NOISE.mean()),
                    "wall_seconds": el,
                    "oracle_logical": int(bank.n_calls - oc0)})

    # 判据
    print()
    verdict = ("长期决策有值得学的空间" if (lo > 0 and gain.mean() > NOISE.mean())
               else "长期决策的增益不大于生成噪声，不支持上 RL")
    print(f"判据: 平均两步增益 {gain.mean():+.6f} vs 生成噪声 {NOISE.mean():.6f}，"
          f"CI 下界 {lo:+.6f}\n-> **{verdict}**")
    rows[0]["verdict"] = verdict

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    keys = ["scope", "n", "differ_rate", "strict_differ_rate",
            "mean_two_step_gain", "ci_lo", "ci_hi", "one_step_regret",
            "noise_mean", "wall_seconds", "oracle_logical", "verdict"]
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    (op.parent / "lookahead_raw.json").write_text(
        json.dumps(recs, ensure_ascii=False))
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
