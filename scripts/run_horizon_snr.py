"""Horizon-SNR Pilot：真实的反事实动作优势噪声随 horizon 怎么变。

**纠正一个此前的错误测量。** 我曾用 `DynamicRouter − TwoPhase` 整条轨迹的
分子内 SD（0.004566）当作 A(s,a) 的标签噪声，并据此判 EXP 10 不可行。
那是错的：那两条策略从第 0 步就分叉、走完全不同的 20 步，解耦彻底；
而 A(s,a) 的两个分支**共享 s 之前的全部前缀，之后跟随同一个策略 B**，
共同前缀和共同续策略会抵消大量方差。0.004566 是个（很可能很松的）上界。

这里测真的：
    从同一个状态快照 s 出发
      分支 a ∈ {4 个 expert}：先执行 a，之后一律跟随 TwoPhase 到第 20 步
      每个分支换 R 个 proposal seed 重复
    一次 rollout 记录全部 horizon 的累计 ΔHV（1/2/3/5/10/20），
    不为每个 horizon 重跑 —— 这把成本降了一个数量级。

对每个 horizon h 报：
    signal   跨动作的优势幅度 = mean_s[ max_a Ā(s,a) ]   （Ā 是 R 次均值）
    noise    同 (s,a) 跨 seed 的 SD
    SNR      signal / noise
    rank_agree  单次 seed 的 argmax 与 R 次均值 argmax 的一致率
    agree_with_1step  horizon h 的最优动作与 horizon 1 的最优动作一致率

**SNR < 1 不作为否决门槛。** 标签带噪但无偏时，跨分子的稳定规律仍可能
被学出来；最终判据是泛化收益。这里只给出噪声随 horizon 的实际形状，
供决定采哪个 horizon。

用法:
  python scripts/run_horizon_snr.py --n-mol 40 --reps 5
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                              # noqa: E402

from run_oracle_router import pool_env_kw               # noqa: E402
from src.env import MolEnv                              # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig       # noqa: E402
from src.oracles import OracleBank                      # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 960000
HORIZONS = (1, 2, 3, 5, 10, 20)


def branch(env, a, two, max_steps, seed_shift):
    """从 env 的状态克隆，先走 a，之后跟随 TwoPhase 到 max_steps。

    返回长度 max_steps+1 的累计 ΔHV 曲线（相对分支起点）。
    env 本身不变。
    """
    e1, e2, k = two
    c = env.clone()
    c.episode_seed += seed_shift
    h0 = c.hv
    cur = [0.0]
    act = a
    while c.t < max_steps:
        _, _, done, info = c.step(act)
        cur.append(float(info["hv"] - h0))
        act = e1 if c.t < k else e2       # 第一步之后一律 TwoPhase
        if done:
            break
    while len(cur) < max_steps + 1:       # 提前结束就补平
        cur.append(cur[-1])
    return np.asarray(cur)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("--n-mol", type=int, default=40)
    ap.add_argument("--states", default="0,4,9,14")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--behaviors", default="random,round_robin,bandit")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", default="results_v2/horizon/horizon_snr.csv")
    args = ap.parse_args()

    env_kw = pool_env_kw(args.pool)
    two = tuple(json.loads(
        (ROOT / "results_v2/baselines/baseline_meta.json").read_text())["two_phase"])
    steps = [int(x) for x in args.states.split(",")]
    behs = args.behaviors.split(",")
    pols = pickle.loads((ROOT / f"results_{args.pool}/policies.pkl").read_bytes())
    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    nstate = len(mols) * len(behs) * len(steps)
    print(f"pool {args.pool}  续策略 = TwoPhase({EXPERTS[two[0]]}->"
          f"{EXPERTS[two[1]]}@{two[2]})")
    print(f"{len(mols)} 分子 x {len(behs)} 行为策略 x {len(steps)} 个 step "
          f"= {nstate} 个状态")
    print(f"每状态 {NE} 动作 x {args.reps} 次重复 = {NE * args.reps} 条续轨迹；"
          f"一次 rollout 记录全部 {len(HORIZONS)} 个 horizon\n")

    # CUM[状态, 动作, 重复, horizon]
    CUM, meta = [], []
    t0, oc0 = time.perf_counter(), bank.n_calls
    for bi, b in enumerate(behs):
        pol = pols[b]
        for mi, m in enumerate(mols):
            ep = SEED_BASE + bi * 100000 + mi
            obs = env.reset(m, ep)
            if hasattr(pol, "reset"):
                pol.reset(ep)
            for t in range(args.max_steps):
                if t in steps:
                    blk = np.zeros((NE, args.reps, len(HORIZONS)))
                    for a in range(NE):
                        for r in range(args.reps):
                            cur = branch(env, a, two, args.max_steps,
                                         seed_shift=7919 * (r + 1))
                            rel = [cur[min(t + h, args.max_steps)] for h in HORIZONS]
                            blk[a, r] = rel
                    CUM.append(blk)
                    meta.append({"smiles": m, "behavior": b, "step": t,
                                 "current_smiles": env.current})
                a = pol.act(obs, t)
                obs, _, done, _ = env.step(a)
                if done:
                    break
        print(f"  {b}: 累计 {len(CUM)} 个状态 ({time.perf_counter() - t0:.0f}s)",
              flush=True)

    C = np.asarray(CUM)                       # (S, A, R, H)
    S = len(C)
    el = time.perf_counter() - t0
    print(f"\n墙钟 {el / 60:.1f} 分钟；逻辑 oracle {bank.n_calls - oc0}"
          f"（缓存命中 {bank.n_hits}）\n")

    mean_a = C.mean(2)                        # (S, A, H) R 次均值
    sd_a = C.std(2, ddof=1)                   # (S, A, H) 同 (s,a) 跨 seed 的 SD
    base_h1 = mean_a[:, :, 0].argmax(1)       # horizon=1 的最优动作

    print(f"{'horizon':<9}{'signal':>11}{'noise':>11}{'SNR':>8}"
          f"{'单次 argmax 一致':>16}{'与 h=1 同选':>13}{'优势 SD(跨状态)':>16}")
    print("-" * 86)
    rows = []
    for hi, h in enumerate(HORIZONS):
        mu = mean_a[:, :, hi]
        # signal：最优动作相对该状态平均动作的优势
        sig = float(np.mean(mu.max(1) - mu.mean(1)))
        noise = float(np.mean(sd_a[:, :, hi]))
        # 单次 seed 的 argmax 与 R 次均值 argmax 的一致率
        single = C[:, :, :, hi].argmax(1)     # (S, R)
        agree = float(np.mean(single == mu.argmax(1)[:, None]))
        agree1 = float(np.mean(mu.argmax(1) == base_h1))
        spread = float(np.std(mu.max(1) - mu.mean(1)))
        snr = sig / noise if noise > 0 else float("inf")
        print(f"{h:<9}{sig:>11.6f}{noise:>11.6f}{snr:>8.2f}"
              f"{agree:>16.0%}{agree1:>13.0%}{spread:>16.6f}")
        rows.append({"horizon": h, "signal": sig, "noise": noise, "snr": snr,
                     "single_seed_argmax_agree": agree,
                     "agree_with_h1": agree1, "advantage_sd_across_states": spread,
                     "n_state": S, "reps": args.reps})

    # 单次重复下的标签噪声（EXP 10 真正会用的配置）
    print(f"\n若每个分支只跑 1 次（EXP 10 的原始配置），标签的 SE = 上面的 noise；")
    print(f"跑 R 次取均值则 SE = noise/sqrt(R)。各 horizon 下达到 "
          f"SE <= signal/2 所需的重复次数：")
    for r in rows:
        need = int(np.ceil((2 * r["noise"] / r["signal"]) ** 2)) if r["signal"] > 0 else -1
        print(f"  horizon {r['horizon']:<3} 需要 {need:>4} 次/分支"
              f"（1000 状态 x 4 动作 = {1000 * 4 * need:,} 条续轨迹）")
        r["reps_needed_for_half_signal_se"] = need

    print(f"\n参照：此前用 Router-vs-TwoPhase 整条轨迹算的 0.004566 是**上界**，"
          f"\n      真实 horizon=20 的反事实噪声是上表的 {rows[-1]['noise']:.6f}"
          f"（{rows[-1]['noise'] / 0.004566:.0%} of 它）")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(op.parent / "horizon_cum.npy", C)
    (op.parent / "horizon_meta.json").write_text(json.dumps({
        "pool": args.pool, "two_phase": list(two), "horizons": list(HORIZONS),
        "n_state": S, "reps": args.reps, "states": steps, "behaviors": behs,
        "n_mol": len(mols), "wall_seconds": el,
        "oracle_logical": int(bank.n_calls - oc0),
        "meta": meta}, ensure_ascii=False))
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
