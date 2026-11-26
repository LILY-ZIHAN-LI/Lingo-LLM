"""EXP 3 / Step 4：逐步反事实采集 —— 同一状态下四个 expert 的真实收益。

为什么必须重新采而不能用已有轨迹：之前那个"免费 step-wise 初筛"
（最优 expert 随 t 从 gsk3b 迁移到 qed/jnk3，逐步 oracle 比全程 oracle
高 +0.0013~0.0019）是沿**各 expert 自己的轨迹**算的边际增益 —— 四条轨迹
在第 t 步的分子根本不是同一个，所以那只能当信号探测，不能当标签。

这里的做法：跑一条行为策略轨迹，在指定的几个 step 上把环境
`clone()` 四份（共享 bank，见 src/env.py:clone 的说明），四个 expert
各从**逐字相同**的 (current, archive, seen, t, 历史计数) 出发走一步，
记下各自的 ΔHV 和产出。连续执行四个动作构造标签是错的。

Pilot 规模（按 Step 4 的表）：
  30 分子 x 3 行为策略 x 5 个状态 = 450 个状态，450 x 4 = 1800 次反事实调用
  正式：200 分子 -> 3000 个状态，12000 次反事实调用
上面不含生成基础轨迹本身的成本，脚本会把两部分分开报。

Pilot 要先回答三个问题（Step 4 的要求）：
  1. 同一状态下四个 expert 的收益是否有明显差异
  2. 最优 expert 是否随状态变化
  3. 这些差异是否大于生成噪声（同一状态同一 expert 换 seed 重跑的波动）

用法:
  python scripts/collect_counterfactuals.py --pool sbt_first --n-mol 30
  python scripts/collect_counterfactuals.py --pool sbt_first --n-mol 200 \
      --out results_v2/counterfactuals/cf_sbt_first_full.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                       # noqa: E402

from run_loo_selector import EDITOR_KWARGS       # noqa: E402
from src.env import MolEnv                       # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank                # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
BEHAVIORS = ("random", "round_robin", "bandit")   # bandit = LinUCB
SEED_BASE = 700000                                # 与 train(600000)/val(500000) 不重叠


def probe(env, action, seed_shift=0):
    """从 env 的当前状态克隆一份，执行 action 一步，返回这一步的结果。

    env 本身不被修改 —— 调用方可以对同一个状态连续 probe 四个 action。
    """
    c = env.clone()
    if seed_shift:
        c.episode_seed = c.episode_seed + seed_shift
    hv0 = c.hv
    _, r, _, info = c.step(action)
    lg = c.logs[-1]
    return {
        "d_hv": float(info["hv"] - hv0),
        "reward": float(r),
        "n_proposed": int(lg.n_proposed),
        "n_valid": int(lg.n_valid),
        "n_eligible": int(lg.n_scaffold_ok),
        "n_novel": int(lg.n_novel),
        "entered": bool(lg.entered_archive),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("--n-mol", type=int, default=30)
    ap.add_argument("--states", default="0,4,9,14,19",
                    help="在哪些 step 上做反事实探查")
    ap.add_argument("--noise-reps", type=int, default=2,
                    help="同一状态同一 expert 换 seed 重跑几次，用来量生成噪声")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = f"results_{args.pool}"
    ekw = EDITOR_KWARGS[run_dir]
    env_kw = dict(ekw)
    env_kw.update({"sample_by_template": args.pool.startswith("sbt"),
                   "product_pick": {"sbt_first": "first", "sbt_last": "last",
                                    "sbt_random": "random"}.get(args.pool, "random")})
    states = [int(x) for x in args.states.split(",")]
    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]

    pol_all = pickle.loads((ROOT / run_dir / "policies.pkl").read_bytes()) \
        if (ROOT / run_dir / "policies.pkl").exists() else None
    if pol_all is None:
        raise SystemExit(f"缺 {run_dir}/policies.pkl —— LinUCB 行为策略取不到")
    pols = {b: pol_all[b] for b in BEHAVIORS}

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    print(f"pool {args.pool}  env {env_kw}")
    print(f"{len(mols)} 分子 x {len(BEHAVIORS)} 行为策略 x {len(states)} 个状态"
          f" = {len(mols) * len(BEHAVIORS) * len(states)} 个状态")
    print(f"反事实调用 {len(mols) * len(BEHAVIORS) * len(states) * NE}"
          f"（+ 噪声复查 {args.noise_reps} 次/状态/expert）\n")

    t0 = time.perf_counter()
    oracle_before = bank.n_calls
    samples = []
    base_steps = 0

    for bi, b in enumerate(BEHAVIORS):
        pol = pols[b]
        for mi, m in enumerate(mols):
            ep_seed = SEED_BASE + bi * 100000 + mi
            obs = env.reset(m, ep_seed)
            if hasattr(pol, "reset"):
                pol.reset(ep_seed)
            for t in range(args.max_steps):
                if t in states:
                    rec = {
                        "seed_smiles": m, "behavior": b, "episode_seed": ep_seed,
                        "step": t, "current_smiles": env.current,
                        "archive_size": len(env.archive),
                        "n_seen": len(env.seen),
                        "budget_left": 1.0 - t / args.max_steps,
                        "obs": np.asarray(obs).tolist(),
                        "calls_so_far": env.calls_so_far.tolist(),
                        "gain_so_far": env.gain_so_far.tolist(),
                        "actions": {},
                    }
                    for a in range(NE):
                        r0 = probe(env, a)
                        # 噪声复查：同一状态同一 action 换 proposal seed
                        reps = [probe(env, a, seed_shift=10_000 * (k + 1))["d_hv"]
                                for k in range(args.noise_reps)]
                        r0["d_hv_reps"] = reps
                        rec["actions"][EXPERTS[a]] = r0
                    samples.append(rec)
                a = pol.act(obs, t)
                obs, _, done, _ = env.step(a)
                base_steps += 1
                if done:
                    break
        print(f"  {b}: 累计 {len(samples)} 个状态  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    el = time.perf_counter() - t0
    n_cf = len(samples) * NE * (1 + args.noise_reps)
    print(f"\n墙钟 {el / 60:.1f} 分钟")
    print(f"基础轨迹 {base_steps} 步；反事实探查 {n_cf} 次"
          f"（含噪声复查）")
    print(f"逻辑 oracle 打分 {bank.n_calls - oracle_before}"
          f"（缓存命中 {bank.n_hits}，去重分子 {len(bank._cache)}）")

    out = ROOT / (args.out or
                  f"results_v2/counterfactuals/cf_{args.pool}_{len(mols)}mol.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False))
    (out.parent / (out.stem + "_meta.json")).write_text(json.dumps({
        "pool": args.pool, "env_kw": env_kw, "n_mol": len(mols),
        "behaviors": list(BEHAVIORS), "states": states,
        "noise_reps": args.noise_reps, "n_state": len(samples),
        "n_counterfactual_probe": n_cf, "base_trajectory_steps": base_steps,
        "wall_seconds": el, "oracle_logical_calls": bank.n_calls - oracle_before,
        "oracle_cache_hits": bank.n_hits, "seed_base": SEED_BASE,
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
