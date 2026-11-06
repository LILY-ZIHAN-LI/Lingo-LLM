"""Check that env runs and produce a first HV-vs-budget comparison
(random vs round-robin).

Usage: python scripts/smoke_test_env.py --seeds data/seeds_val.txt -n 10
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def run(env, mols, policy_fn, name):
    curves, finals = [], []
    for i, m in enumerate(mols):
        env.reset(m, episode_seed=i)
        rng = random.Random(1000 + i)
        hv0 = env.hv
        curve = [0.0]
        for t in range(env.cfg.max_steps):
            a = policy_fn(rng, t)
            _, _, done, info = env.step(a)
            curve.append(info["hv"] - hv0)
            if done:
                break
        curves.append(curve)
        finals.append(curve[-1])
    arr = np.array(curves)
    print(f"{name:14} final HV gain {np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
          f"(area under curve {np.mean(arr.sum(axis=1)):.3f})")
    return np.array(finals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seeds_val.txt")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    mols = Path(args.seeds).read_text().split()[: args.n]
    env = MolEnv(OracleBank(), EpisodeConfig())
    print(f"obs_dim = {env.obs_dim}, {len(mols)} seeds, {env.cfg.max_steps} steps\n")

    t0 = time.perf_counter()
    f_rand = run(env, mols, lambda rng, t: rng.randrange(len(EXPERTS)), "random")
    f_rr = run(env, mols, lambda rng, t: t % len(EXPERTS), "round-robin")
    dt = time.perf_counter() - t0

    diff = f_rr - f_rand
    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    print(f"\npaired diff (round-robin - random): {np.mean(diff):+.4f}")
    print(f"round-robin better on {wins}/{len(diff)} seeds ({ties} ties)")

    print(f"\n{2*len(mols)} episodes in {dt:.1f}s "
          f"-> {dt/(2*len(mols)):.2f}s/episode on one core")
    print(f"main experiment 6 policies x 100 seeds x 3 repeats = 1800 episodes "
          f"-> {1800*dt/(2*len(mols))/60:.0f} min on one core")
    print(f"oracle: {env.bank.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
