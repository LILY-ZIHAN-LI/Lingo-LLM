"""EXP 3 / Step 4 第 3 项：完整 20 步 One-step Oracle 轨迹。

为什么必须真跑：pilot 里"逐步最优 − 固定 sa"在 5 个采样 step 上分别是
+0.00195 / +0.00094 / +0.00032 / +0.00022 / +0.00011，把它们加起来当最终
性能是错的 —— 那 5 个数来自**同一条行为策略轨迹**上的不同状态，逐步选
最优会把分子带到完全不同的轨迹上，后面的状态分布随之改变。

所以这里真跑：每一步把环境 clone 四份，四个 expert 各探一步，选即时
ΔHV 最大的那个**真的执行**，然后继续。20 步跑完得到一条完整轨迹。

这条曲线是**诊断上界**，不是公平基线：
  - 它每步付 4 次探查（= 5 倍的编辑算力和 oracle 打分），成本单列
  - 它只贪即时收益，不是全局最优策略的上界，也不是下界
  - 平局按 canonical 顺序取先，和 env 的候选选择规则保持一致

同时跑的对照（共享同一批 (分子, episode_seed)，严格配对）：
  SBS            train 上选出的固定 expert
  Random / RR    随机 / 轮询
  Two-phase      前 k 步一个 expert、之后换另一个；(k, e1, e2) 在 train 上定
  FixedProb      按 train 上各 expert 的最优频率抽样
  StepOnly       只看 step、不看分子的查表策略；train 上每个 step 的最优 expert

用法:
  python scripts/run_oracle_router.py --pool sbt_first --n-mol 50
"""
from __future__ import annotations

import argparse
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
from src.analysis import hv_at_eligible              # noqa: E402
from src.env import MolEnv                           # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig    # noqa: E402
from src.oracles import OracleBank                   # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 800000


def pool_env_kw(pool: str) -> dict:
    kw = dict(EDITOR_KWARGS[f"results_{pool}"])
    kw.update({"sample_by_template": pool.startswith("sbt"),
               "product_pick": {"sbt_first": "first", "sbt_last": "last",
                                "sbt_random": "random"}.get(pool, "random")})
    return kw


def rollout(env, chooser, smiles, ep_seed, max_steps):
    """chooser(env, t) -> (action, n_probe)。n_probe 计入算力成本。"""
    env.reset(smiles, ep_seed)
    hv0 = env.hv
    curve, probes, acts = [0.0], 0, []
    for t in range(max_steps):
        a, np_ = chooser(env, t)
        probes += np_
        acts.append(a)
        _, _, done, info = env.step(a)
        curve.append(info["hv"] - hv0)
        if done:
            break
    elig = sum(lg.n_scaffold_ok for lg in env.logs)
    dead = sum(1 for lg in env.logs if lg.n_scaffold_ok == 0)
    return {"curve": curve, "final": curve[-1], "auc": float(np.sum(curve)),
            "eligible": elig, "proposed": sum(lg.n_proposed for lg in env.logs),
            "n_valid": sum(lg.n_valid for lg in env.logs),
            "dead_calls": dead, "probes": probes, "actions": acts,
            "eligible_curve": list(np.cumsum(
                [0] + [lg.n_scaffold_ok for lg in env.logs]))}


def oracle_chooser(env, t):
    """每步 clone 四份、各探一步，取即时 ΔHV 最大。"""
    best, ba = -np.inf, 0
    for a in range(NE):
        c = env.clone()
        h0 = c.hv
        _, _, _, info = c.step(a)
        g = info["hv"] - h0
        if g > best:            # 平局保留下标小的，和 env 的 canonical 取先一致
            best, ba = g, a
    return ba, NE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_val.txt")
    ap.add_argument("--train-labels", default=None)
    ap.add_argument("--n-mol", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kw = pool_env_kw(args.pool)
    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]

    # ---- 非学习型基线的参数全部在 train 标签上定 ----
    tl = args.train_labels or f"results_v2/labels/train_{args.pool}.json"
    rows = json.loads((ROOT / tl).read_text())
    g = defaultdict(list)
    for r in rows:
        g[(r["smiles"], r["policy"])].append(r)
    tmols = sorted({m for m, _ in g})
    Y = np.array([[g[(m, f"only_{e}")][0]["final"] for e in EXPERTS]
                  for m in tmols])
    sbs = int(np.argmax(Y.mean(0)))
    freq = np.bincount(Y.argmax(1), minlength=NE) / len(Y)
    print(f"pool {args.pool}  env {env_kw}")
    print(f"train({len(tmols)} 分子) 上定的参数: SBS={EXPERTS[sbs]}  "
          f"最优频率={dict(zip(EXPERTS, np.round(freq, 3)))}")

    # StepOnly / Two-phase 需要逐 step 的最优 expert。train 标签只有整条
    # episode 的终点 HV，没有逐步信息，所以用 pilot 的反事实数据来定 —— 它
    # 是 train 分子上采的，不碰 val。
    cf = ROOT / f"results_v2/counterfactuals/cf_{args.pool}_30mol.json"
    step_best = None
    if cf.exists():
        S = json.loads(cf.read_text())
        D = np.array([[s["actions"][e]["d_hv"] for e in EXPERTS] for s in S])
        st = np.array([s["step"] for s in S])
        usable = D.max(1) > 0
        tot = defaultdict(lambda: np.zeros(NE))
        for i in np.where(usable)[0]:
            tot[int(st[i])] += D[i]
        step_best = {t: int(np.argmax(v)) for t, v in tot.items()}
        print(f"逐 step 最优（来自 train 分子的反事实 pilot）: "
              f"{ {k: EXPERTS[v] for k, v in sorted(step_best.items())} }")
    k_switch = 5

    def nearest_step_best(t):
        if not step_best:
            return sbs
        k = min(step_best, key=lambda x: abs(x - t))
        return step_best[k]

    choosers = {
        "OneStepOracle": oracle_chooser,
        f"SBS ({EXPERTS[sbs]})": lambda e, t: (sbs, 0),
        "Random": lambda e, t: (int(rng.integers(NE)), 0),
        "RoundRobin": lambda e, t: (t % NE, 0),
        "FixedProb": lambda e, t: (int(rng.choice(NE, p=freq)), 0),
        "StepOnly": lambda e, t: (nearest_step_best(t), 0),
        "TwoPhase": lambda e, t: (nearest_step_best(0) if t < k_switch else sbs, 0),
    }

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    print(f"\n{len(mols)} 个 {Path(args.seeds).stem} 分子 x {len(choosers)} 策略\n")

    res = defaultdict(list)
    t0 = time.perf_counter()
    for nm, ch in choosers.items():
        oc0, t1 = bank.n_calls, time.perf_counter()
        for i, m in enumerate(mols):
            res[nm].append(rollout(env, ch, m, SEED_BASE + i, args.max_steps))
        print(f"  {nm:<20}{time.perf_counter() - t1:>6.0f}s  "
              f"逻辑 oracle +{bank.n_calls - oc0}", flush=True)

    names = list(choosers)
    base = np.array([r["final"] for r in res[names[1]]])   # SBS
    print(f"\n=== val/dev {len(mols)} 分子，完整 {args.max_steps} 步，严格配对 ===")
    print(f"{'策略':<20}{'HV@20':>9}{'Δ vs SBS':>11}{'95% CI':>22}{'HV@5':>9}"
          f"{'合格候选':>9}{'死调用':>8}{'探查':>7}")
    print("-" * 97)
    out = {}
    for nm in names:
        f = np.array([r["final"] for r in res[nm]])
        d = f - base
        lo, hi = boot_ci(d, rng) if nm != names[1] else (0.0, 0.0)
        h5 = [hv_at_eligible(r, 5) for r in res[nm]]
        h5ok = [v for v in h5 if v is not None]
        print(f"{nm:<20}{f.mean():>9.5f}{d.mean():>+11.5f}"
              f"{'*' if lo > 0 else ' '}[{lo:+.5f},{hi:+.5f}]"
              f"{(np.mean(h5ok) if h5ok else float('nan')):>9.5f}"
              f"{np.mean([r['eligible'] for r in res[nm]]):>9.1f}"
              f"{np.mean([r['dead_calls'] for r in res[nm]]):>8.1f}"
              f"{np.mean([r['probes'] for r in res[nm]]):>7.0f}")
        out[nm] = {"hv20": float(f.mean()), "d_vs_sbs": float(d.mean()),
                   "ci": [lo, hi], "hv5": float(np.mean(h5ok)) if h5ok else None,
                   "hv5_support": len(h5ok),
                   "eligible": float(np.mean([r["eligible"] for r in res[nm]])),
                   "dead_calls": float(np.mean([r["dead_calls"] for r in res[nm]])),
                   "probes": float(np.mean([r["probes"] for r in res[nm]])),
                   "coverage5": len(h5ok) / len(mols)}

    print(f"\n总墙钟 {(time.perf_counter() - t0) / 60:.1f} 分钟；"
          f"逻辑 oracle 打分 {bank.n_calls}（缓存命中 {bank.n_hits}）")
    print("注意 OneStepOracle 每步额外付 4 次探查 = 5 倍编辑算力，"
          "它是诊断上界而非公平基线。")

    op = ROOT / (args.out or f"results_v2/dynamic/oracle_router_{args.pool}.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({"pool": args.pool, "env_kw": env_kw,
                              "seeds": args.seeds, "n_mol": len(mols),
                              "sbs_expert": EXPERTS[sbs],
                              "best_freq": freq.tolist(),
                              "step_best": step_best, "summary": out,
                              "raw": {k: v for k, v in res.items()}},
                             ensure_ascii=False, default=float))
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
