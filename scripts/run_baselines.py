"""阶段二 Step 3+4：完整的强非学习型基线套件 -> baseline_comparison.csv。

为什么要这一步：Random (+0.00381*) 和 RoundRobin (+0.00431*) 已经超过
Static Selector 的 +0.00222*，所以"超过 SBS"不再是有内容的说法。要证明
学习型调度有价值，对标必须是这一族里最强的那个。

策略清单（全部只用决策前信息，参数全部在 train 分子上定）：
  SBS               固定一个 expert
  Random            随机
  RoundRobin        轮询
  FixedProb         按 train 上各 expert 的"每分子最优"频率抽样
  StepOnly          只看 step 的查表；每个 step 的最优 expert 来自 train 反事实
  TwoPhase-deg      **保留**退化版（两阶段都选到同一个 expert）作为单独一行
  TwoPhase          非退化：强制 e1 != e2，(e1, e2, k) 在 train 上网格选
  AvailabilityRule  按 template_sites 取 argmax，并跳过上一次空手而归的 expert
  GapRule           MAMO 的 target-gap 规则（src/policies.py:GapPolicy）
  OneStepOracle     探查全部 4 个 expert 再选 —— 诊断上界，成本单列

统一口径（第 8 项）：同一批 (分子, episode_seed)、同一个 editor 池、
同样的失败重试、同样 20 次常规调用预算。OneStepOracle 的 80 次探查
单列在 probes 一栏，不计入调用预算，但在成本分析里必须算。

AvailabilityRule 的"耗尽记忆"只用自己这条 episode 已经观察到的结果
（上一次调用该 expert 是否产出合格候选），不看未来、不额外探查。

用法:
  python scripts/run_baselines.py --pool sbt_first --fit-twophase
  python scripts/run_baselines.py --pool sbt_first --seeds data/seeds_val.txt
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

from rdkit import RDLogger                              # noqa: E402

from run_loo_selector import EDITOR_KWARGS, boot_ci     # noqa: E402
from run_oracle_router import oracle_chooser, pool_env_kw, rollout  # noqa: E402
from src.analysis import hv_at_eligible                 # noqa: E402
from src.env import MolEnv                              # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig       # noqa: E402
from src.oracles import OracleBank                      # noqa: E402
from src.policies import GapPolicy                      # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 800000          # 与 run_oracle_router.py 一致，两个脚本的结果可配对


# --------------------------------------------------------------------------
def train_stats(pool: str):
    """从 train 标签和 train 分子的反事实数据里取出所有基线需要的参数。"""
    rows = json.loads((ROOT / f"results_v2/labels/train_{pool}.json").read_text())
    g = defaultdict(list)
    for r in rows:
        g[(r["smiles"], r["policy"])].append(r)
    mols = sorted({m for m, _ in g})
    Y = np.array([[g[(m, f"only_{e}")][0]["final"] for e in EXPERTS] for m in mols])
    sbs = int(np.argmax(Y.mean(0)))
    freq = np.bincount(Y.argmax(1), minlength=NE) / len(Y)

    step_best, step_rank = None, None
    cf = ROOT / f"results_v2/counterfactuals/cf_{pool}_200mol.json"
    if not cf.exists():
        cf = ROOT / f"results_v2/counterfactuals/cf_{pool}_30mol.json"
    if cf.exists():
        S = json.loads(cf.read_text())
        D = np.array([[s["actions"][e]["d_hv"] for e in EXPERTS] for s in S])
        st = np.array([s["step"] for s in S])
        tot = defaultdict(lambda: np.zeros(NE))
        for i in range(len(S)):
            tot[int(st[i])] += D[i]
        step_best = {t: int(np.argmax(v)) for t, v in tot.items()}
        step_rank = {t: np.argsort(-v).tolist() for t, v in tot.items()}
    return mols, Y, sbs, freq, step_best, step_rank


def make_choosers(sbs, freq, step_best, two_phase, rng, editors, max_steps):
    """返回 {名字: chooser}。chooser(env, t) -> (action, n_probe)。"""
    def nearest(t):
        if not step_best:
            return sbs
        k = min(step_best, key=lambda x: abs(x - t))
        return step_best[k]

    # AvailabilityRule：用 template_sites（0.20 ms，便宜）取 argmax，
    # 并把"上一次调用后颗粒无收"的 expert 降权。只看已观察到的历史。
    def availability(env, t):
        sites = np.array([sum(editors[e].template_sites(env.current)["heavy"])
                          if editors[e].template_sites(env.current)["shared"] or
                          editors[e].template_sites(env.current)["active_heavy"]
                          else sum(editors[e].template_sites(env.current)["light"])
                          for e in EXPERTS], dtype=float)
        # 本 episode 内该 expert 最近一次调用是否产出合格候选
        last = {}
        for lg in env.logs:
            last[lg.expert] = lg.n_scaffold_ok
        pen = np.array([0.0 if last.get(e, 1) > 0 else -1e9 for e in EXPERTS])
        sc = sites + pen
        if np.all(sc < 0):           # 全都空手过 -> 退回纯 sites
            sc = sites
        return int(np.argmax(sc)), 0

    gap = GapPolicy()

    def gap_rule(env, t):
        return int(gap.act(env._obs(), t)), 0

    ch = {
        f"SBS({EXPERTS[sbs]})": lambda e, t: (sbs, 0),
        "Random": lambda e, t: (int(rng.integers(NE)), 0),
        "RoundRobin": lambda e, t: (t % NE, 0),
        "FixedProb": lambda e, t: (int(rng.choice(NE, p=freq)), 0),
        "StepOnly": lambda e, t: (nearest(t), 0),
        "TwoPhase-deg": lambda e, t: (nearest(0) if t < 5 else sbs, 0),
        "AvailabilityRule": availability,
        "GapRule": gap_rule,
        "OneStepOracle": oracle_chooser,
    }
    if two_phase:
        e1, e2, k = two_phase
        ch[f"TwoPhase({EXPERTS[e1]}->{EXPERTS[e2]}@{k})"] = \
            lambda e, t, e1=e1, e2=e2, k=k: (e1 if t < k else e2, 0)
    return ch


def fit_two_phase(pool, env_kw, tmols, max_steps, n_fit, workers_note=""):
    """在 train 分子上网格搜 (e1, e2, k)，强制 e1 != e2。"""
    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=max_steps), **env_kw)
    sub = tmols[:n_fit]
    best, bs = None, -np.inf
    grid = [(a, b, k) for a in range(NE) for b in range(NE) if a != b
            for k in (3, 5, 7, 10)]
    print(f"  在 train 的 {len(sub)} 个分子上搜 {len(grid)} 个 (e1,e2,k) 组合"
          f"{workers_note} ...")
    t0 = time.perf_counter()
    scores = []
    for a, b, k in grid:
        f = [rollout(env, lambda e, t, a=a, b=b, k=k: (a if t < k else b, 0),
                     m, SEED_BASE + 500000 + i, max_steps)["final"]
             for i, m in enumerate(sub)]
        s = float(np.mean(f))
        scores.append(((a, b, k), s))
        if s > bs:
            best, bs = (a, b, k), s
    scores.sort(key=lambda x: -x[1])
    print(f"  {time.perf_counter() - t0:.0f}s  前 5:")
    for (a, b, k), s in scores[:5]:
        print(f"    {EXPERTS[a]:>6} -> {EXPERTS[b]:<6} @k={k:<3} {s:.5f}")
    return best, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_val.txt")
    ap.add_argument("--n-mol", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--fit-twophase", action="store_true")
    ap.add_argument("--n-fit", type=int, default=60,
                    help="搜 TwoPhase 用多少个 train 分子")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default="results_v2/baselines/baseline_comparison.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kw = pool_env_kw(args.pool)
    tmols, Y, sbs, freq, step_best, step_rank = train_stats(args.pool)
    print(f"pool {args.pool}  env {env_kw}")
    print(f"train({len(tmols)}) 上定的参数: SBS={EXPERTS[sbs]}  "
          f"freq={dict(zip(EXPERTS, np.round(freq, 3)))}")
    print(f"逐 step 最优（train 反事实）: "
          f"{ {k: EXPERTS[v] for k, v in sorted(step_best.items())} }\n")

    two_phase, tp_scores = None, None
    if args.fit_twophase:
        print("拟合非退化 TwoPhase（强制 e1 != e2，参数只看 train）:")
        two_phase, tp_scores = fit_two_phase(args.pool, env_kw, tmols,
                                             args.max_steps, args.n_fit)
        print(f"  选中: {EXPERTS[two_phase[0]]} -> {EXPERTS[two_phase[1]]} "
              f"@ k={two_phase[2]}\n")

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]
    ch = make_choosers(sbs, freq, step_best, two_phase, rng, env.editors,
                       args.max_steps)

    print(f"{len(mols)} 个 {Path(args.seeds).stem} 分子 x {len(ch)} 策略，"
          f"严格配对\n")
    res, wall, ocall = {}, {}, {}
    for nm, fn in ch.items():
        t1, o0 = time.perf_counter(), bank.n_calls
        res[nm] = [rollout(env, fn, m, SEED_BASE + i, args.max_steps)
                   for i, m in enumerate(mols)]
        wall[nm] = time.perf_counter() - t1
        ocall[nm] = bank.n_calls - o0
        print(f"  {nm:<28}{wall[nm]:>6.0f}s  oracle +{ocall[nm]}", flush=True)

    names = list(ch)
    ref = f"SBS({EXPERTS[sbs]})"
    base = np.array([r["final"] for r in res[ref]])

    # 共同支撑：所有策略都攒够 5 个合格候选的分子
    common = np.ones(len(mols), bool)
    for nm in names:
        common &= np.array([hv_at_eligible(r, 5) is not None for r in res[nm]])

    print(f"\n=== {len(mols)} 个 {Path(args.seeds).stem} 分子，完整 "
          f"{args.max_steps} 步 ===")
    print(f"HV@5 共同支撑 = {common.sum()}/{len(mols)} 个分子\n")
    hdr = (f"{'策略':<28}{'HV@20':>9}{'Δ vs SBS':>11}{'95% CI':>22}{'HV-AUC':>9}"
           f"{'HV@5(共支)':>11}{'覆盖率':>7}{'候选':>7}{'死调用':>7}{'探查':>6}{'秒':>6}")
    print(hdr)
    print("-" * len(hdr) * 1)
    rowsout = []
    for nm in names:
        rs = res[nm]
        f = np.array([r["final"] for r in rs])
        d = f - base
        lo, hi = boot_ci(d, rng) if nm != ref else (0.0, 0.0)
        h5c = np.array([hv_at_eligible(r, 5) for r in rs], dtype=object)
        cov = float(np.mean([v is not None for v in h5c]))
        h5v = h5c[common].astype(float).mean() if common.sum() else float("nan")
        rec = {
            "policy": nm, "hv20": f.mean(), "d_vs_sbs": d.mean(),
            "ci_lo": lo, "ci_hi": hi, "sig": int(lo > 0),
            "hv_auc": np.mean([r["auc"] for r in rs]),
            "hv5_common": h5v, "coverage5": cov,
            "eligible": np.mean([r["eligible"] for r in rs]),
            "valid": np.mean([r["n_valid"] for r in rs]),
            "proposed": np.mean([r["proposed"] for r in rs]),
            "dead_calls": np.mean([r["dead_calls"] for r in rs]),
            "probes": np.mean([r["probes"] for r in rs]),
            "oracle_logical": ocall[nm], "wall_seconds": wall[nm],
        }
        rowsout.append(rec)
        print(f"{nm:<28}{rec['hv20']:>9.5f}{rec['d_vs_sbs']:>+11.5f}"
              f"{'*' if lo > 0 else ' '}[{lo:+.5f},{hi:+.5f}]{rec['hv_auc']:>9.4f}"
              f"{h5v:>11.5f}{cov:>7.0%}{rec['eligible']:>7.1f}"
              f"{rec['dead_calls']:>7.1f}{rec['probes']:>6.0f}{wall[nm]:>6.0f}")

    # 对最强非学习型基线的配对差（第 8 项要的核心对标）
    cand = [r for r in rowsout if r["policy"] != "OneStepOracle"]
    strongest = max(cand, key=lambda r: r["hv20"])["policy"]
    sb = np.array([r["final"] for r in res[strongest]])
    print(f"\n最强非学习型基线 = {strongest}（HV@20 {sb.mean():.5f}）")
    print(f"{'策略':<28}{'Δ vs 最强基线':>14}{'95% CI':>22}")
    print("-" * 64)
    for nm in names:
        if nm == strongest:
            continue
        d = np.array([r["final"] for r in res[nm]]) - sb
        lo, hi = boot_ci(d, rng)
        print(f"{nm:<28}{d.mean():>+14.5f}{'*' if lo > 0 else ' '}"
              f"[{lo:+.5f},{hi:+.5f}]")
        next(r for r in rowsout if r["policy"] == nm)["d_vs_strongest"] = d.mean()
        next(r for r in rowsout if r["policy"] == nm)["ci_strongest"] = [lo, hi]

    op = ROOT / args.out_csv
    op.parent.mkdir(parents=True, exist_ok=True)
    keys = ["policy", "hv20", "d_vs_sbs", "ci_lo", "ci_hi", "sig", "hv_auc",
            "hv5_common", "coverage5", "eligible", "valid", "proposed",
            "dead_calls", "probes", "oracle_logical", "wall_seconds",
            "d_vs_strongest"]
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rowsout:
            w.writerow({k: r.get(k, "") for k in keys})
    (op.parent / "baseline_meta.json").write_text(json.dumps({
        "pool": args.pool, "env_kw": env_kw, "seeds": args.seeds,
        "n_mol": len(mols), "max_steps": args.max_steps,
        "sbs_expert": EXPERTS[sbs], "best_freq": freq.tolist(),
        "step_best": step_best, "two_phase": two_phase,
        "two_phase_grid_top": tp_scores[:10] if tp_scores else None,
        "strongest_nonlearned": strongest,
        "hv5_common_support": int(common.sum()),
        "seed_base": SEED_BASE, "rng_seed": args.seed,
    }, indent=2, ensure_ascii=False, default=str))
    print(f"\n-> {op}\n-> {op.parent / 'baseline_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
