"""Step 8：验证集上的完整 20 步实验 —— Dynamic Router vs 全部强基线。

公平性约束（Step 8 + Step 3 的"统一耗尽处理"）：
  - 同一批 (分子, episode_seed)、同一 editor 池、同一 EpisodeConfig
  - 所有普通策略 20 次常规调用，0 次探查
  - OneStepOracle 单独标记：它每步探查 4 个 expert = 80 次额外编辑调用，
    不是公平预算下的普通策略
  - 耗尽处理报两套：`raw`（各策略原样）和 `stale`（统一叠加"跳过本
    episode 内调用过且零收益的 expert"）。这样"ML 是不是只靠少死调用赢"
    可以直接从两套的差里读出来

指标（Step 8 全量）：HV@20 / HV@5(共同支撑) / 覆盖率 / HV-AUC /
HV vs 生成尝试 / HV vs 评分预算 / 合格候选 / 死调用 / 分段收益(1-5,6-10,
11-20) / 墙钟 / 训练与采集成本。

用法:
  python scripts/run_rollout_comparison.py --pool sbt_first
  python scripts/run_rollout_comparison.py --exhaustion stale
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                                   # noqa: E402

from run_baselines import make_choosers, train_stats         # noqa: E402
from run_loo_selector import boot_ci                         # noqa: E402
from run_oracle_router import pool_env_kw, rollout           # noqa: E402
from src.analysis import hv_at_eligible                      # noqa: E402
from src.dynamic_router import DynamicRouter, stale_mask     # noqa: E402
from src.env import MolEnv                                   # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig            # noqa: E402
from src.oracles import OracleBank                           # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 800000           # 与 run_baselines / run_oracle_router 一致


def wrap_stale(fn):
    """给任意 chooser 叠一层统一的耗尽处理：跳过已调用且零收益的 expert。

    实现方式是"若选中的 expert 已 stale，就换成未 stale 的里分数最高的"。
    对不给分数的规则型 chooser，退化为"换成第一个未 stale 的"。
    """
    def g(env, t):
        a, p = fn(env, t)
        bad = stale_mask(env)
        if bad.all() or not bad[a]:
            return a, p
        alt = [j for j in range(NE) if not bad[j]]
        return int(alt[0]) if alt else a, p
    return g


def seg_gain(curve, lo, hi):
    """curve[i] 是走完第 i 步后的累计 HV 增量；取 (lo, hi] 区间的增量。"""
    c = np.asarray(curve, float)
    hi = min(hi, len(c) - 1)
    lo = min(lo, len(c) - 1)
    return float(c[hi] - c[lo])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_val.txt")
    ap.add_argument("--n-mol", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--router", default=None,
                    help="默认读 results_v2/dynamic_router/models/router_meta.json")
    ap.add_argument("--exhaustion", default="both",
                    choices=["raw", "stale", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results_v2/dynamic_router")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kw = pool_env_kw(args.pool)
    tmols, Y, sbs, freq, step_best, _ = train_stats(args.pool)
    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]

    # ---- 基线（含非退化 TwoPhase，参数从 baseline_meta.json 读回）----
    bm = ROOT / "results_v2/baselines/baseline_meta.json"
    two = None
    if bm.exists():
        two = json.loads(bm.read_text()).get("two_phase")
        if isinstance(two, str):
            two = json.loads(two.replace("(", "[").replace(")", "]"))
        if two is not None:
            two = tuple(int(x) for x in two)
    ch = make_choosers(sbs, freq, step_best, two, rng, env.editors, args.max_steps)

    # Availability 组合基线（Step 3 额外要求）
    avail = ch["AvailabilityRule"]

    def nearest(t):
        k = min(step_best, key=lambda x: abs(x - t))
        return step_best[k]

    def avail_step(e, t):
        bad = stale_mask(e)
        want = nearest(t)
        if not bad[want] or bad.all():
            return want, 0
        return avail(e, t)

    def avail_freq(e, t):
        bad = stale_mask(e)
        want = int(rng.choice(NE, p=freq))
        if not bad[want] or bad.all():
            return want, 0
        return avail(e, t)

    ch["Avail+StepOnly"] = avail_step
    ch["Avail+FixedProb"] = avail_freq

    # ---- Dynamic Router ----
    meta_p = ROOT / "results_v2/dynamic_router/models/router_meta.json"
    rp = args.router or (json.loads(meta_p.read_text())["path"]
                         if meta_p.exists() else None)
    router_info = None
    if rp and (ROOT / rp).exists():
        r_none = DynamicRouter.load(ROOT / rp, availability="none")
        r_stale = DynamicRouter.load(ROOT / rp, availability="stale_penalty")
        ch["DynamicRouter"] = r_none.chooser()
        ch["DynamicRouter+Avail"] = r_stale.chooser()
        router_info = {"path": str(rp), "kind": r_none.kind,
                       "group": r_none.group,
                       "feat_cost_ms": r_none.feat_cost_ms}
        print(f"Router: {r_none.kind} / {r_none.group}  "
              f"特征 {r_none.feat_cost_ms:.2f} ms/状态")
    else:
        print("!! 没找到训练好的 Router，只跑基线")

    modes = (["raw", "stale"] if args.exhaustion == "both"
             else [args.exhaustion])
    print(f"\n{len(mols)} 个 {Path(args.seeds).stem} 分子 x {len(ch)} 策略 "
          f"x {len(modes)} 种耗尽处理，严格配对\n")

    allrows, res_by_mode = [], {}
    for mode in modes:
        res, wall, oc = {}, {}, {}
        for nm, fn in ch.items():
            f2 = wrap_stale(fn) if mode == "stale" else fn
            t1, o0 = time.perf_counter(), bank.n_calls
            res[nm] = [rollout(env, f2, m, SEED_BASE + i, args.max_steps)
                       for i, m in enumerate(mols)]
            wall[nm], oc[nm] = time.perf_counter() - t1, bank.n_calls - o0
            print(f"  [{mode}] {nm:<24}{wall[nm]:>6.0f}s  oracle +{oc[nm]}",
                  flush=True)
        res_by_mode[mode] = res

        common = np.ones(len(mols), bool)
        for nm in res:
            common &= np.array([hv_at_eligible(r, 5) is not None
                                for r in res[nm]])
        # 「最强普通基线」必须排除学习型策略和带额外探查的诊断上界 ——
        # 否则 Router 会被选成自己的基线，所有 Δ 都对自己算（踩过）。
        LEARNED = ("DynamicRouter",)
        cand = [n for n in res
                if n != "OneStepOracle"
                and not any(n.startswith(k) for k in LEARNED)]
        strongest = max(cand, key=lambda n: np.mean(
            [r["final"] for r in res[n]]))
        sb = np.array([r["final"] for r in res[strongest]])
        base = np.array([r["final"] for r in res[f"SBS({EXPERTS[sbs]})"]])

        print(f"\n=== [{mode}] HV@5 共同支撑 {common.sum()}/{len(mols)}；"
              f"最强普通基线 = {strongest} ===")
        hdr = (f"{'策略':<24}{'HV@20':>9}{'Δ强基线':>10}{'95% CI':>22}"
               f"{'HV@5(共支)':>11}{'覆盖':>6}{'AUC':>8}{'候选':>6}"
               f"{'死调':>6}{'1-5步':>9}{'6-10':>9}{'11-20':>9}{'探查':>6}")
        print(hdr)
        print("-" * 150)
        for nm in res:
            rs = res[nm]
            f = np.array([r["final"] for r in rs])
            d = f - sb
            lo, hi = boot_ci(d, rng) if nm != strongest else (0.0, 0.0)
            h5 = np.array([hv_at_eligible(r, 5) for r in rs], dtype=object)
            cov = float(np.mean([v is not None for v in h5]))
            h5c = (h5[common].astype(float).mean() if common.sum()
                   else float("nan"))
            s1 = float(np.mean([seg_gain(r["curve"], 0, 5) for r in rs]))
            s2 = float(np.mean([seg_gain(r["curve"], 5, 10) for r in rs]))
            s3 = float(np.mean([seg_gain(r["curve"], 10, 20) for r in rs]))
            rec = {
                "mode": mode, "policy": nm, "hv20": f.mean(),
                "d_vs_sbs": float((f - base).mean()),
                "d_vs_strongest": d.mean(), "ci_lo": lo, "ci_hi": hi,
                "sig_vs_strongest": int(lo > 0), "strongest": strongest,
                "hv5_common": h5c, "coverage5": cov,
                "hv_auc": float(np.mean([r["auc"] for r in rs])),
                "eligible": float(np.mean([r["eligible"] for r in rs])),
                "valid": float(np.mean([r["n_valid"] for r in rs])),
                "proposed": float(np.mean([r["proposed"] for r in rs])),
                "hv_per_attempt": float(np.mean(
                    [r["final"] / max(r["proposed"], 1) for r in rs])),
                "hv_per_valid": float(np.mean(
                    [r["final"] / max(r["n_valid"], 1) for r in rs])),
                "dead_calls": float(np.mean([r["dead_calls"] for r in rs])),
                "gain_1_5": s1, "gain_6_10": s2, "gain_11_20": s3,
                "probes": float(np.mean([r["probes"] for r in rs])),
                "oracle_logical": oc[nm], "wall_seconds": wall[nm],
            }
            allrows.append(rec)
            print(f"{nm:<24}{rec['hv20']:>9.5f}{d.mean():>+10.5f}"
                  f"{'*' if lo > 0 else ' '}[{lo:+.5f},{hi:+.5f}]"
                  f"{h5c:>11.5f}{cov:>6.0%}{rec['hv_auc']:>8.4f}"
                  f"{rec['eligible']:>6.1f}{rec['dead_calls']:>6.1f}"
                  f"{s1:>9.5f}{s2:>9.5f}{s3:>9.5f}{rec['probes']:>6.0f}")
        print()

    od = ROOT / args.outdir
    od.mkdir(parents=True, exist_ok=True)
    keys = list(allrows[0])
    with (od / "rollout_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)

    # 预算口径对照表
    bud = []
    for r in allrows:
        bud.append({k: r[k] for k in
                    ("mode", "policy", "hv20", "hv5_common", "coverage5",
                     "hv_auc", "hv_per_attempt", "hv_per_valid", "proposed",
                     "valid", "probes", "oracle_logical", "wall_seconds")})
    with (od / "budget_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bud[0]))
        w.writeheader()
        w.writerows(bud)

    (od / "rollout_meta.json").write_text(json.dumps({
        "pool": args.pool, "env_kw": env_kw, "seeds": args.seeds,
        "n_mol": len(mols), "max_steps": args.max_steps,
        "sbs_expert": EXPERTS[sbs], "two_phase": two,
        "router": router_info, "modes": modes,
        "seed_base": SEED_BASE, "rng_seed": args.seed,
        "note": "OneStepOracle 每步探查 4 个 expert = 80 次额外编辑调用，"
                "不是公平预算下的普通策略",
    }, indent=2, ensure_ascii=False, default=str))
    print(f"-> {od / 'rollout_comparison.csv'}")
    print(f"-> {od / 'budget_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
