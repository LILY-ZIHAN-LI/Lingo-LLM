"""Step 6：Hybrid Router（前段 ML、后段人工规则）+ 必要的对照。

动机：轨迹实验显示两者在不同阶段各有优势
    1-5 步    Router 0.01016   TwoPhase 0.00593
    6-10 步   Router 0.00255   TwoPhase 0.00488

**但阶段收益不能直接相加，而且理由比"两条独立轨迹"更强**：TwoPhase 在
6-10 步拿到 0.00488 是在**它自己前 5 步造出的状态**上拿的。Router 前 5 步
拿 0.01016 的同时把分子带到 archive 已经更满的状态，TwoPhase 的后半段
换到那里未必还有效。所以 Hybrid 很可能两头落空 —— 必须实跑。

对照（缺一不可）：
  Hybrid-k        前 k 步 Router，之后 TwoPhase 的第二段 expert
  PhaseSplit-k    前 k 步 Router，之后**还是 Router** —— 纯阶段划分对照，
                  用来确认收益不是来自"在第 k 步做了个切换"这件事本身
  RevHybrid-k     前 k 步 TwoPhase，之后 Router —— 反向，检验"ML 擅长前段"
                  这个解释是否成立（若反向也赢，那就不是前段擅长）
  纯 Router / 纯 TwoPhase

切换点 k 必须在 **train** 分子上定；这里把 train 上的结果和 val 上的
结果分开报，不拿 val 选 k。

用法:
  python scripts/run_hybrid.py --fit-on-train
  python scripts/run_hybrid.py --k 5
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

from rdkit import RDLogger                              # noqa: E402

from run_baselines import train_stats                   # noqa: E402
from run_loo_selector import boot_ci                    # noqa: E402
from run_oracle_router import pool_env_kw, rollout      # noqa: E402
from run_rollout_comparison import seg_gain             # noqa: E402
from src.analysis import hv_at_eligible                 # noqa: E402
from src.dynamic_router import DynamicRouter            # noqa: E402
from src.env import MolEnv                              # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig       # noqa: E402
from src.oracles import OracleBank                      # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
NE = len(EXPERTS)
SEED_BASE = 800000          # 与 rollout 一致，可配对
TRAIN_SEED_BASE = 810000    # train 上选 k 用，和 val 的 seed 空间分开


def build(router, two, sbs):
    """两段式策略族。two = (e1, e2, k0) 来自 baseline_meta。"""
    e1, e2, _ = two
    rfn = router.chooser()

    def hybrid(k):
        return lambda e, t: (rfn(e, t) if t < k else (e2, 0))

    def phase_split(k):
        # 纯阶段划分对照：两段都是 Router，只是形式上"切换"
        return lambda e, t: rfn(e, t)

    def rev_hybrid(k):
        return lambda e, t: ((e1, 0) if t < k else rfn(e, t))

    def two_phase(k):
        return lambda e, t: ((e1, 0) if t < k else (e2, 0))

    return hybrid, phase_split, rev_hybrid, two_phase


def evaluate(env, choosers, mols, seed_base, max_steps):
    out = {}
    for nm, fn in choosers.items():
        t0 = time.perf_counter()
        out[nm] = [rollout(env, fn, m, seed_base + i, max_steps)
                   for i, m in enumerate(mols)]
        print(f"    {nm:<22}{np.mean([r['final'] for r in out[nm]]):.5f}"
              f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--router", default=None)
    ap.add_argument("--ks", default="3,5,7,10")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_v2/dynamic_router/hybrid_comparison.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kw = pool_env_kw(args.pool)
    tmols_all, Y, sbs, freq, step_best, _ = train_stats(args.pool)
    two = tuple(json.loads(
        (ROOT / "results_v2/baselines/baseline_meta.json").read_text())["two_phase"])
    meta = json.loads(
        (ROOT / "results_v2/dynamic_router/models/router_meta.json").read_text())
    router = DynamicRouter.load(ROOT / (args.router or meta["path"]))
    print(f"Router {router.kind}/{router.group}   "
          f"TwoPhase = {EXPERTS[two[0]]}->{EXPERTS[two[1]]}@{two[2]}\n")

    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    hybrid, phase_split, rev_hybrid, two_phase = build(router, two, sbs)
    ks = [int(x) for x in args.ks.split(",")]

    # ---------- 在 train 上选 k ----------
    tmols = tmols_all[:args.n_train]
    print(f"在 train 的 {len(tmols)} 个分子上选切换点 k ...")
    ch_tr = {f"Hybrid-{k}": hybrid(k) for k in ks}
    ch_tr["Router"] = router.chooser()
    ch_tr[f"TwoPhase@{two[2]}"] = two_phase(two[2])
    res_tr = evaluate(env, ch_tr, tmols, TRAIN_SEED_BASE, args.max_steps)
    hyb = {k: float(np.mean([r["final"] for r in res_tr[f"Hybrid-{k}"]]))
           for k in ks}
    best_k = max(hyb, key=hyb.get)
    print(f"  -> train 上最好的 k = {best_k}（HV {hyb[best_k]:.5f}）\n")

    # ---------- 在 val 上评 ----------
    vmols = [ln.strip() for ln in (ROOT / "data/seeds_val.txt").read_text()
             .splitlines() if ln.strip()][:args.n_val]
    print(f"在 val 的 {len(vmols)} 个分子上评估 ...")
    ch = {
        "Router": router.chooser(),
        f"TwoPhase@{two[2]}": two_phase(two[2]),
        f"Hybrid-{best_k}": hybrid(best_k),
        f"PhaseSplit-{best_k}": phase_split(best_k),
        f"RevHybrid-{best_k}": rev_hybrid(best_k),
    }
    res = evaluate(env, ch, vmols, SEED_BASE, args.max_steps)

    base = np.array([r["final"] for r in res["Router"]])
    tp = np.array([r["final"] for r in res[f"TwoPhase@{two[2]}"]])
    common = np.ones(len(vmols), bool)
    for nm in res:
        common &= np.array([hv_at_eligible(r, 5) is not None for r in res[nm]])

    print(f"\n=== val {len(vmols)} 分子，HV@5 共同支撑 {common.sum()}/{len(vmols)} ===")
    hdr = (f"{'策略':<20}{'HV@20':>9}{'Δ vs Router':>13}{'95% CI':>22}"
           f"{'Δ vs TwoPhase':>15}{'HV@5':>9}{'1-5':>9}{'6-10':>9}{'11-20':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for nm in res:
        f = np.array([r["final"] for r in res[nm]])
        d, d2 = f - base, f - tp
        lo, hi = boot_ci(d, rng) if nm != "Router" else (0.0, 0.0)
        h5 = np.array([hv_at_eligible(r, 5) for r in res[nm]], dtype=object)
        h5c = h5[common].astype(float).mean() if common.sum() else float("nan")
        s = [float(np.mean([seg_gain(r["curve"], a, b) for r in res[nm]]))
             for a, b in ((0, 5), (5, 10), (10, 20))]
        print(f"{nm:<20}{f.mean():>9.5f}{d.mean():>+13.5f}"
              f"{'*' if lo > 0 else ' '}[{lo:+.5f},{hi:+.5f}]"
              f"{d2.mean():>+15.5f}{h5c:>9.5f}"
              + "".join(f"{v:>9.5f}" for v in s))
        rows.append({"policy": nm, "hv20": f.mean(),
                     "d_vs_router": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "d_vs_twophase": d2.mean(), "hv5_common": h5c,
                     "gain_1_5": s[0], "gain_6_10": s[1], "gain_11_20": s[2],
                     "best_k_on_train": best_k,
                     "train_hv": hyb.get(int(nm.split("-")[-1]), "")
                     if nm.startswith("Hybrid") else ""})

    print(f"\n判读要点")
    hv = {r["policy"]: r["hv20"] for r in rows}
    hb, ps = hv[f"Hybrid-{best_k}"], hv[f"PhaseSplit-{best_k}"]
    print(f"  Hybrid-{best_k} {hb:.5f}  vs  纯阶段划分对照 {ps:.5f}  "
          f"差 {hb - ps:+.5f}")
    print(f"  -> {'切换到人工规则确有贡献' if hb > ps else '收益不来自切换到人工规则'}")
    rv = hv[f"RevHybrid-{best_k}"]
    print(f"  RevHybrid-{best_k}（反向）{rv:.5f}；若它也赢过纯 Router，"
          f"说明不是『ML 擅长前段』")

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (op.parent / "hybrid_meta.json").write_text(json.dumps({
        "pool": args.pool, "two_phase": list(two), "ks_tried": ks,
        "best_k_on_train": best_k, "train_hv_by_k": hyb,
        "n_train": len(tmols), "n_val": len(vmols),
        "router": {"kind": router.kind, "group": router.group},
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
