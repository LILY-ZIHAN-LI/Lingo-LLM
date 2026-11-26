"""最终独立测试：冻结的策略集合 vs test_v2 的 200 个分子。

**这个脚本只跑一次。** 跑完不再据其结果调参、改主指标或换主对照。

预先声明（见 docs/test_v2_preregistration.md，先于本脚本运行写入）：
  主模型   DynamicRouter_v1  (R5 / random forest, 102 维, 1.62 ms/决策)
  主对照   TwoPhase(sa->gsk3b@7)
  主指标   HV@20
  主判据   按分子配对 bootstrap 的 95% CI 下界 > 0
  次要检验 Wilcoxon 符号秩（照报，不用于判定）
  限定条款 若只在固定调用预算下赢，结论限定在固定调用预算下

所有策略的参数**只能**来自 train（200 分子）和 val（50 分子）：
  SBS / FixedProb / StepOnly       train 标签 + train 反事实
  TwoPhase (e1,e2,k)               train 的 60 个分子上网格选
  StrongRule                       train 上拟合规则族
  StaticSelector                   train 上训练（本脚本内重建并存盘）
  DynamicRouter_v1                 已冻结的 pkl
  LinUCB                           results_sbt_first/policies.pkl（历史训练）

用法:
  python scripts/run_test_v2.py            # 真跑
  python scripts/run_test_v2.py --dry-run  # 只检查配置和泄漏，不跑
"""
from __future__ import annotations

import argparse
import csv

import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger                                  # noqa: E402
from scipy.stats import trim_mean, wilcoxon                 # noqa: E402


RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from run_baselines import make_choosers, train_stats
    from run_loo_selector import boot_ci, load_labels
    from run_nested_selector import RuleFamily
    from run_oracle_router import pool_env_kw, rollout
    from run_rollout_comparison import seg_gain, wrap_stale
    from src.analysis import hv_at_eligible
    from src.chem import murcko_scaffold
    from src.dynamic_router import DynamicRouter
    from src.editors import make_editors
    from src.env import MolEnv
    from src.features import build_matrix
    from src.interfaces import EXPERTS, EpisodeConfig
    from src.oracles import OracleBank
    from run_loo_selector import fit_predict as _fit_predict

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sbt_first")
    ap.add_argument("--seeds", default="data/seeds_test_v2.txt")
    ap.add_argument("--n-mol", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="results_v2/test_v2")
    args = ap.parse_args()
    NE = len(EXPERTS)
    SEED_BASE = 950000
    rng = np.random.default_rng(0)
    od = ROOT / args.outdir
    od.mkdir(parents=True, exist_ok=True)

    # ---------- 泄漏核验（硬失败） ----------
    def scafs(f):
        return {murcko_scaffold(s) or s
                for s in (ROOT / f).read_text().split() if s.strip()}
    tv = scafs(args.seeds)
    for other in ("data/seeds_train.txt", "data/seeds_val.txt",
                  "data/seeds_test.txt"):
        inter = tv & scafs(other)
        if inter:
            raise SystemExit(f"scaffold 泄漏：{args.seeds} 与 {other} 有 "
                             f"{len(inter)} 个重叠")
    print(f"泄漏核验通过：{len(tv)} 个 scaffold，与 train/val/test 交集为 0")

    env_kw = pool_env_kw(args.pool)
    tmols, Y, sbs, freq, step_best, _ = train_stats(args.pool)
    two = tuple(json.loads(
        (ROOT / "results_v2/baselines/baseline_meta.json").read_text())["two_phase"])
    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=args.max_steps), **env_kw)
    ch = make_choosers(sbs, freq, step_best, two, rng, env.editors, args.max_steps)

    # ---------- 冻结的学习型策略 ----------
    v1p = ROOT / "results_v2/dynamic_router/models/v1/DynamicRouter_v1.pkl"
    router = DynamicRouter.load(v1p)
    ch["DynamicRouter_v1"] = router.chooser()
    sha = hashlib.sha256(v1p.read_bytes()).hexdigest()

    # StrongRule（train 上拟合）
    ed_feat = make_editors(qed_disjoint=True)
    bank_f = OracleBank()
    Xtr_f, names_f = build_matrix(tmols, ed_feat, bank_f, "F5")
    rule = RuleFamily(names_f).fit(Xtr_f, Y)

    # StaticSelector（train 上训练；每个分子开局选一次，之后不变）
    #  —— 之前只在 dev 上评过、没存模型，这里重建并存盘，作为预先声明的次要对照
    sel_pick_cache: dict[str, int] = {}

    def _static_for(smiles):
        if smiles not in sel_pick_cache:
            Xq, _ = build_matrix([smiles], ed_feat, bank_f, "F5")
            P = _fit_predict("C", Xtr_f, Y, Xq, rng)
            sel_pick_cache[smiles] = int(np.asarray(P).reshape(1, -1).argmax(1)[0])
        return sel_pick_cache[smiles]

    def static_selector(e, t):
        return _static_for(e.seed_smiles), 0

    def strong_rule(e, t):
        if not hasattr(e, "_sr_cache"):
            e._sr_cache = {}
        if e.seed_smiles not in e._sr_cache:
            Xq, _ = build_matrix([e.seed_smiles], ed_feat, bank_f, "F5")
            e._sr_cache[e.seed_smiles] = int(rule.predict(Xq)[0])
        return e._sr_cache[e.seed_smiles], 0

    ch["StaticSelector"] = static_selector
    ch["StrongRule"] = strong_rule

    # LinUCB（历史训练好的）
    pols = pickle.loads((ROOT / f"results_{args.pool}/policies.pkl").read_bytes())
    lin = pols["bandit"]
    ch["LinUCB"] = lambda e, t: (int(lin.act(e._obs(), t)), 0)

    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()][:args.n_mol]
    jobs = [(m, SEED_BASE + rep * 10000 + i)
            for rep in range(args.repeats) for i, m in enumerate(mols)]

    prereg = {
        "primary_model": "DynamicRouter_v1", "primary_model_sha256": sha,
        "primary_baseline": f"TwoPhase({EXPERTS[two[0]]}->{EXPERTS[two[1]]}@{two[2]})",
        "primary_metric": "HV@20",
        "primary_criterion": "paired-by-molecule bootstrap 95% CI lower bound > 0",
        "secondary_test": "Wilcoxon signed-rank (reported, not decisive)",
        "scope_clause": "若只在固定调用预算下赢，结论限定在固定调用预算下",
        "n_mol": len(mols), "repeats": args.repeats,
        "policies": list(ch), "env_kw": env_kw, "seed_base": SEED_BASE,
        "sbs_expert": EXPERTS[sbs], "two_phase": list(two),
        "strong_rule": [str(rule.best[0]), float(rule.best[1])],
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (od / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, ensure_ascii=False))
    print(f"预注册 -> {od / 'preregistration.json'}")
    print(f"{len(mols)} 分子 x {args.repeats} 重复 x {len(ch)} 策略 = "
          f"{len(jobs) * len(ch)} 条 episode")
    if args.dry_run:
        print("\n--dry-run：配置已冻结，未执行")
        return 0

    # ---------- 执行 ----------
    res, wall, oc = {}, {}, {}
    ck = od / "ckpt"
    ck.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for nm, fn in ch.items():
        # rollout 对 (分子, episode_seed, 策略) 是确定性的，所以已存的
        # checkpoint 可以直接复用 —— 上一轮崩在第 12 条策略上，11 条
        # 已跑完的结果却因为只在最后统一写盘而全部丢失，不再犯第二次。
        cp = ck / (nm.replace("/", "_").replace("(", "_").replace(")", "_")
                   .replace(">", "_") + ".json")
        if cp.exists():
            blob = json.loads(cp.read_text())
            res[nm], wall[nm], oc[nm] = blob["rows"], blob["wall"], blob["oracle"]
            print(f"  {nm:<24}[复用 checkpoint]  "
                  f"HV {np.mean([r['final'] for r in res[nm]]):.5f}", flush=True)
            continue
        t1, o0 = time.perf_counter(), bank.n_calls
        res[nm] = [rollout(env, fn, m, s, args.max_steps) for m, s in jobs]
        wall[nm], oc[nm] = time.perf_counter() - t1, bank.n_calls - o0
        cp.write_text(json.dumps({"rows": res[nm], "wall": wall[nm],
                                  "oracle": oc[nm]}, ensure_ascii=False,
                                 default=float))
        print(f"  {nm:<24}{wall[nm]:>6.0f}s  oracle +{oc[nm]:<8} "
              f"HV {np.mean([r['final'] for r in res[nm]]):.5f}", flush=True)

    # 按分子聚合（3 个重复取均值，重复不算独立样本）
    nmol = len(mols)
    def per_mol(nm, key="final"):
        a = np.array([r[key] for r in res[nm]]).reshape(args.repeats, nmol)
        return a.mean(0)

    REF = prereg["primary_baseline"]
    base = per_mol(REF)
    prim = per_mol("DynamicRouter_v1")
    d = prim - base
    lo, hi = boot_ci(d, rng, n=50000)
    w = wilcoxon(prim, base)

    print(f"\n{'=' * 78}\n主判据\n{'=' * 78}")
    print(f"DynamicRouter_v1 {prim.mean():.5f}  vs  {REF} {base.mean():.5f}")
    print(f"  Δ = {d.mean():+.5f}   (相对 {100 * d.mean() / base.mean():+.1f}%)")
    print(f"  按分子配对 bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  -> 主判据 {'**通过**' if lo > 0 else '**未通过**'}（下界 {lo:+.5f}）")
    print(f"  次要：Wilcoxon p={w.pvalue:.4f}  赢面 {np.mean(d > 0):.0%}  "
          f"中位数 {np.median(d):+.5f}  winsor10 {trim_mean(d, 0.1):+.5f}")

    common = np.ones(nmol, bool)
    h5 = {}
    for nm in res:
        a = np.array([hv_at_eligible(r, 5) for r in res[nm]], dtype=object) \
            .reshape(args.repeats, nmol)
        ok = np.array([[v is not None for v in row] for row in a])
        h5[nm] = (a, ok)
        common &= ok.all(0)
    print(f"\nHV@5 全策略共同支撑 {common.sum()}/{nmol}")

    rows = []
    print(f"\n{'策略':<24}{'HV@20':>9}{'Δ 主对照':>11}{'95% CI':>22}"
          f"{'HV@5':>9}{'覆盖':>7}{'候选':>7}{'死调':>6}{'探查':>6}{'秒':>7}")
    print("-" * 110)
    for nm in sorted(res, key=lambda n: -per_mol(n).mean()):
        f = per_mol(nm)
        dd = f - base
        l2, h2 = boot_ci(dd, rng) if nm != REF else (0.0, 0.0)
        a, ok = h5[nm]
        v5 = (np.nanmean(np.where(ok, a, np.nan).astype(float), axis=0)[common].mean()
              if common.sum() else float("nan"))
        rec = {"policy": nm, "hv20": f.mean(), "d_vs_primary_baseline": dd.mean(),
               "ci_lo": l2, "ci_hi": h2, "sig": int(l2 > 0),
               "hv5_common": v5, "coverage5": float(ok.mean()),
               "median_d": float(np.median(dd)),
               "winsor10_d": float(trim_mean(dd, 0.1)),
               "winrate": float(np.mean(dd > 0)),
               "eligible": float(np.mean([r["eligible"] for r in res[nm]])),
               "dead_calls": float(np.mean([r["dead_calls"] for r in res[nm]])),
               "proposed": float(np.mean([r["proposed"] for r in res[nm]])),
               "hv_auc": float(np.mean([r["auc"] for r in res[nm]])),
               "gain_1_5": float(np.mean([seg_gain(r["curve"], 0, 5) for r in res[nm]])),
               "gain_6_10": float(np.mean([seg_gain(r["curve"], 5, 10) for r in res[nm]])),
               "gain_11_20": float(np.mean([seg_gain(r["curve"], 10, 20) for r in res[nm]])),
               "probes": float(np.mean([r["probes"] for r in res[nm]])),
               "oracle_logical": oc[nm], "wall_seconds": wall[nm]}
        rows.append(rec)
        print(f"{nm:<24}{f.mean():>9.5f}{dd.mean():>+11.5f}"
              f"{'*' if l2 > 0 else ' '}[{l2:+.5f},{h2:+.5f}]{v5:>9.5f}"
              f"{ok.mean():>7.0%}{rec['eligible']:>7.1f}{rec['dead_calls']:>6.1f}"
              f"{rec['probes']:>6.0f}{wall[nm]:>7.0f}")

    with (od / "test_v2_results.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    (od / "test_v2_headline.json").write_text(json.dumps({
        "preregistration": prereg,
        "primary": {"model": "DynamicRouter_v1", "baseline": REF,
                    "hv20_model": float(prim.mean()), "hv20_baseline": float(base.mean()),
                    "delta": float(d.mean()), "ci": [lo, hi],
                    "criterion_passed": bool(lo > 0),
                    "wilcoxon_p": float(w.pvalue), "win_rate": float(np.mean(d > 0)),
                    "median": float(np.median(d)), "winsor10": float(trim_mean(d, 0.1))},
        "hv5_common_support": int(common.sum()),
        "total_wall_seconds": time.perf_counter() - t0,
        "total_oracle_logical": int(sum(oc.values())),
    }, indent=2, ensure_ascii=False))
    (od / "raw_episodes.json").write_text(json.dumps(
        {k: [{kk: vv for kk, vv in r.items() if kk != "actions"} for r in v]
         for k, v in res.items()}, ensure_ascii=False, default=float))
    print(f"\n总墙钟 {(time.perf_counter() - t0) / 60:.1f} 分钟；"
          f"逻辑 oracle {sum(oc.values())}")
    print(f"-> {od / 'test_v2_results.csv'}")
    print(f"-> {od / 'test_v2_headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
