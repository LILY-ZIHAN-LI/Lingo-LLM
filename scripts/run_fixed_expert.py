"""单 expert 基线：整条 episode 不路由，只调同一个 expert。

proposal 审稿意见 #1：
  "Run each editor alone for the full budget and select the best fixed editor
   on validation data. Otherwise beating random or cyclic routing might mainly
   reflect avoiding a weak editor."

这条基线堵的是解释漏洞，不是为了赢。两种可能的结果都有内容：
  - 最强单 expert >= 所有 router  ->  "路由"这件事本身没有产生价值，
    router 之间的差只是在不同程度地逼近"别用弱 expert"。
  - 所有 router > 最强单 expert   ->  切换确实带来了单一 expert 拿不到的东西。

选哪个 expert 必须在 val 上定。在 test 上挑最好的那个，等于做了 4 次比较
只报 1 次，p 值不成立 —— 所以这里跑两趟：val 选，test 比。

不需要训练：SingleExpertPolicy 没有参数。所以这个脚本只做评估，
复用已有 results 目录里的 (分子, episode_seed) 列表来保证严格配对。

用法:
  python scripts/run_fixed_expert.py --results results_base0 --workers 8
  python scripts/run_fixed_expert.py --results results_gen_router \
      --editor-kind gen --gen-ckpt results_gen/generator_delta.pt --workers 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, episode_key, hv_at_eligible, hv_at_step,
    pick_candidate_budget, print_pair_table, summarize_pair,
)
from src.interfaces import EXPERTS  # noqa: E402
from src.policies import make_fixed_experts  # noqa: E402
from scripts.run_experiment import _worker_eval, _worker_init  # noqa: E402


def evaluate(jobs, policies, cfg_dict, env_kw, workers):
    t0 = time.perf_counter()
    with Pool(workers, initializer=_worker_init,
              initargs=(cfg_dict, pickle.dumps(policies), env_kw)) as pool:
        rows = pool.map(_worker_eval, jobs, chunksize=4)
    print(f"   {(time.perf_counter() - t0) / 60:.1f} 分钟\n")
    return rows


def summary_table(rows, names):
    print(f"{'policy':14}{'终点HV':>9}{'AUC':>8}{'合格候选':>9}{'HV/候选':>10}")
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        if not sub:
            continue
        print(f"{n:14}{np.mean([r['final'] for r in sub]):9.4f}"
              f"{np.mean([r['auc'] for r in sub]):8.3f}"
              f"{np.mean([r['eligible'] for r in sub]):9.1f}"
              f"{np.mean([r['hv_per_eligible'] for r in sub]):10.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", required=True,
                    help="已有的 run_experiment 输出目录，提供配对用的 episode 列表")
    ap.add_argument("--out", default=None, help="默认写回 --results")
    ap.add_argument("--val-repeats", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--editor-kind", choices=["rdkit", "gen"], default="rdkit")
    ap.add_argument("--gen-ckpt", default="results_gen/generator.pt")
    ap.add_argument("--pool-overlap", type=float, default=0.0)
    ap.add_argument("--qed-disjoint", action="store_true")
    ap.add_argument("--sample-by-template", action="store_true",
                    help="仅 rdkit：k 个提案名额分给 k 条不同模板，而不是从\n                         全部产物里均匀抽。见 docs/findings.md §16b")
    ap.add_argument("--product-pick", default="random",
                    choices=["random", "first", "last"],
                    help="仅 --sample-by-template：每条模板出哪个产物")
    ap.add_argument("--pool-design", default=None)
    ap.add_argument("--escape-bits", action="store_true")
    ap.add_argument("--skip-val", action="store_true",
                    help="已经跑过 val 时跳过，直接读回 fixed_val.json")
    ap.add_argument("--skip-test", action="store_true",
                    help="已经跑过 test 时跳过，直接读回 fixed_raw.json 重算表格")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.out) if args.out else res
    out.mkdir(parents=True, exist_ok=True)

    router_rows = json.loads((res / "raw.json").read_text())
    routers = sorted({r["policy"] for r in router_rows})
    # 严格配对：直接沿用 router 评估过的那批 (分子, episode_seed)
    test_jobs_keys = sorted({episode_key(r) for r in router_rows})
    print(f"复用 {res}/raw.json：{len(routers)} 个 router，"
          f"{len(test_jobs_keys)} 个 test episode\n")

    cfg_dict = {"max_steps": args.max_steps, "lam": args.lam}
    env_kw = {"use_escape_bits": args.escape_bits,
              "editor_kind": args.editor_kind, "gen_ckpt": args.gen_ckpt,
              "pool_overlap": args.pool_overlap,
              "qed_disjoint": args.qed_disjoint,
              "pool_design": args.pool_design,
              "sample_by_template": args.sample_by_template,
              "product_pick": args.product_pick}
    fixed = make_fixed_experts()
    fixed_names = list(fixed)

    # ---------- 1. val：选 expert ----------
    val_path = out / "fixed_val.json"
    if args.skip_val and val_path.exists():
        val_rows = json.loads(val_path.read_text())
        print(f"1. 跳过 val，读回 {val_path}\n")
    else:
        val_seeds = Path(args.data).joinpath("seeds_val.txt").read_text().split()
        val_jobs = [(n, m, 500000 + rep * 10000 + i)
                    for n in fixed_names
                    for rep in range(args.val_repeats)
                    for i, m in enumerate(val_seeds)]
        print(f"1. val 选 expert：{len(val_seeds)} 个种子 x {args.val_repeats} "
              f"repeat x 4 expert = {len(val_jobs)} episode...")
        val_rows = evaluate(val_jobs, fixed, cfg_dict, env_kw, args.workers)
        val_path.write_text(json.dumps(val_rows))

    summary_table(val_rows, fixed_names)

    # 选择口径必须和最终比较口径一致：同一候选预算下的 HV。
    # 用终点 HV 选会偏袒产出率高的 expert —— 那正是我们一直在控制的混淆。
    xv = pick_candidate_budget(val_rows)
    supp_v = common_support(val_rows, xv)
    val_score = {}
    for n in fixed_names:
        vals = [hv_at_eligible(r, xv) for r in val_rows
                if r["policy"] == n and episode_key(r) in supp_v]
        vals = [v for v in vals if v is not None]
        val_score[n] = float(np.mean(vals)) if vals else float("-inf")
    best = max(val_score, key=val_score.get)
    # 终点 HV 口径另选一个。两个口径选出不同的 expert 是常态：
    # 产出率低的 expert 在"每个候选换多少 HV"上好看，在"20 步之后走多远"上不好看。
    # 哪张表就用哪张表自己口径下 val 的赢家，否则等于拿一个被别的指标选出来的
    # 基线去打这张表。
    val_end = {n: float(np.mean([r["final"] for r in val_rows
                                 if r["policy"] == n])) for n in fixed_names}
    best_end = max(val_end, key=val_end.get)
    print(f"\nval 候选预算 x={xv}，公共支撑 {len(supp_v)} episode")
    print(f"  {'expert':14}{'@x 候选':>10}{'终点HV':>10}")
    for n in fixed_names:
        m1 = " <-x" if n == best else "    "
        m2 = " <-终点" if n == best_end else ""
        print(f"  {n:14}{val_score[n]:+10.5f}{m1}{val_end[n]:10.5f}{m2}")

    # ---------- 2. test ----------
    test_path = out / "fixed_raw.json"
    if args.skip_test and test_path.exists():
        test_rows = json.loads(test_path.read_text())
        print(f"\n2. 跳过 test，读回 {test_path}")
    else:
        print(f"\n2. test 评估 {len(fixed_names) * len(test_jobs_keys)} 个 episode...")
        test_jobs = [(n, m, s) for n in fixed_names for m, s in test_jobs_keys]
        test_rows = evaluate(test_jobs, fixed, cfg_dict, env_kw, args.workers)
        test_path.write_text(json.dumps(test_rows))

    # ---------- 3. 合并比较 ----------
    rows = router_rows + test_rows
    all_names = routers + fixed_names
    print("test 全表（4 条单 expert 基线 + 已有 router）\n")
    summary_table(rows, all_names)

    def budget_for(subset):
        """只在参与这张表的 policy 之间定预算和公共支撑。

        把全部 10 个 policy 混在一起定预算是错的：产出率最低的那个 expert
        （RDKit 池里 only_sa 每 episode 只有 5.6 个合格候选）会把 x 压到
        它够得着的水平，连带把 router 之间的比较也挪到一个和 §11a 不同的
        预算点上。每张表各自用自己那批 policy 定 x，表内仍然是严格配对的。
        """
        sub = [r for r in rows if r["policy"] in subset]
        xx = pick_candidate_budget(sub)
        return xx, common_support(sub, xx), coverage_at(sub, xx)

    def at_end(r):
        return hv_at_step(r, args.max_steps)

    # ---- 主表：val 选出的单 expert vs 每个 router ----
    x, supp, cov = budget_for(set(routers) | {best})
    print(f"\n候选预算 x={x}（在 {len(routers)} 个 router + {best} 之间定），"
          f"公共支撑 {len(supp)} episode（{cov['_common']:.0%}）")

    def at_x(r):
        return hv_at_eligible(r, x)

    print_pair_table(
        [summarize_pair(rows, a, best, at_x, restrict=supp) for a in routers],
        f"配对比较 @ 累计 {x} 个合格候选： router - {best}（val 选出的最强单 expert）")

    print_pair_table(
        [summarize_pair(rows, a, best_end, at_end) for a in routers],
        f"配对比较 @ 第 {args.max_steps} 次 editor 调用： router - {best_end}"
        f"（终点口径下 val 的赢家）")

    # ---- 参考表：四个单 expert 之间差多少，就是路由本来有多少可赚 ----
    x2, supp2, cov2 = budget_for(set(fixed_names) | {"random"})
    print(f"\n候选预算 x={x2}（在 4 个单 expert + random 之间定，"
          f"被产出率最低的那个压住），公共支撑 {len(supp2)} episode"
          f"（{cov2['_common']:.0%}）")
    print_pair_table(
        [summarize_pair(rows, a, "random", lambda r: hv_at_eligible(r, x2),
                        restrict=supp2) for a in fixed_names],
        f"参考 @ 累计 {x2} 个合格候选： 单 expert - random")

    (out / "fixed_summary.json").write_text(json.dumps({
        "val_budget": xv, "val_score": val_score, "val_final": val_end,
        "best_on_val": best, "best_on_val_terminal": best_end,
        "test_budget": x, "common_support": len(supp), "coverage": cov,
        "router_minus_best": [summarize_pair(rows, a, best, at_x, restrict=supp)
                              for a in routers],
        "router_minus_best_at_end": [summarize_pair(rows, a, best_end, at_end)
                                     for a in routers],
        "fixed_budget": x2, "fixed_common_support": len(supp2),
        "fixed_minus_random": [
            summarize_pair(rows, a, "random", lambda r: hv_at_eligible(r, x2),
                           restrict=supp2) for a in fixed_names],
    }, indent=2))
    print(f"\n写入 {out}/fixed_raw.json, fixed_val.json, fixed_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
