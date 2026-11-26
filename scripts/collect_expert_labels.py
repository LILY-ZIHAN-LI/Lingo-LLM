"""EXP 2 / Step 2：采集 train 分子上四个固定 expert 的轨迹，给 Static Selector 当标签。

和 run_fixed_expert.py 的分工：那个脚本跑 val / test，并且要复用已有
results 目录里的 episode 列表来保证与 router 严格配对；这里跑的是 **train**
分子，没有 router 要配对，所以直接从 data/seeds_train.txt 起。

配对规则：同一个分子的四个 expert 用**同一个 episode_seed**，这样
proposal cache 命中、四条轨迹面对的随机性一致，per-molecule 的 argmax
才是干净的比较。

重复次数：§EXP1 实测 val 上的标签噪声是单次 SD 中位数 0.00019（两次均值
SE 0.00014），相对 +0.003 量级的效应是 1/25，所以主采集用 1 次重复；
另抽 --noise-check 个分子多跑 2 次，用来确认噪声在 train 分布上同样低
（不能假设 val 上的噪声水平直接搬过来）。

输出格式与 fixed_val.json 逐字段一致，可以直接喂
scripts/run_loo_selector.py / run_nested_selector.py 的 load_labels()。

用法:
  # 先估成本（只跑 5 个分子，不写出）
  python scripts/collect_expert_labels.py --pilot 5 --qed-disjoint --sample-by-template --product-pick first
  # 正式采集
  python scripts/collect_expert_labels.py --qed-disjoint --sample-by-template --product-pick first \
      --out results_v2/labels/train_sbt_first.json --workers 8
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

from src.oracles import OracleBank            # noqa: E402
from src.policies import make_fixed_experts   # noqa: E402
from scripts.run_experiment import _worker_eval, _worker_init  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = 600000   # train 专用，和 val(500000)/test 的 seed 空间不重叠


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("--out", default="results_v2/labels/train_labels.json")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--noise-check", type=int, default=25,
                    help="额外多跑 2 次重复的分子数，用来在 train 分布上验标签噪声")
    ap.add_argument("--pilot", type=int, default=0,
                    help="只跑前 N 个分子并打印成本外推，不写出结果")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--qed-disjoint", action="store_true")
    ap.add_argument("--sample-by-template", action="store_true")
    ap.add_argument("--product-pick", default="random",
                    choices=["random", "first", "last"])
    ap.add_argument("--pool-overlap", type=float, default=0.0)
    ap.add_argument("--pool-design", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mols = [ln.strip() for ln in (ROOT / args.seeds).read_text().splitlines()
            if ln.strip()]
    if args.pilot:
        mols = mols[:args.pilot]
    rng = np.random.default_rng(args.seed)

    fixed = make_fixed_experts()
    names = list(fixed)

    # 主采集：每个分子一个 seed，四个 expert 共用
    jobs = [(n, m, SEED_BASE + rep * 1000 + i)
            for rep in range(args.repeats)
            for i, m in enumerate(mols)
            for n in names]
    # 噪声检查：抽一批分子多跑 2 次
    extra_idx = []
    if args.noise_check and not args.pilot:
        extra_idx = sorted(rng.choice(len(mols),
                                      size=min(args.noise_check, len(mols)),
                                      replace=False).tolist())
        jobs += [(n, mols[i], SEED_BASE + (args.repeats + r) * 1000 + i)
                 for r in range(2) for i in extra_idx for n in names]

    cfg_dict = {"max_steps": args.max_steps, "lam": args.lam}
    env_kw = {"qed_disjoint": args.qed_disjoint,
              "sample_by_template": args.sample_by_template,
              "product_pick": args.product_pick,
              "pool_overlap": args.pool_overlap,
              "pool_design": args.pool_design}

    print(f"分子 {len(mols)}  x expert {len(names)}  x 重复 {args.repeats}"
          f"  (+ 噪声检查 {len(extra_idx)} 分子 x 2 次)  = {len(jobs)} 条 episode")
    print(f"env: {env_kw}")
    print(f"workers {args.workers}\n")

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_worker_init,
              initargs=(cfg_dict, pickle.dumps(fixed), env_kw)) as pool:
        rows = pool.map(_worker_eval, jobs, chunksize=4)
    el = time.perf_counter() - t0

    # 成本汇总。注意 oracle 调用在 worker 进程里，主进程的 bank 读不到，
    # 所以用每条 episode 的 proposed / n_valid 作代理，并单独报墙钟。
    prop = sum(r["proposed"] for r in rows)
    valid = sum(r.get("n_valid", 0) for r in rows)
    elig = sum(r["eligible"] for r in rows)
    print(f"墙钟 {el / 60:.1f} 分钟 ({el / len(jobs):.2f} s/episode, "
          f"{args.workers} 进程)")
    print(f"生成尝试 {prop}  合法 {valid}  合格 {elig}  "
          f"(合格率 {100 * elig / max(prop, 1):.1f}%)")
    print(f"逻辑 oracle 打分上界 {4 * valid}（去重缓存后实际更低）")

    if args.pilot:
        full = 200 * len(names) * args.repeats
        scale = full / len(jobs)
        print(f"\n--- 外推到 200 个分子 ---")
        print(f"episode {full}  预计墙钟 {el * scale / 60:.0f} 分钟"
              f" ({el * scale / 3600:.1f} 小时)")
        print(f"生成尝试 {int(prop * scale)}  逻辑 oracle 打分上界 {int(4 * valid * scale)}")
        print(f"加上 {args.noise_check} 个分子 x 2 次重复："
              f"+{int(el * scale * 2 * args.noise_check / 200 / 60)} 分钟")
        print("\n（未写出结果；去掉 --pilot 正式采集）")
        return 0

    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rows, ensure_ascii=False))
    meta = {"n_mol": len(mols), "experts": names, "repeats": args.repeats,
            "noise_check_idx": extra_idx, "env_kw": env_kw,
            "cfg": cfg_dict, "seed_base": SEED_BASE,
            "wall_seconds": el, "n_episode": len(jobs),
            "generation_attempts": prop, "n_valid": valid, "n_eligible": elig}
    (op.parent / (op.stem + "_meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n-> {op}\n-> {op.parent / (op.stem + '_meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
