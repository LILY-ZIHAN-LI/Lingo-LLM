"""按实测的模板效果，把模板分配给四个 expert，使 action 空间有效维度最大。

这是 findings §12 那条线的"正向使用"：维度是便宜的代理目标（这个脚本几秒，
一次完整路由实验 53 分钟），所以 expert 池可以先在维度上筛，只把最好的拿去跑。

目标函数
--------
expert e 的行向量近似为 `max_{t in pool_e} T[t, :]` —— 取 max 而不是 mean，
因为 env 的 transition rule 和 §1b 的统计量取的都是**最佳候选**。
这只是代理：真实维度必须用 scripts/analyze_action_space.py 复测。

**只最大化 participation ratio 是错的**，这一点是搜出来才看清的：PR 只奖励
四行方向不同、长度相近，完全不管这些方向是不是朝着目标变好的。第二轮搜索
给出的"gsk3b expert"在 gsk3b 上是 −0.056，PR 却有 3.71。一个让四个目标各自
正交地**变差**的池子可以拿满分。维度是个合格的诊断量，不是个合格的设计目标。

所以目标改成字典序：
  1. 先最大化"领头的目标数" —— expert e 在目标 e 上是不是四个 expert 里最强，
     且 M[e,e] > 0。这正是 analyze_action_space.py 检验的 specialization。
  2. 同分再比 participation ratio。
  3. 再同分比 trace（四个对角线之和），偏好真的推得动的池子。

约束（不是装饰，每一条都是实测教的）
------------------------------------
1. 每个池至少 `--min-pool` 条模板、并集触发率不低于 `--min-coverage`。
   §14 测出产出率低的 expert 在"每次 editor 调用"口径下输得很惨，而且会把
   公共支撑压到很小。不加约束优化器会给出四个各含一条高选择性模板的池子 ——
   维度很漂亮，跑起来全是空转。
2. 每个池至多 `--max-pool` 条模板。第一次搜索没有这条上界，给出的 sa 池有
   25 条模板、代理维度 3.62，实测只有 **1.79**（比手写池的 2.20 还差）。
   原因是 `propose` 从全部产物里均匀抽 k=4 个，池子越大越稀释。
   现在 editor 支持按模板抽样（`sample_by_template`），再把池子压到 <= k
   附近，"池的方向 = 成员的逐目标最大值"这个代理才成立。
3. 每条模板自己的触发率不低于 `--min-template-coverage`。只在 1% 分子上
   能触发的模板方向再漂亮也进不了提案。

搜索
----
初值按"模板最擅长的那个目标"分配，然后爬山：反复尝试把一条模板移到别的
expert（或丢弃），接受任何在满足约束下提高目标函数的改动。73 条模板的规模
不需要更聪明的做法。多个随机重启取最好。

  python scripts/design_pool.py --profile data/template_profile.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import participation_ratio  # noqa: E402
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402

DROP = -1


def pool_matrix(assign, T):
    """M[e, :] = 该 expert 名下模板的逐目标最大值。空池记全 0。"""
    M = np.zeros((len(EXPERTS), T.shape[1]))
    for e in range(len(EXPERTS)):
        idx = [i for i, a in enumerate(assign) if a == e]
        if idx:
            M[e] = T[idx].max(axis=0)
    return M


def union_coverage(idx, cov):
    """并集触发率，按独立近似。模板之间当然不独立，但作为下限够用。"""
    if not idx:
        return 0.0
    return float(1.0 - np.prod([1.0 - cov[i] for i in idx]))


def feasible(assign, cov, min_pool, max_pool, min_cov):
    for e in range(len(EXPERTS)):
        idx = [i for i, a in enumerate(assign) if a == e]
        if not (min_pool <= len(idx) <= max_pool):
            return False
        if union_coverage(idx, cov) < min_cov:
            return False
    return True


def n_leading(M):
    """有几个目标由对应的 expert 领头（且确实是正向的）。"""
    n = 0
    for e in range(M.shape[0]):
        if M[e, e] > 0 and M[e, e] >= M[:, e].max() - 1e-12:
            n += 1
    return n


def score(assign, T, cov, min_pool, max_pool, min_cov):
    if not feasible(assign, cov, min_pool, max_pool, min_cov):
        return -1.0
    M = pool_matrix(assign, T)
    # 字典序压成一个数：领头数 (0..4) 权重最大，其次 PR (1..4)，最后 trace。
    return (100.0 * n_leading(M) + participation_ratio(M)
            + 0.01 * float(np.trace(M)))


def hill_climb(assign, T, cov, min_pool, max_pool, min_cov, rng, n_sweep=60):
    best = score(assign, T, cov, min_pool, max_pool, min_cov)
    n = len(assign)
    for _ in range(n_sweep):
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            cur = assign[i]
            for tgt in list(range(len(EXPERTS))) + [DROP]:
                if tgt == cur:
                    continue
                assign[i] = tgt
                s = score(assign, T, cov, min_pool, max_pool, min_cov)
                if s > best + 1e-9:
                    best, cur, improved = s, tgt, True
                else:
                    assign[i] = cur
        if not improved:
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="data/template_profile.json")
    ap.add_argument("--out", default="data/designed_pool.json")
    ap.add_argument("--min-pool", type=int, default=3)
    ap.add_argument("--max-pool", type=int, default=5,
                    help="<= k 附近，否则按模板抽样也盖不全，代理会失真")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-template-coverage", type=float, default=0.10,
                    help="单条模板的触发率下限")
    ap.add_argument("--k", type=int, default=4, help="每次调用的提案数")
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prof = json.loads(Path(args.profile).read_text())
    rows = [r for r in prof["templates"]
            if r["coverage"] >= args.min_template_coverage]
    print(f"触发率 >= {args.min_template_coverage:.0%} 的模板 "
          f"{len(rows)}/{len(prof['templates'])} 条")
    T = np.array([r["delta_norm"] for r in rows])      # 已按目标尺度归一
    cov = np.array([r["coverage"] for r in rows])
    smarts = [r["smarts"] for r in rows]
    print(f"{len(rows)} 条模板，来自 {prof['n_seeds']} 个 "
          f"{prof['seeds_file']} 种子\n")

    # 参照点 1：现在在用的手写池
    from src.editors import _GSK3B, _GROW, _JNK3, _QED_SWAP, _SHRINK
    hand = {"gsk3b": _GSK3B, "jnk3": _JNK3, "qed": _QED_SWAP + _GROW,
            "sa": _SHRINK}
    hand_assign = [DROP] * len(rows)
    for e, name in enumerate(EXPERTS):
        for s in hand[name]:
            if s in smarts:
                hand_assign[smarts.index(s)] = e
    Mh = pool_matrix(hand_assign, T)
    print(f"手写池（qed_disjoint=True 口径）：领头 {n_leading(Mh)}/4，"
          f"代理维度 {participation_ratio(Mh):.2f}")

    # 参照点 2：全部模板给所有人 —— 构造上应该塌到 1
    allsame = pool_matrix([0] * len(rows), T)
    allsame = np.repeat(allsame[:1], len(EXPERTS), axis=0)
    print(f"四池全同（零假设）的代理维度         "
          f"= {participation_ratio(allsame):.2f}\n")

    # ---- 搜索 ----
    rng = random.Random(args.seed)
    best_assign, best_score = None, -1.0
    for r in range(args.restarts):
        if r == 0:
            # 初值：每条模板归到它推得最多的那个目标
            a = [int(np.argmax(T[i])) for i in range(len(rows))]
        else:
            # 随机重启：随机挑 4*max_pool 条分给四个 expert，其余丢弃。
            # 均匀乱分会让每个池都超过 max_pool，开局就不可行，爬不出来。
            a = [DROP] * len(rows)
            pick = rng.sample(range(len(rows)),
                              min(len(rows), len(EXPERTS) * args.max_pool))
            for n_, i_ in enumerate(pick):
                a[i_] = n_ % len(EXPERTS)
        s = hill_climb(a, T, cov, args.min_pool, args.max_pool,
                       args.min_coverage, rng)
        if s > best_score:
            best_score, best_assign = s, list(a)
    assign = best_assign

    M = pool_matrix(assign, T)
    nl = n_leading(M)
    pr = participation_ratio(M)
    print(f"领头的目标数 = {nl}/4（expert e 在目标 e 上是最强的那个且为正）")
    print(f"代理维度     = {pr:.2f} / 4.00   （可达上限 3.45）\n")
    print(f"{'expert':<8}{'模板数':>7}{'并集触发率':>12}   "
          + "".join(f"{o:>9}" for o in OBJECTIVES))
    pools = {}
    for e, name in enumerate(EXPERTS):
        idx = [i for i, a in enumerate(assign) if a == e]
        pools[name] = [smarts[i] for i in idx]
        print(f"{name:<8}{len(idx):>7}{union_coverage(idx, cov):>12.0%}   "
              + "".join(f"{M[e, j]:>9.3f}" for j in range(len(OBJECTIVES))))
    n_drop = sum(1 for a in assign if a == DROP)
    print(f"\n丢弃 {n_drop} 条模板（对任何 expert 都只会稀释方向）")

    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    C = Mn @ Mn.T
    print("\n设计后的 expert 余弦相似度（代理）")
    print("  " + " " * 8 + "".join(f"{e:>9}" for e in EXPERTS))
    for e, name in enumerate(EXPERTS):
        print(f"  {name:<8}" + "".join(f"{C[e, j]:>9.3f}"
                                       for j in range(len(EXPERTS))))

    print("\n各 expert 拿到的模板")
    for name in EXPERTS:
        print(f"\n  [{name}]")
        for s in pools[name]:
            i = smarts.index(s)
            d = " ".join(f"{v:+6.3f}" for v in T[i])
            print(f"    {s[:46]:48}{cov[i]:>6.0%}  {d}")

    Path(args.out).write_text(json.dumps({
        "source_profile": args.profile,
        "proxy_eff_dim": pr, "n_leading": nl, "search_score": best_score,
        "min_pool": args.min_pool, "max_pool": args.max_pool,
        "min_coverage": args.min_coverage,
        "min_template_coverage": args.min_template_coverage,
        "sample_by_template": True,
        "pools": pools,
    }, indent=2))
    print(f"\n写入 {args.out}")
    print("代理维度只是筛选用的；真实维度要用 "
          "scripts/analyze_action_space.py --pool-design 复测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
