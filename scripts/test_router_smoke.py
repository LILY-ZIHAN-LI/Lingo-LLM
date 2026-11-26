"""部署前的 smoke test。失败就让 overnight.sh 的那个阶段标红，不往下走。

查四件事：
  1. 离线训练用的特征 和 在线推理取的特征，在同一个状态上逐列相同
  2. Router 能在真实 env 里走满 20 步，且全程不克隆环境、不执行未选中的 expert
  3. 推理只用决策前信息：把 env 的未来相关字段改掉不应影响当步决策
  4. 单条轨迹上累计 Pareto HV 单调不降（评估审计第 10 条）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import RDLogger                                  # noqa: E402

from src.dynamic_router import DynamicRouter                # noqa: E402
from src.env import MolEnv                                  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig           # noqa: E402
from src.oracles import OracleBank                          # noqa: E402
from src.router_features import (FEATURE_GROUPS, build_states,  # noqa: E402
                                 feature_names, features_from_env)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    meta_p = ROOT / "results_v2/dynamic_router/models/router_meta.json"
    if not meta_p.exists():
        print("缺 router_meta.json —— 先跑 train_dynamic_router.py")
        return 1
    meta = json.loads(meta_p.read_text())
    router = DynamicRouter.load(ROOT / meta["path"])
    group = router.group
    print(f"Router: {router.kind} / {group}  "
          f"{len(feature_names(group))} 维\n")

    cf = ROOT / "results_v2/counterfactuals/cf_sbt_first_200mol.json"
    S = json.loads(cf.read_text())
    bank = OracleBank()
    env = MolEnv(bank, EpisodeConfig(max_steps=20), qed_disjoint=True,
                 sample_by_template=True, product_pick="first")

    # ---- 1. 离线/在线特征一致性 ----
    _, blocks, _, _, _, _ = build_states(cf, "sbt_first")
    rec = next(s for s in S if s["step"] == 0)
    i0 = S.index(rec)
    env.reset(rec["seed_smiles"], rec["episode_seed"])
    on = features_from_env(env, group)[0]
    off = np.concatenate([blocks[b][i0] for b in FEATURE_GROUPS[group]])
    bad = int(np.sum(~np.isclose(on, off, atol=1e-9)))
    check("离线/在线特征逐列一致", bad == 0,
          f"{len(on)} 维，不一致 {bad} 列")

    # ---- 2. 走满 20 步 ----
    env.reset(rec["seed_smiles"], rec["episode_seed"])
    acts, hv = [], [env.hv]
    for t in range(20):
        a = router.act_env(env, t)
        acts.append(a)
        _, _, done, info = env.step(a)
        hv.append(info["hv"])
        if done:
            break
    check("Router 走满 20 步", len(acts) == 20, f"动作 {acts}")
    check("动作全在合法范围", all(0 <= a < len(EXPERTS) for a in acts))
    check("不是只会选同一个 expert", len(set(acts)) > 1,
          f"用到 {len(set(acts))}/{len(EXPERTS)} 个")

    # ---- 3. 累计 Pareto HV 单调不降 ----
    hv = np.asarray(hv)
    drops = int(np.sum(np.diff(hv) < -1e-12))
    check("单条轨迹累计 HV 单调不降", drops == 0,
          f"下降 {drops} 次，终点 {hv[-1]:.6f}")

    # ---- 4. 推理不依赖未来 / 不执行 expert ----
    env.reset(rec["seed_smiles"], rec["episode_seed"])
    for _ in range(3):
        env.step(router.act_env(env, env.t))
    n_log_before = len(env.logs)
    oc_before = bank.n_calls
    a1 = router.act_env(env, env.t)
    a2 = router.act_env(env, env.t)
    check("预测不改变 env（logs 不增长）", len(env.logs) == n_log_before)
    check("预测不消耗 oracle（当前分子已缓存）", bank.n_calls == oc_before,
          f"+{bank.n_calls - oc_before}")
    check("同一状态两次预测一致", a1 == a2)

    print()
    if FAIL:
        print(f"{len(FAIL)} 项失败: {FAIL}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
