"""实测每条反应模板在四个目标上的效果，为数据驱动的 expert 池设计提供输入。

动机（findings §12 + §14）
--------------------------
§12：routing 收益随 action 空间有效维度单调上升，斜率 +0.0017 HV/维。
§14：在 2.20 维的池子上，学出来的 router 打不过零参数的单 expert 基线。
合起来给出一个可证伪的设计任务 —— 把维度推上去，看收益落不落到那条线上。

原来的四个池是手写的，gsk3b 和 jnk3 都是"往芳环 CH 上挂取代基"，方向重合。
这里改成：先测每条模板真的把四个目标各推动多少，再按**选择性**分配。
谁属于哪个 expert 由数据定。

测什么
------
对模板 t 和种子分子 m：跑 t，保留骨架不变的产物，取"该产物相对母体在目标 j
上的提升"的最大值（和 env 的 transition rule、§1b 的统计量口径一致 ——
取最佳候选而不是平均候选）。对 m 求均值得到 T[t, j]。

全部除以 `objective_scales`（可达变化的标准差），否则 qed 的尺度会独占
奇异值，见 §10d。

成本：oracle 调用有缓存，但第一遍是实打实的。--n-seeds 100 / --max-prod 3
大约一万次打分。

  python scripts/profile_templates.py --n-seeds 100 --out data/template_profile.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RDLogger.DisableLog("rdApp.*")

from src.analysis import objective_scales  # noqa: E402
from src.chem import canonical, is_reasonable, scaffold_preserved  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402
from src.templates import ALL_TEMPLATES, FAMILY_OF  # noqa: E402


def products_of(rxn, mol, parent: str, max_prod: int) -> list[str]:
    """跑一条模板，返回保留骨架的合理产物（已 canonical、已去重）。"""
    try:
        outcomes = rxn.RunReactants((mol,))
    except Exception:
        return []
    seen, out = {parent}, []
    for tup in outcomes:
        for p in tup:
            try:
                Chem.SanitizeMol(p)
                smi = Chem.MolToSmiles(p)
            except Exception:
                continue
            if smi in seen or not is_reasonable(smi):
                continue
            seen.add(smi)
            if not scaffold_preserved(parent, smi):
                continue
            out.append(smi)
            if len(out) >= max_prod:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_train.txt",
                    help="用 train 而不是 test：池子的设计也是一种拟合，"
                         "在 test 上设计等于把 test 用成了开发集")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--max-prod", type=int, default=3,
                    help="每条模板每个分子最多打分几个产物")
    ap.add_argument("--out", default="data/template_profile.json")
    args = ap.parse_args()

    data = Path(args.data)
    seeds = data.joinpath(args.seeds_file).read_text().split()[: args.n_seeds]
    bank = OracleBank()
    scales = objective_scales(str(data / "gen_pairs_delta.jsonl"))
    if scales is None:
        scales = np.ones(len(OBJECTIVES))
        print("! 找不到 pair 数据，不做归一化 —— 结果会被 qed 的尺度主导")
    print(f"{len(seeds)} 个种子 x {len(ALL_TEMPLATES)} 条模板，"
          f"归一化尺度 {np.round(scales, 4)}\n")

    rxns = {s: AllChem.ReactionFromSmarts(s) for s in ALL_TEMPLATES}
    n_o = len(OBJECTIVES)
    # 每条模板累计：命中的分子数，以及各目标最大提升之和
    hits = {s: 0 for s in ALL_TEMPLATES}
    tot = {s: np.zeros(n_o) for s in ALL_TEMPLATES}
    n_prod = {s: 0 for s in ALL_TEMPLATES}

    t0 = time.perf_counter()
    for si, smi in enumerate(seeds):
        parent = canonical(smi)
        if parent is None:
            continue
        base = bank.vector(parent)
        if base is None:
            continue
        base = np.asarray(base)
        mol = Chem.MolFromSmiles(parent)
        if mol is None:
            continue
        for s in ALL_TEMPLATES:
            prods = products_of(rxns[s], mol, parent, args.max_prod)
            if not prods:
                continue
            best = None
            for p in prods:
                v = bank.vector(p)
                if v is None:
                    continue
                d = np.asarray(v) - base
                best = d if best is None else np.maximum(best, d)
            if best is None:
                continue
            hits[s] += 1
            n_prod[s] += len(prods)
            tot[s] += best
        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(seeds)}  oracle 调用 {bank.n_calls:,}  "
                  f"{time.perf_counter()-t0:.0f}s")

    # 对"命中的分子"求均值。用命中数而不是全部种子数做分母：
    # 一条只在 5% 分子上能触发的模板，在那 5% 上的方向仍然是它真实的方向，
    # 触发率另外单独记（coverage），设计时两个都要看。
    rows = []
    for s in ALL_TEMPLATES:
        if hits[s] == 0:
            continue
        mean = tot[s] / hits[s]
        rows.append({
            "smarts": s,
            "family": FAMILY_OF.get(s, "?"),
            "coverage": hits[s] / len(seeds),
            "prod_per_hit": n_prod[s] / hits[s],
            "delta": mean.tolist(),
            "delta_norm": (mean / scales).tolist(),
        })

    out = Path(args.out)
    out.write_text(json.dumps({
        "objectives": list(OBJECTIVES),
        "scales": scales.tolist(),
        "n_seeds": len(seeds), "seeds_file": args.seeds_file,
        "max_prod": args.max_prod,
        "oracle_calls": bank.n_calls,
        "templates": rows,
    }, indent=2))

    print(f"\n{len(rows)}/{len(ALL_TEMPLATES)} 条模板至少命中一个分子，"
          f"oracle 调用 {bank.n_calls:,}，{(time.perf_counter()-t0)/60:.1f} 分钟")
    print(f"\n{'模板':52}{'族':16}{'覆盖':>7}  归一化 Δ (gsk3b jnk3 qed sa)")
    for r in sorted(rows, key=lambda r: -max(r["delta_norm"])):
        d = " ".join(f"{v:+6.3f}" for v in r["delta_norm"])
        print(f"{r['smarts'][:50]:52}{r['family']:16}{r['coverage']:>7.0%}  {d}")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
