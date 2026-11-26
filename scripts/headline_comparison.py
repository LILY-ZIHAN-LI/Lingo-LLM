"""头条配对比较：DynamicRouter vs 最强非学习型基线。

单独成脚本的理由：这是整个研究的主判据，不能只活在 rollout 汇总表里。
均值 bootstrap 与 Wilcoxon 符号秩在这里给出不同答案（HV 重尾），两个都报，
不挑有利的那个。匹配候选预算只在参与比较的策略之间取共同支撑 ——
rollout 表里的 25/50 是全部 13 条策略的交集，被 SBS 的 50% 覆盖率拖窄了。

输出 results_v2/dynamic_router/headline_comparison.json
"""
import sys, json, pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scipy.stats import wilcoxon
from src.analysis import hv_at_eligible
from src.chem import murcko_scaffold
rng=np.random.default_rng(0)
OUT = {}

# 重跑 raw 模式下 Router 与 TwoPhase 的配对差（直接从 rollout 脚本的逻辑复用）
from run_rollout_comparison import seg_gain
from run_oracle_router import pool_env_kw, rollout
from run_baselines import make_choosers, train_stats
from src.dynamic_router import DynamicRouter
from src.env import MolEnv
from src.oracles import OracleBank
from src.interfaces import EpisodeConfig, EXPERTS

env_kw=pool_env_kw('sbt_first')
tmols,Y,sbs,freq,step_best,_=train_stats('sbt_first')
bank=OracleBank(); env=MolEnv(bank, EpisodeConfig(max_steps=20), **env_kw)
mols=[l.strip() for l in open(ROOT / "data/seeds_val.txt") if l.strip()][:50]
two=tuple(json.load(open(ROOT / "results_v2/baselines/baseline_meta.json"))['two_phase'])
ch=make_choosers(sbs,freq,step_best,two,rng,env.editors,20)
meta=json.load(open(ROOT / "results_v2/dynamic_router/models/router_meta.json"))
r=DynamicRouter.load(ROOT / meta["path"], availability='none')
ch['DynamicRouter']=r.chooser()
tp_name=[k for k in ch if k.startswith('TwoPhase(')][0]

res={}
for nm in ('DynamicRouter', tp_name, 'OneStepOracle', 'RoundRobin'):
    res[nm]=[rollout(env,ch[nm],m,800000+i,20) for i,m in enumerate(mols)]

A=np.array([x['final'] for x in res['DynamicRouter']])
B=np.array([x['final'] for x in res[tp_name]])
d=A-B
scaf=np.array([murcko_scaffold(m) or m for m in mols])
print(f"DynamicRouter vs {tp_name}  (n={len(mols)}, 50 个互不相同 scaffold)")
print(f"  HV@20  {A.mean():.5f} vs {B.mean():.5f}   Δ = {d.mean():+.5f}")
for nb in (4000, 50000):
    bs=np.array([d[i].mean() for i in rng.integers(0,len(d),(nb,len(d)))])
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"  按分子 bootstrap n={nb:>6}: [{lo:+.5f}, {hi:+.5f}]  p(Δ<=0)={np.mean(bs<=0):.3f}")
# scaffold 聚类 bootstrap（这里每个分子一个 scaffold，等价，但按要求显式做一遍）
ug=np.unique(scaf); idx={g:np.where(scaf==g)[0] for g in ug}
bs=np.array([d[np.concatenate([idx[g] for g in ug[rng.integers(0,len(ug),len(ug))]])].mean()
             for _ in range(20000)])
print(f"  按 scaffold 聚类 bootstrap:   [{np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f}]")
w=wilcoxon(A,B)
print(f"  Wilcoxon 符号秩: p={w.pvalue:.4f}   赢面 {np.mean(d>0):.0%}  中位数 {np.median(d):+.5f}")
OUT["headline"] = {
    "a": "DynamicRouter", "b": tp_name, "n_mol": len(mols),
    "hv20_a": float(A.mean()), "hv20_b": float(B.mean()),
    "delta": float(d.mean()),
    "ci_molecule_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "wilcoxon_p": float(w.pvalue), "win_rate": float(np.mean(d > 0)),
    "median_delta": float(np.median(d)),
}
print()
print("分段收益")
for nm in ('DynamicRouter', tp_name, 'OneStepOracle', 'RoundRobin'):
    rs=res[nm]
    print(f"  {nm:<24}" + "  ".join(
        f"{a}-{b}: {np.mean([seg_gain(x['curve'],a-1,b) for x in rs]):+.5f}"
        for a,b in ((1,5),(6,10),(11,20))))
print()
print("匹配候选预算（共同支撑）")
h5={nm:np.array([hv_at_eligible(x,5) for x in res[nm]],dtype=object) for nm in res}
common=np.ones(len(mols),bool)
for nm in res: common &= np.array([v is not None for v in h5[nm]])
print(f"  共同支撑 {common.sum()}/50")
b5=h5[tp_name][common].astype(float)
for nm in res:
    v=h5[nm][common].astype(float)
    dd=v-b5; lo,hi=np.percentile([dd[i].mean() for i in rng.integers(0,len(dd),(20000,len(dd)))],[2.5,97.5])
    print(f"  {nm:<24}{v.mean():.5f}   Δ vs {tp_name[:9]} {dd.mean():+.5f} [{lo:+.5f},{hi:+.5f}]")
    OUT.setdefault("matched_candidate", {})[nm] = {
        "hv5": float(v.mean()), "d_vs_strongest": float(dd.mean()),
        "ci": [float(lo), float(hi)], "support": int(common.sum())}

import json as _json
_p = ROOT / "results_v2/dynamic_router/headline_comparison.json"
OUT["stage_gain"] = {nm: {f"{a}-{b}": float(np.mean(
    [seg_gain(x["curve"], a - 1, b) for x in res[nm]]))
    for a, b in ((1, 5), (6, 10), (11, 20))} for nm in res}
open(_p, "w").write(_json.dumps(OUT, indent=2, ensure_ascii=False))
print(f"\n-> {_p}")
