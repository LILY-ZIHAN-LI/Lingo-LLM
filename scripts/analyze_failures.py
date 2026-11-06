"""Failure analysis: how a locally useful edit damages later progress.

analyze_editor_depletion.py gives the mechanism: a step with 0 eligible candidates
leaves the molecule unchanged, a deterministic policy re-picks the same editor, and
gets 0 again. This script grounds the mechanism in concrete molecules:

  1. Select episodes from raw.json that lead early and lose at the end (greedy
     beats random at step 3 but loses the endpoint), which operationalizes
     "myopic gain traded for long-term stagnation".
  2. Replay both trajectories for the top-ranked cases, printing editor, current
     molecule, eligible count and HV per step.
  3. Draw how the structure changes, ready for the report.

  python scripts/analyze_failures.py --top 3
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import MolEnv  # noqa: E402
from src.interfaces import EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def replay(env, policy, smiles: str, episode_seed: int):
    """Replay one trajectory, recording each step. env is identical for every
    policy, so this is reproducible."""
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    steps = []
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        obs, _, done, info = env.step(a)
        lg = env.logs[-1]
        steps.append({
            "t": t, "expert": lg.expert, "parent": lg.current,
            "n_eligible": lg.n_scaffold_ok, "hv_gain": info["hv"] - hv0,
            "entered": lg.entered_archive,
            "best_single_gain": lg.best_single_gain,
            "next": env.current,
            "moved": env.current != lg.current,
        })
        if done:
            break
    return steps


def print_side_by_side(sa, sb, na, nb) -> None:
    print(f"\n  {'t':>2}  {na:<34}  {nb:<34}")
    print(f"  {'':>2}  {'editor  cand  HV      moved':<34}  "
          f"{'editor  cand  HV      moved':<34}")
    for t in range(max(len(sa), len(sb))):
        def cell(s):
            if t >= len(s):
                return " " * 34
            r = s[t]
            mark = "->" if r["moved"] else "x"
            return (f"{LABEL[r['expert']]:<7}{r['n_eligible']:>4}  "
                    f"{r['hv_gain']:>7.4f}  {mark}").ljust(34)
        print(f"  {t:>2}  {cell(sa)}  {cell(sb)}")
    print(f"  {'':>2}  x = 0 eligible candidates, molecule unchanged (absorbing state)")


def draw(steps, path: Path, title: str) -> bool:
    """Draw the steps where the molecule actually changed."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping plot: {exc})")
        return False
    mols, legends = [], []
    first = steps[0]["parent"]
    m = Chem.MolFromSmiles(first)
    if m is not None:
        mols.append(m)
        legends.append("seed")
    for r in steps:
        if not r["moved"]:
            continue
        m = Chem.MolFromSmiles(r["next"])
        if m is None:
            continue
        mols.append(m)
        legends.append(f"t={r['t']} via {LABEL[r['expert']]}\nHV +{r['hv_gain']:.4f}")
    if len(mols) < 2:
        return False
    img = Draw.MolsToGridImage(mols[:8], molsPerRow=min(4, len(mols)),
                               subImgSize=(260, 220), legends=legends[:8])
    # The return type varies by RDKit version and environment: PIL Image, a
    # wrapper with .data, or raw bytes
    if hasattr(img, "save"):
        img.save(str(path))
    else:
        data = img.data if hasattr(img, "data") else img
        path.write_bytes(data.encode() if isinstance(data, str) else data)
    print(f"  structure figure -> {path}  ({title})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--a", default="greedy", help="policy that leads early")
    ap.add_argument("--b", default="random", help="policy that overtakes at the end")
    ap.add_argument("--early", type=int, default=3, help="early budget point")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--out", default="results/failure_cases.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    A = {(r["smiles"], r["episode_seed"]): r for r in rows if r["policy"] == args.a}
    B = {(r["smiles"], r["episode_seed"]): r for r in rows if r["policy"] == args.b}
    keys = sorted(set(A) & set(B))

    def early(r):
        return r["curve"][min(args.early, len(r["curve"]) - 1)]

    # ---------- How common is this overall ----------
    lead = [k for k in keys if early(A[k]) > early(B[k])]
    lost = [k for k in lead if A[k]["final"] < B[k]["final"]]
    print(f"{args.a} vs {args.b}: {len(keys)} paired episodes")
    print(f"  {args.a} leading at step {args.early}: {len(lead)} "
          f"({len(lead)/max(len(keys),1):.0%})")
    print(f"  of those, overtaken by the end: {len(lost)} "
          f"({len(lost)/max(len(lead),1):.0%} of the leaders)")

    if lost:
        dead_a = np.mean([
            float(np.mean(np.diff(A[k]["eligible_curve"]) == 0)) for k in lost])
        dead_b = np.mean([
            float(np.mean(np.diff(B[k]["eligible_curve"]) == 0)) for k in lost])
        print(f"\n  share of dead steps in these episodes: {args.a} {dead_a:.0%}  "
              f"vs  {args.b} {dead_b:.0%}")
        print("  -> the early lead is given back alongside many zero-yield steps.")

    # ---------- Replay the most representative cases ----------
    scored = sorted(lost, key=lambda k: (early(A[k]) - early(B[k]))
                    + (B[k]["final"] - A[k]["final"]), reverse=True)
    picks = scored[: args.top]
    if not picks:
        print("\nno episodes match the criteria.")
        return 0

    policies = pickle.loads(Path(args.policies).read_bytes())
    env = MolEnv(OracleBank(), EpisodeConfig(max_steps=args.max_steps))
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for n, (smi, eseed) in enumerate(picks, 1):
        print(f"\n{'='*78}\ncase {n}: {smi}  (episode_seed={eseed})")
        print(f"  step {args.early}: {args.a} {early(A[(smi,eseed)]):.4f}  vs  "
              f"{args.b} {early(B[(smi,eseed)]):.4f}")
        print(f"  endpoint:  {args.a} {A[(smi,eseed)]['final']:.4f}  vs  "
              f"{args.b} {B[(smi,eseed)]['final']:.4f}")
        sa = replay(env, policies[args.a], smi, eseed)
        sb = replay(env, policies[args.b], smi, eseed)
        print_side_by_side(sa, sb, args.a, args.b)

        n_dead_a = sum(1 for r in sa if not r["moved"])
        stuck_from = next((r["t"] for r in sa
                           if all(not q["moved"] for q in sa[r["t"]:])), None)
        print(f"\n  {args.a}: {n_dead_a}/{len(sa)} steps without movement"
              + (f", stuck from t={stuck_from} onwards" if stuck_from is not None else ""))
        print(f"  {args.b}: {sum(1 for r in sb if not r['moved'])}/{len(sb)} steps without movement")

        draw(sa, figdir / f"fig6_case{n}_{args.a}.png", f"case {n} / {args.a}")
        draw(sb, figdir / f"fig6_case{n}_{args.b}.png", f"case {n} / {args.b}")
        cases.append({
            "smiles": smi, "episode_seed": eseed,
            f"early_{args.a}": early(A[(smi, eseed)]),
            f"early_{args.b}": early(B[(smi, eseed)]),
            f"final_{args.a}": A[(smi, eseed)]["final"],
            f"final_{args.b}": B[(smi, eseed)]["final"],
            "stuck_from": stuck_from,
            f"{args.a}_trace": [{k: v for k, v in r.items()} for r in sa],
            f"{args.b}_trace": [{k: v for k, v in r.items()} for r in sb],
        })

    Path(args.out).write_text(json.dumps({
        "a": args.a, "b": args.b, "early_step": args.early,
        "n_pairs": len(keys), "n_lead": len(lead), "n_lead_then_lost": len(lost),
        "cases": cases,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
