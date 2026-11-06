"""How much does the absorbing state cost: add a retry rule at evaluation time,
with no retraining.

The rule is small: if the editor the router picked has already produced 0 eligible
candidates on the current molecule, pick the next editor that has not failed yet
(in fixed order, so it stays deterministic). Everything else is unchanged: the
same editors, the same scoring, the same transition rule, the same 20-call budget.

This gives an upper-bound estimate of the mechanism's cost: nothing is learned, so
the masked-minus-unmasked difference is how much HV comes back from merely not
spinning on a dead editor. It is the counterpart to the escape-bit arm, which
feeds the same information into the observation and lets the policy learn: one is
a structural fix, the other a representational one.

  python scripts/eval_masking.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import bootstrap_ci  # noqa: E402
from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, N_ACTIONS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def rollout_masked(env, policy, smiles, episode_seed, mask: bool):
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    curve, actions, n_overridden = [0.0], [], 0
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        if mask and a in env.failed_on_current:
            # Find the next editor that has not failed on this molecule, in fixed
            # order; if all have failed, keep the original choice, since every
            # option is a dead step then
            for d in range(1, N_ACTIONS):
                alt = (a + d) % N_ACTIONS
                if alt not in env.failed_on_current:
                    a = alt
                    n_overridden += 1
                    break
        obs, _, done, info = env.step(a)
        actions.append(EXPERTS[a])
        curve.append(info["hv"] - hv0)
        if done:
            break
    eligible = [lg.n_scaffold_ok for lg in env.logs]
    return {
        "curve": curve, "final": float(curve[-1]), "actions": actions,
        "eligible": int(sum(eligible)),
        "eligible_curve": np.cumsum([0] + eligible).tolist(),
        "dead_steps": int(sum(1 for e in eligible if e == 0)),
        "overridden": n_overridden,
    }


def _init(cfg_dict, blob):
    global _ENV, _POL
    _ENV = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict))
    _POL = pickle.loads(blob)


def _work(job):
    name, smi, eseed, mask = job
    r = rollout_masked(_ENV, _POL[name], smi, eseed, mask)
    r.update({"policy": name, "smiles": smi, "episode_seed": eseed,
              "mask": mask})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/masking.json")
    ap.add_argument("--raw-out", default="results/masking_raw.json",
                    help="raw rows under the retry rule, ready for compare_budgets.py")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    policies = pickle.loads(Path(args.policies).read_bytes())
    seeds = Path(args.data).joinpath("seeds_test.txt").read_text().split()
    seeds = seeds[: args.n_test]
    cfg = {"max_steps": args.max_steps, "lam": 0.0}

    jobs = [(n, m, rep * 10000 + i, mask)
            for n in policies for mask in (False, True)
            for rep in range(args.repeats) for i, m in enumerate(seeds)]
    print(f"evaluating {len(jobs)} episodes with {args.workers} processes...")
    with Pool(args.workers, initializer=_init,
              initargs=(cfg, pickle.dumps(policies))) as pool:
        rows = pool.map(_work, jobs, chunksize=4)

    def pick(n, mask, key="final"):
        return {(r["smiles"], r["episode_seed"]): r[key] for r in rows
                if r["policy"] == n and r["mask"] == mask}

    print(f"\n{'policy':13}{'baseHV':>9}{'retryHV':>10}{'diff':>10}"
          f"{'95% CI':>22}{'dead':>13}{'override':>10}")
    report = {}
    for n in policies:
        a, b = pick(n, True), pick(n, False)
        ks = sorted(set(a) & set(b))
        d = np.array([a[k] - b[k] for k in ks])
        lo, hi = bootstrap_ci(d)
        d0 = np.mean([r["dead_steps"] for r in rows
                      if r["policy"] == n and not r["mask"]]) / args.max_steps
        d1 = np.mean([r["dead_steps"] for r in rows
                      if r["policy"] == n and r["mask"]]) / args.max_steps
        ov = np.mean([r["overridden"] for r in rows
                      if r["policy"] == n and r["mask"]]) / args.max_steps
        sig = "*" if (lo > 0 or hi < 0) else " "
        report[n] = {"base": float(np.mean(list(b.values()))),
                     "masked": float(np.mean(list(a.values()))),
                     "diff": float(d.mean()), "lo": lo, "hi": hi,
                     "dead_base": float(d0), "dead_masked": float(d1),
                     "override_rate": float(ov), "n": len(ks)}
        print(f"{n:13}{np.mean(list(b.values())):>9.4f}"
              f"{np.mean(list(a.values())):>10.4f}{d.mean():>+10.5f}"
              f"  [{lo:+.5f}, {hi:+.5f}]{sig}"
              f"{d0:>7.0%}->{d1:<5.0%}{ov:>10.0%}")

    print("\n  'override' is the share of steps where the retry rule changed the")
    print("  original choice. A significantly positive paired difference means the")
    print("  absorbing state really costs HV, and that it can be recovered without")
    print("  learning anything.")

    # Under the retry rule, do the learned routers still beat random
    print(f"\nunder the retry rule, vs random (both with retry)")
    base = pick("random", True)
    for n in policies:
        if n == "random":
            continue
        cur = pick(n, True)
        ks = sorted(set(base) & set(cur))
        d = np.array([cur[k] - base[k] for k in ks])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        report.setdefault("vs_random_masked", {})[n] = {
            "mean": float(d.mean()), "lo": lo, "hi": hi}
        print(f"  {n:13}{d.mean():>+10.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    # The retry rule gives policies more candidates, so a conclusion under the
    # editor-call budget only holds if it also holds under the candidate-count
    # budget. Saved in raw.json format for
    # compare_budgets.py --raw results/masking_raw.json to cross-check.
    Path(args.raw_out).write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "mask"}
         for r in rows if r["mask"]]))
    print(f"wrote {args.raw_out} (retry-rule rows only)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
