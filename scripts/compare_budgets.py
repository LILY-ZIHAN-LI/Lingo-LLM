"""Side-by-side comparison of the two budget definitions. Runs on an existing
results/raw.json without rerunning the experiment.

  python scripts/compare_budgets.py --raw results/raw.json
  python scripts/compare_budgets.py --raw results_s8/raw.json --cand-budget 20

Why this script exists: the HV-per-candidate ratio systematically favours policies
with few candidates (HV gain saturates, so a smaller denominator inflates the
ratio) and cannot support a conclusion. The clean approach is to fix a candidate
budget x and compare every policy's HV at the same x. This script does that and
prints the same comparison under the editor-call budget alongside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, episode_key, hv_at_eligible, hv_at_step,
    pick_candidate_budget, print_pair_table, summarize_pair,
)

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--cand-budget", type=int, nargs="*", default=None,
                    help="candidate budget points; defaults to a conservative point plus a few common ones")
    ap.add_argument("--min-common", type=float, default=0.8,
                    help="required common-support coverage at a candidate budget point")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="editor-call budget points to compare")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    names = [n for n in ORDER if any(r["policy"] == n for r in rows)]
    max_steps = max(len(r["curve"]) for r in rows) - 1
    steps = args.steps or sorted({3, 5, max_steps // 2, max_steps})

    reach = np.array([r["eligible_curve"][-1] for r in rows])
    x_list = args.cand_budget or sorted(
        {pick_candidate_budget(rows, args.min_common), 10, 20, 30})
    x_list = [x for x in x_list if x > 0]
    # The level table uses the most conservative point (largest common support)
    x_cand = x_list[0]
    keys = common_support(rows, x_cand)
    cov = coverage_at(rows, x_cand)

    print(f"{args.raw}:  {len(rows)} episodes, {max_steps} steps, "
          f"{len(names)} policies")
    print(f"final eligible candidates: median {np.median(reach):.0f}, "
          f"25th pct {np.percentile(reach, 25):.0f}, min {reach.min():.0f}")
    print(f"candidate budget x = {x_cand}, common support {len(keys)} episodes "
          f"({cov['_common']:.0%})")
    print("  fraction of episodes reaching x per policy: " + "  ".join(
        f"{n}:{cov[n]:.0%}" for n in names))
    print("  Common support is the set of episodes where every policy reached x;"
          " every candidate-budget comparison uses only these, which avoids the"
          " selection bias of dropping low-yield trajectories.\n")

    # ---------- Levels under both budget definitions ----------
    print(f"{'policy':13}{'finalHV':>9}{'eligible':>10}"
          f"{'HV@x=' + str(x_cand):>11}{'reach':>8}")
    levels = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        fin = np.mean([r["final"] for r in sub])
        el = np.mean([r["eligible_curve"][-1] for r in sub])
        at = [hv_at_eligible(r, x_cand) for r in sub
              if episode_key(r) in keys]
        ok = [v for v in at if v is not None]
        levels[n] = {"final": float(fin), "eligible": float(el),
                     "hv_at_cand": float(np.mean(ok)) if ok else None,
                     "frac_reaching": float(cov[n])}
        hv_s = f"{np.mean(ok):.4f}" if ok else "  n/a"
        print(f"{n:13}{fin:>9.4f}{el:>9.1f}{hv_s:>11}{cov[n]:>8.0%}")

    print("\n  Disagreement between 'finalHV' and 'HV@x' is a budget-definition flip."
          "\n  The former charges editor calls and favours high-yield editors; the"
          "\n  latter charges candidate molecules.")

    pairs = [(n, "random") for n in names if n != "random"]
    pairs += [("bandit", "greedy"), ("dqn", "bandit"), ("greedy", "round_robin")]
    pairs = [(a, b) for a, b in pairs if a in names and b in names]

    report = {"raw": args.raw, "x_cand": x_cand, "coverage": cov,
              "n_common_support": len(keys), "levels": levels,
              "by_step": {}, "by_candidate": None}

    for x in steps:
        res = [summarize_pair(rows, a, b, lambda r: hv_at_step(r, x))
               for a, b in pairs]
        print_pair_table(res, f"[editor-call budget] paired comparison @ call {x}")
        report["by_step"][str(x)] = res

    report["by_candidate"] = {}
    print("\nCoverage trade-off across candidate budgets (larger budgets"
          " discriminate better but fewer episodes reach them):")
    print(f"  {'x':>5}  {'common':>8}  " + "  ".join(f"{n:>11}" for n in names))
    for x in x_list:
        c = coverage_at(rows, x)
        print(f"  {x:>5}  {c['_common']:>8.0%}  "
              + "  ".join(f"{c[n]:>11.0%}" for n in names))
    for x in x_list:
        k = common_support(rows, x)
        if len(k) < 30:
            print(f"\n[candidate budget] skipping x={x}: common support is only {len(k)} episodes")
            continue
        res = [summarize_pair(rows, a, b, lambda r: hv_at_eligible(r, x),
                              restrict=k) for a, b in pairs]
        print_pair_table(res, f"[candidate budget] paired comparison @ {x} "
                              f"eligible candidates (common support n={len(k)})")
        report["by_candidate"][str(x)] = res

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
