"""How much routing gain is left when the action space collapses into two groups?

Motivation: the four conditions of the generative pool collapse into two groups by
cosine similarity ({gsk3b, jnk3} 0.994, {qed, sa} 0.998, 0.92-0.95 between
groups), and the RDKit pool shows the same pairing more weakly. If there are
really only two groups, random's 25/25/25/25 already covers both, while any
"pick the best editor" rule concentrates within a group. Concentrating on an
editor that is nearly equivalent to its neighbour gains nothing and loses
cross-group coverage.

Test, keeping policy identity out of it: look at the relation between group
balance and final HV within a single policy's own episodes. A policy's action
sequence varies only with episode_seed, so balance fluctuates at random and the
comparison stays within one policy. Comparing balance across policies would also
compare which policy it is, which says nothing about the mechanism.

Reads raw.json only; no oracle calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Groups come from the measured cosine matrix, not from hand-splitting into
# "two docking objectives plus two drug-likeness objectives". Both editor
# families give the same split, so one constant serves for both.
GROUPS = {"gsk3b": 0, "jnk3": 0, "qed": 1, "sa": 1}
POLICIES = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def balance(actions: list[str]) -> float:
    """Group balance: 1.0 means an even split, 0.0 means everything in one group."""
    if not actions:
        return float("nan")
    g = np.array([GROUPS[a] for a in actions])
    p1 = float((g == 1).mean())
    return 2.0 * min(p1, 1.0 - p1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Computed here to avoid a scipy dependency. Rank correlation is more robust
    than Pearson on HV's heavy-tailed distribution."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def boot_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = [spearman(x[i], y[i])
           for i in (rng.integers(0, n, size=n) for _ in range(n_boot))]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="results directories, in arm order")
    args = ap.parse_args()

    for d in args.dirs:
        rows = json.loads(Path(d, "raw.json").read_text())
        print(f"\n{'=' * 74}\n{d}\n{'=' * 74}")

        print(f"{'policy':13}{'balance':>9}{'finalHV':>9}"
              f"{'in-group':>11}   actions (g1/g2)")
        for name in POLICIES:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["final"] for r in rs])
            acts = [a for r in rs for a in r["actions"]]
            g1 = np.mean([GROUPS[a] == 0 for a in acts])
            # In-group concentration: how unevenly the two editors in a group are
            # used, 1.0 meaning all calls go to one of them
            in1 = [a for a in acts if GROUPS[a] == 0]
            in2 = [a for a in acts if GROUPS[a] == 1]
            def skew(xs, pair):
                if not xs:
                    return float("nan")
                f = np.mean([x == pair[0] for x in xs])
                return abs(2 * f - 1)
            conc = np.nanmean([skew(in1, ("gsk3b", "jnk3")),
                               skew(in2, ("qed", "sa"))])
            print(f"{name:13}{b.mean():>9.3f}{hv.mean():>9.4f}"
                  f"{conc:>11.3f}   {g1:.0%}/{1 - g1:.0%}")

        # --- Mechanism test: balance vs final HV within each policy ---
        print("\n  within-policy rank correlation of balance vs HV:")
        for name in POLICIES:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["final"] for r in rs])
            if np.nanstd(b) < 1e-9:
                print(f"    {name:13}   balance has no variance (fixed action sequence)")
                continue
            rho = spearman(b, hv)
            lo, hi = boot_ci(b, hv)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"    {name:13}{rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}"
                  f"  n={len(rs)}")

        print("\n  same, controlling for yield: balance vs HV per eligible candidate")
        for name in ["random", "greedy", "bandit", "dqn"]:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["hv_per_eligible"] for r in rs])
            if np.nanstd(b) < 1e-9:
                continue
            rho = spearman(b, hv)
            lo, hi = boot_ci(b, hv)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"    {name:13}{rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
