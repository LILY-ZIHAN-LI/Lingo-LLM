"""Routing gain vs effective action-space dimension, as a curve rather than two
points.

Why this script instead of reading each run's log:
  1. Each run picks its own candidate budget x (results_base0 picked 10,
     seed1/seed2 picked 5), which makes the variants incomparable. Here x is
     fixed and everything is recomputed.
  2. Each variant has 3 training seeds. Measured retraining noise of 0.00355
     exceeds every effect in the sweep, so single-seed points are unusable. This
     reports the cross-seed mean and range, and only calls routing beneficial at a
     point when the sign is consistent.

The effective dimension is the per-objective normalized one; the raw version is
dominated by the objective with the largest reachable change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (                                      # noqa: E402
    common_support, episode_key, hv_at_eligible,
)

# Variants: label -> (effective-dimension json, [results dir per training seed])
# alpha=0 reuses the three existing rdkit seeds and was not rerun.
MANIFEST = {
    "overlap 1.00": ("results_sweep/action_space_a100.json",
                     ["results_sweep/a100_s0", "results_sweep/a100_s1",
                      "results_sweep/a100_s2"]),
    "overlap 0.50": ("results_sweep/action_space_a050.json",
                     ["results_sweep/a050_s0", "results_sweep/a050_s1",
                      "results_sweep/a050_s2"]),
    "overlap 0.25": ("results_sweep/action_space_a025.json",
                     ["results_sweep/a025_s0", "results_sweep/a025_s1",
                      "results_sweep/a025_s2"]),
    "qdis + a0.07": ("results_sweep/action_space_q007.json",
                     ["results_sweep/q007_s0", "results_sweep/q007_s1",
                      "results_sweep/q007_s2"]),
    "qdis + a0.03": ("results_sweep/action_space_q003.json",
                     ["results_sweep/q003_s0", "results_sweep/q003_s1",
                      "results_sweep/q003_s2"]),
    "overlap 0 (base)": ("results_sweep/action_space_a000.json",
                         ["results_base0", "results_seed1", "results_seed2"]),
    "qed disjoint": ("results_sweep/action_space_qdis.json",
                     ["results_sweep/qdis_s0", "results_sweep/qdis_s1",
                      "results_sweep/qdis_s2"]),
    # Generative branch: a different editor family, so mark it separately in the
    # figure rather than treating it as a point on the same curve
    "generative": ("results_gen/conditioning_delta.json",
                   ["results_gen_router"]),
}
POLICIES = ["round_robin", "gap", "greedy", "bandit", "dqn"]


def eff_dim(path: str) -> float | None:
    """Both diagnostic scripts write the same key (shared implementation in
    src/analysis.py)."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("effective_action_dim_scaled")


def gain_vs_random(rows: list[dict], x: float) -> dict:
    """Paired difference against random at a fixed candidate budget x, with
    bootstrap CIs."""
    keys = common_support(rows, x)
    if not keys:
        return {}
    at = {}
    for r in rows:
        k = episode_key(r)
        if k in keys:
            v = hv_at_eligible(r, x)
            if v is not None:
                at.setdefault(r["policy"], {})[k] = v
    if "random" not in at:
        return {}
    out = {"_n": len(keys)}
    rng = np.random.default_rng(0)
    for name in POLICIES:
        if name not in at:
            continue
        shared = sorted(set(at[name]) & set(at["random"]))
        if not shared:
            continue
        d = np.array([at[name][k] - at["random"][k] for k in shared])
        idx = rng.integers(0, len(d), size=(10000, len(d)))
        m = d[idx].mean(axis=1)
        out[name] = (float(d.mean()),
                     float(np.percentile(m, 2.5)),
                     float(np.percentile(m, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-x", "--budget", type=float, default=5.0,
                    help="fixed candidate budget; 5 gives ~80%% common support in every run")
    ap.add_argument("--out", default="results_sweep/dimension_curve.json")
    args = ap.parse_args()

    table = {}
    for label, (dim_path, dirs) in MANIFEST.items():
        ed = eff_dim(dim_path)
        per_seed = []
        for d in dirs:
            f = Path(d, "raw.json")
            if not f.exists():
                continue
            per_seed.append(gain_vs_random(json.loads(f.read_text()),
                                           args.budget))
        if not per_seed:
            print(f"!! {label}: no results yet, skipping")
            continue
        table[label] = {"eff_dim": ed, "n_seeds": len(per_seed),
                        "seeds": per_seed}

    print(f"\nfixed candidate budget x = {args.budget:g} (the same x for every "
          f"variant, so they are comparable)")
    print("values are paired differences in final HV gain against random\n")
    hdr = f"{'variant':<18}{'eff dim':>8}{'seeds':>6}  "
    hdr += "".join(f"{p:>22}" for p in ["greedy", "bandit", "dqn"])
    print(hdr)
    print("-" * len(hdr))
    for label, rec in table.items():
        ed = rec["eff_dim"]
        eds = f"{ed:.2f}" if ed is not None else "  ? "
        cells = []
        for p in ["greedy", "bandit", "dqn"]:
            vals = [s[p] for s in rec["seeds"] if p in s]
            if not vals:
                cells.append(f"{'-':>22}")
                continue
            mu = np.mean([v[0] for v in vals])
            sig = sum(1 for v in vals if v[1] > 0 or v[2] < 0)
            # Sign consistency is the criterion, not an individual CI
            same = all(v[0] > 0 for v in vals) or all(v[0] < 0 for v in vals)
            mark = ("+" if mu > 0 else "-") if same else "~"
            cells.append(f"{mu:>+12.5f} {mark}{sig}/{len(vals)}      "[:22])
        print(f"{label:<18}{eds:>8}{rec['n_seeds']:>6}  " + "".join(cells))

    print("\n  The +/-/~ marks cross-seed sign consistency (~ means the signs")
    print("  disagree and the point is inconclusive); n/m counts seeds whose 95% CI")
    print("  excludes 0. The criterion is the former, not the latter.")
    print("  The 'generative' row is a different editor family with only one seed,")
    print("  so it is a reference point rather than a point on the same curve.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"budget": args.budget, "table": table}, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
