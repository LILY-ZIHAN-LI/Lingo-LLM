"""Training-seed variance: rerun the same configuration with different training
seeds and see whether the effect or the noise is larger.

An 8-step ablation showed greedy-random flipping from +0.0008 to -0.0012 after a
single retrain. Any claim that a learned router is better must therefore come with
a cross-seed band, or it reports training noise rather than a property of the
method.

Two things happen here:
  1. For several results directories (same configuration, different training
     seeds), compute the policy-minus-random paired difference in each and report
     the cross-seed min/max band.
  2. escape-bit A/B: per-episode paired differences between the escape run and its
     same-seed control. Both run the same (smiles, episode_seed) set, so they pair
     directly.

  python scripts/compare_seeds.py \
      --runs results_base0 results_seed1 results_seed2 \
      --escape results_escape0:results_base0 results_escape1:results_seed1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import bootstrap_ci, episode_key  # noqa: E402

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def load(d: str) -> list[dict]:
    return json.loads((Path(d) / "raw.json").read_text())


def by_key(rows, policy, field="final"):
    return {episode_key(r): r[field] for r in rows if r["policy"] == policy}


def paired(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi, "n": len(ks),
            "sig": lo > 0 or hi < 0}


def seed_band(runs: list[str]) -> dict:
    """Per policy and per training seed: HV and the (policy - random) paired difference."""
    per_run = {}
    for d in runs:
        rows = load(d)
        rnd = by_key(rows, "random")
        pol = {}
        for n in ORDER:
            cur = by_key(rows, n)
            pol[n] = {"hv": float(np.mean(list(cur.values()))),
                      "vs_random": paired(cur, rnd) if n != "random" else None}
        per_run[d] = pol

    print(f"\n{'='*78}\nAcross training seeds: absolute HV\n{'='*78}")
    print(f"{'policy':13}" + "".join(f"{Path(d).name:>16}" for d in runs)
          + f"{'spread':>10}")
    for n in ORDER:
        vs = [per_run[d][n]["hv"] for d in runs]
        print(f"{n:13}" + "".join(f"{v:>16.4f}" for v in vs)
              + f"{max(vs)-min(vs):>10.4f}")

    print(f"\n{'='*78}\nAcross training seeds: (policy - random) paired difference\n"
          f"  * = 95% CI excludes 0 for that seed\n{'='*78}")
    print(f"{'policy':13}" + "".join(f"{Path(d).name:>16}" for d in runs)
          + f"{'band':>22}{'same sign':>11}")
    band = {}
    for n in ORDER:
        if n == "random":
            continue
        ds = [per_run[d][n]["vs_random"] for d in runs]
        ms = [x["mean"] for x in ds]
        cells = "".join(f"{x['mean']:>+15.5f}{'*' if x['sig'] else ' '}"
                        for x in ds)
        same = "yes" if (min(ms) > 0 or max(ms) < 0) else "no"
        band[n] = {"per_seed": ms, "lo": min(ms), "hi": max(ms),
                   "sign_consistent": same == "yes",
                   "n_sig": sum(x["sig"] for x in ds)}
        print(f"{n:13}{cells}  [{min(ms):+.5f}, {max(ms):+.5f}]{same:>9}")

    print("\n  'band' is the cross-seed min/max, not a CI, and must be reported")
    print("  alongside the per-seed CI: the CI captures episode sampling noise,")
    print("  the band captures training noise. A policy whose sign is not")
    print("  consistent cannot be called better than random; that is seed luck.")
    return {"per_run": per_run, "band": band}


def escape_ab(pairs: list[str]) -> dict:
    """escape-bit run vs its same-seed control, paired per episode."""
    print(f"\n{'='*78}\nescape-bit A/B (4 observation bits for editors that already\n"
          f"failed on the current molecule)\n{'='*78}")
    out = {}
    for spec in pairs:
        esc_d, base_d = spec.split(":")
        esc, base = load(esc_d), load(base_d)
        print(f"\n{Path(esc_d).name} vs {Path(base_d).name}")
        print(f"  {'policy':13}{'control':>9}{'escape':>9}{'diff':>11}"
              f"{'95% CI':>24}")
        for n in ORDER:
            a, b = by_key(esc, n), by_key(base, n)
            if not a or not b:
                continue
            r = paired(a, b)
            print(f"  {n:13}{np.mean(list(b.values())):>9.4f}"
                  f"{np.mean(list(a.values())):>9.4f}{r['mean']:>+11.5f}"
                  f"  [{r['lo']:+.5f}, {r['hi']:+.5f}]"
                  f"{'*' if r['sig'] else ' '}")
            out.setdefault(spec, {})[n] = r
    print("\n  random and round_robin are negative controls: they never read these")
    print("  bits, so their rows reflect retraining noise alone and can be read as")
    print("  the noise floor.")
    return out


def decompose(runs: list[str]) -> dict:
    """Split the band into its two noise sources.

    --seed is also passed to make_baselines(seed=...), so RandomPolicy is a
    different realization in each run (src/policies.py uses self.seed in reset).
    The cross-seed band of (policy_i - random_i) therefore mixes two things:
      (a) learning noise: the difference from retraining the policy itself;
      (b) baseline noise: the difference between realizations of random.
    round_robin, gap and bandit are deterministic (spread is exactly 0.0000), so
    their band holds only (b); greedy and dqn have both. Taking the full i-by-j
    grid separates them: dispersion across rows at a fixed column is (a), and
    across columns at a fixed row is (b).
    """
    rows = {d: load(d) for d in runs}
    rnd = {d: by_key(rows[d], "random") for d in runs}

    print(f"\n{'='*78}\nNoise decomposition: learning noise vs random-baseline\n"
          f"realization noise\n{'='*78}")
    # Baseline noise floor: random_i vs random_j, two realizations of random
    floor = [paired(rnd[a], rnd[b])["mean"]
             for i, a in enumerate(runs) for b in runs[i + 1:]]
    print(f"\nrandom_i - random_j (pure baseline noise, no learning):")
    print("  " + "  ".join(f"{v:+.5f}" for v in floor)
          + f"   |max| = {max(abs(v) for v in floor):.5f}")

    out = {"baseline_floor": floor,
           "baseline_floor_abs_max": max(abs(v) for v in floor)}
    print(f"\n{'policy':13}{'learn noise':>13}{'base noise':>12}"
          f"{'full-grid band':>26}{'same sign':>11}")
    for n in ORDER:
        if n == "random":
            continue
        M = np.array([[paired(by_key(rows[i], n), rnd[j])["mean"]
                       for j in runs] for i in runs])
        # Dispersion across rows at a fixed column (same random) = learning noise
        learn = float(np.mean(M.max(axis=0) - M.min(axis=0)))
        # Dispersion across columns at a fixed row (same trained policy) = baseline noise
        base = float(np.mean(M.max(axis=1) - M.min(axis=1)))
        same = "yes" if (M.min() > 0 or M.max() < 0) else "no"
        out[n] = {"learning_noise": learn, "baseline_noise": base,
                  "lo": float(M.min()), "hi": float(M.max()),
                  "sign_consistent": same == "yes", "matrix": M.tolist()}
        print(f"{n:13}{learn:>13.5f}{base:>12.5f}"
              f"      [{M.min():+.5f}, {M.max():+.5f}]{same:>11}")

    print("\n  Learning noise is the swing from changing the training seed under")
    print("  one random baseline; baseline noise is the swing from changing the")
    print("  random realization under one trained policy. Deterministic policies")
    print("  (round_robin / gap / bandit) should show exactly 0 learning noise.")
    print("  Only a sign-consistent full-grid band shows the effect is not noise.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["results_base0", "results_seed1", "results_seed2"])
    ap.add_argument("--escape", nargs="*", default=[],
                    help="escape_dir:control_dir, repeatable")
    ap.add_argument("--out", default="results/seed_variance.json")
    args = ap.parse_args()

    report = {"runs": args.runs}
    report.update(seed_band(args.runs))
    report["decomposition"] = decompose(args.runs)
    if args.escape:
        report["escape_ab"] = escape_ab(args.escape)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
