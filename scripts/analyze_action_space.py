"""Is the action space degenerate: how many distinct directions do the four
editors actually have?

Background: check_editor_specialization.py reports that only 2 of 4 objectives are
led by their own editor, but that is a point estimate with no uncertainty. This
script does three things:

  1. For each objective j, test whether editor j really beats the strongest other
     editor, with seed-paired bootstrap confidence intervals.
  2. Compute cosine similarities between editors' improvement vectors. Two editors
     with similarity near 1 are interchangeable to the router, and choosing between
     them makes no difference.
  3. Report the effective dimension of the action space via the participation
     ratio: nominally 4 actions, how many distinct directions in practice.

This is a precondition for the whole project: if the action space is itself
degenerate, then "learned routing shows no significant advantage" is a conclusion
about the editing pool, not about routing. The two must be reported separately.

  python scripts/analyze_action_space.py --n-seeds 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    objective_scales, participation_ratio,
)
from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.gen_editors import make_gen_editors  # noqa: E402
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def best_delta_per_seed(seeds, editors, bank, k: int):
    """D[s, e, j] = largest gain on objective j among editor e's candidates for seed s.

    Uses the best candidate rather than the mean, because that is what the
    environment's transition rule takes. A call with no eligible candidate records
    NaN rather than 0, since 0 would disguise "proposed nothing" as "proposed
    something with no gain".
    """
    n_e, n_o = len(EXPERTS), len(OBJECTIVES)
    D = np.full((len(seeds), n_e, n_o), np.nan)
    for s, smi in enumerate(seeds):
        parent = canonical(smi)
        if parent is None:
            continue
        base = bank.vector(parent)
        if base is None:
            continue
        base = np.asarray(base)
        for ei, e in enumerate(EXPERTS):
            best = None
            for cand in editors[e].propose(parent, k, seed=s):
                c = canonical(cand)
                if c is None or not scaffold_preserved(parent, c):
                    continue
                v = bank.vector(c)
                if v is None:
                    continue
                d = np.asarray(v) - base
                best = d if best is None else np.maximum(best, d)
            if best is not None:
                D[s, ei] = best
        if (s + 1) % 20 == 0:
            print(f"  {s+1}/{len(seeds)}")
    return D


def boot_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = np.nanmean(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_test.txt")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="results/action_space.json")
    ap.add_argument("--pairs", default="data/gen_pairs_delta.jsonl",
                    help="source of per-objective reachable-change scales, used to normalize eff dim")
    ap.add_argument("--editors", choices=["rdkit", "gen"], default="rdkit",
                    help="rdkit = deterministic reaction pool (the control); "
                         "gen = conditional generator (the main experiment)")
    ap.add_argument("--ckpt", default="results_gen/generator.pt")
    ap.add_argument("--pool-overlap", type=float, default=0.0,
                    help="rdkit only: pool overlap ratio, see src/editors.py:_mix")
    ap.add_argument("--qed-disjoint", action="store_true",
                    help="rdkit only: use _QED_SWAP instead of _SHRINK for heavy molecules")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = (make_gen_editors(args.ckpt, bank) if args.editors == "gen"
               else make_editors(overlap=args.pool_overlap,
                                 qed_disjoint=args.qed_disjoint))
    print(f"editor pool: {args.editors}"
          + (f" ({args.ckpt})" if args.editors == "gen" else ""))

    print(f"measuring best-candidate gains per editor over {len(seeds)} seeds...")
    D = best_delta_per_seed(seeds, editors, bank, args.k)

    cover = (~np.isnan(D[:, :, 0])).mean(axis=0)
    print("\nfraction of seeds where each editor proposed at least one eligible candidate:")
    print("  " + "  ".join(f"{LABEL[e]}:{c:.0%}" for e, c in zip(EXPERTS, cover)))

    M = np.nanmean(D, axis=0)          # 4 editors x 4 objectives
    print("\nmean best-gain matrix (rows = editor, columns = objective)")
    print(f"  {'editor':<8}" + "".join(f"{LABEL[o]:>10}" for o in OBJECTIVES))
    for ei, e in enumerate(EXPERTS):
        print(f"  {LABEL[e]:<8}" + "".join(f"{M[ei, j]:>10.4f}"
                                           for j in range(len(OBJECTIVES))))

    # ---------- 1. Per objective: is the matching editor really the strongest ----------
    print("\npaired test: is editor j stronger on objective j than the best other editor")
    print("  (paired by seed, bootstrap 95% CI; only seeds where both sides have")
    print("  an eligible candidate are counted)")
    spec = {}
    for j, o in enumerate(OBJECTIVES):
        own = D[:, j, j]
        others = np.delete(D[:, :, j], j, axis=1)
        rival_idx = int(np.nanargmax(np.nanmean(others, axis=0)))
        rival_name = [e for i, e in enumerate(EXPERTS) if i != j][rival_idx]
        rival = others[:, rival_idx]
        ok = ~np.isnan(own) & ~np.isnan(rival)
        d = own[ok] - rival[ok]
        lo, hi = boot_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        win = "+" if lo > 0 else ("-" if hi < 0 else "~")
        spec[o] = {"rival": rival_name, "mean": float(d.mean()),
                   "lo": lo, "hi": hi, "n": int(ok.sum()),
                   "specialized": bool(lo > 0)}
        print(f"  {LABEL[o]:<6} vs {LABEL[rival_name]:<6}"
              f"{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}  n={ok.sum():<4}{win}")
    n_spec = sum(v["specialized"] for v in spec.values())
    print(f"\n  editors genuinely specialized on their own objective: {n_spec}/4"
          "  (+ = significantly stronger than the best rival, ~ = no significant"
          " difference, - = significantly weaker)")

    # ---------- 2. Pairwise editor similarity ----------
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    C = Mn @ Mn.T
    print("\ncosine similarity of editor improvement vectors (near 1 = interchangeable)")
    print("  " + " " * 8 + "".join(f"{LABEL[e]:>9}" for e in EXPERTS))
    for ei, e in enumerate(EXPERTS):
        print(f"  {LABEL[e]:<8}" + "".join(f"{C[ei, j]:>9.3f}"
                                           for j in range(len(EXPERTS))))
    pairs = [(EXPERTS[i], EXPERTS[j], float(C[i, j]))
             for i in range(len(EXPERTS)) for j in range(i + 1, len(EXPERTS))]
    for a, b, v in sorted(pairs, key=lambda t: -t[2])[:2]:
        print(f"  most similar pair: {LABEL[a]} - {LABEL[b]} = {v:.3f}")

    # ---------- 3. Effective dimension of the action space ----------
    sv = np.linalg.svd(M, compute_uv=False)
    eff = participation_ratio(M)
    # Normalize by each objective's reachable-change scale first; otherwise the
    # objective with the largest dy scale dominates the singular values (see
    # analysis.objective_scales).
    s = objective_scales(args.pairs)
    eff_n = participation_ratio(M / s) if s is not None else float("nan")
    print(f"\n  singular values {np.round(sv, 4).tolist()}")
    print(f"  effective dimension (raw)        = {eff:.2f} / 4.00")
    print(f"  effective dimension (normalized) = {eff_n:.2f} / 4.00"
          f"   reachable upper bound 3.45")
    print(f"  -> 4 editors nominally, about {eff_n:.1f} distinct edit directions.")

    Path(args.out).write_text(json.dumps({
        "n_seeds": len(seeds), "coverage": cover.round(4).tolist(),
        "mean_best_delta": M.round(5).tolist(),
        "specialization": spec,
        "editor_cosine": C.round(4).tolist(),
        "singular_values": sv.round(5).tolist(),
        "effective_action_dim": eff,
        "effective_action_dim_scaled": eff_n,
        "achievable_eff_dim_scaled": 3.45,
        "n_specialized": int(n_spec),
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
