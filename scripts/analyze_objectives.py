"""How many dimensions do the four objectives actually span: Spearman
correlations plus PCA effective dimension.

Why this matters: a common concern is that too few objectives leave no room for
routing. The four objectives used here (GSK3B / JNK3 / QED / SA) are the standard
combination shared by MAMO, MultiMol and RationaleRL. But four objectives are not
four independent directions: if GSK3B and JNK3 are highly correlated (both are
random-forest kinase classifiers on ECFP6 features) and QED and SA are highly
correlated (both prefer small, simple molecules), the real trade-off space is only
2-dimensional and routing has little room to work in. As far as we know this has
not been verified.

Method: take the score matrix over the test seeds and their one-step edit
candidates, compute pairwise Spearman correlations, run PCA on the standardized
scores, and report the effective dimension via the participation ratio.

  python scripts/analyze_objectives.py --n-seeds 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def spearman(X: np.ndarray) -> np.ndarray:
    """Rank correlation: objective scores are heavily skewed (JNK3 is mostly near
    0) and Pearson would be dominated by the tail."""
    R = np.apply_along_axis(lambda c: np.argsort(np.argsort(c)), 0, X).astype(float)
    R -= R.mean(axis=0)
    denom = np.sqrt((R ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    return (R.T @ R) / np.outer(denom, denom)


def effective_dim(evals: np.ndarray) -> float:
    """Participation ratio: (sum of eigenvalues)^2 / sum of squares.

    Four fully independent objectives with equal variance give 4.0; two pairs of
    perfectly coincident objectives give 2.0. More continuous than "components
    needed for 90% of the variance" and free of a threshold.
    """
    return float(evals.sum() ** 2 / np.maximum((evals ** 2).sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_test.txt")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="results/objective_analysis.json")
    ap.add_argument("--npz", default="results/objective_scores.npz")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = make_editors()

    rows, origin = [], []
    print(f"scoring {len(seeds)} seeds and their one-step edit candidates...")
    for i, smi in enumerate(seeds):
        c0 = canonical(smi)
        if c0 is None:
            continue
        v = bank.vector(c0)
        if v is not None:
            rows.append(v)
            origin.append("seed")
        for e in EXPERTS:
            for cand in editors[e].propose(c0, args.k, seed=i):
                c = canonical(cand)
                if c is None or not scaffold_preserved(c0, c):
                    continue
                v = bank.vector(c)
                if v is not None:
                    rows.append(v)
                    origin.append(e)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(seeds)}  {len(rows)} molecules so far")

    X = np.array(rows, dtype=float)
    origin = np.array(origin)
    print(f"\nscore matrix {X.shape} ({int((origin=='seed').sum())} seeds + "
          f"{int((origin!='seed').sum())} eligible candidates)\n")

    # ---------- Distribution per objective ----------
    print("Distribution per objective (all oriented so higher is better)")
    print(f"  {'objective':<10}{'mean':>8}{'std':>8}{'median':>8}{'p90':>8}{'>0.5':>10}")
    dist = {}
    for j, o in enumerate(OBJECTIVES):
        c = X[:, j]
        dist[o] = {"mean": float(c.mean()), "std": float(c.std()),
                   "median": float(np.median(c)), "p90": float(np.percentile(c, 90)),
                   "frac_above_half": float((c > 0.5).mean())}
        print(f"  {LABEL[o]:<10}{c.mean():>8.3f}{c.std():>8.3f}"
              f"{np.median(c):>8.3f}{np.percentile(c,90):>8.3f}"
              f"{(c>0.5).mean():>10.1%}")
    print("\n  Low mean and variance on GSK3B/JNK3 means most molecules are near 0"
          "\n  on both kinases, so the archive HV is carried mainly by QED and SA.")

    # ---------- Pairwise correlation ----------
    S = spearman(X)
    print("\nPairwise Spearman correlation")
    print("  " + " " * 8 + "".join(f"{LABEL[o]:>9}" for o in OBJECTIVES))
    for i, o in enumerate(OBJECTIVES):
        print(f"  {LABEL[o]:<8}" + "".join(f"{S[i, j]:>9.3f}"
                                           for j in range(len(OBJECTIVES))))

    pairs = [(i, j) for i in range(len(OBJECTIVES))
             for j in range(i + 1, len(OBJECTIVES))]
    strong = [(OBJECTIVES[i], OBJECTIVES[j], S[i, j]) for i, j in pairs
              if abs(S[i, j]) > 0.3]
    if strong:
        print("\n  objective pairs with |rho| > 0.3:")
        for a, b, v in sorted(strong, key=lambda t: -abs(t[2])):
            print(f"    {LABEL[a]} - {LABEL[b]}: {v:+.3f}")

    # ---------- PCA ----------
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
    C = np.cov(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    ratio = evals / evals.sum()

    print("\nPCA on standardized scores (eigendecomposition of the correlation matrix)")
    print(f"  {'PC':<5}{'eigenval':>9}{'var %':>10}{'cum %':>9}   loadings")
    for i in range(len(evals)):
        load = "  ".join(f"{LABEL[o]}:{evecs[j, i]:+.2f}"
                         for j, o in enumerate(OBJECTIVES))
        print(f"  PC{i+1:<4}{evals[i]:>9.3f}{ratio[i]:>10.1%}"
              f"{ratio[:i+1].sum():>9.1%}   {load}")

    eff = effective_dim(evals)
    n90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
    print(f"\n  effective dimension (participation ratio) = {eff:.2f} / 4.00")
    print(f"  {n90} principal components explain 90% of the variance")
    print(f"  -> 4 objectives nominally, a trade-off space of about {eff:.1f} dimensions.")

    # ---------- Per-editor distribution: does each editor own a dimension ----------
    print("\nMean scores of each editor\'s candidates (does conditioning work)")
    print(f"  {'editor':<8}{'n':>7}" + "".join(f"{LABEL[o]:>9}" for o in OBJECTIVES))
    by_editor = {}
    base = X[origin == "seed"].mean(axis=0)
    for e in EXPERTS:
        m = origin == e
        if m.sum() < 10:
            continue
        mu = X[m].mean(axis=0)
        by_editor[e] = {"n": int(m.sum()), "mean": mu.round(4).tolist(),
                        "delta_vs_seed": (mu - base).round(4).tolist()}
        print(f"  {LABEL[e]:<8}{int(m.sum()):>7}"
              + "".join(f"{v:>9.3f}" for v in mu))
    print(f"  {'(seeds)':<8}{int((origin=='seed').sum()):>7}"
          + "".join(f"{v:>9.3f}" for v in base))
    print("\n  Each editor should be highest relative to the seeds in its own"
          "\n  column; if not, the objective-conditioned editors have not diverged.")

    report = {
        "n_molecules": int(X.shape[0]),
        "n_seeds": int((origin == "seed").sum()),
        "objectives": list(OBJECTIVES),
        "distribution": dist,
        "spearman": S.round(4).tolist(),
        "pca_eigenvalues": evals.round(4).tolist(),
        "pca_variance_ratio": ratio.round(4).tolist(),
        "pca_loadings": evecs.round(4).tolist(),
        "effective_dim": eff,
        "n_components_90pct": n90,
        "by_editor": by_editor,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    np.savez(args.npz, X=X, origin=origin)
    print(f"\nwrote {args.out} and {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
