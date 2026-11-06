"""Generator quality diagnostic: does changing the condition actually change the
generated distribution?

This is a precondition for the router experiment. If the four conditions produce
the same distribution, the action space collapses to 1 dimension, the same problem
as the RDKit pool's 2.05/4, and comparing routers is pointless. This script must
pass before run_experiment is run.

Criteria, all paired on the same seed set:
  1. Own-objective gain greater than the other three conditions' gain on that
     objective (specialization, the same statistic analyze_action_space.py uses,
     so it compares directly against 2.05/4).
  2. Validity, scaffold retention and uniqueness. The RDKit pool is 100% on these
     by construction; the generator is not, so they must be measured.
  3. Novelty: whether outputs appear in the generator's own training corpus
     (recitation vs generation).

  python scripts/diagnose_conditioning.py --n-seeds 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    bootstrap_ci, objective_scales, participation_ratio,
)
from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.gen_editors import make_gen_editors  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_gen/generator.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_val.txt",
                    help="val by default; test is reserved for the final router comparison")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bump", type=int, default=1,
                    help="bins to push the own objective up; >=n_bins means the top bin")
    ap.add_argument("--corpus", default="data/gen_corpus.jsonl")
    ap.add_argument("--pairs", default="data/gen_pairs_delta.jsonl",
                    help="source of per-objective reachable-change scales, used to normalize eff dim")
    ap.add_argument("--out", default="results_gen/conditioning.json")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = make_gen_editors(args.ckpt, bank, bump=args.bump)

    train_smiles = set()
    if Path(args.corpus).exists():
        for line in Path(args.corpus).read_text().splitlines():
            train_smiles.add(json.loads(line)["smiles"])
    print(f"{len(seeds)} seeds, K={args.k}, bump={args.bump}, "
          f"corpus {len(train_smiles)} rows\n")

    n_obj = len(OBJECTIVES)
    # D[s, e, j] = largest gain on objective j over condition e's eligible candidates
    D = np.full((len(seeds), n_obj, n_obj), np.nan)
    counts = {e: dict(attempts=0, valid=0, scaf=0, uniq=0, novel=0)
              for e in OBJECTIVES}

    for si, smi in enumerate(seeds):
        base = bank.vector(smi)
        if base is None:
            continue
        for ei, ename in enumerate(OBJECTIVES):
            props = editors[ename].propose(smi, args.k, seed=1000 + si)
            c = counts[ename]
            c["attempts"] += args.k        # exactly K attempts; invalid ones count
            c["valid"] += len(props)
            seen_ok = []
            for p in props:
                if scaffold_preserved(smi, p):
                    seen_ok.append(p)
            c["scaf"] += len(seen_ok)
            c["uniq"] += len(set(seen_ok))
            c["novel"] += sum(1 for p in set(seen_ok) if p not in train_smiles)
            gains = []
            for p in set(seen_ok):
                v = bank.vector(p)
                if v is not None:
                    gains.append(np.array(v) - np.array(base))
            if gains:
                D[si, ei] = np.max(np.stack(gains), axis=0)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(seeds)}")

    print(f"\n{'='*74}\n1) generation quality (K={args.k} attempts per call, invalid included)\n{'='*74}")
    print(f"{'condition':11}{'valid':>9}{'scaffold':>12}{'unique':>9}{'novel':>8}")
    qual = {}
    for e in OBJECTIVES:
        c = counts[e]
        a = max(c["attempts"], 1)
        qual[e] = {k: c[k] / a for k in ("valid", "scaf", "uniq", "novel")}
        qual[e]["attempts"] = c["attempts"]
        print(f"{e:11}{c['valid']/a:>9.1%}{c['scaf']/a:>12.1%}"
              f"{c['uniq']/a:>9.1%}{c['novel']/a:>8.1%}")
    print("\n  'novel' means absent from the generator training corpus; a low value"
          "\n  means the model is reciting rather than generating.")

    print(f"\n{'='*74}\n2) does the condition change the distribution (paired, bootstrap 95% CI)\n{'='*74}")
    print("  The rival is the mean of the others, not their maximum: own is a single")
    print("  condition, and a max-of-3 would systematically win even if all four")
    print("  conditions were equivalent (selection bias of extrema), so such a test")
    print("  could never pass.")
    print(f"\n{'objective':11}{'own':>11}{'rival mean':>11}{'diff':>11}{'95% CI':>24}")
    spec = {}
    n_spec = 0
    for j, o in enumerate(OBJECTIVES):
        own = D[:, j, j]
        rivals = np.delete(D[:, :, j], j, axis=1)
        mean_rival = np.nanmean(rivals, axis=1)
        ok = ~np.isnan(own) & ~np.isnan(mean_rival)
        if ok.sum() < 5:
            print(f"{o:11}  too few samples (n={int(ok.sum())})")
            continue
        d = own[ok] - mean_rival[ok]
        lo, hi = bootstrap_ci(d)
        sig = lo > 0 or hi < 0
        n_spec += int(sig and d.mean() > 0)
        spec[o] = {"own": float(np.nanmean(own[ok])),
                   "rival": float(np.nanmean(mean_rival[ok])),
                   "mean": float(d.mean()), "lo": lo, "hi": hi,
                   "n": int(ok.sum()),
                   "specialised": bool(sig and d.mean() > 0)}
        star = "*" if sig else " "
        print(f"{o:11}{np.nanmean(own[ok]):>11.4f}"
              f"{np.nanmean(mean_rival[ok]):>11.4f}"
              f"{d.mean():>+11.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")

    # 2b) Remove the edit-magnitude confound: a condition that simply edits harder
    # wins on every objective. Row-normalizing first and then checking diagonal
    # dominance asks whether the conditions actually divide the work.
    Mraw = np.nan_to_num(np.nanmean(D, axis=0))
    rown = Mraw / (np.abs(Mraw).sum(axis=1, keepdims=True) + 1e-12)
    diag_wins = [int(rown[:, j].argmax() == j) for j in range(n_obj)]
    print(f"\n  strongest condition per objective after row normalization:")
    for j, o in enumerate(OBJECTIVES):
        best = OBJECTIVES[int(rown[:, j].argmax())]
        print(f"    {o:6} -> {best:6}{'  (own)' if diag_wins[j] else ''}")
    print(f"  diagonal dominant on {sum(diag_wins)}/{n_obj} (chance is 1 in 4)")

    # 3) Effective action dimension, directly comparable to the RDKit pool's 2.05/4
    M = np.nanmean(D, axis=0)                      # [editor, objective]
    M = np.nan_to_num(M)
    sv = np.linalg.svd(M, compute_uv=False)
    eff = float(sv.sum() ** 2 / max((sv ** 2).sum(), 1e-12))
    norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = norm @ norm.T

    # Normalize by each objective's reachable-change scale; otherwise the objective
    # with the largest dy scale (qed, 4.2x jnk3's std) dominates the singular
    # values. See analysis.objective_scales.
    s = objective_scales(args.pairs)
    eff_n = participation_ratio(M / s) if s is not None else float("nan")

    print(f"\n{'='*74}\n3) action space\n{'='*74}")
    print(f"singular values {np.round(sv, 4).tolist()}")
    print(f"effective action dim (raw)        {eff:.2f} / {n_obj}"
          f"   (RDKit reaction pool 2.05)")
    print(f"effective action dim (normalized) {eff_n:.2f} / {n_obj}"
          f"   (RDKit reaction pool 2.20, reachable upper bound 3.45)")
    print("\ncosine similarity between conditions (near 1 means one editor):")
    print("        " + "".join(f"{o:>9}" for o in OBJECTIVES))
    for i, o in enumerate(OBJECTIVES):
        print(f"{o:8}" + "".join(f"{cos[i, j]:>9.3f}" for j in range(n_obj)))

    verdict = (eff >= 3.0 and n_spec >= 3)
    print(f"\n{'='*74}")
    print(f"verdict: {n_spec}/4 conditions win on their own objective, "
          f"effective dim {eff:.2f}/4")
    print("PASS - ready for the router comparison." if verdict else
          "FAIL - the action space is still degenerate; fix the generator or the "
          "conditioning before discussing routing.")
    print("="*74)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "quality": qual, "specialization": spec, "n_specialized": n_spec,
        "effective_action_dim": eff,
        "effective_action_dim_scaled": eff_n,
        "singular_values": sv.tolist(),
        "editor_cosine": cos.tolist(), "objectives": list(OBJECTIVES),
        "mean_best_delta": M.tolist(), "n_seeds": len(seeds),
        "k": args.k, "bump": args.bump,
        "row_normalized": rown.tolist(), "diag_wins": sum(diag_wins),
        "passes": bool(verdict), "rdkit_baseline_eff_dim": 2.05,
        "rdkit_baseline_eff_dim_scaled": 2.20,
        "achievable_eff_dim_scaled": 3.45,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
