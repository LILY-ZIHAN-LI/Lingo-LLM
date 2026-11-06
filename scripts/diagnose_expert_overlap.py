"""How similar are the four generative experts?

The generative branch measures an effective action dimension of only 1.80, but
that number comes from the downstream gain matrix and cannot tell apart two very
different causes: (i) the condition never reaches the model, or (ii) it does, but
decoding washes the difference out. The fixes differ by an order of magnitude, so
two things are measured directly:

  A) Condition sensitivity in logit space. Teacher-force a parent molecule's own
     token string and change only the bin tokens in the prefix, then compare the
     next-token distributions. A pure forward pass, so the condition is isolated
     exactly. Reported as total variation distance (TV in [0,1]).
     bump is not a sweepable axis: with neutral bin 2 and n_bins=4,
     min(2+bump, 3) saturates for any bump >= 1, so the largest available contrast
     is bin 0 vs bin 3 on one dimension.
     The positive control replaces the source molecule in the prefix instead; that
     changes the context entirely, so its TV must be large, otherwise the
     measurement is invalid rather than the condition.

  B) End-to-end proposal-set overlap: call propose() on the same (molecule, seed)
     with the experiment's decoding parameters and take pairwise Jaccard between
     the experts' proposal sets. Large A with large B points at decoding; small A
     points at the condition itself.

  python scripts/diagnose_expert_overlap.py --n-seeds 60
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical                              # noqa: E402
from src.gen_editors import TEMPERATURE, TOP_K              # noqa: E402
from src.gen_editors import load_backbone, make_gen_editors  # noqa: E402
from src.generator import make_prefix, tokenize             # noqa: E402
from src.interfaces import OBJECTIVES                       # noqa: E402
from src.oracles import OracleBank                          # noqa: E402

PAIRS = list(itertools.combinations(range(len(OBJECTIVES)), 2))


def tv_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-position total variation distance. a and b are [T, V] probability matrices."""
    return 0.5 * np.abs(a - b).sum(axis=1)


def dists(model, prefix: list[int], cont: list[int]) -> np.ndarray | None:
    """Teacher-force cont and return the next-token distribution at each step [T, V].

    Position alignment: the output at position i predicts token i+1, so the slice
    [len(prefix)-1, len(prefix)+len(cont)-2] covers exactly the positions whose
    next token belongs to the molecule itself.
    """
    ids = prefix + cont
    if len(ids) > model.cfg.max_len:
        return None
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))[0]
    lo, hi = len(prefix) - 1, len(prefix) + len(cont) - 1
    return torch.softmax(logits[lo:hi].float(), dim=-1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_gen/generator_delta.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_val.txt")
    ap.add_argument("--n-seeds", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="results_gen/expert_overlap.json")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]

    model, vocab, meta = load_backbone(args.ckpt)
    n_bins = meta["n_bins"]
    mode = meta.get("conditioning", "absolute")
    if mode != "delta":
        print(f"!! this ckpt uses the {mode} scheme; part A's neutral reference is "
              f"only defined for delta")
    neutral = [meta["neutral_bin"][o] for o in OBJECTIVES]
    top = n_bins - 1
    saturated = all(min(v + 1, top) == top for v in neutral)
    print(f"ckpt={args.ckpt}  mode={mode}  n_bins={n_bins}  "
          f"neutral={neutral}  top={top}")
    print(f"bump saturates at bump=1: {saturated}"
          f"{'  => bump is not a sweepable axis' if saturated else ''}")
    print(f"{len(seeds)} parent molecules\n")

    # ---- A) logit space -------------------------------------------------
    acc = {"pair": [], "pair_max": [], "vs_neutral": [[] for _ in OBJECTIVES],
           "min_vs_max": [[] for _ in OBJECTIVES], "control": []}
    n_used = 0
    for si, smi in enumerate(seeds):
        parent = canonical(smi)
        if parent is None:
            continue
        src = tokenize(parent, vocab)
        cont = src + [vocab["<eos>"]]
        base = dists(model, make_prefix(neutral, src, vocab), cont)
        if base is None:
            continue

        P = []            # each expert's actual condition: own dim at top, rest neutral
        ok = True
        for j in range(len(OBJECTIVES)):
            tgt = list(neutral)
            tgt[j] = top
            d = dists(model, make_prefix(tgt, src, vocab), cont)
            # Lowest bin on the same dimension: the largest available contrast
            lo_t = list(neutral)
            lo_t[j] = 0
            dlo = dists(model, make_prefix(lo_t, src, vocab), cont)
            if d is None or dlo is None:
                ok = False
                break
            P.append(d)
            acc["vs_neutral"][j].append(float(tv_rows(d, base).mean()))
            acc["min_vs_max"][j].append(float(tv_rows(d, dlo).mean()))
        if not ok:
            continue

        # Positive control: keep the bins, replace only the source molecule in the
        # prefix. The continuation is still this molecule, so positions stay aligned.
        alt = canonical(seeds[(si + 1) % len(seeds)])
        dctl = (dists(model, make_prefix(neutral, tokenize(alt, vocab), vocab),
                      cont) if alt and alt != parent else None)
        if dctl is not None:
            acc["control"].append(float(tv_rows(dctl, base).mean()))

        pv = [tv_rows(P[i], P[j]) for i, j in PAIRS]
        acc["pair"].append(float(np.mean([x.mean() for x in pv])))
        acc["pair_max"].append(float(np.max([x.max() for x in pv])))
        n_used += 1

    pair = float(np.mean(acc["pair"]))
    pmax = float(np.mean(acc["pair_max"]))
    ctl = float(np.mean(acc["control"])) if acc["control"] else float("nan")
    vn = [float(np.mean(v)) for v in acc["vs_neutral"]]
    mm = [float(np.mean(v)) for v in acc["min_vs_max"]]
    secA = {"pair_tv": pair, "pair_tv_max": pmax, "vs_neutral": vn,
            "min_vs_max": mm, "positive_control_tv": ctl,
            "n_control": len(acc["control"])}

    print(f"{'='*72}\nA) condition sensitivity (logit space, TV in [0,1], "
          f"{n_used} parents)\n{'='*72}")
    print(f"  pairwise TV across the four experts (6 pairs, all positions) {pair:>8.4f}")
    print(f"  largest single-step TV                                      {pmax:>8.4f}")
    print(f"  positive control: source molecule swapped in the prefix     {ctl:>8.4f}"
          f"   <- must be large")
    print(f"\n  {'objective':11}{'top vs neutral':>16}{'top vs lowest':>16}")
    for j, o in enumerate(OBJECTIVES):
        print(f"  {o:11}{vn[j]:>16.4f}{mm[j]:>16.4f}")
    print("\n  The context is identical except for the 4 bin tokens, so these numbers")
    print("  are how much changing the condition changes the output distribution,")
    print("  with no decoding randomness.")

    # ---- B) End-to-end proposal-set overlap ----------------------------
    print(f"\n{'='*72}\nB) proposal-set overlap (real decoding, temperature={TEMPERATURE} "
          f"top_k={TOP_K} K={args.k})\n{'='*72}")
    # A low Jaccard on its own does not mean the experts are specialized: four
    # independent samples from the same distribution also give a low Jaccard. A
    # null control is required: the Jaccard between two proposal sets from the same
    # expert under different sampling seeds. If the between-expert Jaccard matches
    # it, expert identity does nothing beyond changing the seed.
    bank = OracleBank()
    eds = make_gen_editors(args.ckpt, bank, bump=1)
    jac, only_one, n_prop, same_ed = [], [], [], []
    for si, smi in enumerate(seeds):
        sets = [set(eds[o].propose(smi, args.k, seed=1000 + si))
                for o in OBJECTIVES]
        n_prop.append(float(np.mean([len(s) for s in sets])))
        for i, j in PAIRS:
            u = sets[i] | sets[j]
            if u:
                jac.append(len(sets[i] & sets[j]) / len(u))
        for o, s0 in zip(OBJECTIVES, sets):
            s1 = set(eds[o].propose(smi, args.k, seed=7000 + si))
            u = s0 | s1
            if u:
                same_ed.append(len(s0 & s1) / len(u))
        allm = set().union(*sets)
        if allm:
            cnt = {m: sum(m in s for s in sets) for m in allm}
            only_one.append(sum(1 for v in cnt.values() if v == 1) / len(allm))
    secB = {"jaccard": float(np.mean(jac)) if jac else None,
            "jaccard_same_editor_other_seed": float(np.mean(same_ed)),
            "frac_unique_to_one": float(np.mean(only_one)),
            "mean_proposals_per_call": float(np.mean(n_prop)),
            "bump": 1}
    print(f"  pairwise Jaccard between experts        {secB['jaccard']:.3f}")
    print(f"  null control: same expert, other seed  "
          f"{secB['jaccard_same_editor_other_seed']:.3f}   <- control")
    print(f"  molecules proposed by exactly one      "
          f"{secB['frac_unique_to_one']:.3f}")
    print(f"  valid proposals per call               "
          f"{secB['mean_proposals_per_call']:.2f}/{args.k}")

    # ---- Interpretation -------------------------------------------------
    print(f"\n{'='*72}\ninterpretation\n{'='*72}")
    worst = max(mm)
    jb, jc = secB["jaccard"], secB["jaccard_same_editor_other_seed"]
    if not (ctl > 0.05):
        verdict = (f"Measurement invalid: the positive control is only {ctl:.4f}, so "
                   f"even swapping the source molecule did not change the "
                   f"distribution and a small TV cannot be read as an ineffective "
                   f"condition. Check the teacher-force position alignment first.")
    elif jb <= jc * 1.3 and worst < 0.1 * ctl:
        verdict = (
            f"Expert identity amounts to a different sampling seed. The "
            f"between-expert Jaccard {jb:.3f} is no higher than the same-expert "
            f"other-seed {jc:.3f}, and the largest condition contrast {worst:.4f} is "
            f"only 1/{ctl/max(worst,1e-9):.0f} of the positive control {ctl:.4f}. The "
            f"1.80 dimension therefore comes from the condition barely changing the "
            f"distribution, and the low Jaccard is sampling noise rather than "
            f"specialization. Cheap fixes (bump is already saturated, logit bias, "
            f"lower temperature) cannot address this; the weights must change, via a "
            f"per-objective adapter or a cross-expert exclusivity term in training.")
    elif worst < 0.1 * ctl:
        verdict = (f"The condition has little effect on the distribution (largest "
                   f"contrast {worst:.4f} vs control {ctl:.4f}), yet the "
                   f"between-expert Jaccard {jb:.3f} is below the same-expert "
                   f"other-seed {jc:.3f}. The two disagree, so check part B's "
                   f"implementation first.")
    elif jb > jc * 1.3:
        verdict = (f"The condition has a real effect (largest contrast {worst:.4f}) "
                   f"but the proposal sets are more alike than independent resampling "
                   f"({jb:.3f} > {jc:.3f}), so decoding is the bottleneck: lower the "
                   f"temperature or tighten top-k first.")
    else:
        verdict = (f"The condition has a real effect (largest contrast {worst:.4f}) "
                   f"and the experts are genuinely more dispersed than same-expert "
                   f"resampling ({jb:.3f} < {jc:.3f}). The 1.80 dimension then means "
                   f"different proposals with the same property effect, which calls "
                   f"for a change to the training objective.")
    print("  " + verdict)
    print(f"  Reference: generative effective dimension 1.80, RDKit 2.20, "
          f"reachable upper bound 3.45.")

    out = {"ckpt": args.ckpt, "n_parents": n_used, "n_seeds": len(seeds),
           "k": args.k, "temperature": TEMPERATURE, "top_k": TOP_K,
           "objectives": list(OBJECTIVES), "neutral_bin": neutral,
           "bump_saturated_at_1": saturated,
           "logit_sensitivity": secA, "proposal_overlap": secB,
           "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
