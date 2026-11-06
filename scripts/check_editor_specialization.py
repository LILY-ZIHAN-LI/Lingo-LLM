"""Go/no-go diagnostic: are the four editors actually specialized?

Run this before training DQN. If an editor is no better on its own objective than
the others, routing cannot pay off in principle and both DQN and the bandit can
only produce a null result. Knowing that early leaves room to change the editing
pool instead of explaining the null in the report.

Usage: python scripts/check_editor_specialization.py --seeds data/seeds_train.txt -n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    mols = Path(args.seeds).read_text().split()[: args.n]
    editors = make_editors()
    bank = OracleBank()

    # Three quantities; only best_delta is what the environment actually uses:
    #   mean_delta  - mean gain over all candidates (diluted by bad edits)
    #   best_delta  - gain of the best candidate per call (what the transition rule takes)
    #   p_positive  - fraction of calls producing at least one positive gain
    mean_d = {e: {o: [] for o in OBJECTIVES} for e in editors}
    best_d = {e: {o: [] for o in OBJECTIVES} for e in editors}
    stats = {e: {"proposed": 0, "valid": 0, "scaf_ok": 0} for e in editors}

    for idx, parent in enumerate(mols):
        base = bank.vector(parent)
        if base is None:
            continue
        for ename, ed in editors.items():
            cands = ed.propose(parent, args.k, seed=idx)
            stats[ename]["proposed"] += args.k
            per_call = {o: [] for o in OBJECTIVES}
            for c in cands:
                v = bank.vector(c)
                if v is None:
                    continue
                stats[ename]["valid"] += 1
                if not scaffold_preserved(parent, c):
                    continue
                stats[ename]["scaf_ok"] += 1
                for j, o in enumerate(OBJECTIVES):
                    d = v[j] - base[j]
                    mean_d[ename][o].append(d)
                    per_call[o].append(d)
            for o in OBJECTIVES:
                if per_call[o]:
                    best_d[ename][o].append(max(per_call[o]))

    def _show(table, title):
        print(f"\n{title}")
        print(f"{'':8}" + "".join(f"{o:>12}" for o in OBJECTIVES))
        mat = []
        for ename in editors:
            row = [float(np.mean(table[ename][o])) if table[ename][o] else 0.0
                   for o in OBJECTIVES]
            mat.append(row)
            print(f"{ename:8}" + "".join(f"{v:>12.4f}" for v in row))
        return np.array(mat)

    _show(mean_d, "mean delta (all candidates)")
    mat = _show(best_d, "best-candidate delta (what the environment uses)")

    # Read column-wise: for each objective, is one editor clearly stronger?
    names = list(editors)
    print("\nStrongest editor per objective (column maximum):")
    col_wins = 0
    for j, o in enumerate(OBJECTIVES):
        col = mat[:, j]
        order = np.argsort(-col)
        top, second = names[order[0]], col[order[0]]
        gap = col[order[0]] - col[order[1]]
        hit = "ok" if top == o else "  "
        print(f"  {o:6} -> {top:8} ({second:+.4f}, lead over runner-up {gap:.4f}) {hit}")
        if top == o:
            col_wins += 1

    print("\nFraction of calls with at least one positive gain:")
    for ename in editors:
        row = " ".join(
            f"{o}={np.mean([d > 0 for d in best_d[ename][o]]):.0%}" if best_d[ename][o]
            else f"{o}=n/a" for o in OBJECTIVES)
        print(f"  {ename:8} {row}")

    print("\nYield:")
    for ename, s in stats.items():
        p = s["proposed"] or 1
        print(f"  {ename:8} valid {s['valid']/p:5.1%}  scaffold_ok {s['scaf_ok']/p:5.1%}")

    diag_wins = col_wins
    print(f"\nObjectives led by their own editor: {diag_wins}/4")
    print(f"\noracle: {bank.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())