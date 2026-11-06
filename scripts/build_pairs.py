"""Turn the molecule corpus into edit pairs: analog pairs sharing a
Bemis-Murcko scaffold.

Pairs rather than single molecules because an editor modifies the current
molecule instead of rebuilding one from a scaffold. Conditioned on the current
molecule, learning (src, target bins) -> dst only requires copying most of the
input and changing a little, which makes scaffold preservation learnable. Under a
scaffold-only condition the smoke test preserved the scaffold 0% of the time (see
the comment on src/generator.py:make_prefix).

Same-scaffold analog pairs are exactly the analog series of lead optimization, so
this construction is the right formalization of the task, not a concession to the
model.

Both directions are kept (A->B and B->A). Keeping only the improving direction
would make the condition bins redundant and the model would learn to ignore them
and always improve, leaving routing nothing to choose between.

  python scripts/build_pairs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interfaces import OBJECTIVES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/gen_corpus.jsonl")
    ap.add_argument("--out", default="data/gen_pairs.jsonl")
    ap.add_argument("--meta-in", default="data/gen_meta.json")
    ap.add_argument("--meta-out", default="data/gen_pairs_meta.json")
    ap.add_argument("--max-per-scaffold", type=int, default=30,
                    help="cap per scaffold, since large families swamp the training set")
    ap.add_argument("--max-len", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.corpus).read_text().splitlines()]
    meta = json.loads(Path(args.meta_in).read_text())

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["scaffold"]].append(r)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"{len(rows)} molecules -> {len(groups)} scaffolds, "
          f"{len(multi)} of them with >=2 members")

    rng = np.random.default_rng(args.seed)
    pairs = []
    for scaf, members in multi.items():
        n = len(members)
        # All-pairs explodes as n^2 on large families, so sample up to the cap
        budget = min(args.max_per_scaffold, n * (n - 1))
        seen: set[tuple[int, int]] = set()
        tries = 0
        while len(seen) < budget and tries < budget * 10:
            tries += 1
            i, j = rng.integers(0, n, 2)
            if i == j or (i, j) in seen:
                continue
            seen.add((int(i), int(j)))
        for i, j in seen:
            a, b = members[i], members[j]
            if len(a["smiles"]) > args.max_len or len(b["smiles"]) > args.max_len:
                continue
            pairs.append({"src": a["smiles"], "dst": b["smiles"],
                          "scaffold": scaf, "bins": b["bins"],
                          "src_bins": a["bins"],
                          "dy": [float(y - x) for x, y in zip(a["y"], b["y"])],
                          "split": b["split"]})

    rng.shuffle(pairs)
    tr = sum(1 for p in pairs if p["split"] == "train")
    print(f"built {len(pairs)} edit pairs (train {tr} / val {len(pairs)-tr})")

    # How informative the condition is: the fraction of pairs whose bins differ
    changed = sum(1 for p in pairs if p["bins"] != p["src_bins"])
    print(f"  pairs whose target bins differ from the source: {changed}/{len(pairs)} = "
          f"{changed/max(len(pairs),1):.1%}")
    print("  if this fraction is too low the condition is redundant and the model")
    print("  will learn to ignore it.")
    per_obj = {o: sum(1 for p in pairs if p["bins"][j] != p["src_bins"][j])
               for j, o in enumerate(OBJECTIVES)}
    for o, c in per_obj.items():
        print(f"    {o:6} bin changed {c/max(len(pairs),1):>6.1%}")

    with Path(args.out).open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    meta.update({"n_pairs": len(pairs), "n_pairs_train": tr,
                 "pair_max_per_scaffold": args.max_per_scaffold,
                 "frac_bins_changed": changed / max(len(pairs), 1),
                 "per_objective_bin_change": per_obj,
                 "format": "(src molecule, target bins) -> dst molecule, same scaffold"})
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out} and {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
