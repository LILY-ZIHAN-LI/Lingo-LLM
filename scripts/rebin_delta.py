"""Re-bin the condition from absolute property quantiles to change quantiles.

Absolute bin edges are fitted on ChEMBL percentiles, and ChEMBL is overwhelmingly
inactive against GSK3B and JNK3. Seed molecules are biased towards activity, so
across 40 val seeds all are already in the top gsk3b bin and 65% in the top jnk3
bin, where `min(cur + bump, n_bins - 1)` equals the current bin: the gsk3b
editor's condition becomes identical to "leave it as is". Headroom
(qed 100% > sa 70% > jnk3 35% > gsk3b 0%) matches the specialization ranking
exactly.

With change quantiles, dy = 0 always falls in a middle bin, so headroom is always
100% wherever a molecule sits in property space. The condition also no longer
depends on the current molecule's oracle scores, so train and inference binning
cannot disagree.

The 4 bins x 4 objectives token layout is kept, so only the meaning of a bin
changes. Edges are fitted on the train split only, matching
scripts/prepare_generator_data.py.

  python scripts/rebin_delta.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interfaces import OBJECTIVES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/gen_pairs.jsonl")
    ap.add_argument("--meta-in", default="data/gen_pairs_meta.json")
    ap.add_argument("--out", default="data/gen_pairs_delta.jsonl")
    ap.add_argument("--meta-out", default="data/gen_pairs_delta_meta.json")
    ap.add_argument("--n-bins", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.pairs).read_text().splitlines()]
    meta = json.loads(Path(args.meta_in).read_text())
    print(f"{len(rows)} edit pairs, {args.n_bins} change-quantile bins")

    # Fit edges on the train split only
    DY_tr = np.array([r["dy"] for r in rows if r["split"] == "train"])
    qs = np.linspace(0, 100, args.n_bins + 1)[1:-1]
    edges = {o: np.percentile(DY_tr[:, j], qs).tolist()
             for j, o in enumerate(OBJECTIVES)}

    def to_bins(dy):
        return [int(np.searchsorted(edges[o], dy[j], side="right"))
                for j, o in enumerate(OBJECTIVES)]

    # Which bin dy=0 falls into; this is the "leave the other objectives alone" bin
    neutral = {o: int(np.searchsorted(edges[o], 0.0, side="right"))
               for o in OBJECTIVES}

    print(f"\n{'objective':11}{'change-quantile edges':>34}{'dy=0 bin':>10}{'top usable':>12}")
    for o in OBJECTIVES:
        e = "[" + ", ".join(f"{v:+.4f}" for v in edges[o]) + "]"
        top_ok = neutral[o] < args.n_bins - 1
        print(f"{o:11}{e:>34}{'bin' + str(neutral[o]):>10}"
              f"{'yes' if top_ok else 'no <- still saturated!':>12}")
    assert all(neutral[o] < args.n_bins - 1 for o in OBJECTIVES), \
        "dy=0 lands in the top bin, so the condition still cannot ask for an " \
        "improvement; the binning needs to change"

    counts = np.zeros((len(OBJECTIVES), args.n_bins), int)
    with Path(args.out).open("w") as fh:
        for r in rows:
            b = to_bins(r["dy"])
            for j, v in enumerate(b):
                counts[j, v] += 1
            r["bins_abs"] = r["bins"]          # kept so both schemes can be compared
            r["bins"] = b
            fh.write(json.dumps(r) + "\n")

    print(f"\nbin shares (should be near {1/args.n_bins:.0%})")
    print(f"{'objective':11}" + "".join(f"{'bin'+str(v):>9}" for v in
                                        range(args.n_bins)))
    n = len(rows)
    for j, o in enumerate(OBJECTIVES):
        print(f"{o:11}" + "".join(f"{counts[j, v]/n:>9.1%}"
                                  for v in range(args.n_bins)))

    meta.update({
        "conditioning": "delta",          # src/gen_editors.py switches on this
        "delta_edges": edges,
        "neutral_bin": neutral,
        "n_bins": args.n_bins,
        "format": "(src molecule, per-objective change quantiles) -> dst molecule, same scaffold",
    })
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out} and {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
