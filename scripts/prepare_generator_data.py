"""Generator training corpus: (scaffold, property targets) -> SMILES
reconstruction data cut from ChEMBL.

Trains a compact SMILES transformer to recover molecular SMILES from scaffold
SMILES and discretized property targets using token cross-entropy.

Leakage control:
  - Split by Bemis-Murcko scaffold; any molecule sharing a scaffold with a
    val/test seed is dropped. The router's held-out seeds must not appear in the
    generator's training set in any form.
  - Scaffolds of the training seeds are kept, since the router trains on them too.
  - Molecules with an empty scaffold are dropped, matching seeds_meta.json.

Property binning: n_bins per objective, with edges fitted on the training split
only and recorded in meta. At inference one objective's bin is pushed up a notch
and the rest are held fixed, which gives the objective-conditioned editors.

  python scripts/prepare_generator_data.py --n-target 100000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RDLogger.DisableLog("rdApp.*")

from src.chem import canonical, murcko_scaffold  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

# Very long SMILES blow up the batch and are mostly peptides or natural products,
# which are not lead optimization targets
MAX_LEN = 110
MIN_LEN = 12


def held_out_scaffolds(data: Path) -> set[str]:
    """Scaffolds of the val and test seeds; none of these may enter the generator
    training set."""
    out: set[str] = set()
    for name in ("seeds_val.txt", "seeds_test.txt"):
        for smi in data.joinpath(name).read_text().split():
            sc = murcko_scaffold(smi)
            if sc:
                out.add(sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl", default="data/chembl_v29.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/gen_corpus.jsonl")
    ap.add_argument("--meta", default="data/gen_meta.json")
    ap.add_argument("--n-target", type=int, default=100000,
                    help="target corpus size; the plan calls for 50k-100k")
    ap.add_argument("--n-scan", type=int, default=600000,
                    help="maximum number of ChEMBL rows to scan")
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    data = Path(args.data)
    banned = held_out_scaffolds(data)
    print(f"held-out scaffolds (val+test seeds): {len(banned)}, will be dropped")

    bank = OracleBank()
    rows: list[dict] = []
    seen: set[str] = set()
    n_scanned = n_badlen = n_leak = n_noscaf = 0

    with open(args.chembl) as fh:
        rd = csv.DictReader(fh)
        for rec in rd:
            n_scanned += 1
            if n_scanned > args.n_scan or len(rows) >= args.n_target:
                break
            smi = rec.get("smiles") or ""
            if not (MIN_LEN <= len(smi) <= MAX_LEN):
                n_badlen += 1
                continue
            cs = canonical(smi)
            if cs is None or cs in seen or len(cs) > MAX_LEN:
                continue
            mol = Chem.MolFromSmiles(cs)
            if mol is None:
                continue
            sc = murcko_scaffold(cs)
            if not sc:
                n_noscaf += 1
                continue
            if sc in banned:          # leakage control: never keep a held-out scaffold
                n_leak += 1
                continue
            seen.add(cs)
            rows.append({"smiles": cs, "scaffold": sc,
                         "y": [float(v) for v in bank.vector(cs)]})
            if len(rows) % 10000 == 0:
                print(f"  collected {len(rows)} / scanned {n_scanned}")

    print(f"\nscanned {n_scanned} rows -> kept {len(rows)}")
    print(f"  bad length {n_badlen} / no scaffold {n_noscaf} / "
          f"held-out scaffold {n_leak}")

    # --- Property binning: edges fitted on the training split only -----------
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val_frac)
    val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]

    Y = np.array([rows[i]["y"] for i in tr_idx])
    qs = np.linspace(0, 100, args.n_bins + 1)[1:-1]
    edges = {o: np.percentile(Y[:, j], qs).tolist()
             for j, o in enumerate(OBJECTIVES)}
    print("\nproperty bin edges (fitted on the training split only):")
    for o in OBJECTIVES:
        print(f"  {o:6} {[round(v, 4) for v in edges[o]]}")

    def to_bins(y):
        return [int(np.searchsorted(edges[o], y[j], side="right"))
                for j, o in enumerate(OBJECTIVES)]

    out = Path(args.out)
    with out.open("w") as fh:
        for i, r in enumerate(rows):
            r["bins"] = to_bins(r["y"])
            r["split"] = "val" if i in val_idx else "train"
            fh.write(json.dumps(r) + "\n")

    meta = {
        "n_total": len(rows), "n_val": n_val, "n_bins": args.n_bins,
        "bin_edges": edges, "objectives": list(OBJECTIVES),
        "max_len": MAX_LEN, "n_scanned": n_scanned,
        "n_dropped_heldout_scaffold": n_leak,
        "banned_scaffolds": len(banned),
        "leakage_rule": "drop every molecule sharing a scaffold with a val/test seed",
        "oracle_calls": bank.n_calls,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out} ({len(rows)} rows, val {n_val}) and {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
