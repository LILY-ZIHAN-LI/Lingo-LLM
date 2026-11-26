"""Build scaffold-split seed molecules: 200 train / 50 val / 100 test.

Two selection principles:
1. There must be an activity handle. Random ChEMBL molecules score near 0 on the
   GSK3B/JNK3 predictors, and substituent edits starting from zero activity do not
   create activity, so HV ends up driven by QED and SA alone and the four
   objectives degenerate into two. Molecules with signal are therefore preferred.
2. There must be headroom. ChEMBL molecules already sit in the QED optimum, where
   any edit can only lower it, so a QED cap keeps the router with work to do.

Usage: python scripts/make_seeds.py --out data/ --scan 40000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from src.chem import canonical, murcko_scaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_CANDIDATE_DATASETS = ["ChEMBL_V29", "ChEMBL", "ZINC", "MOSES"]

_BANK = None


def _init_worker():
    global _BANK
    from src.oracles import OracleBank
    _BANK = OracleBank()


def _score_one(smi: str):
    v = _BANK.vector(smi)
    return (smi, v)


def load_pool(n_raw: int) -> tuple[list[str], str]:
    from tdc.generation import MolGen

    last_err = None
    for name in _CANDIDATE_DATASETS:
        try:
            print(f"trying to download {name} ...")
            df = MolGen(name=name).get_data()
            col = "smiles" if "smiles" in df.columns else df.columns[0]
            return df[col].astype(str).tolist()[:n_raw], name
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  {name} unavailable: {type(e).__name__}")
    raise RuntimeError(f"no usable dataset; last error: {last_err}")


def lead_like(smi: str) -> bool:
    """Cheap RDKit prefilter, to shrink the scan before calling the oracles."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    if not (250 <= Descriptors.MolWt(mol) <= 480):
        return False
    if mol.GetNumHeavyAtoms() > 42:
        return False
    # Needs an aromatic CH, otherwise the growth editors have nothing to act on
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[cH]")):
        return False
    # Needs a removable single-atom ring substituent, otherwise the shrink editors
    # produce no candidates
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R][!R;D1]")):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scan", type=int, default=40000, help="molecules sent to the oracles")
    ap.add_argument("--raw", type=int, default=400000, help="raw rows read from the dataset")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--max-qed", type=float, default=0.70,
                    help="QED cap; already drug-like molecules have no headroom")
    ap.add_argument("--min-act", type=float, default=0.15,
                    help="activity floor; below this the oracle is flat and edits have no gradient to follow")
    ap.add_argument("--max-act", type=float, default=0.55,
                    help="activity ceiling; above this the oracle is near saturation and any edit only hurts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="exclude scaffolds from these existing seed files")
    ap.add_argument("--only", default=None, choices=["train", "val", "test"],
                    help="write only the selected split")
    ap.add_argument("--suffix", default="",
                    help="append this suffix to the split and metadata filenames")
    args = ap.parse_args()

    if args.only:
        need = {"train": args.n_train, "val": args.n_val,
                "test": args.n_test}[args.only]
    else:
        need = args.n_train + args.n_val + args.n_test

    banned: set[str] = set()
    for f in args.exclude:
        p = Path(f)
        if not p.exists():
            raise SystemExit(f"--exclude file does not exist: {p}")
        n0 = len(banned)
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if s:
                sc = murcko_scaffold(s)
                if sc:
                    banned.add(sc)
        print(f"excluded {p}: +{len(banned) - n0} scaffolds")
    if banned:
        print(f"excluded {len(banned)} existing scaffolds\n")

    raw, source = load_pool(args.raw)
    print(f"source: {source}, {len(raw)} raw rows")

    # --- cheap prefilter ---
    pre: list[str] = []
    seen: set[str] = set()
    for smi in raw:
        c = canonical(smi)
        if c is None or c in seen or not lead_like(c):
            continue
        seen.add(c)
        pre.append(c)
        if len(pre) >= args.scan:
            break
    print(f"{len(pre)} molecules after prefilter, scoring with {args.workers} processes...")

    # --- parallel scoring ---
    with Pool(args.workers, initializer=_init_worker) as pool:
        scored = pool.map(_score_one, pre, chunksize=200)

    # vector order: gsk3b, jnk3, qed, sa
    rows = [(s, v) for s, v in scored if v is not None]
    print(f"scored {len(rows)} molecules")

    act = [max(v[0], v[1]) for _, v in rows]
    print(f"  activity signal: median max(gsk3b,jnk3) {sorted(act)[len(act)//2]:.3f}, "
          f"{sum(a > 0.1 for a in act)} above 0.1, "
          f"{sum(a > 0.3 for a in act)} above 0.3")

    # --- Selection: stay inside the "signal but not saturated" activity band
    # rather than taking the most active molecules. The most active ones make every
    # editor's activity delta negative, i.e. no headroom, which is exactly the
    # mirror image of the already-optimal QED problem.
    cands = [(s, v) for s, v in rows
             if v[2] <= args.max_qed
             and args.min_act <= max(v[0], v[1]) <= args.max_act]
    random.Random(args.seed).shuffle(cands)
    print(f"  {len(cands)} molecules with QED<={args.max_qed} and activity in "
          f"[{args.min_act}, {args.max_act}]")

    picked: list[tuple[str, list[float]]] = []
    used_scaffolds: set[str] = set(banned)
    for s, v in cands:
        scaf = murcko_scaffold(s)
        if not scaf or scaf in used_scaffolds:
            continue
        used_scaffolds.add(scaf)
        picked.append((s, v))
        if len(picked) >= need:
            break

    if len(picked) < need:
        print(f"warning: only picked {len(picked)}/{need}; raise --scan or relax --max-qed")

    # --- split with no scaffold overlap ---
    random.Random(args.seed).shuffle(picked)
    splits: dict[str, list[str]] = {}
    if args.only:
        splits[args.only] = [s for s, _ in picked[:need]]
    else:
        i = 0
        for name, n in [("train", args.n_train), ("val", args.n_val),
                        ("test", args.n_test)]:
            splits[name] = [s for s, _ in picked[i:i + n]]
            i += n
        print(f"  {name}: {len(splits[name])}")

    sel_act = [max(v[0], v[1]) for _, v in picked]
    sel_qed = [v[2] for _, v in picked]
    print(f"\nselected seeds: median activity {sorted(sel_act)[len(sel_act)//2]:.3f}, "
          f"median QED {sorted(sel_qed)[len(sel_qed)//2]:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, mols in splits.items():
        (out / f"seeds_{name}{args.suffix}.txt").write_text("\n".join(mols) + "\n")
    (out / f"seeds_meta{args.suffix}.json").write_text(json.dumps({
        "source": source,
        "split_seed": args.seed,
        "excluded_files": list(args.exclude),
        "n_excluded_scaffold": len(banned),
        "only_split": args.only,
        "n_scanned": len(pre),
        "n_scored": len(rows),
        "max_qed": args.max_qed,
        "selection": "QED<=max_qed, activity band on max(gsk3b,jnk3), one per scaffold",
        "sizes": {k: len(v) for k, v in splits.items()},
        "scaffold_overlap": 0,
        "empty_scaffold_rule": "dropped",
        "median_activity": sorted(sel_act)[len(sel_act) // 2] if sel_act else None,
        "median_qed": sorted(sel_qed)[len(sel_qed) // 2] if sel_qed else None,
    }, indent=2))
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
