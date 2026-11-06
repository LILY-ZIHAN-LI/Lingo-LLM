"""Go/no-go check: the four oracles run and throughput is sufficient.

A clean run of this script completes the oracle audit milestone.
Usage: python scripts/smoke_test_oracles.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracles import OBJECTIVES, OracleBank  # noqa: E402

# A few known molecules plus one deliberately malformed SMILES
PROBES = [
    "CC(=O)Oc1ccccc1C(=O)O",                      # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",               # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 # ibuprofen
    "c1ccccc1" * 12,                              # huge but valid
    "C1=CC=CC=C",                                 # invalid
]


def main() -> int:
    print("Loading oracles (the first run downloads the GSK3B/JNK3 RF models)...")
    t0 = time.perf_counter()
    bank = OracleBank()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    print(f"{'SMILES':<34} " + " ".join(f"{o:>7}" for o in OBJECTIVES))
    for smi in PROBES:
        v = bank.vector(smi)
        label = (smi[:31] + "...") if len(smi) > 34 else smi
        if v is None:
            print(f"{label:<34} {'INVALID':>7}")
        else:
            print(f"{label:<34} " + " ".join(f"{x:7.3f}" for x in v))

    # Throughput determines how large the experiment can be
    print("\nThroughput test (200 unique molecules)...")
    base = "CC(=O)Oc1ccccc1C(=O)O"
    mols = [base.replace("CC(=O)", "C" * (i % 5 + 1) + "C(=O)") for i in range(200)]
    t0 = time.perf_counter()
    for m in mols:
        bank.score(m)
    dt = time.perf_counter() - t0
    print(f"  {dt:.2f}s -> {len(mols) / dt:.1f} molecules/s/core")
    print(f"  stats: {bank.stats()}")

    budget = 6 * 100 * 3 * 20 * 4        # policies x seeds x repeats x steps x candidates
    print(f"\nmain experiment is at most {budget:,} scorings -> "
          f"about {budget / (len(mols) / dt) / 3600:.1f} h on one core, "
          f"{budget / (len(mols) / dt) / 3600 / 8:.1f} h on 8 cores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
