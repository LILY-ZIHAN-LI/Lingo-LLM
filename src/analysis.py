"""Shared helpers for paired comparison, used by run_experiment.py and the
analysis scripts.

There are two budget definitions:

  1. Editor calls: x-axis is the step count. This is the matched proposal budget,
     but it is confounded: eligible yield differs by a factor of two between
     editors (gsk3b/jnk3 are near 100%, qed/sa around 50%), so a policy that
     favours high-yield editors sees more candidate molecules at the same step
     count.

  2. Eligible candidates: x-axis is the cumulative count of eligible candidates.
     This is the budget that actually aligns "how many molecules were searched",
     and it is what drives oracle calls.

When the two definitions disagree, both must be reported; the disagreement is
itself a result.
"""
from __future__ import annotations

import numpy as np


def episode_key(r: dict):
    """Pairing unit: the same seed molecule and the same episode random seed."""
    return (r["smiles"], r["episode_seed"])


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------
# Readouts under the two budget definitions
# --------------------------------------------------------------------------
def hv_at_step(r: dict, x: int) -> float | None:
    """HV gain after the x-th editor call; clamps to the end of the trajectory."""
    c = r["curve"]
    if x < 0:
        return None
    return float(c[min(x, len(c) - 1)])


def hv_at_eligible(r: dict, x: float) -> float | None:
    """HV gain once x eligible candidates have accumulated, linearly interpolated
    between steps.

    None means this trajectory never reached x candidates. Extrapolating there
    would disguise a low yield as a low HV and erase the effect being measured.
    """
    if "eligible_curve" not in r:
        return None
    xs = np.asarray(r["eligible_curve"], dtype=float)
    ys = np.asarray(r["curve"], dtype=float)
    L = min(len(xs), len(ys))
    xs, ys = xs[:L], ys[:L]
    if xs[-1] < x:
        return None
    return float(np.interp(x, xs, ys))


# --------------------------------------------------------------------------
# Paired differences
# --------------------------------------------------------------------------
def common_support(rows: list[dict], x: float) -> set:
    """Episodes in which every policy reached x eligible candidates.

    Dropping unreachable episodes per policy would introduce selection bias: the
    dropped trajectories are exactly the low-yield ones, and yield is the variable
    being controlled for. Fixing one common subset keeps every pair on the same
    episodes.
    """
    by_key: dict = {}
    policies = {r["policy"] for r in rows}
    for r in rows:
        k = episode_key(r)
        by_key.setdefault(k, {})[r["policy"]] = r["eligible_curve"][-1]
    return {k for k, d in by_key.items()
            if len(d) == len(policies) and min(d.values()) >= x}


def coverage_at(rows: list[dict], x: float) -> dict:
    """Fraction of episodes per policy that reach x candidates, plus the size of
    the common support."""
    out = {}
    for n in {r["policy"] for r in rows}:
        sub = [r for r in rows if r["policy"] == n]
        out[n] = float(np.mean([r["eligible_curve"][-1] >= x for r in sub]))
    n_ep = len({episode_key(r) for r in rows})
    out["_common"] = len(common_support(rows, x)) / max(n_ep, 1)
    return out


def pick_candidate_budget(rows: list[dict], min_common: float = 0.8,
                          step: int = 5) -> int:
    """Largest candidate budget whose common-support coverage is still >= min_common.

    A larger budget discriminates better but fewer episodes reach it; this picks a
    point between the two.
    """
    for thresh in (min_common, 0.6, 0.4):
        best = 0
        for x in range(step, 200, step):
            if coverage_at(rows, x)["_common"] >= thresh:
                best = x
            else:
                break
        if best:
            return best
    # Even the smallest budget is under-covered (tiny --quick runs): fall back to
    # the minimum and let the caller report coverage
    return step


def paired_diff(rows: list[dict], a: str, b: str, value, restrict: set | None = None):
    """Policy a minus policy b, paired on (molecule, episode seed).

    value: a function r -> float|None. Episodes where either side is None are
    dropped as a pair, and the drop count is returned so tables can report how
    many episodes fell short of this budget.
    restrict: use only these episodes (the common support), so every pair is
    computed on the same subset.
    """
    va = {episode_key(r): value(r) for r in rows if r["policy"] == a}
    vb = {episode_key(r): value(r) for r in rows if r["policy"] == b}
    keys = sorted(set(va) & set(vb))
    if restrict is not None:
        keys = [k for k in keys if k in restrict]
    kept = [k for k in keys if va[k] is not None and vb[k] is not None]
    d = np.array([va[k] - vb[k] for k in kept], dtype=float)
    return d, len(keys) - len(kept)


def summarize_pair(rows, a, b, value, label="", restrict=None):
    d, dropped = paired_diff(rows, a, b, value, restrict=restrict)
    if len(d) == 0:
        return {"a": a, "b": b, "label": label, "n": 0, "dropped": dropped}
    lo, hi = bootstrap_ci(d)
    return {
        "a": a, "b": b, "label": label,
        "n": int(len(d)), "dropped": int(dropped),
        "mean": float(d.mean()), "lo": lo, "hi": hi,
        "wins": int((d > 0).sum()),
        "sig": bool(lo > 0 or hi < 0),
    }


def print_pair_table(res: list[dict], title: str) -> None:
    print(f"\n{title}")
    for r in res:
        if r["n"] == 0:
            print(f"  {r['a']:7}-{r['b']:11} (no paired episodes)")
            continue
        sig = "*" if r["sig"] else " "
        drop = f"  dropped {r['dropped']}" if r["dropped"] else ""
        print(f"  {r['a']:7}-{r['b']:11}{r['mean']:+.5f}  "
              f"[{r['lo']:+.5f}, {r['hi']:+.5f}]{sig}  "
              f"won {r['wins']}/{r['n']}{drop}  {r['label']}")


def participation_ratio(M: np.ndarray) -> float:
    """Participation ratio of the singular values, i.e. the effective dimension.
    M is the [editor, objective] matrix of mean gains."""
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv.sum() ** 2 / max((sv ** 2).sum(), 1e-12))


def objective_scales(pairs_path: str) -> np.ndarray | None:
    """Per-objective standard deviation of the reachable change, used to normalize
    the input of participation_ratio.

    Without it the objective with the largest dy scale dominates the singular
    values: measured on within-scaffold edit pairs, dy itself scores 1.51/4 raw
    and 3.45/4 after per-dimension normalization (qed's dy std is 0.137, 4.2x
    jnk3's). Four mutually orthogonal actions would then measure as ~1.5
    dimensions.

    The normalizer comes from the edit-pair data, so it is a property of the
    environment rather than of the editor under test, and the RDKit pool and the
    generator variants are directly comparable. 3.45 is also the practical upper
    bound of this statistic and a better reference than 4.0.
    """
    import json
    from pathlib import Path
    p = Path(pairs_path)
    if not p.exists():
        return None
    dy = np.array([json.loads(l)["dy"] for l in p.read_text().splitlines()])
    s = dy.std(axis=0)
    return s if np.all(s > 0) else None
