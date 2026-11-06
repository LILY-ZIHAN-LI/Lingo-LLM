"""Does an editor's candidate pool run dry: how yield changes with consecutive
calls to the same editor.

Motivation: DQN collapses to 68% gsk3b, and gsk3b is a high-yield editor (about
100% eligible), so it should collect more candidates than random. Measured, DQN
averages only 45.5 eligible candidates against random's 53.9. Yield is therefore
not a fixed property of an editor but decays with consecutive calls: editors are
deterministic, so repeatedly calling the same one on the same molecule keeps
proposing the same edits.

If that holds, it gives a mechanism for why routing matters that has nothing to do
with objective trade-offs: rotating editors avoids proposal exhaustion.

Uses results/raw.json only; does not rerun the experiment.
  python scripts/analyze_editor_depletion.py --raw results/raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERTS = ("gsk3b", "jnk3", "qed", "sa")
ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def per_step_eligible(r: dict) -> np.ndarray:
    """eligible_curve is cumulative; differencing recovers the per-step count."""
    return np.diff(np.asarray(r["eligible_curve"], dtype=float))


def run_lengths(actions: list[str]) -> list[int]:
    """How many consecutive times the current expert has been called at step t
    (1 on the first call)."""
    out, run = [], 0
    prev = None
    for a in actions:
        run = run + 1 if a == prev else 1
        out.append(run)
        prev = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--out", default="results/editor_depletion.json")
    ap.add_argument("--max-run", type=int, default=6)
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    names = [n for n in ORDER if any(r["policy"] == n for r in rows)]

    # (expert, run length) -> eligible candidate count at that step
    by_run: dict = defaultdict(list)
    by_expert_run: dict = defaultdict(list)
    # Control for step position: yield may simply decay with t as molecules get
    # harder to edit
    by_step: dict = defaultdict(list)
    for r in rows:
        el = per_step_eligible(r)
        acts = r["actions"]
        runs = run_lengths(acts)
        for t, (a, run) in enumerate(zip(acts, runs)):
            if t >= len(el):
                break
            by_run[min(run, args.max_run)].append(el[t])
            by_expert_run[(a, min(run, args.max_run))].append(el[t])
            by_step[t].append(el[t])

    k = 4  # k_per_call
    print(f"{args.raw}: {len(rows)} episodes\n")
    print("eligible candidates per step vs consecutive calls to that expert (k=4 proposals/call)")
    print(f"  {'run':>6}{'eligible':>10}{'rate':>9}{'n':>9}")
    rows_run = {}
    for run in sorted(by_run):
        v = np.array(by_run[run])
        rows_run[run] = {"mean": float(v.mean()), "n": int(len(v))}
        print(f"  {run:>6}{v.mean():>10.2f}{v.mean()/k:>9.0%}{len(v):>9}")

    first = rows_run[1]["mean"]
    last = rows_run[max(rows_run)]["mean"]
    print(f"\n  first call {first:.2f} -> consecutive call {max(rows_run)} {last:.2f}"
          f" ({(last/max(first,1e-9)-1):+.0%})")

    print("\nControl: eligible candidates vs step position (rules out 'later steps")
    print("are simply harder')")
    print(f"  {'step':>6}{'eligible':>10}{'n':>9}")
    step_means = {}
    for t in sorted(by_step)[:args.max_run + 2]:
        v = np.array(by_step[t])
        step_means[t] = float(v.mean())
        print(f"  {t:>6}{v.mean():>10.2f}{len(v):>9}")

    # ---- Control for the "stuck" confound ----
    # The transition rule replaces current with the largest-gain candidate whenever
    # one exists, so "the previous step had an eligible candidate" means "the
    # molecule actually moved". If yield still falls with consecutive calls after
    # the molecule moved, the pool genuinely runs dry; if it recovers, the decline
    # is only the selection effect of a deterministic policy spinning in a dead
    # state.
    moved: dict = defaultdict(list)
    stuck: dict = defaultdict(list)
    for r in rows:
        el = per_step_eligible(r)
        runs = run_lengths(r["actions"])
        for t in range(1, min(len(el), len(runs))):
            bucket = moved if el[t - 1] > 0 else stuck
            bucket[min(runs[t], args.max_run)].append(el[t])

    print("\nControlling for whether the molecule moved (eligible candidate last step)")
    print(f"  {'run':>6}{'moved':>10}{'n':>8}{'stuck':>10}{'n':>8}")
    ctrl = {}
    for run in sorted(set(moved) | set(stuck)):
        a = np.array(moved.get(run, []))
        b = np.array(stuck.get(run, []))
        ctrl[run] = {"moved": float(a.mean()) if len(a) else None,
                     "n_moved": int(len(a)),
                     "stuck": float(b.mean()) if len(b) else None,
                     "n_stuck": int(len(b))}
        am = f"{a.mean():.2f}" if len(a) else "-"
        bm = f"{b.mean():.2f}" if len(b) else "-"
        print(f"  {run:>6}{am:>10}{len(a):>8}{bm:>10}{len(b):>8}")
    mv = [ctrl[r]["moved"] for r in sorted(ctrl)
          if ctrl[r]["moved"] is not None and ctrl[r]["n_moved"] >= 30]
    if len(mv) >= 2:
        print(f"\n  steps where the molecule moved: {mv[0]:.2f} -> {mv[-1]:.2f}"
              f" ({(mv[-1]/max(mv[0],1e-9)-1):+.0%})")
        print("  a clear decline in this column means genuine proposal exhaustion;"
              "\n  a flat column means the decline comes from deterministic policies"
              "\n  spinning in dead states.")

    print("\nPer expert (rows = expert, columns = consecutive call index)")
    header = "".join(f"{r:>9}" for r in sorted(by_run))
    print(f"  {'expert':<8}{header}")
    per_expert = {}
    for e in EXPERTS:
        cells, vals = [], {}
        for run in sorted(by_run):
            v = by_expert_run.get((e, run), [])
            if len(v) < 20:
                cells.append(f"{'-':>9}")
                continue
            m = float(np.mean(v))
            vals[run] = {"mean": m, "n": len(v)}
            cells.append(f"{m:>9.2f}")
        per_expert[e] = vals
        print(f"  {e:<8}" + "".join(cells))

    # Each policy's repeat rate vs the total candidates it collected
    print("\nPer-policy repetition and total yield")
    print(f"  {'policy':<13}{'mean run':>10}{'repeat %':>11}{'eligible':>10}")
    pol = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        runs = [run_lengths(r["actions"]) for r in sub]
        mean_run = float(np.mean([np.mean(x) for x in runs]))
        rep = float(np.mean([np.mean([v > 1 for v in x]) for x in runs]))
        el = float(np.mean([r["eligible_curve"][-1] for r in sub]))
        pol[n] = {"mean_run": mean_run, "frac_repeat": rep, "eligible": el}
        print(f"  {n:<13}{mean_run:>10.2f}{rep:>11.0%}{el:>10.1f}")

    # ---- Dead-step budget accounting ----
    # The control above shows the absorbing state: a step with 0 eligible
    # candidates leaves the molecule in place, a deterministic policy re-picks the
    # same editor, and gets 0 again. Price it directly in budget terms.
    print("\nDead-step accounting (a dead step has 0 eligible candidates: spends a")
    print("call and yields nothing)")
    print(f"  {'policy':<13}{'dead %':>9}{'repeat prev':>14}"
          f"{'first@':>10}{'left':>9}")
    dead = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        fr, rep_fr, first, tail = [], [], [], []
        for r in sub:
            el = per_step_eligible(r)
            acts = r["actions"][: len(el)]
            d = el == 0
            fr.append(d.mean())
            idx = [t for t in range(1, len(el)) if d[t]]
            if idx:
                rep_fr.append(np.mean([acts[t] == acts[t - 1] for t in idx]))
            if d.any():
                f = int(np.argmax(d))
                first.append(f)
                tail.append(len(el) - f)
        dead[n] = {"frac_dead": float(np.mean(fr)),
                   "frac_dead_repeating": float(np.mean(rep_fr)) if rep_fr else None,
                   "first_dead_step": float(np.mean(first)) if first else None,
                   "steps_after_first_dead": float(np.mean(tail)) if tail else None}
        rp = f"{np.mean(rep_fr):.0%}" if rep_fr else "-"
        fs = f"{np.mean(first):.1f}" if first else "-"
        tl = f"{np.mean(tail):.1f}" if tail else "-"
        print(f"  {n:<13}{np.mean(fr):>9.0%}{rp:>14}{fs:>10}{tl:>9}")
    print("  A high 'repeat prev' means dead steps are spent spinning on the same"
          "\n  editor rather than scattered one-off failures, which is the signature"
          "\n  of the absorbing state.")

    x = np.array([pol[n]["frac_repeat"] for n in names])
    y = np.array([pol[n]["eligible"] for n in names])
    if len(names) > 2:
        c = float(np.corrcoef(x, y)[0, 1])
        print(f"\n  correlation of repeat share with final candidate count: r = {c:+.2f}"
              f" ({len(names)} policies)")

    Path(args.out).write_text(json.dumps(
        {"by_run": rows_run, "by_step": step_means, "moved_vs_stuck": ctrl,
         "by_expert_run": {e: {str(r): v for r, v in d.items()}
                           for e, d in per_expert.items()},
         "by_policy": pol, "dead_steps": dead}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
