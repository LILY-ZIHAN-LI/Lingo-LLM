"""Main experiment: train, then evaluate paired on held-out seeds.

The three learned policies are trained differently on purpose, so the comparison
is interpretable:
- greedy : offline fit on random trajectories, immediate reward only, no exploration
- bandit : online UCB exploration, immediate reward only
- dqn    : online epsilon-greedy exploration, discounted future reward

greedy -> bandit measures the value of exploration.
bandit -> dqn measures the value of delayed credit assignment.

Usage:
  python scripts/run_experiment.py --quick
  python scripts/run_experiment.py --n-test 200 --repeats 10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, hv_at_eligible, hv_at_step, pick_candidate_budget,
    print_pair_table, summarize_pair,
)
from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402
from src.policies import (  # noqa: E402
    DQNPolicy, GapPolicy, GreedyPolicy, LinUCBBandit, make_baselines,
)


# --------------------------------------------------------------------------
def rollout(env, policy, smiles: str, episode_seed: int, record: bool = False):
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    curve, trans = [0.0], []
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        nxt, r, done, info = env.step(a)
        if record:
            trans.append({"obs": np.asarray(obs), "action": a, "reward": r,
                          "next_obs": np.asarray(nxt), "done": float(done)})
        obs = nxt
        curve.append(info["hv"] - hv0)
        if done:
            break
    return np.array(curve), trans, [lg.expert for lg in env.logs], env.logs


def collect(env, policy, seeds, repeats, tag=""):
    buf = []
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            _, tr, _, _ = rollout(env, policy, m, rep * 10000 + i, record=True)
            buf.extend(tr)
    print(f"  {tag}: {len(buf)} transitions")
    return buf


def train_bandit_online(env, bandit, seeds, repeats):
    """Run training trajectories under the bandit's own UCB policy, updating as it goes.

    This is the only difference from greedy: greedy passively fits random's data,
    while the bandit decides which (state, action) pairs to look at.
    """
    bandit.explore = True
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            obs = env.reset(m, rep * 10000 + i)
            for t in range(env.cfg.max_steps):
                a = bandit.act(obs, t)
                nxt, r, done, _ = env.step(a)
                bandit.update(obs, a, r)
                obs = nxt
                if done:
                    break
    bandit.explore = False
    return bandit


def train_dqn_online(env, dqn, seeds, repeats, updates_per_epoch=300,
                     eps_start=1.0, eps_end=0.1, eps_frac=0.6):
    """Decay epsilon linearly to 0.1 over the first 60% of training rather than
    the first 10%.

    Exploration is the main risk under a sparse HV reward, and slow decay is the
    cheapest mitigation.
    """
    buf, losses = [], []
    total = repeats * len(seeds)
    step = 0
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            frac = step / max(total * eps_frac, 1)
            dqn.epsilon = max(eps_end, eps_start + (eps_end - eps_start) * frac)
            _, tr, _, _ = rollout(env, dqn, m, rep * 10000 + i, record=True)
            buf.extend(tr)
            step += 1
        losses += dqn.train_on(buf, n_updates=updates_per_epoch)
        print(f"    epoch {rep+1}/{repeats}  eps={dqn.epsilon:.2f}  "
              f"buffer={len(buf)}  loss={np.mean(losses[-100:]):.4f}")
    dqn.epsilon = 0.0
    return dqn, losses


# --------------------------------------------------------------------------
def _worker_init(cfg_dict, blob, env_kw):
    global _ENV, _POLICIES
    # Parallelism comes from the Pool, so torch must be single-threaded: otherwise
    # 8 workers times torch's default 8 threads means 64 threads on 8 cores. The
    # RDKit pool never touches torch, but with editor_kind="gen" every worker runs
    # autoregressive sampling.
    import torch
    torch.set_num_threads(1)
    _ENV = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict), **env_kw)
    _POLICIES = pickle.loads(blob)


def _worker_eval(job):
    name, smiles, episode_seed = job
    curve, _, actions, logs = rollout(_ENV, _POLICIES[name], smiles, episode_seed)
    eligible = sum(lg.n_scaffold_ok for lg in logs)
    return {
        "policy": name, "smiles": smiles, "episode_seed": episode_seed,
        "curve": curve.tolist(),
        "final": float(curve[-1]),
        "auc": float(curve.sum()),
        "actions": actions,
        "eligible": eligible,
        "proposed": sum(lg.n_proposed for lg in logs),
        "entered": sum(1 for lg in logs if lg.entered_archive),
        # The key confound control: HV gained per eligible candidate
        "hv_per_eligible": float(curve[-1]) / max(eligible, 1),
        "eligible_curve": np.cumsum([0] + [lg.n_scaffold_ok for lg in logs]).tolist(),
        # Secondary metrics. StepLog has always recorded both, but only
        # n_scaffold_ok was propagated, so validity could not be recovered from an
        # existing raw.json.
        "n_valid": sum(lg.n_valid for lg in logs),
        "n_novel": sum(lg.n_novel for lg in logs),
        # Uniqueness, novelty and Morgan internal diversity need the molecules
        # themselves, which counts cannot recover. The archive is the Pareto front,
        # a few dozen molecules, so raw.json grows by a few MB and saves a rerun.
        "archive": [s for s, _ in _ENV.archive],
    }


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--train-repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; varying it quantifies retraining variance")
    ap.add_argument("--editor-kind", choices=["rdkit", "gen"], default="rdkit",
                    help="gen = the conditional generator main experiment")
    ap.add_argument("--gen-ckpt", default="results_gen/generator.pt")
    ap.add_argument("--pool-overlap", type=float, default=0.0,
                    help="overlap ratio of the rdkit pools: 0 = disjoint as designed, "
                         "1 = all four editors equivalent; sweeps action-space dimension")
    ap.add_argument("--qed-disjoint", action="store_true",
                    help="use _QED_SWAP instead of the sa _SHRINK pool for heavy molecules, raising the dimension")
    ap.add_argument("--escape-bits", action="store_true",
                    help="add 4 observation bits for editors that already failed on the current molecule")
    args = ap.parse_args()

    if args.quick:
        args.n_train, args.n_test = 40, 20
        args.repeats, args.train_repeats = 1, 2

    data = Path(args.data)
    train_seeds = data.joinpath("seeds_train.txt").read_text().split()[: args.n_train]
    test_seeds = data.joinpath("seeds_test.txt").read_text().split()[: args.n_test]
    cfg_dict = {"max_steps": args.max_steps, "lam": args.lam}
    env_kw = {"use_escape_bits": args.escape_bits,
              "editor_kind": args.editor_kind, "gen_ckpt": args.gen_ckpt,
              "pool_overlap": args.pool_overlap,
              "qed_disjoint": args.qed_disjoint}
    env = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict), **env_kw)

    print(f"train {len(train_seeds)} / test {len(test_seeds)} seeds, "
          f"{args.max_steps} steps, lam={args.lam}, seed={args.seed}, "
          f"escape_bits={args.escape_bits}, editors={args.editor_kind}, "
          f"overlap={args.pool_overlap}, qed_disjoint={args.qed_disjoint}\n")

    policies = make_baselines(seed=args.seed)

    # ---------- 1. random trajectories, used by greedy and gap ----------
    t0 = time.perf_counter()
    print("1. collecting random trajectories...")
    buf = collect(env, make_baselines(seed=7 + 13 * args.seed)["random"],
                  train_seeds, args.train_repeats, "random")
    policies["gap"] = GapPolicy().fit(buf)
    policies["greedy"] = GreedyPolicy().fit(buf)
    print(f"   {time.perf_counter() - t0:.0f}s\n")

    # ---------- 2. bandit: online UCB ----------
    t0 = time.perf_counter()
    print("2. training bandit online...")
    bandit = LinUCBBandit()
    bandit._init(env.obs_dim)
    policies["bandit"] = train_bandit_online(env, bandit, train_seeds,
                                             args.train_repeats)
    print(f"   {time.perf_counter() - t0:.0f}s\n")

    # ---------- 3. DQN: online epsilon-greedy ----------
    t0 = time.perf_counter()
    print("3. training DQN online...")
    dqn, losses = train_dqn_online(env,
                                   DQNPolicy(obs_dim=env.obs_dim, seed=args.seed),
                                   train_seeds, args.train_repeats)
    policies["dqn"] = dqn
    print(f"   {time.perf_counter() - t0:.0f}s  "
          f"loss {np.mean(losses[:100]):.4f} -> {np.mean(losses[-100:]):.4f}\n")

    # ---------- 4. paired evaluation ----------
    jobs = [(name, m, rep * 10000 + i)
            for name in policies
            for rep in range(args.repeats)
            for i, m in enumerate(test_seeds)]
    print(f"4. evaluating {len(jobs)} episodes with {args.workers} processes...")
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_worker_init,
              initargs=(cfg_dict, pickle.dumps(policies), env_kw)) as pool:
        rows = pool.map(_worker_eval, jobs, chunksize=4)
    dt = time.perf_counter() - t0
    print(f"   {dt/60:.1f} min\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(rows))
    (out / "policies.pkl").write_bytes(pickle.dumps(policies))

    def pick(name, key="final"):
        return {(r["smiles"], r["episode_seed"]): r[key]
                for r in rows if r["policy"] == name}

    # ---------- 5. summary ----------
    print(f"{'policy':13}{'finalHV':>9}{'AUC':>8}{'eligible':>10}{'HV/cand':>10}"
          f"  action distribution")
    for name in policies:
        sub = [r for r in rows if r["policy"] == name]
        f = np.mean([r["final"] for r in sub])
        a = np.mean([r["auc"] for r in sub])
        el = np.mean([r["eligible"] for r in sub])
        hpe = np.mean([r["hv_per_eligible"] for r in sub])
        counts = {e: 0 for e in EXPERTS}
        for r in sub:
            for act in r["actions"]:
                counts[act] += 1
        tot = max(sum(counts.values()), 1)
        dist = " ".join(f"{e}:{counts[e]/tot:.0%}" for e in EXPERTS)
        print(f"{name:13}{f:>9.4f}{a:>8.3f}{el:>9.1f}{hpe:>10.5f}  {dist}")

    print("\nThe 'eligible' column is a confound: editor yields differ by a factor"
          "\nof two, so a policy favouring high-yield experts automatically sees more"
          "\ncandidates. 'HV/cand' controls for it; when the two orderings disagree,"
          "\ntrust the latter.\n")

    print(f"paired comparison vs random (bootstrap 95% CI, n={len(pick('random'))})")
    base = pick("random")
    for name in policies:
        if name == "random":
            continue
        cur = pick(name)
        keys = sorted(set(base) & set(cur))
        d = np.array([cur[k] - base[k] for k in keys])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {name:13}{d.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}  "
              f"won {int((d > 0).sum())}/{len(d)}")

    print("\nkey comparisons:")
    for a_, b_, why in [("bandit", "greedy", "value of exploration"),
                        ("dqn", "bandit", "value of delayed credit assignment"),
                        ("dqn", "greedy", "both combined")]:
        pa, pb = pick(a_), pick(b_)
        keys = sorted(set(pa) & set(pb))
        d = np.array([pa[k] - pb[k] for k in keys])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {a_:7}-{b_:9}{d.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}  "
              f"won {int((d > 0).sum())}/{len(d)}   ({why})")

    # ---------- 6. intermediate budgets and the candidate budget ----------
    # The HV curve saturates around step 5, so looking only at the endpoint dilutes
    # policy differences. Repeat the same paired comparison at earlier budgets and
    # under the candidate-count budget.
    pairs = [(n, "random") for n in policies if n != "random"]
    pairs += [("bandit", "greedy"), ("dqn", "bandit")]

    mid_points = sorted({3, 5, args.max_steps // 2, args.max_steps})
    for x in mid_points:
        res = [summarize_pair(rows, a, b, lambda r: hv_at_step(r, x))
               for a, b in pairs]
        print_pair_table(res, f"paired comparison @ editor call {x}")

    # Candidate budget: pick a point most episodes can reach
    if "eligible_curve" in rows[0]:
        # Common support: use only episodes where every policy reached x
        # candidates. Dropping unreachable episodes per policy would introduce
        # selection bias, since the dropped trajectories are the low-yield ones and
        # yield is the variable being controlled for.
        x_cand = pick_candidate_budget(rows, 0.8)
        keys = common_support(rows, x_cand)
        cov = coverage_at(rows, x_cand)
        print(f"\ncandidate budget x = {x_cand}, common support {len(keys)} episodes "
              f"({cov['_common']:.0%})")
        res = [summarize_pair(rows, a, b, lambda r: hv_at_eligible(r, x_cand),
                              restrict=keys)
               for a, b in pairs]
        print_pair_table(
            res, f"paired comparison @ {x_cand} eligible candidates (yield controlled)")
        print("\n  Read alongside the editor-call tables above: if the two disagree,"
              "\n  the difference comes from yield rather than routing quality, and"
              "\n  that disagreement is itself a result.")

    (out / "summary.json").write_text(json.dumps({
        "n_train": len(train_seeds), "n_test": len(test_seeds),
        "repeats": args.repeats, "train_repeats": args.train_repeats,
        "max_steps": args.max_steps, "lam": args.lam,
        "seed": args.seed, "escape_bits": args.escape_bits,
        "editor_kind": args.editor_kind,
        "eval_minutes": dt / 60,
    }, indent=2))
    print(f"\nwrote {out}/raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())