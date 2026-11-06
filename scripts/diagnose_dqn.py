"""DQN failure diagnostic: separate "failed to learn" from "learned, but
sequentiality does not help".

DQN collapses to a near-constant policy at evaluation (68% gsk3b). Before claiming
that sequential learning does not help, an implementation problem has to be ruled
out. The two explanations leave very different Q-value fingerprints:

  A. The network learned no distinction (implementation or optimization problem)
     -> Q is nearly equal across actions in a state (relative range < 1%) and
        barely varies with state. The argmax is set by random initialization bias.

  B. The network learned, and the optimal policy really is near-constant
     (a scientific conclusion)
     -> Q differs clearly across actions within a state and varies with state;
        the argmax simply happens to point at the same action most of the time.

GreedyPolicy is the control: same features, same reward, same training data, but a
linear model on immediate gain only. If greedy's relative range is far larger than
DQN's, DQN has an optimization problem rather than an absent signal.

Usage:
  python scripts/diagnose_dqn.py
  python scripts/diagnose_dqn.py --n-seeds 30 --max-steps 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

EPS = 1e-12


# --------------------------------------------------------------------------
def q_values(dqn, obs: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        x = torch.tensor(np.asarray(obs, dtype=np.float32))
        return dqn.q(x).numpy().astype(float)


def greedy_values(greedy, obs: np.ndarray) -> np.ndarray:
    """GreedyPolicy's predicted immediate reward, as a control for whether the
    signal exists at all."""
    x = np.append(np.asarray(obs, dtype=float), 1.0)
    return np.asarray(greedy.W @ x, dtype=float)


def collect_states(env, policy, seeds, episode_seed_base=0):
    """Collect the states visited along the policy's own trajectories.

    Use the policy's own state distribution rather than synthetic states: a
    collapse only matters on the states DQN actually reaches. random's state
    distribution is collected as a control, ruling out the explanation that DQN
    merely walked into a narrow region where all actions are equivalent anyway.
    """
    states = []
    for i, smi in enumerate(seeds):
        obs = env.reset(smi, episode_seed_base + i)
        if hasattr(policy, "reset"):
            policy.reset(episode_seed_base + i)
        for t in range(env.cfg.max_steps):
            states.append(np.asarray(obs, dtype=float))
            a = policy.act(obs, t)
            obs, _, done, _ = env.step(a)
            if done:
                break
    return np.array(states)


# --------------------------------------------------------------------------
def spread_stats(V: np.ndarray) -> dict:
    """V: (n_states, n_actions). Quantifies how much actions differ within a state.

    The relative range is divided by the mean |Q| rather than some max-min scale,
    because the question is precisely whether the difference is negligible
    relative to the magnitude of Q.
    """
    rng = V.max(axis=1) - V.min(axis=1)          # per-state range across actions
    mag = np.abs(V).mean(axis=1)                 # per-state magnitude of Q
    rel = rng / np.maximum(mag, EPS)
    argmax = V.argmax(axis=1)
    counts = np.bincount(argmax, minlength=len(EXPERTS))
    return {
        "n_states": int(V.shape[0]),
        "q_mean": float(V.mean()),
        "q_abs_mean": float(np.abs(V).mean()),
        # Variation of each action's Q across states; near 0 means a constant function
        "per_action_std_across_states": V.std(axis=0).round(6).tolist(),
        "per_action_mean": V.mean(axis=0).round(6).tolist(),
        "range_mean": float(rng.mean()),
        "range_median": float(np.median(rng)),
        "rel_range_mean": float(rel.mean()),
        "rel_range_median": float(np.median(rel)),
        "frac_states_rel_below_1pct": float((rel < 0.01).mean()),
        "argmax_counts": counts.tolist(),
        "argmax_frac": (counts / max(V.shape[0], 1)).round(4).tolist(),
        "argmax_entropy_bits": float(
            -sum(p * np.log2(p) for p in counts / max(V.shape[0], 1) if p > 0)
        ),
    }


def advantage_stats(V: np.ndarray) -> dict:
    """Decompose Q(s,a) into V(s) + A(s,a) to see what the network learned.

    The first diagnostic round found Q's cross-state std (~4.4) far larger than the
    within-state range across actions (~0.5), meaning nearly all capacity goes to
    fitting how much HV a state can still gain (V) rather than which editor to call
    (A). If A also barely varies with state, it amounts to a fixed action
    preference table with no state dependence at all, which explains the constant
    collapse at evaluation.
    """
    Vs = V.mean(axis=1, keepdims=True)           # V(s): state value
    A = V - Vs                                   # A(s,a): advantage
    total = V.var()
    return {
        "var_total": float(total),
        # How much of Q's variance comes from differences between states versus
        # differences between actions within a state
        "frac_var_from_state": float(
            np.repeat(Vs, V.shape[1], axis=1).var() / max(total, EPS)),
        "frac_var_from_action": float(A.var() / max(total, EPS)),
        "adv_mean_per_action": A.mean(axis=0).round(6).tolist(),
        # Advantage's cross-state std against its mean absolute value: far below 1
        # means the advantage is essentially a constant bias with no state dependence
        "adv_std_per_action": A.std(axis=0).round(6).tolist(),
        "adv_std_over_abs_mean": float(
            A.std(axis=0).mean() / max(np.abs(A.mean(axis=0)).mean(), EPS)),
    }


def print_advantage(title: str, ad: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  Q variance split: between states {ad['frac_var_from_state']:.2%}  /  "
          f"between actions {ad['frac_var_from_action']:.2%}")
    print("  advantage mean             " + "  ".join(
        f"{e}:{v:+.4f}" for e, v in zip(EXPERTS, ad["adv_mean_per_action"])))
    print("  advantage std over states  " + "  ".join(
        f"{e}:{v:.4f}" for e, v in zip(EXPERTS, ad["adv_std_per_action"])))
    print(f"  std / |mean| ratio         {ad['adv_std_over_abs_mean']:.2f}"
          "   (<1 = near-constant bias; >1 = state dependent)")


def print_block(title: str, st: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  states                     {st['n_states']}")
    print(f"  mean |Q|                   {st['q_abs_mean']:.4f}")
    print(f"  action range (mean)        {st['range_mean']:.6f}")
    print(f"  relative range (mean/med)  {st['rel_range_mean']:.4%} / "
          f"{st['rel_range_median']:.4%}")
    print(f"  states with rel range <1%  {st['frac_states_rel_below_1pct']:.1%}")
    print("  per-action mean            " + "  ".join(
        f"{e}:{v:+.4f}" for e, v in zip(EXPERTS, st["per_action_mean"])))
    print("  per-action std over states " + "  ".join(
        f"{e}:{v:.5f}" for e, v in zip(EXPERTS, st["per_action_std_across_states"])))
    print("  argmax distribution        " + "  ".join(
        f"{e}:{p:.0%}" for e, p in zip(EXPERTS, st["argmax_frac"]))
          + f"   (entropy {st['argmax_entropy_bits']:.2f}/2.00 bits)")


def sample_table(V: np.ndarray, n: int = 8) -> None:
    print(f"\n  raw Q values for the first {n} states:")
    print("    " + "".join(f"{e:>12}" for e in EXPERTS) + "     range      rel")
    for row in V[:n]:
        rng = row.max() - row.min()
        rel = rng / max(np.abs(row).mean(), EPS)
        print("    " + "".join(f"{v:>12.5f}" for v in row)
              + f"  {rng:>8.5f}  {rel:>6.2%}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/dqn_diagnosis.json")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=20)
    args = ap.parse_args()

    policies = pickle.loads(Path(args.policies).read_bytes())
    dqn = policies["dqn"]
    greedy = policies.get("greedy")

    test_seeds = Path(args.data).joinpath("seeds_test.txt").read_text().split()
    test_seeds = test_seeds[: args.n_seeds]

    cfg = EpisodeConfig(max_steps=args.max_steps)
    env = MolEnv(OracleBank(), cfg)

    print(f"DQN diagnostic: {len(test_seeds)} test seeds, {args.max_steps} steps")
    print(f"reward_scale={dqn.reward_scale}  gamma={dqn.gamma}  "
          f"epsilon={dqn.epsilon}")
    # Rewards are multiplied by reward_scale, so Q should be on the order of the
    # per-step HV gain times scale, summed under the gamma discount
    print(f"  -> with a per-step HV gain near 1e-3, Q should be around "
          f"{1e-3 * dqn.reward_scale / (1 - dqn.gamma):.1f}")

    report = {"config": {"n_seeds": len(test_seeds), "max_steps": args.max_steps,
                         "reward_scale": dqn.reward_scale, "gamma": dqn.gamma}}

    # DQN's own state distribution
    print("\ncollecting states along DQN's own trajectories...")
    S_dqn = collect_states(env, dqn, test_seeds, 0)
    V = np.array([q_values(dqn, s) for s in S_dqn])
    st = spread_stats(V)
    print_block("Q values at states DQN visits", st)
    sample_table(V)
    report["dqn_on_own_states"] = st

    # random's state distribution, ruling out a too-narrow state region
    print("\ncollecting states along random trajectories as a control...")
    S_rnd = collect_states(env, policies["random"], test_seeds, 500000)
    V_rnd = np.array([q_values(dqn, s) for s in S_rnd])
    st_rnd = spread_stats(V_rnd)
    print_block("Q values at states random visits", st_rnd)
    report["dqn_on_random_states"] = st_rnd

    # Q = V(s) + A(s,a), separating a learned state value from learned routing
    ad = advantage_stats(V)
    print_advantage("Q decomposition at DQN's own states", ad)
    report["advantage_decomposition"] = ad

    # Control: how well a linear model on the same features and reward separates
    if greedy is not None and getattr(greedy, "W", None) is not None:
        G = np.array([greedy_values(greedy, s) for s in S_rnd])
        st_g = spread_stats(G)
        print_block("control: GreedyPolicy immediate-reward prediction, same states", st_g)
        report["greedy_on_random_states"] = st_g
        ratio = st_g["rel_range_mean"] / max(st_rnd["rel_range_mean"], EPS)
        print(f"\n  greedy separates actions {ratio:.0f}x more than DQN, relatively")
        report["greedy_over_dqn_rel_spread"] = float(ratio)
        agree = float((G.argmax(axis=1) == V_rnd.argmax(axis=1)).mean())
        print(f"  argmax agreement {agree:.1%} (chance baseline 25%)")
        report["dqn_greedy_argmax_agreement"] = agree

    # ---------------- Verdict ----------------
    rel = st["rel_range_mean"]
    const_fn = max(st["per_action_std_across_states"]) < 0.01 * st["q_abs_mean"]
    state_dep_adv = ad["adv_std_over_abs_mean"] > 1.0
    print("\n" + "=" * 70)
    print("verdict")
    print("=" * 70)
    if rel < 0.01 or const_fn:
        verdict = ("A: the network learned no action distinction "
                   "(implementation/optimization failure, not a scientific result)")
        print(f"  relative range across actions {rel:.4%} < 1%: Q is nearly identical")
        print("  for every action, and the argmax is set by initialization noise.")
        print("  -> This does not support 'sequential learning does not help', only")
        print("     'DQN did not learn under this budget'.")
    elif not state_dep_adv:
        verdict = ("B1: V(s) was learned but the advantage is a near-constant bias "
                   "- the fit succeeded, the policy has no state dependence")
        print(f"  relative range across actions {rel:.4%}: Q does separate actions,")
        print("  which rules out explanation A.")
        print(f"  But {ad['frac_var_from_state']:.0%} of Q's variance is between")
        print(f"  states and only {ad['frac_var_from_action']:.0%} between actions")
        print("  within a state.")
        print(f"  Advantage cross-state std / |mean| = "
              f"{ad['adv_std_over_abs_mean']:.2f} < 1, i.e. essentially a fixed")
        print("  preference table.")
        print("  -> The network spent its capacity predicting how much HV a state can")
        print("     still gain (value) and failed to learn which editor to call")
        print("     (routing). The collapse follows directly from that split.")
        print("  -> This supports 'Q fit succeeded but the routing signal is very")
        print("     weak', which is stronger than 'failed to learn' but still not")
        print("     'sequentiality is useless in principle'.")
    else:
        verdict = ("B2: state-dependent routing was learned; implementation failure "
                   "is ruled out, the routing signal is real but very small")
        print(f"  relative range across actions {rel:.4%} >= 1%, and the advantage")
        print(f"  varies with state (std/|mean| = {ad['adv_std_over_abs_mean']:.2f} > 1).")
        print(f"  The argmax distribution has entropy {st['argmax_entropy_bits']:.2f}")
        print("  bits (0 for a pure constant), so strictly it is not constant.")
        print(f"  But action-related variance is only {ad['frac_var_from_action']:.2%}")
        print("  of Q's total variance.")
        print("  -> The conclusion is 'the implementation is fine, the routing signal")
        print("     is two orders of magnitude smaller than the value signal', not")
        print("     'DQN cannot learn'. This is consistent with greedy and bandit")
        print("     holding only a marginal edge under the editor-call budget.")
    report["verdict"] = verdict
    report["constant_function"] = bool(const_fn)
    report["state_dependent_advantage"] = bool(state_dep_adv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
