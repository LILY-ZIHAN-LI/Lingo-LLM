"""Six routers, all implementing interfaces.Policy: act(obs, t) -> action index.

Three groups:
- no learning: RandomPolicy, RoundRobinPolicy, GapPolicy
- fitted on training trajectories: GreedyPolicy (immediate gain only),
  LinUCBBandit (adds exploration)
- sequential learning: DQNPolicy

GreedyPolicy against DQNPolicy is the central comparison: whether delayed credit
assignment pays off. The bandit sits in between as a diagnostic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .interfaces import EXPERTS, N_ACTIONS, N_OBJ


# ==========================================================================
# Baselines without learning
# ==========================================================================
@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def reset(self, episode_seed: int) -> None:
        self._rng = random.Random(self.seed * 100003 + episode_seed)

    def act(self, obs, t: int) -> int:
        return self._rng.randrange(N_ACTIONS)

    def update(self, *a, **k) -> None:
        pass


@dataclass
class RoundRobinPolicy:
    name: str = "round_robin"

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        return t % N_ACTIONS

    def update(self, *a, **k) -> None:
        pass


@dataclass
class GapPolicy:
    """Call the expert for whichever objective is furthest from target after
    normalization.

    Without normalization this degenerates into always calling the same expert:
    the objectives live on very different scales (JNK3 is usually near 0, SA near
    0.9), so the ranking of 1-score barely varies across molecules. The scale is
    estimated per objective on the training set.
    """

    name: str = "gap"
    target: float = 1.0
    scale: np.ndarray | None = None

    def fit(self, transitions) -> "GapPolicy":
        X = np.array([tr["obs"][:N_OBJ] for tr in transitions], dtype=float)
        # Normalize by the mean distance to 1.0, which separates "low everywhere"
        # from "low on this molecule"
        self.scale = np.clip((self.target - X).mean(axis=0), 1e-3, None)
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        cur = np.asarray(obs[:N_OBJ], dtype=float)
        s = self.scale if self.scale is not None else np.ones(N_OBJ)
        return int(np.argmax((self.target - cur) / s))

    def update(self, *a, **k) -> None:
        pass


# ==========================================================================
# Fitted on training trajectories
# ==========================================================================
@dataclass
class GreedyPolicy:
    """One ridge regression per action predicting the immediate reward, then
    argmax. No exploration.

    This is the greedy expected-immediate-gain router fitted on training
    transitions. Its only difference from DQN is whether the future is considered.
    """

    name: str = "greedy"
    alpha: float = 1.0
    W: np.ndarray | None = None

    def fit(self, transitions) -> "GreedyPolicy":
        d = len(transitions[0]["obs"])
        self.W = np.zeros((N_ACTIONS, d + 1))
        for a in range(N_ACTIONS):
            rows = [tr for tr in transitions if tr["action"] == a]
            if not rows:
                continue
            X = np.array([np.append(tr["obs"], 1.0) for tr in rows])
            y = np.array([tr["reward"] for tr in rows])
            A = X.T @ X + self.alpha * np.eye(X.shape[1])
            self.W[a] = np.linalg.solve(A, X.T @ y)
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        return int(np.argmax(self.W @ x))

    def update(self, *a, **k) -> None:
        pass


@dataclass
class LinUCBBandit:
    """Contextual bandit: explores during training, pure argmax at evaluation.

    It uses the same features and the same immediate reward as GreedyPolicy; the
    only difference is exploration during training. The difference against DQN is
    sequentiality.
    """

    name: str = "bandit"
    alpha: float = 0.5           # exploration strength
    ridge: float = 1.0
    explore: bool = False        # set False at evaluation time

    A: list = field(default_factory=list)
    b: list = field(default_factory=list)

    def _init(self, d: int) -> None:
        self.A = [self.ridge * np.eye(d + 1) for _ in range(N_ACTIONS)]
        self.b = [np.zeros(d + 1) for _ in range(N_ACTIONS)]

    def fit(self, transitions) -> "LinUCBBandit":
        d = len(transitions[0]["obs"])
        self._init(d)
        for tr in transitions:
            x = np.append(np.asarray(tr["obs"], dtype=float), 1.0)
            a = tr["action"]
            self.A[a] += np.outer(x, x)
            self.b[a] += tr["reward"] * x
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        scores = []
        for a in range(N_ACTIONS):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mu = float(theta @ x)
            if self.explore:
                mu += self.alpha * float(np.sqrt(x @ Ainv @ x))
            scores.append(mu)
        return int(np.argmax(scores))

    def update(self, obs, action, reward, *a, **k) -> None:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


# ==========================================================================
# DQN
# ==========================================================================
class DQNPolicy:
    """Small MLP with experience replay and a target network; epsilon decays
    linearly during training.

    Two choices mitigate the sparse reward and exploration problem:
    1. epsilon decays over 60% of training rather than bottoming out in the first 10%
    2. rewards are rescaled, since HV gains are on the order of 1e-3 and the MLP
       cannot learn from them unscaled
    """

    name = "dqn"

    def __init__(self, obs_dim: int, hidden: int = 128, lr: float = 1e-3,
                 gamma: float = 0.95, reward_scale: float = 2000.0, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.epsilon = 0.0
        self._rng = random.Random(seed)

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        import torch
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.randrange(N_ACTIONS)
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs, dtype=np.float32))
            return int(self.q(x).argmax().item())

    def update(self, *a, **k) -> None:
        pass

    # ---- training ----
    def train_on(self, buffer, n_updates: int = 2000, batch: int = 64,
                 target_every: int = 100) -> list[float]:
        import torch
        losses = []
        obs = torch.tensor(np.array([t["obs"] for t in buffer], dtype=np.float32))
        act = torch.tensor(np.array([t["action"] for t in buffer]), dtype=torch.long)
        rew = torch.tensor(
            np.array([t["reward"] for t in buffer], dtype=np.float32)
        ) * self.reward_scale
        nxt = torch.tensor(np.array([t["next_obs"] for t in buffer], dtype=np.float32))
        done = torch.tensor(np.array([t["done"] for t in buffer], dtype=np.float32))

        n = len(buffer)
        for i in range(n_updates):
            idx = torch.randint(0, n, (min(batch, n),))
            q = self.q(obs[idx]).gather(1, act[idx].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(nxt[idx]).max(dim=1).values
                target = rew[idx] + self.gamma * (1 - done[idx]) * q_next
            loss = torch.nn.functional.smooth_l1_loss(q, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
            if (i + 1) % target_every == 0:
                self.target.load_state_dict(self.q.state_dict())
        return losses


@dataclass
class SingleExpertPolicy:
    """整条 episode 只调一个 expert，不路由。

    proposal 审稿意见 #1 要的那条基线。它堵的是这个解释漏洞：
    router 赢过 random，可能只是因为它学会了躲开弱 expert ——
    那样的话"全程只用最强的那个 expert"应该赢得一样多甚至更多，
    这时"学会排序"就不是一个有内容的说法。
    选哪个 expert 必须在 val 上定（scripts/run_fixed_expert.py 就是这么做的），
    在 test 上挑最好的那个等于把 4 次比较当 1 次报。
    """

    action: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"only_{EXPERTS[self.action]}"

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        return self.action

    def update(self, *a, **k) -> None:
        pass


def make_baselines(seed: int = 0) -> dict:
    """The two baselines that do no learning at all. gap must be fitted first to
    estimate its normalization scale."""
    return {
        "random": RandomPolicy(seed=seed),
        "round_robin": RoundRobinPolicy(),
    }


def make_fixed_experts() -> dict:
    """四条单 expert 基线，不需要任何训练。"""
    return {f"only_{e}": SingleExpertPolicy(action=a)
            for a, e in enumerate(EXPERTS)}
