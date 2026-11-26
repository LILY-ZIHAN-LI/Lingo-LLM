"""Step 7：可部署的 Dynamic Router。

推理路径的硬约束：
  - 只读当前 env 状态，**不克隆环境、不执行任何 expert**
  - 不接触反事实标签、未选中 expert 的真实收益、未来状态
  - 特征走 src/router_features.features_from_env()，和训练时逐字同一份
    实现（已有一致性测试：同一状态下离线/在线 102 维逐列相同）

可用性规则（Step 7 第 4 条）：
  预测只给出四个 expert 的预期 ΔHV，再叠一层只用**已观察到**的历史的
  可用性过滤 —— 本 episode 内调用过且至今零收益的 expert 降权。这一层
  必须和 AvailabilityRule 基线用同一个定义，否则"ML 的增量"里混进了
  不同的耗尽处理，无法归因（Step 3 的"统一耗尽处理机制"）。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .interfaces import EXPERTS
from .router_features import FEATURE_GROUPS, features_from_env

NE = len(EXPERTS)


def stale_mask(env) -> np.ndarray:
    """本 episode 内调用过、但至今零累计收益的 expert -> True。

    只看已经发生的事，不探查、不看未来。和 AvailabilityRule 基线共用。
    """
    calls = np.asarray(env.calls_so_far, dtype=float)
    gain = np.asarray(env.gain_so_far, dtype=float)
    return (calls > 0) & (gain <= 0)


class DynamicRouter:
    """从 train_dynamic_router.py 存下的 bundle 里恢复，逐步给出动作。"""

    def __init__(self, bundle: dict, availability: str = "stale_penalty"):
        self.kind = bundle["kind"]
        self.group = bundle["feature_group"]
        self.scaler = bundle["scaler"]
        self.models = bundle["models"]
        self.experts = bundle["experts"]
        self.feat_cost_ms = bundle.get("feat_cost_ms")
        if self.experts != list(EXPERTS):
            raise ValueError(f"expert 顺序不一致: {self.experts} vs {EXPERTS}")
        if self.group not in FEATURE_GROUPS:
            raise ValueError(f"未知特征组 {self.group}")
        if availability not in ("none", "stale_penalty"):
            raise ValueError(availability)
        self.availability = availability

    @classmethod
    def load(cls, path, **kw):
        return cls(pickle.loads(Path(path).read_bytes()), **kw)

    # ---- 预测 ----
    def _scores(self, X: np.ndarray) -> np.ndarray:
        A = self.scaler.transform(X)
        out = np.zeros((len(A), NE))
        # ranking：存的是 6 个 expert 对的分类器，Borda 计分
        if self.models and isinstance(self.models[0], tuple) \
                and self.models[0][0] == "rank":
            pair = self.models[0][1]
            for (a, b), (kind, m) in pair.items():
                if kind == "const":
                    out[:, a if m == 1.0 else b] += 1.0
                    continue
                p = m.predict_proba(A)[:, 1]
                out[:, a] += p
                out[:, b] += 1.0 - p
            return out
        for j, m in enumerate(self.models):
            if isinstance(m, tuple) and m and m[0] == "twostage":
                _, clf, reg, fallback, const = m
                p = (np.full(len(A), const) if clf is None
                     else clf.predict_proba(A)[:, 1])
                q = (np.full(len(A), fallback) if reg is None
                     else reg.predict(A))
                out[:, j] = p * np.maximum(q, 0.0)
            else:
                out[:, j] = m.predict(A)
        return out

    def predict_env(self, env) -> np.ndarray:
        """当前状态下四个 expert 的预测 ΔHV。"""
        return self._scores(features_from_env(env, self.group))[0]

    # ---- 决策 ----
    def act_env(self, env, t: int | None = None) -> int:
        s = self.predict_env(env).copy()
        if self.availability == "stale_penalty":
            bad = stale_mask(env)
            if not bad.all():                 # 全都 stale 时不做过滤
                s[bad] = -np.inf
        return int(np.argmax(s))

    # ---- 给 run_oracle_router.rollout 用的 chooser 接口 ----
    def chooser(self):
        def fn(env, t):
            return self.act_env(env, t), 0    # 0 次探查：不执行任何 expert
        return fn
