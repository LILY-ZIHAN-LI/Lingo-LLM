"""按起始分子 / scaffold 分组的交叉验证切分。

单独成模块的理由：3000 个反事实状态只有 200 个独立起始分子，所有脚本
必须用**同一套** fold，否则跨模型、跨特征组的比较不成立 —— 之前
probe_learnable_space.py 在特征组循环内部重新 `rng.permutation`，
两组特征拿到的是不同的划分，连 StepOnly 这条参照线都不一样
（0.000935 vs 0.000904）。

分组粒度：Murcko scaffold。seed 分子是 scaffold-split 选出来的，所以
一个 scaffold 基本对应一个起始分子；按 scaffold 分组同时也挡住了
"同一分子的不同 step 落到训练和测试两边" 这个泄漏。
"""
from __future__ import annotations

import numpy as np

from .chem import murcko_scaffold


def scaffold_of(smiles_list) -> np.ndarray:
    """返回每个样本的分组键；拿不到 scaffold 的退回 SMILES 自身。"""
    cache: dict[str, str] = {}
    out = []
    for s in smiles_list:
        if s not in cache:
            cache[s] = murcko_scaffold(s) or s
        out.append(cache[s])
    return np.asarray(out)


def grouped_folds(group_keys, n_folds: int, seed: int = 0) -> np.ndarray:
    """-> 每个样本的 fold 下标。同一个 group 的全部样本落在同一个 fold。

    确定性：只依赖 (group 名字的排序, n_folds, seed)，不依赖样本顺序，
    所以不同脚本、不同特征组、不同模型拿到的划分逐字相同。
    """
    g = np.asarray(group_keys)
    uniq = np.unique(g)                      # np.unique 已排序 -> 顺序确定
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[j]: i % n_folds for i, j in enumerate(perm)}
    return np.array([fold_of[x] for x in g], dtype=int)


def inner_grouped_folds(group_keys, mask, n_folds: int, seed: int = 0) -> np.ndarray:
    """在 mask 选出的训练子集内部再切一次（给超参选择用）。

    返回长度与 mask.sum() 相同的 fold 下标；同样按 group 分组。
    """
    g = np.asarray(group_keys)[np.asarray(mask)]
    return grouped_folds(g, n_folds, seed)
