"""Shared edit-and-score environment. Every policy (random / round-robin / gap /
greedy / bandit / DQN) runs on this class, which is what makes the comparison paired.

Rules:
- every eligible candidate enters the archive, which keeps the non-dominated set
- next molecule = candidate with the largest single-candidate HV gain, ties broken
  by canonical SMILES order
- no eligible candidate means the current molecule is kept
- reward = HV(A_{t+1}) - HV(A_t) - lambda * c_t

The transition rule is identical for every policy; the only thing a policy
changes is the action sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .chem import canonical, scaffold_preserved
from .editors import make_editors
from .interfaces import EXPERTS, N_OBJ, OBJECTIVES, Candidate, EpisodeConfig, StepLog
from .oracles import OracleBank

RDLogger.DisableLog("rdApp.*")

_FP_BITS = 64  # 1024 bits would swamp the other features and DQN cannot learn from 200 seeds
_OCC_BINS = 4
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=_FP_BITS)


# --------------------------------------------------------------------------
# Hypervolume
# --------------------------------------------------------------------------
def hypervolume(points: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    """HV for maximization. pymoo minimizes, so use 1-f and move the reference to 1-ref."""
    if not points:
        return 0.0
    from pymoo.indicators.hv import HV

    F = 1.0 - np.asarray(points, dtype=float)
    ref_min = 1.0 - np.asarray(ref, dtype=float)
    # Points outside the reference point contribute nothing; pymoo handles them
    return float(HV(ref_point=ref_min)(F))


def nondominated(points: list[tuple[str, tuple[float, ...]]]):
    """Non-dominated subset. The point count is small (<100), so O(n^2) is fine."""
    keep = []
    for i, (si, vi) in enumerate(points):
        dominated = False
        for j, (sj, vj) in enumerate(points):
            if i == j:
                continue
            if all(a >= b for a, b in zip(vj, vi)) and any(
                a > b for a, b in zip(vj, vi)
            ):
                dominated = True
                break
        if not dominated:
            keep.append((si, vi))
    return keep


def _fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_FP_BITS, dtype=np.float32)
    return np.asarray(_fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
@dataclass
class MolEnv:
    """One episode is one trajectory from one seed molecule."""

    bank: OracleBank
    cfg: EpisodeConfig = field(default_factory=EpisodeConfig)
    use_fingerprint: bool = True
    # Fix for the absorbing state found by scripts/analyze_editor_depletion.py:
    # a step with 0 eligible candidates leaves the molecule unchanged, a
    # deterministic policy then sees the same obs, picks the same editor and gets
    # 0 again. 45% of gap's steps were stuck this way. These bits record which
    # editors already failed on the current molecule and clear when it changes.
    # Off by default: enabling it changes obs_dim and breaks old policies.pkl.
    use_escape_bits: bool = False
    # Include archive shape and per-expert history. This changes obs_dim.
    use_rich_state: bool = False
    # Editor pool: 'rdkit' is the deterministic reaction pool (supplementary
    # control), 'gen' is the conditional SMILES transformer (main experiment).
    # Both honour the same Editor contract, so the rest of env is unchanged.
    editor_kind: str = "rdkit"
    gen_ckpt: str = "results_gen/generator.pt"
    # Only meaningful for editor_kind="rdkit": tunes how much the four pools
    # overlap, which turns "routing gain vs action-space dimension" from two
    # points into a curve. See src/editors.py:_mix. The default reproduces the
    # original behaviour.
    pool_overlap: float = 0.0
    qed_disjoint: bool = False
    pool_design: str | None = None
    sample_by_template: bool = False
    product_pick: str = "random"

    def __post_init__(self) -> None:
        if self.editor_kind == "gen":
            from .gen_editors import make_gen_editors
            # Reuse env's own bank: the current molecule is already scored, so this hits the cache
            self.editors = make_gen_editors(self.gen_ckpt, self.bank)
        elif self.editor_kind == "rdkit":
            self.editors = make_editors(overlap=self.pool_overlap,
                                        qed_disjoint=self.qed_disjoint,
                                        pool_design=self.pool_design,
                                        sample_by_template=self.sample_by_template,
                                        product_pick=self.product_pick)
        else:
            raise ValueError(f"unknown editor_kind: {self.editor_kind}")
        self.reset_counters()

    def clone(self):
        """Copy episode state for a counterfactual probe.

        Share the immutable editor pools and the scoring cache across branches.
        """
        import copy
        bank, editors = self.bank, self.editors
        self.bank = self.editors = None
        try:
            c = copy.deepcopy(self)
        finally:
            self.bank, self.editors = bank, editors
        c.bank, c.editors = bank, editors
        return c

    def reset_counters(self) -> None:
        self.n_generation_attempts = 0
        self.n_oracle_calls_at_start = self.bank.n_calls

    # ---- episode lifecycle ----
    def reset(self, seed_smiles: str, episode_seed: int):
        self.seed_smiles = canonical(seed_smiles) or seed_smiles
        self.episode_seed = episode_seed
        self.current = self.seed_smiles
        self.t = 0
        self.prev_action = -1
        self.failed_on_current: set[int] = set()
        self.logs: list[StepLog] = []
        self.calls_so_far = np.zeros(len(EXPERTS), dtype=np.float32)
        self.gain_so_far = np.zeros(len(EXPERTS), dtype=np.float32)
        self.seen: set[str] = {self.seed_smiles}
        self.n_generation_attempts = 0

        v = self.bank.vector(self.seed_smiles)
        if v is None:
            raise ValueError(f"invalid seed molecule: {seed_smiles}")
        self.archive: list[tuple[str, tuple[float, ...]]] = [
            (self.seed_smiles, tuple(v))
        ]
        self.hv = hypervolume([tuple(v)], self.cfg.ref_point)
        self.seed_hv = self.hv
        return self._obs()

    def _obs(self) -> np.ndarray:
        cur = self.bank.vector(self.current) or [0.0] * N_OBJ
        prev = np.zeros(len(EXPERTS) + 1, dtype=np.float32)
        prev[self.prev_action if self.prev_action >= 0 else len(EXPERTS)] = 1.0
        arch = np.asarray([v for _, v in self.archive], dtype=np.float32)
        feats = [
            np.asarray(cur, dtype=np.float32),            # four objective scores of the current molecule
            prev,                                          # one-hot of the previous expert
            np.asarray([1.0 - self.t / self.cfg.max_steps], dtype=np.float32),
            np.asarray([self.hv], dtype=np.float32),
            arch.max(axis=0),                              # per-objective archive maximum
            np.asarray([len(self.archive) / 20.0], dtype=np.float32),
        ]
        if self.use_escape_bits:
            bits = np.zeros(len(EXPERTS), dtype=np.float32)
            for a in self.failed_on_current:
                bits[a] = 1.0
            feats.append(bits)
        if self.use_rich_state:
            # 占据直方图：各目标 4 个等宽格子里前沿点的比例。
            # HV + max 看不出前沿是挤成一团还是铺开，这个能。
            occ = np.zeros((N_OBJ, _OCC_BINS), dtype=np.float32)
            for j in range(N_OBJ):
                b = np.clip((arch[:, j] * _OCC_BINS).astype(int), 0, _OCC_BINS - 1)
                np.add.at(occ[j], b, 1.0)
            feats.append((occ / max(len(arch), 1)).ravel())
            feats.append(arch.min(axis=0))
            feats.append(arch.mean(axis=0))
            # history-aware：谁被调过、谁换来过 HV。
            # HV 增益的量级是 1e-3，乘 100 和其余特征放到同一个尺度上。
            feats.append(self.calls_so_far / self.cfg.max_steps)
            feats.append(self.gain_so_far * 100.0)
        if self.use_fingerprint:
            feats.append(_fingerprint(self.current))
        return np.concatenate(feats)

    @property
    def obs_dim(self) -> int:
        return N_OBJ + len(EXPERTS) + 1 + 1 + 1 + N_OBJ + 1 + (
            len(EXPERTS) if self.use_escape_bits else 0
        ) + (
            N_OBJ * _OCC_BINS + 2 * N_OBJ + 2 * len(EXPERTS)
            if self.use_rich_state else 0
        ) + (_FP_BITS if self.use_fingerprint else 0)

    # ---- one step ----
    def step(self, action: int):
        expert = EXPERTS[action]
        parent = self.current
        base = self.bank.vector(parent)
        hv_before = self.hv
        oracle_before = self.bank.n_calls

        # The editor seed is a function of (episode, t), so every policy gets the
        # same candidates for the same (molecule, expert) pair.
        raw = self.editors[expert].propose(
            parent, self.cfg.k_per_call, seed=self.episode_seed * 1000 + self.t
        )
        self.n_generation_attempts += self.cfg.k_per_call

        cands: list[Candidate] = []
        for smi in raw:
            c = canonical(smi)
            if c is None:
                cands.append(Candidate(None, parent, expert, False, False))
                continue
            ok_scaf = (not self.cfg.enforce_scaffold) or scaffold_preserved(parent, c)
            v = self.bank.vector(c) if ok_scaf else None
            cands.append(
                Candidate(c, parent, expert, True, ok_scaf,
                          tuple(v) if v is not None else None)
            )

        eligible = [c for c in cands if c.eligible and c.scores is not None]
        n_novel = sum(1 for c in eligible if c.smiles not in self.seen)

        # --- archive update: every eligible candidate enters ---
        pool = self.archive + [(c.smiles, c.scores) for c in eligible]
        # Deduplicate, then keep the non-dominated set
        dedup: dict[str, tuple[float, ...]] = {}
        for s, v in pool:
            dedup[s] = v
        new_archive = nondominated(sorted(dedup.items()))
        hv_after = hypervolume([v for _, v in new_archive], self.cfg.ref_point)

        # --- transition: largest single-candidate HV gain, ties by canonical SMILES ---
        best_smi, best_gain = None, -1.0
        for c in sorted(eligible, key=lambda x: x.smiles):
            hv_c = hypervolume(
                [v for _, v in nondominated(
                    sorted({**dedup_base(self.archive), c.smiles: c.scores}.items()))],
                self.cfg.ref_point,
            )
            gain = hv_c - hv_before
            if gain > best_gain + 1e-12:
                best_gain, best_smi = gain, c.smiles

        entered = len(new_archive) != len(self.archive) or any(
            s not in {a for a, _ in self.archive} for s, _ in new_archive
        )

        best_single_gain = 0.0
        if eligible and base is not None:
            best_single_gain = max(
                max(c.scores[j] - base[j] for j in range(N_OBJ)) for c in eligible
            )

        self.archive = new_archive
        self.hv = hv_after
        if best_smi is not None and best_smi != parent:
            # The molecule moved, so past failures no longer apply
            self.current = best_smi
            self.failed_on_current = set()
        elif not eligible:
            # This editor produced nothing on the current molecule
            self.failed_on_current.add(action)
        for c in eligible:
            self.seen.add(c.smiles)

        # --- reward ---
        if self.cfg.cost_unit == "oracle":
            cost = self.bank.n_calls - oracle_before
        else:
            cost = self.cfg.k_per_call
        reward = (hv_after - hv_before) - self.cfg.lam * cost

        self.logs.append(StepLog(
            t=self.t, expert=expert, current=parent,
            n_proposed=self.cfg.k_per_call,
            n_valid=sum(1 for c in cands if c.valid),
            n_scaffold_ok=sum(1 for c in cands if c.eligible),
            n_novel=n_novel,
            hv_before=hv_before, hv_after=hv_after, reward=reward,
            archive_size=len(self.archive), entered_archive=entered,
            best_single_gain=best_single_gain,
        ))

        self.calls_so_far[action] += 1.0
        self.gain_so_far[action] += hv_after - hv_before
        self.prev_action = action
        self.t += 1
        done = self.t >= self.cfg.max_steps
        return self._obs(), reward, done, {
            "hv": hv_after,
            "hv_gain_vs_seed": hv_after - self.seed_hv,
            "generation_attempts": self.n_generation_attempts,
        }


def dedup_base(archive):
    return {s: v for s, v in archive}
