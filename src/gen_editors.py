"""Generative objective-conditioned editors: one shared backbone, four conditions.

This is the main editor family; the RDKit pool is its supplementary control.
The difference is structural:

  RDKit: an action is a set of reaction templates. qed and sa share literally the
         same pool on 70 of 100 seeds, so the effective action dimension is only
         2.05/4.
  Here:  an action is a condition vector. The four editors necessarily receive
         four different condition prefixes, and whether that actually yields
         different distributions is testable; see scripts/diagnose_conditioning.py.

Budget: each call samples exactly K times. Invalid, duplicate and
scaffold-breaking proposals still count as one generation attempt, with no
oversampling compensation. Validity is therefore a measured quantity rather than
100% by construction as in the RDKit pool.

Determinism (required by interfaces.Editor): sampling uses a torch.Generator
seeded from a blake2b digest of (smiles, editor name, seed). Python's hash() is
salted per process and would break the paired comparison.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from .chem import canonical
from .generator import (
    CondSmilesTransformer, GenConfig, detokenize, make_prefix, tokenize,
)
from .interfaces import OBJECTIVES
from .oracles import OracleBank

# Decoding parameters, frozen before the router comparison
TEMPERATURE = 0.9
TOP_K = 24

# Proposal cache keyed by (ckpt, editor name, molecule, k, seed, bump).
# Per-process: the evaluation stage of run_experiment uses multiprocessing.Pool
# so each worker holds its own copy, while training is single-process and gets
# the highest hit rate across repeats.
_PROP_CACHE: dict = {}
_CACHE_MAX = 300_000


@lru_cache(maxsize=4)
def load_backbone(ckpt_path: str):
    """Cached by path: the four editors share one set of weights, loaded once."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GenConfig(**blob["cfg"])
    model = CondSmilesTransformer(cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob["vocab"], blob["gen_meta"]


def _stable_seed(*parts) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


@dataclass
class GenEditor:
    """name must be one of OBJECTIVES; it selects which objective bin is pushed up."""

    name: str
    ckpt: str
    bank: OracleBank
    # How many bins to push the target up. 1 is too weak: adjacent gsk3b/jnk3
    # bins differ by only ~0.014 in the underlying property, so the condition is
    # barely asking for anything. bump >= n_bins means "top bin", which maximizes
    # the difference between the four conditions.
    bump: int = 1
    _m: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._m = load_backbone(self.ckpt)
        _, _, meta = self._m
        self.obj_idx = list(meta["objectives"]).index(self.name)
        self.n_bins = meta["n_bins"]
        # "delta": bins are quantiles of the change dy (scripts/rebin_delta.py)
        # "absolute": bins are quantiles of the property itself (older scheme,
        # kept only to read archived checkpoints)
        self.mode = meta.get("conditioning", "absolute")
        if self.mode == "delta":
            self.neutral = meta["neutral_bin"]
        else:
            self.edges = meta["bin_edges"]

    def _abs_bins(self, smiles: str) -> list[int] | None:
        """Absolute property bins of the current molecule (absolute scheme only).
        Goes through the bank cache: env already scored this molecule, so this is
        a cache hit and costs no extra oracle budget."""
        v = self.bank.vector(smiles)
        if v is None:
            return None
        out = []
        for j, o in enumerate(OBJECTIVES):
            # Must match the training-time binning exactly:
            # scripts/prepare_generator_data.py uses
            # np.searchsorted(edges, y, side="right"), i.e. y >= edge counts.
            # Using v[j] > x differs only exactly on a boundary, but TDC scores
            # for gsk3b/jnk3 are highly discrete (jnk3's first edge is 0.0) and
            # 70.7% of molecules then got a bin token never seen in training on at
            # least one dimension (jnk3 63%, gsk3b 27%, qed/sa ~0%).
            out.append(int(sum(1 for x in self.edges[o] if v[j] >= x)))
        return out

    def _target(self, smiles: str) -> list[int] | None:
        """Condition vector for this editor: push its own dimension up, leave the
        others unchanged."""
        if self.mode == "delta":
            # "unchanged" means dy in the neutral bin; independent of the current
            # molecule, so there is no saturation problem
            tgt = [self.neutral[o] for o in OBJECTIVES]
        else:
            tgt = self._abs_bins(smiles)
            if tgt is None:
                return None
        tgt[self.obj_idx] = min(tgt[self.obj_idx] + self.bump, self.n_bins - 1)
        return tgt

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        model, vocab, meta = self._m
        tgt = self._target(smiles)
        if tgt is None:
            return []

        # Condition = the current molecule plus the raised target bin
        parent = canonical(smiles) or smiles
        prefix = make_prefix(tgt, tokenize(parent, vocab), vocab)
        if len(prefix) >= model.cfg.max_len:
            return []

        # Proposal cache across policies and repeats. env requires every policy
        # to get the same candidates for the same (molecule, expert, seed), so
        # this is already a pure function; without the cache 6 policies times 3
        # repeats would redo identical sampling. OracleBank caches scores, but
        # nothing cached proposals before.
        ck = (self.ckpt, self.name, parent, k, seed, self.bump)
        hit = _PROP_CACHE.get(ck)
        if hit is not None:
            return list(hit)

        inv_eos = vocab["<eos>"]
        inv = {i: t for t, i in vocab.items()}
        out: list[str] = []
        seen = {parent}
        g = torch.Generator()
        g.manual_seed(_stable_seed(parent, self.name, seed))
        # Exactly K attempts, no compensation; sampled as one batch (see generator.sample_batch)
        for ids in model.sample_batch(prefix, k, max_new=meta["max_len"] + 4,
                                      eos=inv_eos, temperature=TEMPERATURE,
                                      top_k=TOP_K, generator=g):
            cs = canonical(detokenize(ids, inv))   # invalid -> None, still costs an attempt
            if cs is None or cs in seen:
                continue
            seen.add(cs)
            out.append(cs)   # env does the scaffold check; rejected proposals are counted

        if len(_PROP_CACHE) < _CACHE_MAX:
            _PROP_CACHE[ck] = list(out)
        return out


def make_gen_editors(ckpt: str, bank: OracleBank,
                     bump: int = 1) -> dict[str, GenEditor]:
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"generator not found at {ckpt}; run scripts/train_generator.py first")
    return {n: GenEditor(name=n, ckpt=ckpt, bank=bank, bump=bump)
            for n in OBJECTIVES}
