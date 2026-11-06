"""Conditional SMILES transformer: (source molecule, property bins) -> SMILES.

A compact generator trained with token cross-entropy as a reconstruction
objective. At inference one objective's bin is pushed up a notch and the rest are
held fixed, which gives four objective-conditioned generative editors that share
the same weights, rather than four separately trained models.

The key difference from the RDKit editing pool:
  In the RDKit pool an action is a set of reaction templates, and two editors may
  share literally the same pool (qed and sa propose identically on 70 of 100
  seeds). Here an action is a condition vector, so the four editors necessarily
  correspond to four different conditions, and whether that yields different
  distributions is testable.

Architecture (trainable on CPU): decoder-only, conditions as prefix tokens.
  [BOS] <bin_gsk3b> <bin_jnk3> <bin_qed> <bin_sa> <SRC> source... <MOL> smiles... [EOS]
The loss applies only to positions after <MOL>; the prefix is the condition, not
a distribution to learn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-character elements and marks must match before single characters,
# otherwise 'Cl' splits into 'C','l'. %10-%99 are two-digit ring closures and
# would otherwise be read as two separate ring numbers.
MULTI = ("Cl", "Br", "Si", "Se", "se", "@@", "+2", "-2", "+3", "-3") + tuple(
    f"%{i}" for i in range(10, 100))
SPECIAL = ["<pad>", "<bos>", "<eos>", "<src>", "<mol>", "<unk>"]


def split_smiles(smi: str) -> list[str]:
    """Character-level split that keeps multi-character marks intact. Vocabulary
    independent, so it can be used to scan a corpus and build the vocabulary."""
    out, i = [], 0
    while i < len(smi):
        for m in MULTI:
            if smi.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            out.append(smi[i])
            i += 1
    return out


def build_vocab(n_bins: int, chars, n_obj: int = 4) -> dict[str, int]:
    """`chars` comes from an actual corpus scan (see scripts/train_generator.py),
    not from a hand-written character table."""
    toks = list(SPECIAL)
    toks += [f"<b{j}_{v}>" for j in range(n_obj) for v in range(n_bins)]
    toks += sorted(set(chars))
    out: dict[str, int] = {}
    for t in toks:
        if t not in out:
            out[t] = len(out)
    return out


def tokenize(smi: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in split_smiles(smi)]


def detokenize(ids: list[int], inv: dict[int, str]) -> str:
    out = []
    for i in ids:
        t = inv.get(i, "")
        if t in ("<eos>", "<pad>"):
            break
        if t.startswith("<"):
            continue
        out.append(t)
    return "".join(out)


@dataclass
class GenConfig:
    vocab_size: int
    n_bins: int = 4
    n_obj: int = 4
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_ff: int = 1024
    max_len: int = 260
    dropout: float = 0.1


class CondSmilesTransformer(nn.Module):
    """Decoder-only with a causal mask; conditions enter as prefix tokens.

    Parameter count is roughly n_layer * 12 * d_model^2, so the default 6x256 is
    about 4.8M. The original plan was 10-30M, but that assumed a 16-32GB GPU;
    this runs on a CPU-only instance, hence below the lower bound. This deviation
    should be stated in the report.
    """

    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.body = nn.TransformerEncoder(layer, cfg.n_layer)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        p = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(p)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.body(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    @torch.no_grad()
    def sample_batch(self, prefix: list[int], n: int, max_new: int, eos: int,
                     temperature: float = 1.0, top_k: int = 0,
                     generator: torch.Generator | None = None
                     ) -> list[list[int]]:
        """Sample n sequences from the same prefix as one batch.

        On CPU this dominates editor cost: sampling one at a time means a batch=1
        forward per step, where the matrices are too small and time goes to memory
        access and operator dispatch rather than FLOPs. Batching dispatches once
        per step.

        Determinism: the whole batch shares one generator, seeded by GenEditor
        from (parent, editor name, seed). The interfaces.Editor requirement that
        the same (smiles, seed) gives the same output still holds; only which n
        molecules come out differs from one-at-a-time sampling.
        """
        self.eval()
        x = torch.tensor([prefix], dtype=torch.long).repeat(n, 1)
        out: list[list[int]] = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool)
        for _ in range(max_new):
            if x.shape[1] >= self.cfg.max_len:
                break
            logits = self(x)[:, -1] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1), 1,
                                    generator=generator).squeeze(1)
            for i in range(n):
                if not done[i]:
                    t = int(nxt[i])
                    if t == eos:
                        done[i] = True
                    else:
                        out[i].append(t)
            if bool(done.all()):
                break
            # Finished rows keep being fed eos; the done mask above ignores their output
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        return out

    @torch.no_grad()
    def sample(self, prefix: list[int], max_new: int, eos: int,
               temperature: float = 1.0, top_k: int = 0,
               generator: torch.Generator | None = None) -> list[int]:
        self.eval()
        x = torch.tensor([prefix], dtype=torch.long)
        out: list[int] = []
        for _ in range(max_new):
            if x.shape[1] >= self.cfg.max_len:
                break
            logits = self(x)[0, -1] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits = logits.masked_fill(logits < v[-1], float("-inf"))
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1,
                                        generator=generator))
            if nxt == eos:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        return out


def make_prefix(bins: list[int], src_ids: list[int],
                vocab: dict[str, int]) -> list[int]:
    """[bos] <b0_v> .. <b3_v> <src> ...current molecule... <mol>

    The condition is the current molecule itself, not its scaffold, for two reasons:
      1. an expert is specified to see the current molecule/scaffold, the current
         property scores and an objective-specific target;
      2. given only the scaffold, the model drops every substituent and each call
         jumps to an arbitrary same-scaffold analog, so trajectories stop
         accumulating and there is nothing left to route. That variant held the
         scaffold 0% of the time in smoke tests.
    Conditioned on the current molecule, the model only has to learn to copy most
    of it and change a little according to the target bin, which makes scaffold
    preservation learnable rather than luck.
    """
    ids = [vocab["<bos>"]]
    ids += [vocab[f"<b{j}_{v}>"] for j, v in enumerate(bins)]
    ids += [vocab["<src>"]] + src_ids + [vocab["<mol>"]]
    return ids


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
