"""Train the conditional SMILES transformer.

  micromamba activate lingollm
  python -u scripts/train_generator.py --epochs 12

The loss applies only to tokens after <mol>: the prefix (property bins plus the
source molecule) is a given condition, not a distribution to model. Including the
prefix would spend capacity on memorizing the scaffold distribution.

CPU-only, so the default is 6 layers x 256 dims (~5M parameters). Set the torch
thread count to the core count.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.generator import (  # noqa: E402
    CondSmilesTransformer, GenConfig, build_vocab, count_params, detokenize,
    make_prefix, split_smiles, tokenize,
)


class ReconDataset(Dataset):
    """One sample = prefix (target bins plus source molecule) + target SMILES.

    The source molecule rather than its scaffold is the condition; see
    src/generator.py:make_prefix for why.
    """

    def __init__(self, rows, vocab, max_len):
        self.rows, self.vocab, self.max_len = rows, vocab, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pre = make_prefix(r["bins"], tokenize(r["src"], self.vocab),
                          self.vocab)
        body = tokenize(r["dst"], self.vocab) + [self.vocab["<eos>"]]
        ids = (pre + body)[: self.max_len]
        # The loss mask marks the positions to be predicted. run_epoch takes
        # M[:, 1:] aligned with tgt = X[:, 1:], so index directly into ids: the
        # prefix contributes no loss and the body (including <eos>) contributes all.
        mask = [0] * len(pre) + [1] * (len(ids) - len(pre))
        return torch.tensor(ids), torch.tensor(mask[: len(ids)])


class LengthBucketSampler(torch.utils.data.Sampler):
    """Form batches by bucketing on length.

    This is the biggest CPU win: attention is O(T^2), and with random batching the
    longest sequence in a batch (~230) far exceeds the median (~110), so most of
    the compute goes to padding. The order is: shuffle, cut into large chunks,
    sort by length within a chunk, cut batches, then shuffle the batch order. That
    keeps randomness while keeping lengths within a batch close.
    """

    def __init__(self, lengths, batch_size: int, chunk: int = 50, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.bs, self.chunk, self.seed, self.epoch = batch_size, chunk, seed, 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        big = self.bs * self.chunk
        batches = []
        for s in range(0, len(idx), big):
            part = idx[s: s + big]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i: i + self.bs].tolist()
                        for i in range(0, len(part), self.bs)]
        rng.shuffle(batches)
        return iter(batches)


def collate(batch, pad: int):
    L = max(len(x) for x, _ in batch)
    X = torch.full((len(batch), L), pad, dtype=torch.long)
    M = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, m) in enumerate(batch):
        X[i, : len(x)] = x
        M[i, : len(m)] = m
    return X, M


def run_epoch(model, loader, pad, opt=None, sched=None, log_every=200):
    train = opt is not None
    model.train(train)
    tot_loss = tot_tok = 0
    t0 = time.time()
    for it, (X, M) in enumerate(loader):
        inp, tgt = X[:, :-1], X[:, 1:]
        msk = M[:, 1:].bool()                     # aligned with the predicted positions
        with torch.set_grad_enabled(train):
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                reduction="none").reshape(tgt.shape)
            n = msk.sum()
            loss = (loss * msk).sum() / n.clamp(min=1)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        tot_loss += float(loss.detach()) * int(n)
        tot_tok += int(n)
        if train and it % log_every == 0 and it:
            el = time.time() - t0
            print(f"    it {it:5d}  loss {tot_loss/tot_tok:.4f}  "
                  f"{tot_tok/el:7.0f} tok/s  lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def sample_quality(model, rows, vocab, n: int, temperature=0.7, top_k=16):
    """Sample a batch each epoch and measure validity and scaffold preservation.

    Token CE is not the metric of interest: a CE of 0.35 over a 60-token SMILES
    puts the probability of a fully correct string near e^-21. Whether the model
    produces valid, scaffold-preserving molecules is what decides if it is usable
    as an editor, so measure that directly rather than discovering it after
    training.
    """
    if not rows or n <= 0:
        return {}
    model.eval()
    inv = {i: tk for tk, i in vocab.items()}
    eos = vocab["<eos>"]
    val = scaf = exact = 0
    g = torch.Generator()
    for i, r in enumerate(rows[:n]):
        pre = make_prefix(r["bins"], tokenize(r["src"], vocab), vocab)
        if len(pre) >= model.cfg.max_len - 4:
            continue
        g.manual_seed(i)
        ids = model.sample(pre, max_new=min(140, model.cfg.max_len - len(pre)),
                           eos=eos, temperature=temperature, top_k=top_k,
                           generator=g)
        cs = canonical(detokenize(ids, inv))
        if cs is None:
            continue
        val += 1
        if scaffold_preserved(r["src"], cs):
            scaf += 1
        if cs == canonical(r["dst"]):
            exact += 1
    m = min(n, len(rows))
    return {"valid": val / m, "scaffold": scaf / m, "exact": exact / m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/gen_pairs_delta.jsonl")
    ap.add_argument("--gen-meta", default="data/gen_pairs_delta_meta.json")
    ap.add_argument("--init-from", default="",
                    help="warm start from existing weights; the bin-token embeddings are re-randomized")
    ap.add_argument("--out", default="results_gen/generator.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows, for smoke tests")
    ap.add_argument("--val-gen", type=int, default=16,
                    help="validation samples drawn per epoch to measure validity; 0 disables")
    ap.add_argument("--save-last", action="store_true",
                    help="also save the last weights each epoch, for diagnosing overfit")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    meta = json.loads(Path(args.gen_meta).read_text())
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"{len(rows)} edit pairs (train {len(tr)} / val {len(va)}), "
          f"{meta['n_bins']} bins x {len(meta['objectives'])} objectives")

    # The vocabulary comes from scanning the corpus, not a hand-written table
    chars = set()
    for r in rows:
        chars.update(split_smiles(r["src"]))
        chars.update(split_smiles(r["dst"]))
    vocab = build_vocab(meta["n_bins"], chars, len(meta["objectives"]))
    print(f"vocabulary {len(vocab)} ({len(chars)} corpus characters)")

    # Longest prefix = 1 + n_obj + 1 + len(source) + 1, plus the body
    max_len = 2 * meta["max_len"] + len(meta["objectives"]) + 8
    cfg = GenConfig(vocab_size=len(vocab), n_bins=meta["n_bins"],
                    n_obj=len(meta["objectives"]), d_model=args.d_model,
                    n_layer=args.n_layer, n_head=args.n_head,
                    d_ff=4 * args.d_model, max_len=max_len)
    model = CondSmilesTransformer(cfg)
    print(f"{count_params(model)/1e6:.2f}M parameters, max_len {max_len}, "
          f"{args.threads} threads")

    if args.init_from:
        # Architecture and corpus are unchanged; only the meaning of the 16 bin
        # tokens changed (absolute quantiles -> change quantiles, see
        # scripts/rebin_delta.py). Copying the SMILES and changing a little is the
        # bulk of what was learned and transfers; only those 16 embedding rows must
        # be re-randomized, or the model reads the new condition with the old meaning.
        blob = torch.load(args.init_from, map_location="cpu",
                          weights_only=False)
        if blob["vocab"] != vocab:
            print(f"!! vocabulary mismatch (old {len(blob['vocab'])} / "
                  f"new {len(vocab)}); skipping warm start, initializing randomly")
        else:
            model.load_state_dict(blob["state"])
            bin_rows = [vocab[f"<b{j}_{v}>"]
                        for j in range(cfg.n_obj) for v in range(cfg.n_bins)]
            with torch.no_grad():
                # Assign through __setitem__: model.tok.weight[bin_rows] is
                # advanced indexing and returns a copy, so calling nn.init on it
                # would do nothing.
                model.tok.weight[bin_rows] = 0.02 * torch.randn(
                    len(bin_rows), cfg.d_model)
            print(f"warm started from {args.init_from} (epoch {blob.get('epoch')}, "
                  f"val {blob.get('val', float('nan')):.4f}), "
                  f"re-randomized {len(bin_rows)} bin-token embeddings")

    pad = vocab["<pad>"]

    def seq_len(r):
        return min(len(r["src"]) + len(r["dst"]) + 8, max_len)

    def mk(rs, bucket):
        ds = ReconDataset(rs, vocab, max_len)
        kw = dict(collate_fn=lambda b: collate(b, pad), num_workers=0)
        if bucket:
            return DataLoader(ds, batch_sampler=LengthBucketSampler(
                [seq_len(r) for r in rs], args.batch), **kw)
        return DataLoader(ds, batch_size=args.batch, shuffle=False, **kw)

    dl_tr, dl_va = mk(tr, True), mk(va, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    steps = max(args.epochs * len(dl_tr), 1)
    warm = max(int(0.03 * steps), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, hist = float("inf"), []
    for ep in range(args.epochs):
        t0 = time.time()
        trl = run_epoch(model, dl_tr, pad, opt, sched)
        val = run_epoch(model, dl_va, pad)
        q = sample_quality(model, va, vocab, args.val_gen)
        hist.append({"epoch": ep, "train": trl, "val": val,
                     "ppl": math.exp(min(val, 20)), "sec": time.time() - t0,
                     **{f"gen_{k}": v for k, v in q.items()}})
        flag = ""
        if val < best:
            best = val
            torch.save({"cfg": cfg.__dict__, "vocab": vocab,
                        "state": model.state_dict(), "gen_meta": meta,
                        "epoch": ep, "val": val}, out)
            flag = "  <- saved"
        qs = (f"  valid {q['valid']:.0%}  scaf {q['scaffold']:.0%}"
              f"  exact {q['exact']:.0%}" if q else "")
        print(f"epoch {ep:2d}  train {trl:.4f}  val {val:.4f}  "
              f"ppl {math.exp(min(val,20)):.2f}{qs}  "
              f"{time.time()-t0:.0f}s{flag}", flush=True)
        Path(str(out) + ".hist.json").write_text(json.dumps(hist, indent=2))
        if args.save_last:      # for diagnostics: best-val may be an early epoch
            torch.save({"cfg": cfg.__dict__, "vocab": vocab,
                        "state": model.state_dict(), "gen_meta": meta,
                        "epoch": ep, "val": val}, str(out) + ".last.pt")

    print(f"\nbest val loss {best:.4f}, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
