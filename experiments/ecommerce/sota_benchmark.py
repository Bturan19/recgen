"""SOTA-style comparison on next-item recommendation, standard protocol.

Protocol (RecBole-style):
- leave-one-out: last interaction per user is the positive test item
- eval candidates: positive + 100 random negatives (fixed seed), rank over
  the 101 candidates -> HR@10, NDCG@10
- baselines: SASRec (self-attention sequential rec), ALS-128, popularity,
  ItemKNN, recgen (frozen-LLM-embedding catalog-aware head)
- cold-start slice: test users with short history (<=7 items)

Run: uv run python experiments/ecommerce/sota_benchmark.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from recgen import CatalogRankingHead, FrozenEncoder

from ecommerce.data import build_splits, prep_data, verbalize_history, verbalize_item
from common import record

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/ecommerce_v2"
INSTRUCTION_HIST = "Given the user's purchase history, infer what they are likely to buy next."
N_USERS = 25000
SEED = 0
N_NEG = 100
MAX_LEN = 10


def sasrec_block(dim, nhead, ff_dim, dropout=0.1):
    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nhead,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )


class SASRec(nn.Module):
    def __init__(self, n_items, dim=64, layers=2, nhead=2, max_len=MAX_LEN, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, dim)
        self.blocks = nn.ModuleList([sasrec_block(dim, nhead, dim * 2, dropout) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, seq):
        mask = (seq != 0)
        x = self.emb(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device).unsqueeze(0))
        x = self.drop(x)
        attn_mask = torch.triu(torch.full((seq.shape[1], seq.shape[1]), float("-inf"), device=seq.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=~mask, src_mask=attn_mask)
        return self.layer_norm(x)


def pad_seq(seq, max_len=MAX_LEN):
    seq = seq[-max_len:]
    return seq + [0] * (max_len - len(seq))


def train_sasrec(model, seqs, targets, val_seqs, val_targets, n_items, n_neg=1, epochs=15, bs=512, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(SEED)
    best = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(seqs))
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            s = torch.tensor([pad_seq(seqs[j]) for j in b], device=dev)
            t = torch.tensor([targets[j] for j in b], device=dev)
            negs = []
            for j in b:
                hist = set(seqs[j]) - {0}
                pool = [it for it in range(1, n_items + 1) if it not in hist and it != targets[j]]
                negs.append(rng.choice(pool, n_neg))
            negs = torch.tensor(negs, device=dev)
            out = model(s)
            seq_lens = (s != 0).sum(1) - 1
            h = out[torch.arange(len(b), device=dev), seq_lens]
            pos_score = (h * model.emb(t)).sum(-1)
            neg_score = (h.unsqueeze(1) * model.emb(negs)).sum(-1)
            loss = loss_fn(pos_score, torch.ones_like(pos_score)) + loss_fn(neg_score, torch.zeros_like(neg_score))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        hr, ndcg = eval_sasrec(model, val_seqs, val_targets, n_items)
        print(f"  sasrec epoch {epoch + 1}: val HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
        if hr > best:
            best = hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_sasrec(model, seqs, targets, n_items, n_neg=100):
    model.eval()
    dev = next(model.parameters()).device
    hits = ndcg = 0.0
    with torch.no_grad():
        for j in range(len(seqs)):
            s = torch.tensor([pad_seq(seqs[j])], device=dev)
            t = targets[j]
            out = model(s)
            h = out[0, int((s != 0).sum(1)) - 1]
            hist = set(seqs[j]) - {0}
            pool = [it for it in range(1, n_items + 1) if it not in hist and it != t]
            negs = np.random.default_rng(SEED + j).choice(pool, n_neg, replace=False)
            cands = np.concatenate([[t], negs])
            scores = (h * model.emb(torch.tensor(cands, device=dev))).sum(-1)
            rank = int((scores[0] < scores[1:]).sum()) + 1
            if rank <= 10:
                hits += 1
                ndcg += 1.0 / np.log2(rank + 1)
    return hits / len(seqs), ndcg / len(seqs)


def hr_ndcg(scores_pos, scores_neg):
    hits = ndcg = 0.0
    n = len(scores_pos)
    for i in range(n):
        rank = int((scores_pos[i] < scores_neg[i]).sum()) + 1
        if rank <= 10:
            hits += 1
            ndcg += 1.0 / np.log2(rank + 1)
    return hits / n, ndcg / n


def main():
    from leakfree import load_splits, leak_mask, split_sets
    splits, meta = load_splits(min_hist=5, max_history=MAX_LEN, n_users=N_USERS, seed=SEED)
    mask = leak_mask(splits, meta=meta)
    n_leaked = int(mask.sum())
    splits = [s for s, m in zip(splits, mask) if not m]
    print(f"leakage filter: removed {n_leaked}/{N_USERS} users (test item in history); remaining {len(splits)}")
    n_tr = int(len(splits) * 0.8)
    n_va = int(len(splits) * 0.1)
    train, val, test = splits[:n_tr], splits[n_tr : n_tr + n_va], splits[n_tr + n_va :]
    all_items = sorted({it for _, _, it in splits})
    cat_idx = {i: j for j, i in enumerate(all_items)}
    n_items = len(all_items)
    print(f"users: train={len(train)} val={len(val)} test={len(test)}, items={n_items}")

    negs = {}
    for j, (u, hist, t) in enumerate(test):
        forbidden = {it for it, _ in hist} | {t}
        pool = np.array([i for i in all_items if i not in forbidden], dtype=object)
        negs[j] = np.random.default_rng(SEED + j).choice(pool, N_NEG, replace=False)

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=512)
    E = encoder.encode_cached([verbalize_item(i, meta) for i in all_items], f"{CACHE_DIR}/item_emb.npy")
    H = encoder.encode_cached([f"{INSTRUCTION_HIST}\n{verbalize_history(h, meta)}" for _, h, _ in splits], f"{CACHE_DIR}/user_emb_noleak.npy")
    H_te = H[n_tr + n_va :]

    head = CatalogRankingHead(dim=H.shape[1], epochs=30, batch_size=256, patience=5)
    head.fit(H[:n_tr], np.array([cat_idx[t] for _, _, t in train]), E)
    print(f"recgen head val mrr@20 = {head.best_mrr_:.4f}")

    r_pos = np.zeros(len(test))
    r_neg = np.zeros((len(test), N_NEG))
    for j, (u, hist, t) in enumerate(test):
        c = np.array([cat_idx[t]] + [cat_idx[i] for i in negs[j]])
        scores = head.predict_scores(H_te[j : j + 1], E[c], item_idx=c)[0]
        r_pos[j] = scores[0]
        r_neg[j] = scores[1:]
    hr, ndcg = hr_ndcg(r_pos, r_neg)
    record("sota_recgen", hr10=hr, ndcg10=ndcg)
    print(f"recgen: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    pop = np.bincount(np.array([cat_idx[t] for _, _, t in train]), minlength=n_items).astype(float)
    p_pos = np.array([pop[cat_idx[t]] for _, _, t in test])
    p_neg = np.array([[pop[cat_idx[i]] for i in negs[j]] for j in range(len(test))])
    hr, ndcg = hr_ndcg(p_pos, p_neg)
    record("sota_popularity", hr10=hr, ndcg10=ndcg)
    print(f"popularity: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    train_items = sorted({it for _, hist, _ in splits for it, _ in hist})
    iidx = {i: j for j, i in enumerate(train_items)}
    rows, cols, data = [], [], []
    for j, (u, hist, _) in enumerate(splits):
        for it, rt in hist:
            rows.append(j)
            cols.append(iidx[it])
            data.append(1.0)
    mat = csr_matrix((data, (rows, cols)), shape=(N_USERS, len(train_items)))

    try:
        from implicit.als import AlternatingLeastSquares
        model = AlternatingLeastSquares(factors=128, iterations=30, regularization=0.1, random_state=SEED, num_threads=1)
        model.fit(mat)
        uf = model.user_factors
        ifs = np.zeros((n_items, 128))
        for it, j in cat_idx.items():
            if it in iidx:
                ifs[j] = model.item_factors[iidx[it]]
        a_pos = np.array([uf[j + n_tr + n_va] @ ifs[cat_idx[t]] for j, (u, h, t) in enumerate(test)])
        a_neg = np.array([[uf[j + n_tr + n_va] @ ifs[cat_idx[i]] for i in negs[j]] for j in range(len(test))])
        hr, ndcg = hr_ndcg(a_pos, a_neg)
        record("sota_als128", hr10=hr, ndcg10=ndcg)
        print(f"ALS-128: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
    except Exception as e:
        print(f"ALS failed: {e}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sas = SASRec(n_items, dim=128, layers=2, nhead=4).to(dev)
    seqs = []
    targets = []
    for u, hist, t in train:
        s = [cat_idx[it] + 1 for it, _ in hist[-MAX_LEN:] if it in cat_idx]
        if not s:
            s = [0]
        seqs.append(s)
        targets.append(cat_idx[t] + 1)
    val_seqs = [[cat_idx[it] + 1 for it, _ in hist[-MAX_LEN:] if it in cat_idx] or [0] for _, hist, _ in val]
    val_targets = [cat_idx[t] + 1 for _, _, t in val]
    t0 = time.time()
    sas = train_sasrec(sas, seqs, targets, val_seqs, val_targets, n_items, n_neg=3, epochs=20, lr=5e-4)
    print(f"sasrec trained in {time.time() - t0:.0f}s")

    s_pos = np.zeros(len(test))
    s_neg = np.zeros((len(test), N_NEG))
    with torch.no_grad():
        for j, (u, hist, t) in enumerate(test):
            s = [cat_idx[it] + 1 for it, _ in hist[-MAX_LEN:] if it in cat_idx] or [0]
            s = torch.tensor([s], device=dev)
            out = sas(s)
            h = out[0, int((s != 0).sum(1)) - 1]
            cands = [cat_idx[t] + 1] + [cat_idx[i] + 1 for i in negs[j]]
            scores = (h * sas.emb(torch.tensor(cands, device=dev))).sum(-1).cpu().numpy()
            s_pos[j] = scores[0]
            s_neg[j] = scores[1:]
    hr, ndcg = hr_ndcg(s_pos, s_neg)
    record("sota_sasrec", hr10=hr, ndcg10=ndcg)
    print(f"SASRec: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    short = [j for j, (u, hist, t) in enumerate(test) if len(hist) <= 7]
    print(f"\ncold-start slice (history <= 7): {len(short)} test users")
    for name, pos, neg in [("recgen", r_pos, r_neg), ("sasrec", s_pos, s_neg), ("als128", a_pos, a_neg), ("popularity", p_pos, p_neg)]:
        hr, ndcg = hr_ndcg(pos[short], neg[short])
        record(f"sota_{name}_coldstart", hr10=hr, ndcg10=ndcg)
        print(f"  {name}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    main()
