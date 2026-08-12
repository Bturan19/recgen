"""GenRec-style MovieLens-1M benchmark: frozen-LLM embeddings + catalog head
vs popularity / ALS / SASRec / BERT4Rec under the SASRec-paper protocol.

Protocol (Kang & McAuley ICDM'18): 5-core, implicit feedback, per-user
temporal split (most recent action = test, second-most-recent = val, rest =
train — ALL users are in training), eval = positive + 100 random negatives,
HR@10 + NDCG@10. MovieLens pairs are unique, so the held-out movie is never
in the history — leak-free by construction.

Usage:
  uv run python experiments/movielens/benchmark.py --backbone smol17
  uv run python experiments/movielens/benchmark.py --backbone qwen7   # ~1-2h encode
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.sparse import csr_matrix

from recgen import CatalogRankingHead, FrozenEncoder

from common import record
from data import (
    INSTRUCTION_HIST,
    build_splits,
    make_negatives,
    prep_data,
    verbalize_history,
    verbalize_item,
)
from models import BERT4Rec, SASRec, eval_bert4rec, eval_sequential, train_bert4rec, train_sasrec

BACKBONES = {
    "smol360": "models/SmolLM2-360M",
    "smol17": "models/SmolLM2-1.7B",
}
SEED = 0
N_NEG = 100
MAX_HIST_LLM = 20
MAX_HIST_SAS = 200


def hr_ndcg(pos, neg):
    hits = ndcg = 0.0
    n = len(pos)
    for i in range(n):
        rank = int((pos[i] < neg[i]).sum()) + 1
        if rank <= 10:
            hits += 1
            ndcg += 1.0 / np.log2(rank + 1)
    return hits / n, ndcg / n


def main(backbone: str = "smol17", n_neg: int = N_NEG, batch_size: int = 16, epochs: int = 40,
         baselines_only: bool = False, skip_baselines: bool = False):
    model_dir = BACKBONES[backbone]
    cache_dir = f".cache/movielens/{backbone}"

    ratings, movies = prep_data(min_actions=5)
    splits = build_splits(ratings, min_history=3)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(splits))
    splits = [splits[i] for i in perm]
    hist, val_items, test_items = (
        [s[1] for s in splits],
        np.array([s[2] for s in splits]),
        np.array([s[3] for s in splits]),
    )
    print(f"users: {len(splits)} (all in training; val/test = per-user temporal holdout)")

    negs, all_items = make_negatives(splits, n_neg=n_neg, seed=SEED)
    all_items = sorted(all_items)
    cat_idx = {i: j for j, i in enumerate(all_items)}
    n_items = len(all_items)
    print(f"catalog: {n_items} movies")

    hist_lens = np.array([len(h) for h in hist])
    print(f"history length: mean={hist_lens.mean():.1f} p50={np.median(hist_lens):.0f} "
          f"p75={np.percentile(hist_lens, 75):.0f} p90={np.percentile(hist_lens, 90):.0f}")

    if not baselines_only:
        encoder = FrozenEncoder(model_dir, pooling="mean", batch_size=batch_size, max_length=1024)
        E = encoder.encode_cached([verbalize_item(i, movies) for i in all_items], f"{cache_dir}/item_emb.npy")
        H = encoder.encode_cached(
            [f"{INSTRUCTION_HIST}\n{verbalize_history(h, movies, max_items=MAX_HIST_LLM)}" for h in hist],
            f"{cache_dir}/user_emb.npy",
        )
        test_texts = [f"{INSTRUCTION_HIST}\n{verbalize_history(h + [v], movies, max_items=MAX_HIST_LLM)}"
                      for h, v in zip(hist, val_items)]
        H_test = encoder.encode_cached(test_texts, f"{cache_dir}/user_emb_test.npy")

        head = CatalogRankingHead(dim=H.shape[1], epochs=epochs, batch_size=256, patience=5)
        head.fit(H, np.array([cat_idx[v] for v in val_items]), E)
        print(f"recgen head val mrr@20 = {head.best_mrr_:.4f}")

        r_pos = np.zeros(len(splits))
        r_neg = np.zeros((len(splits), n_neg))
        for j in range(len(splits)):
            c = np.array([cat_idx[test_items[j]]] + [cat_idx[i] for i in negs[j]])
            scores = head.predict_scores(H_test[j : j + 1], E[c], item_idx=c)[0]
            r_pos[j] = scores[0]
            r_neg[j] = scores[1:]
        hr, ndcg = hr_ndcg(r_pos, r_neg)
        record(f"ml1m_recgen_{backbone}", hr10=hr, ndcg10=ndcg)
        print(f"recgen[{backbone}]: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    pop = np.bincount(np.array([cat_idx[i] for h in hist for i in h]), minlength=n_items).astype(float)
    p_pos = np.array([pop[cat_idx[t]] for t in test_items])
    p_neg = np.array([[pop[cat_idx[i]] for i in negs[j]] for j in range(len(splits))])
    hr, ndcg = hr_ndcg(p_pos, p_neg)
    record("ml1m_popularity", hr10=hr, ndcg10=ndcg)
    print(f"popularity: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    try:
        from implicit.als import AlternatingLeastSquares

        train_items = sorted({i for h in hist for i in h})
        iidx = {i: j for j, i in enumerate(train_items)}
        rows, cols = [], []
        for j, h in enumerate(hist):
            for i in h:
                if i in iidx:
                    rows.append(j)
                    cols.append(iidx[i])
        mat = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(splits), len(train_items)))
        for factors in (32, 64, 128):
            model = AlternatingLeastSquares(factors=factors, iterations=30, regularization=0.1, random_state=SEED, num_threads=1)
            model.fit(mat)
            uf = model.user_factors
            ifs = np.zeros((n_items, factors))
            for i, j in cat_idx.items():
                if i in iidx:
                    ifs[j] = model.item_factors[iidx[i]]
            a_pos = np.array([uf[j] @ ifs[cat_idx[t]] for j, t in enumerate(test_items)])
            a_neg = np.array([[uf[j] @ ifs[cat_idx[i]] for i in negs[j]] for j in range(len(splits))])
            hr, ndcg = hr_ndcg(a_pos, a_neg)
            record(f"ml1m_als{factors}", hr10=hr, ndcg10=ndcg)
            print(f"ALS-{factors}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
    except Exception as e:
        print(f"ALS failed: {e}")

    short = [j for j, h in enumerate(hist) if len(h) <= 30]
    print(f"\ncold-start slice (history <= 30): {len(short)} users")
    if not baselines_only:
        for name, pos, neg in [("recgen", r_pos, r_neg)]:
            hr, ndcg = hr_ndcg(pos[short], neg[short])
            record(f"ml1m_{name}_coldstart", hr10=hr, ndcg10=ndcg)
            print(f"  {name}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    if skip_baselines or baselines_only:
        if baselines_only:
            for name, pos, neg in [("sasrec", s_pos, s_neg), ("bert4rec", b_pos, b_neg), ("popularity", p_pos, p_neg)]:
                hr, ndcg = hr_ndcg(pos[short], neg[short])
                record(f"ml1m_{name}_coldstart", hr10=hr, ndcg10=ndcg)
                print(f"  {name}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
        return

    def to_seq(h):
        s = [cat_idx[i] + 1 for i in h[-MAX_HIST_SAS:] if i in cat_idx]
        return s or [0]

    tr_seqs = [to_seq(h) for h in hist]
    va_seqs = [to_seq(h) for h in hist]
    va_targs = [cat_idx[v] + 1 for v in val_items]

    t0 = time.time()
    sas = SASRec(n_items, dim=50, layers=2, nhead=1, max_len=MAX_HIST_SAS, dropout=0.2).to(dev)
    sas = train_sasrec(sas, tr_seqs, va_targs, va_seqs, va_targs, n_items, n_neg=1, epochs=100, bs=128, lr=1e-3, seed=SEED, device=dev)
    print(f"sasrec trained in {time.time() - t0:.0f}s")

    s_pos = np.zeros(len(splits))
    s_neg = np.zeros((len(splits), n_neg))
    with torch.no_grad():
        for j in range(len(splits)):
            s = torch.tensor([to_seq(hist[j] + [val_items[j]])], device=dev)
            out = sas(s)
            hh = out[0, -1]  # left-padded: last real item at position n-1
            cands = [cat_idx[test_items[j]] + 1] + [cat_idx[i] + 1 for i in negs[j]]
            scores = (hh * sas.emb(torch.tensor(cands, device=dev))).sum(-1).cpu().numpy()
            s_pos[j] = scores[0]
            s_neg[j] = scores[1:]
    hr, ndcg = hr_ndcg(s_pos, s_neg)
    record("ml1m_sasrec", hr10=hr, ndcg10=ndcg)
    print(f"SASRec: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    t0 = time.time()
    bert = BERT4Rec(n_items, dim=64, layers=2, nhead=2, max_len=MAX_HIST_SAS, dropout=0.2).to(dev)
    bert = train_bert4rec(bert, tr_seqs, va_seqs, va_targs, n_items, mask_ratio=0.15, epochs=100, bs=128, lr=1e-3, seed=SEED, device=dev)
    print(f"bert4rec trained in {time.time() - t0:.0f}s")

    b_pos = np.zeros(len(splits))
    b_neg = np.zeros((len(splits), n_neg))
    with torch.no_grad():
        for j in range(len(splits)):
            s = torch.tensor([to_seq(hist[j] + [val_items[j]])], device=dev)
            hh = bert.last_position_logits(s)[0]
            cands = [cat_idx[test_items[j]] + 1] + [cat_idx[i] + 1 for i in negs[j]]
            scores = hh[torch.tensor(cands, device=dev)].cpu().numpy()
            b_pos[j] = scores[0]
            b_neg[j] = scores[1:]
    hr, ndcg = hr_ndcg(b_pos, b_neg)
    record("ml1m_bert4rec", hr10=hr, ndcg10=ndcg)
    print(f"BERT4Rec: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    if not baselines_only:
        for name, pos, neg in [("sasrec", s_pos, s_neg), ("bert4rec", b_pos, b_neg), ("popularity", p_pos, p_neg)]:
            hr, ndcg = hr_ndcg(pos[short], neg[short])
            record(f"ml1m_{name}_coldstart", hr10=hr, ndcg10=ndcg)
            print(f"  {name}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="smol17", choices=list(BACKBONES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--baselines-only", action="store_true", help="run only popularity/ALS/SASRec/BERT4Rec (no LLM encode)")
    ap.add_argument("--skip-baselines", action="store_true", help="run only the recgen head on cached embeddings")
    args = ap.parse_args()
    main(args.backbone, batch_size=args.batch_size, epochs=args.epochs,
         baselines_only=args.baselines_only, skip_baselines=args.skip_baselines)
