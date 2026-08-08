import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from recgen import CatalogRankingHead, FrozenEncoder

from ecommerce.data import build_splits, prep_data, verbalize_history, verbalize_item
from common import record

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/ecommerce"

INSTRUCTION_HIST = "Given the user's purchase history, infer what they are likely to buy next."


def main(n_users: int = 10000, max_history: int = 10, min_hist: int = 5):
    reviews, meta = prep_data()
    splits = build_splits(reviews, max_history=max_history)
    splits = [s for s in splits if len(s[1]) >= min_hist]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(splits))
    splits = [splits[i] for i in perm[:n_users]]
    n_tr = int(n_users * 0.8)
    n_va = int(n_users * 0.1)
    train, val, test = splits[:n_tr], splits[n_tr : n_tr + n_va], splits[n_tr + n_va :]
    print(f"users: train={len(train)} val={len(val)} test={len(test)}")

    hist_texts = [verbalize_history(h, meta) for _, h, _ in train + val + test]
    hist_texts = [f"{INSTRUCTION_HIST}\n{t}" for t in hist_texts]

    all_items = sorted({it for _, _, it in splits})
    item_texts = [verbalize_item(i, meta) for i in all_items]
    print(f"catalog: {len(all_items)} items")

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=512)
    H = encoder.encode_cached(hist_texts, f"{CACHE_DIR}/user_emb.npy")
    E = encoder.encode_cached(item_texts, f"{CACHE_DIR}/item_emb.npy")

    y = np.array([all_items.index(t) for _, _, t in splits])
    H_tr, H_va, H_te = H[:n_tr], H[n_tr : n_tr + n_va], H[n_tr + n_va :]
    y_tr, y_va, y_te = y[:n_tr], y[n_tr : n_tr + n_va], y[n_tr + n_va :]

    t0 = time.time()
    head = CatalogRankingHead(dim=H.shape[1], epochs=30, batch_size=256, patience=5)
    head.fit(H_tr, y_tr, E)
    print(f"ranking head trained in {time.time() - t0:.0f}s, best val mrr@20={head.best_mrr_:.4f}")

    res = head.evaluate(H_te, y_te)
    record("ecom_recgen_rank", **{k: v for k, v in res.items()})
    print(f"recgen ranking head: " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))

    pop = np.bincount(y[:n_tr], minlength=len(all_items))
    pop_order = np.argsort(-pop)
    pop_ranks = np.argsort(pop_order)
    pr = pop_ranks[y_te] + 1
    pop_res = {"recall@10": float(np.mean(pr <= 10)), "recall@20": float(np.mean(pr <= 20)), "mrr@20": float(np.mean(1.0 / pr))}
    record("ecom_popularity", **pop_res)
    print(f"popularity: " + " ".join(f"{k}={v:.4f}" for k, v in pop_res.items()))

    try:
        import implicit
        from implicit.als import AlternatingLeastSquares

        user_ids = sorted({u for u, _, _ in splits})
        uidx = {u: i for i, u in enumerate(user_ids)}
        train_items = sorted({it for _, hist, _ in train for it in hist})
        iidx = {i: j for j, i in enumerate(train_items)}
        rows, cols, data = [], [], []
        for j, (u, hist, _) in enumerate(train):
            for it in hist:
                rows.append(uidx[u])
                cols.append(iidx[it])
                data.append(1.0)
        mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(train_items)))
        model = AlternatingLeastSquares(factors=32, iterations=15, random_state=0, num_threads=1)
        model.fit(mat)
        utr_idx = np.array([uidx[u] for u, _, _ in test])
        user_factors = model.user_factors[utr_idx]
        item_factors = np.zeros((len(all_items), 32))
        for j, it in enumerate(all_items):
            if it in iidx:
                item_factors[j] = model.item_factors[iidx[it]]
        als_scores = user_factors @ item_factors.T
        als_order = np.argsort(-als_scores, axis=1)
        als_ranks = np.zeros(len(y_te), dtype=int)
        for k, o in enumerate(als_order):
            als_ranks[k] = np.where(o == y_te[k])[0][0] + 1
        als_res = {
            "recall@10": float(np.mean(als_ranks <= 10)),
            "recall@20": float(np.mean(als_ranks <= 20)),
            "mrr@20": float(np.mean(1.0 / als_ranks)),
        }
        record("ecom_als", **als_res)
        print(f"ALS: " + " ".join(f"{k}={v:.4f}" for k, v in als_res.items()))
    except Exception as e:
        print(f"ALS failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=10000)
    ap.add_argument("--max-history", type=int, default=10)
    args = ap.parse_args()
    main(args.n_users, args.max_history)
