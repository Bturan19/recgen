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
CACHE_DIR = ".cache/ecommerce_v2"

INSTRUCTION_HIST = "Given the user's purchase history, infer what they are likely to buy next."


def main(n_users: int = 25000, max_history: int = 10, min_hist: int = 5, n_cand: int = 300):
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

    hist_texts = [f"{INSTRUCTION_HIST}\n{verbalize_history(h, meta)}" for _, h, _ in splits]
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
    record("ecom2_recgen_full", **{k: v for k, v in res.items()})
    print(f"recgen full-catalog: " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))

    item_hist_emb = np.zeros((n_users, E.shape[1]))
    for i, (_, hist, _) in enumerate(splits):
        idx = np.array([all_items.index(it) for it, _ in hist if it in all_items], dtype=np.int64)
        if len(idx):
            item_hist_emb[i] = E[idx].mean(axis=0)
    sim = item_hist_emb @ E.T
    pop = np.bincount(y[:n_tr], minlength=len(all_items)).astype(float)
    pop_order = np.argsort(-pop)
    cands = []
    for i in range(n_users):
        sim_ord = np.argsort(-sim[i])
        union = set(pop_order[: n_cand // 3])
        union.update(sim_ord[: n_cand - len(union)])
        cands.append(np.array(sorted(union)))
    cand_res = {"cand_recall@10": 0.0, "cand_recall@20": 0.0, "cand_mrr@20": 0.0}
    for i in range(n_users):
        c = cands[i]
        scores = head.predict_scores(H[i : i + 1], E[c], item_idx=c)
        order = c[np.argsort(-scores[0])]
        pos = np.where(order == y[i])[0]
        if len(pos) == 0:
            continue
        r = pos[0] + 1
        cand_res["cand_recall@10"] += r <= 10
        cand_res["cand_recall@20"] += r <= 20
        cand_res["cand_mrr@20"] += 1.0 / r
    n_te = n_users - n_tr - n_va
    cand_res = {k: v / n_te for k, v in cand_res.items()}
    record("ecom2_recgen_2stage", **cand_res)
    print(f"recgen 2-stage: " + " ".join(f"{k}={v:.4f}" for k, v in cand_res.items()))

    pop_order_all = np.argsort(-pop)
    pop_ranks = np.argsort(pop_order_all)
    pr = pop_ranks[y_te] + 1
    pop_res = {"recall@10": float(np.mean(pr <= 10)), "recall@20": float(np.mean(pr <= 20)), "mrr@20": float(np.mean(1.0 / pr))}
    record("ecom2_popularity", **pop_res)
    print(f"popularity: " + " ".join(f"{k}={v:.4f}" for k, v in pop_res.items()))

    try:
        import implicit
        from implicit.als import AlternatingLeastSquares

        user_ids = sorted({u for u, _, _ in splits})
        uidx = {u: i for i, u in enumerate(user_ids)}
        train_items = sorted({it for _, hist, _ in train for it, _ in hist})
        iidx = {i: j for j, i in enumerate(train_items)}
        rows, cols, data = [], [], []
        for j, (u, hist, _) in enumerate(train):
            for it, rt in hist:
                rows.append(uidx[u])
                cols.append(iidx[it])
                data.append(1.0 + 0.5 * rt)
        mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(train_items)))
        model = AlternatingLeastSquares(factors=64, iterations=20, random_state=0, num_threads=1)
        model.fit(mat)
        utr_idx = np.array([uidx[u] for u, _, _ in test])
        user_factors = model.user_factors[utr_idx]
        item_factors = np.zeros((len(all_items), 64))
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
        record("ecom2_als", **als_res)
        print(f"ALS: " + " ".join(f"{k}={v:.4f}" for k, v in als_res.items()))
    except Exception as e:
        print(f"ALS failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=25000)
    ap.add_argument("--max-history", type=int, default=10)
    ap.add_argument("--n-cand", type=int, default=300)
    args = ap.parse_args()
    main(args.n_users, args.max_history, n_cand=args.n_cand)
