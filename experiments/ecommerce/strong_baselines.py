import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import csr_matrix

from recgen import CatalogRankingHead, FrozenEncoder

from ecommerce.data import build_splits, prep_data, verbalize_history, verbalize_item
from common import record

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/ecommerce_v2"

INSTRUCTION_HIST = "Given the user's purchase history, infer what they are likely to buy next."

N_USERS = 25000
N_CAND = 300


def bootstrap_ci(ranks, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    mrrs = []
    for _ in range(n):
        sample = rng.choice(ranks, size=len(ranks), replace=True)
        inv = np.divide(1.0, sample, out=np.zeros_like(sample, dtype=float), where=sample > 0)
        mrrs.append(np.mean(inv))
    return float(np.percentile(mrrs, 2.5)), float(np.percentile(mrrs, 97.5))


def main():
    from leakfree import load_splits, leak_mask, split_sets
    splits, meta = load_splits(min_hist=5, max_history=10, n_users=N_USERS)
    mask = leak_mask(splits)
    splits = [s for s, m in zip(splits, mask) if not m]
    print(f"leakage filter: removed {int(mask.sum())}/{N_USERS} users; remaining {len(splits)}")
    n_all = len(splits)
    n_tr = int(n_all * 0.8)
    n_va = int(len(splits) * 0.1)
    train, val, test = splits[:n_tr], splits[n_tr : n_tr + n_va], splits[n_tr + n_va :]
    all_items = sorted({it for _, _, it in splits})
    cat_idx = {i: j for j, i in enumerate(all_items)}
    y = np.array([cat_idx[t] for _, _, t in splits])
    y_te = y[n_tr + n_va :]
    n_te = len(test)

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=512)
    E = encoder.encode_cached([verbalize_item(i, meta) for i in all_items], f"{CACHE_DIR}/item_emb.npy")
    H = encoder.encode_cached([f"{INSTRUCTION_HIST}\n{verbalize_history(h, meta)}" for _, h, _ in splits], f"{CACHE_DIR}/user_emb_noleak.npy")
    H_te = H[n_tr + n_va :]

    head = CatalogRankingHead(dim=H.shape[1], epochs=30, batch_size=256, patience=5)
    head.fit(H[:n_tr], y[:n_tr], E)
    print(f"head val mrr@20 = {head.best_mrr_:.4f}")

    train_items = sorted({it for _, hist, _ in splits for it, _ in hist})
    iidx = {i: j for j, i in enumerate(train_items)}
    rows, cols, data = [], [], []
    for j, (u, hist, _) in enumerate(splits):
        for it, rt in hist:
            rows.append(j)
            cols.append(iidx[it])
            data.append(1.0)
    mat = csr_matrix((data, (rows, cols)), shape=(n_all, len(train_items)))
    print(f"matrix (all users' pre-test history): {mat.shape}")

    last_item_embs = np.zeros((n_all, E.shape[1]))
    for j, (u, hist, _) in enumerate(splits):
        last = hist[-1][0]
        if last in cat_idx:
            last_item_embs[j] = E[cat_idx[last]]
    last_te = last_item_embs[n_tr + n_va :]

    pop = np.bincount(y[:n_tr], minlength=len(all_items)).astype(float)
    pop_order = np.argsort(-pop)

    def full_ranks(scores):
        ranks = np.zeros(n_te, dtype=int)
        for k, o in enumerate(np.argsort(-scores, axis=1)):
            ranks[k] = np.where(o == y_te[k])[0][0] + 1
        return ranks

    def summarize(name, scores):
        ranks = full_ranks(scores)
        lo, hi = bootstrap_ci(ranks)
        res = {
            "recall@10": float(np.mean(ranks <= 10)),
            "recall@20": float(np.mean(ranks <= 20)),
            "mrr@20": float(np.mean(1.0 / ranks)),
            "mrr20_ci95": f"[{lo:.4f},{hi:.4f}]",
        }
        record(f"ecom3_{name}", **res)
        print(f"{name}: " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "mrr20_ci95") + f" ci={res['mrr20_ci95']}")

    def summarize_2stage(name, score_fn):
        hits10 = hits20 = 0
        ranks = np.zeros(n_te)
        for k in range(n_te):
            c = cands[k + n_tr + n_va]
            s = score_fn(k, c)
            order = c[np.argsort(-s)]
            pos = np.where(order == y_te[k])[0]
            if len(pos) == 0:
                continue
            r = pos[0] + 1
            ranks[k] = r
            hits10 += r <= 10
            hits20 += r <= 20
        lo, hi = bootstrap_ci(ranks)
        res = {
            "recall@10": hits10 / n_te,
            "recall@20": hits20 / n_te,
            "mrr@20": float(np.mean(np.divide(1.0, ranks, out=np.zeros_like(ranks), where=ranks > 0))),
            "mrr20_ci95": f"[{lo:.4f},{hi:.4f}]",
        }
        record(f"ecom3_{name}_2stage", **res)
        print(f"{name} 2-stage: " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "mrr20_ci95") + f" ci={res['mrr20_ci95']}")

    scores_recgen = head.predict_scores(H_te, E)
    summarize("recgen", scores_recgen)
    summarize("popularity", np.tile(pop, (n_te, 1)))
    summarize("lastitem", last_te @ E.T)

    uf_all, item_factors_als = None, None
    try:
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(factors=128, iterations=30, regularization=0.1, random_state=0, num_threads=1)
        model.fit(mat)
        uf_all = model.user_factors
        item_factors_als = np.zeros((len(all_items), 128))
        for j, it in enumerate(all_items):
            if it in iidx:
                item_factors_als[j] = model.item_factors[iidx[it]]
        summarize("als128", uf_all[n_tr + n_va :] @ item_factors_als.T)
    except Exception as e:
        print(f"ALS failed: {e}")

    try:
        from implicit.nearest_neighbours import ItemItemRecommender

        knn = ItemItemRecommender(num_threads=1)
        knn.fit(mat)
        scores_knn = np.zeros((n_te, len(all_items)))
        for k, (u, hist, _) in enumerate(test):
            for it, _ in hist:
                if it not in iidx:
                    continue
                sims = knn.similar_items(iidx[it], N=100)
                for ii, s in zip(sims[0], sims[1]):
                    if ii in iidx and s > 0:
                        itm = train_items[ii]
                        if itm in cat_idx:
                            scores_knn[k, cat_idx[itm]] += s
        summarize("itemknn", scores_knn)
    except Exception as e:
        print(f"ItemKNN failed: {e}")

    try:
        Xc = np.zeros((n_all, len(all_items)))
        for it, j in cat_idx.items():
            if it in iidx:
                Xc[:, j] = mat[:, iidx[it]].toarray().ravel()
        Gc = Xc.T @ Xc
        W = np.linalg.inv(Gc + 200.0 * np.eye(len(all_items)))
        np.fill_diagonal(W, 0)
        scores_ease = np.zeros((n_te, len(all_items)))
        for k, (u, hist, _) in enumerate(test):
            row = np.zeros(len(all_items))
            for it, _ in hist:
                if it in cat_idx:
                    row += W[cat_idx[it]]
            scores_ease[k] = row
        summarize("ease", scores_ease)
    except Exception as e:
        print(f"EASE failed: {e}")

    mean_hist_embs = np.zeros((n_all, E.shape[1]))
    for j, (u, hist, _) in enumerate(splits):
        idx = np.array([cat_idx[it] for it, _ in hist if it in cat_idx], dtype=np.int64)
        if len(idx):
            mean_hist_embs[j] = E[idx].mean(axis=0)
    sim = mean_hist_embs @ E.T
    cands = []
    for i in range(n_all):
        sim_ord = np.argsort(-sim[i])
        union = set(pop_order[: N_CAND // 3])
        union.update(sim_ord[: N_CAND - len(union)])
        cands.append(np.array(sorted(union)))

    summarize_2stage("recgen", lambda k, c: head.predict_scores(H_te[k : k + 1], E[c], item_idx=c)[0])
    summarize_2stage("lastitem", lambda k, c: last_te[k] @ E[c].T)
    summarize_2stage("popularity", lambda k, c: pop[c])
    if uf_all is not None:
        summarize_2stage("als128", lambda k, c: uf_all[k + n_tr + n_va] @ item_factors_als[c].T)


if __name__ == "__main__":
    main()
