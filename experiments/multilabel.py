"""Multi-label benchmark (AUDIT.md section 4): backs the "multi-label" claim.

Dataset: google-research-datasets/go_emotions (28 emotion labels, genuinely
multi-label ~17% of rows). Frozen SmolLM2-360M encoder + MultiLabelHead vs
TF-IDF+LogReg and LightGBM (binary-relevance) baselines. Same train/test
split for every method. Skips silently if the dataset cannot be fetched.

Run: uv run python experiments/multilabel.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from recgen import FrozenEncoder, MultiLabelHead

from common import record

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/multilabel"

INSTRUCTION = "Classify the emotions expressed in this text."


def load_goemotions(n: int = 6000, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", split="train")
    df = ds.to_pandas().head(n)
    rows = [v if isinstance(v, (list, np.ndarray)) else [v] for v in df["labels"]]
    classes = sorted({int(x) for r in rows for x in r})
    Y = np.zeros((len(rows), len(classes)), dtype=int)
    for i, r in enumerate(rows):
        for x in r:
            Y[i, classes.index(int(x))] = 1
    return df["text"].tolist(), Y


def micro_f1(Y, P):
    return f1_score(Y.ravel(), P.ravel(), zero_division=0)


def main(n: int = 6000):
    texts, Y = load_goemotions(n)
    n_labels = Y.shape[1]
    n_multilabel = int((Y.sum(axis=1) > 1).sum())
    print(f"go_emotions: {len(texts)} rows, {n_labels} labels, {n_multilabel} multi-label rows")
    idx = np.arange(len(texts))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    tr, te = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]
    y_tr, y_te = Y[tr], Y[te]
    X_tr, X_te = [texts[i] for i in tr], [texts[i] for i in te]

    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=256)
    h = encoder.encode_cached(X_tr + X_te, f"{CACHE_DIR}/emb_mean.npy")
    head = MultiLabelHead(epochs=40, hidden=(128,), patience=8)
    head.fit(h[: len(X_tr)], y_tr)
    p_mlh = head.predict(h[len(X_tr) :])
    rec = {
        "mlh_micro_f1": micro_f1(y_te, p_mlh),
        "mlh_exact_acc": float(np.mean((p_mlh == y_te).all(axis=1))),
    }
    record("goemo_mlh", **rec)
    print(f"MultiLabelHead: micro-F1={rec['mlh_micro_f1']:.4f} exact-acc={rec['mlh_exact_acc']:.4f}")

    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    Xtf = vec.fit_transform(X_tr).toarray()
    Xtf_te = vec.transform(X_te).toarray()
    p_tfidf = np.zeros_like(y_te)
    for j in range(n_labels):
        clf = LogisticRegression(max_iter=1000, C=10)
        clf.fit(Xtf, y_tr[:, j])
        p_tfidf[:, j] = clf.predict(Xtf_te)
    rec = {"tfidf_micro_f1": micro_f1(y_te, p_tfidf)}
    record("goemo_tfidf", **rec)
    print(f"TF-IDF+LogReg (binary rel): micro-F1={rec['tfidf_micro_f1']:.4f}")

    try:
        import lightgbm as lgb

        p_lgb = np.zeros_like(y_te)
        for j in range(n_labels):
            m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, num_threads=1, random_state=0)
            m.fit(h[: len(X_tr)], y_tr[:, j])
            p_lgb[:, j] = m.predict(h[len(X_tr) :])
        rec = {"lgb_on_h_micro_f1": micro_f1(y_te, p_lgb)}
        record("goemo_lgb", **rec)
        print(f"LightGBM on h (binary rel): micro-F1={rec['lgb_on_h_micro_f1']:.4f}")
    except Exception as e:
        print(f"LightGBM failed: {e}")

    return rec


if __name__ == "__main__":
    main()
