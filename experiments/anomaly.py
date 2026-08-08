import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from recgen import ClassificationHead, FrozenEncoder, RecgenPipeline, TemplateVerbalizer

from common import record

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/anomaly"


def knn_anomaly_score(H, k=5):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(H)))
    nn.fit(H)
    dists, _ = nn.kneighbors(H)
    return dists[:, -1]


def evaluate(name, H, y, unsupervised=True):
    aucs = {}
    if unsupervised:
        for k in (3, 5, 10):
            s = knn_anomaly_score(H, k=k)
            aucs[f"knn{k}_auc"] = roc_auc_score(y, s)
        iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=1)
        iso.fit(H)
        aucs["isoforest_auc"] = roc_auc_score(y, -iso.score_samples(H))
        record(f"{name}_unsup", **aucs)
        print(f"  {name} unsupervised: " + " ".join(f"{k}={v:.4f}" for k, v in aucs.items()))
    else:
        Xtr, Xva, ytr, yva = train_test_split(H, y, test_size=0.3, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        aucs["logreg_auc"] = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        head = ClassificationHead(epochs=25, hidden=(64, 32))
        head.fit(Xtr, ytr)
        aucs["head_auc"] = roc_auc_score(yva, head.predict_proba(Xva)[:, 1])
        record(f"{name}_supervised", **aucs)
        print(f"  {name} supervised: " + " ".join(f"{k}={v:.4f}" for k, v in aucs.items()))


def sms_spam():
    import urllib.request
    import zipfile

    path = "data/smsspam.zip"
    if not os.path.exists("data/SMSSpamCollection"):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip", path
        )
        with zipfile.ZipFile(path) as z:
            z.extractall("data")
    texts, labels = [], []
    for line in open("data/SMSSpamCollection", encoding="latin-1"):
        lab, _, msg = line.strip().partition("\t")
        texts.append(msg)
        labels.append(1 if lab == "spam" else 0)
    y = np.array(labels)
    print(f"SMS spam: {len(texts)} messages, {y.mean():.3f} spam rate")
    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=256)
    H = encoder.encode_cached(texts, f"{CACHE_DIR}/sms_emb.npy")
    evaluate("sms_spam", H, y)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    Xtf = vec.fit_transform(texts).astype(np.float32).toarray()
    evaluate("sms_tfidf", Xtf, y)
    evaluate("sms_tfidf_sup", Xtf, y, unsupervised=False)
    evaluate("sms_spam_sup", H, y, unsupervised=False)


def cardio():
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(data_id=1560, as_frame=False, return_X_y=True, parser="pandas")
    X = X.astype(np.float32)
    y = (np.asarray(y) == "3").astype(int)
    print(f"cardiotocography: {X.shape}, {y.mean():.3f} pathological rate")

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    evaluate("cardio_raw", Xs, y)

    texts = [
        TemplateVerbalizer(instruction="Describe this patient's fetal health measurements.")
        .fit({"X": X[i]}).transform_rows([{"X": X[i]}])[0]
        for i in range(len(X))
    ]
    encoder = FrozenEncoder(MODEL_DIR, pooling="mean", batch_size=32, max_length=128)
    H = encoder.encode_cached(texts, f"{CACHE_DIR}/cardio_emb.npy")
    evaluate("cardio_llm", H, y)
    evaluate("cardio_llm_sup", H, y, unsupervised=False)

    Hc = np.concatenate([Xs, H], axis=1)
    evaluate("cardio_raw_plus_llm", Hc, y)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sms"
    if which == "sms":
        sms_spam()
    else:
        cardio()
