import argparse

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from recgen import FrozenEncoder, RecgenPipeline, TemplateVerbalizer

from common import lgbm_baseline, lgbm_on_embeddings, record, run_pipeline

MODEL_DIR = "models/SmolLM2-360M"
CACHE_DIR = ".cache/imdb"

INSTRUCTION = "Given a movie review, predict whether the sentiment is positive or negative."


def load_imdb(n_train: int = 12000, n_test: int = 3000):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas().sample(n_train + n_test, random_state=0)
    y = (df["label"] > 0.5).astype(int)
    return df["text"].tolist(), y.to_numpy()


def main(pooling: str = "last"):
    texts, y = load_imdb()
    print(f"IMDB: {len(texts)} reviews, positive rate = {y.mean():.3f}")

    verbalizer = TemplateVerbalizer(instruction=INSTRUCTION)
    encoder = FrozenEncoder(MODEL_DIR, pooling=pooling, batch_size=32, max_length=512)
    pipe = RecgenPipeline(encoder, verbalizer, head="classifier", cache_dir=CACHE_DIR)
    res = run_pipeline(pipe, texts, y, f"imdb_llmhead_{pooling}")
    print(f"LLM-head ({pooling}): acc={res['acc']:.4f} auc={res['auc']:.4f}")

    H = pipe.transform(texts)
    stack = lgbm_on_embeddings(H, y)
    record(f"imdb_lgbm_on_h_{pooling}", acc=stack["acc"], auc=stack["auc"])
    print(f"LightGBM on h ({pooling}): acc={stack['acc']:.4f} auc={stack['auc']:.4f}")

    Xtr, Xva, ytr, yva = train_test_split(texts, y, test_size=0.2, random_state=0, stratify=y)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    Xtf = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=1000, C=10)
    clf.fit(Xtf, ytr)
    p = clf.predict(vec.transform(Xva))
    proba = clf.predict_proba(vec.transform(Xva))[:, 1]
    tfidf_acc = accuracy_score(yva, p)
    tfidf_auc = roc_auc_score(yva, proba)
    record("imdb_tfidf_logreg", acc=tfidf_acc, auc=tfidf_auc)
    print(f"TF-IDF+LogReg: acc={tfidf_acc:.4f} auc={tfidf_auc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = ap.parse_args()
    main(args.pooling)
