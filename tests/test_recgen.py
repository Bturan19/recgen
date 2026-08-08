import numpy as np
import pytest

from recgen import ClassificationHead, RegressionHead, TemplateVerbalizer
from recgen.cache import EmbeddingCache
from recgen.ranking import CatalogRankingHead


def test_classification_head():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] > 0).astype(int)
    head = ClassificationHead(epochs=30, hidden=(64, 32))
    head.fit(X[:240], y[:240])
    acc = (head.predict(X[240:]) == y[240:]).mean()
    assert acc > 0.85


def test_regression_head():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = X[:, 0] * 2 + X[:, 1]
    head = RegressionHead(epochs=30, hidden=(64, 32))
    head.fit(X[:240], y[:240])
    err = np.abs(head.predict(X[240:]) - y[240:]).mean()
    assert err < 1.0


def test_ranking_head():
    rng = np.random.default_rng(0)
    d, n_items, n_users = 16, 50, 400
    E = rng.normal(size=(n_items, d))
    W = rng.normal(size=(d, d))
    H = rng.normal(size=(n_users, d))
    logits = H @ E.T
    y = logits.argmax(axis=1)
    head = CatalogRankingHead(dim=d, epochs=10, batch_size=64, patience=3)
    head.fit(H[:320], y[:320], E)
    res = head.evaluate(H[320:], y[320:])

    assert res["recall@10"] > 0.25


def test_template_verbalizer():
    verb = TemplateVerbalizer(fields=["a", "b"], instruction="Task.")
    verb.fit([{"a": 1, "b": "x"}])
    out = verb.transform_rows([{"a": 2.0, "b": None}])
    assert out == ["Task.\na: 2 | b: unknown"]


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npy"))
    texts = ["one", "two"]
    H = np.zeros((2, 3))
    cache.store(texts, H)
    loaded = cache.load(texts)
    assert loaded.shape == (2, 3)
    assert cache.load(["other"]) is None
