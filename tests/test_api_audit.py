"""API serving audit tests (AUDIT.md section 6): ranking order must be
descending (most similar first), catalog cache key must include model+pooling
(no stale embeddings across backend switches), lazy model load."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from api import config  # noqa: E402
from api.serving import EncoderService  # noqa: E402


class StubEncoder:
    def __init__(self):
        self.dim = 4
        self.device = "cpu"

    def encode(self, texts, progress=False):
        rng = np.random.default_rng(sum(ord(c) for t in texts for c in t) % 2**31)
        return rng.normal(size=(len(texts), self.dim)).astype(np.float32)

    @property
    def pooling(self):
        return config.POOLING


def make_service(monkeypatch, tmp_path):
    svc = EncoderService()
    svc._get_encoder = lambda: StubEncoder()
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    return svc


def test_rank_all_descending_order(monkeypatch, tmp_path):
    svc = make_service(monkeypatch, tmp_path)
    items = ["a", "b", "c"]
    embs = np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 1.0]])
    out = svc.rank_catalog("x", items, embs, top_k=3)
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True), "rankings must be descending"


def test_rank_descending_order(monkeypatch, tmp_path):
    svc = make_service(monkeypatch, tmp_path)
    ctx = "the user likes rock music"
    items = ["punk rock album", "classical piano", "jazz trumpet"]
    out = svc.rank(ctx, items, top_k=3)
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_cache_key_includes_model_and_pooling(monkeypatch, tmp_path):
    svc = make_service(monkeypatch, tmp_path)
    texts = ["hello world"] * 2
    svc.encode(texts, cache_key="catalog_default")
    files = [f for f in os.listdir(tmp_path) if f.endswith(".npy")]
    assert len(files) == 1
    assert "SmolLM2-360M" in files[0] and "mean" in files[0], f"cache key {files[0]} lacks backend identity"


def test_cache_reused_and_correct(monkeypatch, tmp_path):
    svc = make_service(monkeypatch, tmp_path)
    a = np.asarray(svc.encode(["same text", "same text"], cache_key="k"))
    b = np.asarray(svc.encode(["same text", "same text"], cache_key="k"))
    np.testing.assert_allclose(a, b)
    c = np.asarray(svc.encode(["other text", "same text"], cache_key="k"))
    assert not np.allclose(a, c)


def test_health_before_model_load(monkeypatch):
    svc = EncoderService()
    st = svc.status()
    assert st["dim"] is None, "model must load lazily"
